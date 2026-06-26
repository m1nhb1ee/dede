"""Stage 1 prediction utilities + OOF assembly.

This script does NOT re-train. It either:

  (a) Aggregates per-fold OOF parquet files into a single file
      p_text_oof_train.parquet, used by Stage 2 as the `p_text` feature.

  (b) Averages test-set predictions across all folds (per-fold predictions
      saved during training) into p_text_test_ensemble.parquet -- a cheap
      ensemble that usually beats any single fold.

  (c) Re-runs inference from an existing checkpoint on an arbitrary split
      (useful if you change splits without retraining).

Examples:
    # After training all 5 folds:
    python predict.py --assemble_oof
    python predict.py --assemble_test

    # Re-predict using the "full" checkpoint:
    python predict.py --reinfer --ckpt outputs/checkpoints/full --split test

    # Regenerate a missing fold's OOF + test predictions from saved weights
    # (e.g. the lean fold_0_trainable.safetensors). Produces the canonical
    # p_text_oof_fold0.parquet + p_text_test_fold0.parquet that --assemble_*
    # expect:
    python predict.py --reinfer_fold --fold 0 \
        --ckpt fold_0_trainable.safetensors --model_dir <backbone_dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from ver2.stage1.src.config import CKPT_DIR, HW, PRED_DIR
except ImportError:  # flat layout (e.g., Kaggle /kaggle/working/stage1/)
    from config import CKPT_DIR, HW, PRED_DIR


# ─────────────────────────────────────────────────────────────────────────────
# OOF assembly: gather p_text_oof_fold{0..N-1}.parquet -> single train file
# ─────────────────────────────────────────────────────────────────────────────


def assemble_oof(n_folds: int = 3) -> Path:
    """Concatenate per-fold OOF predictions into one parquet keyed by id.

    Each fold's parquet contains predictions ONLY for the validation slice
    of that fold; together they cover every train row exactly once.
    """
    frames = []
    for k in range(n_folds):
        p = PRED_DIR / f"p_text_oof_fold{k}.parquet"
        if not p.exists():
            print(f"  [skip] {p.name} not found")
            continue
        df = pd.read_parquet(p)
        df["fold"] = k
        frames.append(df)
        print(f"  fold {k}: {len(df):,} rows")

    if not frames:
        raise FileNotFoundError("No fold prediction files found in PRED_DIR")

    out = pd.concat(frames, ignore_index=True)
    # Sanity: each id should appear exactly once
    n_dup = int(out["id"].duplicated().sum())
    if n_dup:
        print(f"  [WARN] {n_dup:,} duplicate ids across folds (overlap?)")
    print(f"  total OOF rows: {len(out):,}")

    out_path = PRED_DIR / "p_text_oof_train.parquet"
    out.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"  wrote {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Test ensemble: average per-fold test predictions
# ─────────────────────────────────────────────────────────────────────────────


def assemble_test_ensemble(n_folds: int = 3) -> Path:
    """Average test-set p_text predictions across folds (simple mean ensemble).

    This is the standard cheap ensemble: each fold sees a different train
    subset, predictions on the held-out test set get averaged for variance
    reduction.
    """
    frames = []
    for k in range(n_folds):
        p = PRED_DIR / f"p_text_test_fold{k}.parquet"
        if not p.exists():
            print(f"  [skip] {p.name} not found")
            continue
        df = pd.read_parquet(p).rename(columns={"p_text": f"p_text_fold{k}"})
        frames.append(df.drop(columns=["label"], errors="ignore"))
        print(f"  fold {k}: {len(df):,} rows")

    if not frames:
        raise FileNotFoundError("No fold TEST prediction files found")

    # Merge on id (all should have same id list -> outer for safety)
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="id", how="outer")
    cols = [c for c in merged.columns if c.startswith("p_text_fold")]
    merged["p_text"] = merged[cols].mean(axis=1, skipna=True).astype(np.float32)
    out = merged[["id", "p_text"]]

    out_path = PRED_DIR / "p_text_test_ensemble.parquet"
    out.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"  wrote {out_path}  ({len(out):,} rows, "
          f"avg over {len(cols)} folds)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Re-inference from a saved checkpoint
# ─────────────────────────────────────────────────────────────────────────────


def _load_weights(model, weights_path: str) -> None:
    """Load a state dict into `model` (strict=False) from either a checkpoint
    directory (model.safetensors / pytorch_model.bin) or a single .safetensors
    file (e.g. the lean fold_K_trainable.safetensors = LoRA + head only)."""
    p = Path(weights_path)
    if p.is_dir():
        cands = [p / "model.safetensors", p / "pytorch_model.bin"]
        state_path = next((c for c in cands if c.exists()), None)
        if state_path is None:
            raise FileNotFoundError(f"No weights file in {p}")
    else:
        if not p.exists():
            raise FileNotFoundError(f"Weights file not found: {p}")
        state_path = p

    print(f"Loading weights: {state_path}")
    if state_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(state_path))
    else:
        state = torch.load(str(state_path), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # With a LEAN checkpoint the frozen backbone keys are expected to be
    # "missing" (they come from --model_dir), so only flag head/LoRA gaps.
    trained_missing = [k for k in missing
                       if ("lora_" in k) or k.startswith("head.")]
    print(f"  loaded {len(state)} tensors | missing(total)={len(missing)} "
          f"missing(lora/head)={len(trained_missing)} unexpected={len(unexpected)}")
    if trained_missing:
        raise RuntimeError(
            f"Trained weights missing from checkpoint: {trained_missing[:8]}"
            f"{' ...' if len(trained_missing) > 8 else ''}. Wrong file?"
        )


def reinfer_fold(weights_path: str, fold: int, model_dir: str | None = None) -> dict:
    """Regenerate a single fold's OOF + test predictions from saved weights.

    Mirrors the predict phase of train.train_one for --mode fold, but loads
    weights instead of training. Writes the canonical filenames that
    assemble_oof / assemble_test_ensemble consume:
        p_text_oof_fold{fold}.parquet   (held-out fold slice of train)
        p_text_test_fold{fold}.parquet  (full test set)
    """
    from transformers import Trainer, TrainingArguments
    from transformers.trainer_utils import EvalPrediction

    try:
        from ver2.stage1.src.data import (
            PadCollator, build_or_load_tokenized, make_fold_splits,
        )
        from ver2.stage1.src.model import MentalRoBERTaWithCustomHead
        from ver2.stage1.src.utils import (
            compute_metrics_binary, load_backbone, load_tokenizer,
        )
    except ImportError:  # flat layout (e.g., Kaggle /kaggle/working/stage1/)
        from data import (
            PadCollator, build_or_load_tokenized, make_fold_splits,
        )
        from model import MentalRoBERTaWithCustomHead
        from utils import (
            compute_metrics_binary, load_backbone, load_tokenizer,
        )

    tok, tok_name = load_tokenizer(model_dir)
    backbone, _   = load_backbone(model_dir)
    model = MentalRoBERTaWithCustomHead(backbone, use_lora=True, num_classes=1)
    _load_weights(model, weights_path)

    ds = build_or_load_tokenized(tok, tok_name)
    # fold-aware split: val = train rows where fold==k (the OOF slice), test = test
    _, val_full, test_ds = make_fold_splits(ds, fold=fold)
    print(f"Fold {fold}: oof(val) rows={len(val_full):,}  test rows={len(test_ds):,}")

    args = TrainingArguments(
        output_dir="/tmp/_reinfer_fold",
        # Match train.py's proven predict config (batch 8, not 32): the custom
        # head's output_hidden_states + layer-stack is VRAM-heavy, and batch 32
        # x 512 across 2-GPU DataParallel can thrash/OOM into a hang on T4 16GB.
        per_device_eval_batch_size=HW["per_device_batch"],
        fp16=HW["fp16"],
        report_to=[],
        dataloader_num_workers=HW["dataloader_num_workers"],
        remove_unused_columns=False,
        group_by_length=False,   # MUST stay False: True reorders eval output
                                 # and misaligns p_text with id/label.
    )
    trainer = Trainer(
        model=model, args=args,
        data_collator=PadCollator(tok),
    )

    def _predict_and_save(ds_pred, out_path: Path, name: str) -> dict:
        chunk_size = 10_000
        n_rows = len(ds_pred)
        n_chunks = (n_rows + chunk_size - 1) // chunk_size
        print(f"  [{name}] predicting {len(ds_pred):,} rows "
              f"(batch {HW['per_device_batch']}/device, "
              f"chunk {chunk_size:,} rows)...", flush=True)

        logits_chunks = []
        t_start = time.perf_counter()
        for i, start in enumerate(range(0, n_rows, chunk_size), start=1):
            end = min(start + chunk_size, n_rows)
            t_chunk = time.perf_counter()
            print(
                f"  [{name}] chunk {i}/{n_chunks} start "
                f"rows {start:,}-{end - 1:,}",
                flush=True,
            )
            chunk = ds_pred.select(range(start, end))
            pred = trainer.predict(chunk)
            logits_chunks.append(pred.predictions)
            elapsed = time.perf_counter() - t_start
            chunk_elapsed = time.perf_counter() - t_chunk
            rows_done = end
            rows_per_sec = rows_done / max(elapsed, 1e-9)
            remaining = (n_rows - rows_done) / max(rows_per_sec, 1e-9)
            print(
                f"  [{name}] chunk {i}/{n_chunks} done "
                f"rows {start:,}-{end - 1:,} "
                f"({rows_done:,}/{n_rows:,})  "
                f"chunk={chunk_elapsed/60:.1f}m  "
                f"elapsed={elapsed/60:.1f}m  eta={remaining/60:.1f}m",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logits = np.concatenate(logits_chunks, axis=0)
        if logits.ndim == 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        labels_arr = np.asarray(ds_pred["labels"], dtype=np.int8)
        metrics = compute_metrics_binary(
            EvalPrediction(predictions=logits, label_ids=labels_arr)
        )
        if labels_arr.min() != labels_arr.max():
            gap = probs[labels_arr == 1].mean() - probs[labels_arr == 0].mean()
            print(f"  [{name}] sanity: mean p_text(pos)-(neg) = {gap:+.4f}")
            if gap < 0.05:
                raise RuntimeError(
                    f"[{name}] p_text appears MISALIGNED with labels "
                    f"(gap={gap:+.4f}). NOT saving."
                )
        out = pd.DataFrame({
            "id":     list(ds_pred["id"]),
            "p_text": probs.astype(np.float32),
            "label":  labels_arr,
        })
        out.to_parquet(out_path, engine="pyarrow",
                       compression="snappy", index=False)
        print(f"  [{name}] wrote {out_path}  ({len(out):,} rows)  "
              f"elapsed={(time.perf_counter() - t_start)/60:.1f}m  "
              f"metrics={metrics}")
        return {"n": int(len(out)), "metrics": metrics}

    meta = {
        "oof":  _predict_and_save(
            val_full, PRED_DIR / f"p_text_oof_fold{fold}.parquet", f"oof_fold{fold}"),
        "test": _predict_and_save(
            test_ds, PRED_DIR / f"p_text_test_fold{fold}.parquet", f"test_fold{fold}"),
    }
    return meta


def reinfer(ckpt_dir: str, split: str, model_dir: str | None = None) -> Path:
    """Load a saved Trainer checkpoint and re-predict a given split.

    `split` ∈ {"train", "val", "test"}. Uses the time-based split definitions
    from preprocess/splits.parquet.
    """
    from transformers import Trainer, TrainingArguments

    try:
        from ver2.stage1.src.data import (
            PadCollator, build_or_load_tokenized, make_fold_splits,
        )
        from ver2.stage1.src.model import MentalRoBERTaWithCustomHead
        from ver2.stage1.src.utils import (
            compute_metrics_binary, load_backbone, load_tokenizer,
        )
    except ImportError:  # flat layout (e.g., Kaggle /kaggle/working/stage1/)
        from data import (
            PadCollator, build_or_load_tokenized, make_fold_splits,
        )
        from model import MentalRoBERTaWithCustomHead
        from utils import (
            compute_metrics_binary, load_backbone, load_tokenizer,
        )

    tok, tok_name = load_tokenizer(model_dir)
    backbone, _   = load_backbone(model_dir)
    model = MentalRoBERTaWithCustomHead(backbone, use_lora=True, num_classes=1)
    # Handles both a Trainer checkpoint dir and a lean LoRA+head .safetensors
    # file (e.g. full_trainable.safetensors); validates the trained weights
    # are present rather than silently loading a partial state dict.
    _load_weights(model, ckpt_dir)

    ds = build_or_load_tokenized(tok, tok_name)
    train_ds, val_ds, test_ds = make_fold_splits(ds, fold=None)
    targets = {"train": train_ds, "val": val_ds, "test": test_ds}
    ds_pred = targets[split]
    print(f"Predicting {split}: {len(ds_pred):,} rows")

    args = TrainingArguments(
        output_dir="/tmp/_reinfer",
        # Match the proven train/predict batch (not 32): the custom head's
        # output_hidden_states + layer-stack is VRAM-heavy and batch 32 x 512
        # can OOM on a 6-16GB GPU.
        per_device_eval_batch_size=HW["per_device_batch"],
        fp16=HW["fp16"], report_to=[],
        dataloader_num_workers=HW["dataloader_num_workers"],
        remove_unused_columns=False,
        group_by_length=False,   # MUST stay False: True reorders eval output
                                 # and misaligns p_text with id/label.
    )
    trainer = Trainer(
        model=model, args=args,
        data_collator=PadCollator(tok),
        compute_metrics=compute_metrics_binary,
    )

    # Chunked prediction bounds peak memory on large splits (val 442K / test
    # 213K) and lets a single-GPU box stream through without holding it all.
    chunk_size = 10_000
    n_rows = len(ds_pred)
    n_chunks = (n_rows + chunk_size - 1) // chunk_size
    logits_chunks = []
    for i, start in enumerate(range(0, n_rows, chunk_size), start=1):
        end = min(start + chunk_size, n_rows)
        print(f"  chunk {i}/{n_chunks} rows {start:,}-{end - 1:,}", flush=True)
        logits_chunks.append(trainer.predict(ds_pred.select(range(start, end))).predictions)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logits = np.concatenate(logits_chunks, axis=0)
    if logits.ndim == 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))

    labels_arr = np.asarray(ds_pred["labels"], dtype=np.int8)
    if labels_arr.min() != labels_arr.max():
        gap = probs[labels_arr == 1].mean() - probs[labels_arr == 0].mean()
        print(f"  sanity: mean p_text(pos)-(neg) = {gap:+.4f}")
        if gap < 0.05:
            raise RuntimeError(
                f"p_text appears MISALIGNED with labels (gap={gap:+.4f}). "
                f"Check group_by_length. NOT saving."
            )

    out = pd.DataFrame({
        "id":     list(ds_pred["id"]),
        "p_text": probs.astype(np.float32),
        "label":  labels_arr,
    })
    # Canonical full-mode names consumed by Stage 2 (tech_plan 5.8):
    # p_text_val.parquet / p_text_test.parquet.
    out_path = PRED_DIR / f"p_text_{split}.parquet"
    out.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Wrote {out_path}  ({len(out):,} rows)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1 prediction utilities")
    p.add_argument("--assemble_oof",  action="store_true",
                   help="Concat per-fold OOF predictions into one train file")
    p.add_argument("--assemble_test", action="store_true",
                   help="Average per-fold test predictions into ensemble")
    p.add_argument("--reinfer",       action="store_true",
                   help="Re-run inference from --ckpt on --split")
    p.add_argument("--reinfer_fold",  action="store_true",
                   help="Regenerate fold --fold OOF + test preds from --ckpt")
    p.add_argument("--fold",          type=int, default=None,
                   help="Fold index for --reinfer_fold")
    p.add_argument("--ckpt",          type=str, default=None,
                   help="Checkpoint dir OR lean .safetensors file for --reinfer*")
    p.add_argument("--split",         type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--model_dir",     type=str, default=None,
                   help="Local dir to backbone weights (Kaggle dataset path)")
    p.add_argument("--n_folds",       type=int, default=3)
    args = p.parse_args()

    if args.assemble_oof:  assemble_oof(args.n_folds)
    if args.assemble_test: assemble_test_ensemble(args.n_folds)
    if args.reinfer:
        if not args.ckpt:
            raise SystemExit("--ckpt required with --reinfer")
        reinfer(args.ckpt, args.split, args.model_dir)
    if args.reinfer_fold:
        if not args.ckpt:
            raise SystemExit("--ckpt required with --reinfer_fold")
        if args.fold is None:
            raise SystemExit("--fold required with --reinfer_fold")
        reinfer_fold(args.ckpt, args.fold, args.model_dir)

    if not any([args.assemble_oof, args.assemble_test,
                args.reinfer, args.reinfer_fold]):
        p.print_help()


if __name__ == "__main__":
    main()
