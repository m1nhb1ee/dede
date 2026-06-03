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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from ver2.stage1.src.config import CKPT_DIR, PRED_DIR
except ImportError:  # flat layout (e.g., Kaggle /kaggle/working/stage1/)
    from config import CKPT_DIR, PRED_DIR


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

    # Load weights from the checkpoint
    ckpt_path = Path(ckpt_dir)
    # Trainer saves model under "pytorch_model.bin" or "model.safetensors";
    # let HF load it via from_pretrained-style of the wrapper.
    # We just load the state dict here.
    state_paths = [ckpt_path / "model.safetensors", ckpt_path / "pytorch_model.bin"]
    state_path = next((p for p in state_paths if p.exists()), None)
    if state_path is None:
        raise FileNotFoundError(f"No weights file in {ckpt_path}")
    print(f"Loading weights: {state_path}")
    if state_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(state_path))
    else:
        state = torch.load(str(state_path), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing keys: {len(missing)}, unexpected: {len(unexpected)}")

    ds = build_or_load_tokenized(tok, tok_name)
    train_ds, val_ds, test_ds = make_fold_splits(ds, fold=None)
    targets = {"train": train_ds, "val": val_ds, "test": test_ds}
    ds_pred = targets[split]
    print(f"Predicting {split}: {len(ds_pred):,} rows")

    args = TrainingArguments(
        output_dir="/tmp/_reinfer",
        per_device_eval_batch_size=32,
        fp16=True, report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        group_by_length=False,   # MUST stay False: True reorders eval output
                                 # and misaligns p_text with id/label.
    )
    trainer = Trainer(
        model=model, args=args,
        data_collator=PadCollator(tok),
        compute_metrics=compute_metrics_binary,
    )
    pred = trainer.predict(ds_pred)
    logits = pred.predictions
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
    out_path = PRED_DIR / f"reinfer_{split}_{ckpt_path.name}.parquet"
    out.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Wrote {out_path}")
    print(f"Metrics: {pred.metrics}")
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
    p.add_argument("--ckpt",          type=str, default=None,
                   help="Checkpoint directory for --reinfer")
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

    if not any([args.assemble_oof, args.assemble_test, args.reinfer]):
        p.print_help()


if __name__ == "__main__":
    main()
