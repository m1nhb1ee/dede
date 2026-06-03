"""Stage 1 training entry point.

Usage:
    # Train on time-train set, validate on time-val, predict test (single run)
    python train.py --mode full

    # 3-fold OOF: hold out fold k as validation, train on the other 2 folds.
    # Save predictions on fold k for OOF stacking.
    python train.py --mode fold --fold 0
    python train.py --mode fold --fold 1
    python train.py --mode fold --fold 2

    # Quick local sanity check on a 100K subset:
    STAGE1_PROFILE=local python train.py --mode fold --fold 0 --subsample 100000

Outputs:
    outputs/checkpoints/{full | fold_{k}}/  -- best checkpoint by val f1_macro
    outputs/predictions/p_text_{full_val,full_test,oof_fold{k}}.parquet
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
    from ver2.stage1.src.config import (
        CKPT_DIR, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_THRESHOLD,
        EVAL_STEPS, EVAL_SUBSET_SIZE, HW, LR_HEAD, LR_LORA, LR_SCHEDULER,
        METRIC_FOR_BEST, METRIC_GREATER_IS_BETTER, NUM_EPOCHS, PRED_DIR,
        PROFILE_NAME, QUICK_EVAL_SIZE, QUICK_LOG_STEPS, SAVE_TOTAL_LIMIT,
        SEED, USE_POS_WEIGHT, WARMUP_RATIO, WEIGHT_DECAY,
    )
    from ver2.stage1.src.data import (
        PadCollator, build_or_load_tokenized, compute_pos_weight, make_fold_splits,
    )
    from ver2.stage1.src.model import MentalRoBERTaWithCustomHead
    from ver2.stage1.src.utils import (
        compute_metrics_binary, load_backbone, load_tokenizer, n_gpus, section,
        set_seed, step,
    )
except ImportError:  # flat layout (e.g., Kaggle /kaggle/working/stage1/)
    from config import (
        CKPT_DIR, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_THRESHOLD,
        EVAL_STEPS, EVAL_SUBSET_SIZE, HW, LR_HEAD, LR_LORA, LR_SCHEDULER,
        METRIC_FOR_BEST, METRIC_GREATER_IS_BETTER, NUM_EPOCHS, PRED_DIR,
        PROFILE_NAME, QUICK_EVAL_SIZE, QUICK_LOG_STEPS, SAVE_TOTAL_LIMIT,
        SEED, USE_POS_WEIGHT, WARMUP_RATIO, WEIGHT_DECAY,
    )
    from data import (
        PadCollator, build_or_load_tokenized, compute_pos_weight, make_fold_splits,
    )
    from model import MentalRoBERTaWithCustomHead
    from utils import (
        compute_metrics_binary, load_backbone, load_tokenizer, n_gpus, section,
        set_seed, step,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 training")
    p.add_argument("--mode", choices=["fold", "full"], required=True)
    p.add_argument("--fold", type=int, default=None,
                   help="Fold index 0..4 (required if --mode=fold)")
    p.add_argument("--subsample", type=int, default=None,
                   help="If set, randomly subsample TRAIN to this many rows "
                        "(for quick sanity checks). Val/test kept full.")
    p.add_argument("--model_dir", type=str, default=None,
                   help="Local dir to pre-downloaded MentalRoBERTa (Kaggle Dataset path).")
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--force_retokenize", action="store_true")
    # Resume support: HuggingFace Trainer can pick up from a checkpoint dir.
    # "auto" -> resume from the latest checkpoint in output_dir if any.
    p.add_argument("--resume", type=str, default=None,
                   help='"auto" to resume from latest checkpoint in the run dir, '
                        'or a specific checkpoint path.')
    p.add_argument("--eval_steps", type=int, default=EVAL_STEPS,
                   help="Override default eval cadence.")
    p.add_argument("--undersample_neg", type=float, default=1.0,
                   help="Keep this fraction of negatives in TRAIN only "
                        "(val/test unchanged). 1.0 = no undersampling, "
                        "0.5 = keep 50%% of negatives. Per-fold independent "
                        "sampling with seed=SEED+fold.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training routine
# ─────────────────────────────────────────────────────────────────────────────


def train_one(args: argparse.Namespace) -> dict:
    set_seed(SEED + (args.fold or 0))

    section(f"Stage 1 training | mode={args.mode}  fold={args.fold}  "
            f"profile={PROFILE_NAME}  n_gpus={n_gpus()}")

    # ── Tokenizer + backbone ────────────────────────────────────────────
    tok, tok_name      = load_tokenizer(args.model_dir)
    backbone, bb_name  = load_backbone(args.model_dir)

    # ── Dataset (pre-tokenized + cached) ────────────────────────────────
    ds = build_or_load_tokenized(tok, tok_name, force_rebuild=args.force_retokenize)
    train_ds, val_ds, test_ds = make_fold_splits(ds, fold=args.fold if args.mode == "fold" else None)
    step(f"Split sizes: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    # Optional sanity-check subsample on TRAIN only
    if args.subsample and args.subsample < len(train_ds):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(train_ds), size=args.subsample, replace=False)
        train_ds = train_ds.select(idx)
        step(f"Subsampled train to {len(train_ds):,}")

    # Optional negative-class undersampling on TRAIN only (val/test untouched).
    # Per-fold independent sample (seed = SEED + fold) so the union of negatives
    # used across the N folds still covers ~1 - keep_frac^(N-1) of all negatives.
    if args.undersample_neg < 1.0:
        rng = np.random.default_rng(SEED + (args.fold or 0))
        labels = np.asarray(train_ds["labels"], dtype=np.float32)
        pos_idx = np.where(labels > 0.5)[0]
        neg_idx = np.where(labels <= 0.5)[0]
        n_keep_neg = int(len(neg_idx) * args.undersample_neg)
        keep_neg = rng.choice(neg_idx, size=n_keep_neg, replace=False)
        new_idx = np.sort(np.concatenate([pos_idx, keep_neg]))
        train_ds = train_ds.select(new_idx)
        step(f"Undersampled neg to {args.undersample_neg:.0%}: "
             f"kept {len(pos_idx):,} pos + {n_keep_neg:,} neg "
             f"= {len(train_ds):,} total")

    # ── Stratified val subsets for fast intermediate eval + quick monitor ─
    # Three datasets carved from val_full (the held-out fold):
    #   val_full         : 605K rows, used for final OOF predict
    #   val_ds (subset)  : 60K stratified rows, used by Trainer for early stop
    #   quick_val_ds     : 5K stratified rows, used by QuickEvalCallback every
    #                      QUICK_LOG_STEPS — DISJOINT from val_ds so the two
    #                      metric streams are independent.
    val_full = val_ds  # keep reference for final predict
    val_labels_full = np.asarray(val_full["labels"], dtype=np.float32)
    val_pos_all = np.where(val_labels_full > 0.5)[0]
    val_neg_all = np.where(val_labels_full <= 0.5)[0]
    pos_rate = len(val_pos_all) / len(val_full)
    rng = np.random.default_rng(SEED)

    # --- 1) Eval subset (for Trainer) ---
    if EVAL_SUBSET_SIZE and EVAL_SUBSET_SIZE < len(val_full):
        n_eval_pos = int(round(EVAL_SUBSET_SIZE * pos_rate))
        n_eval_neg = EVAL_SUBSET_SIZE - n_eval_pos
        eval_pos = rng.choice(val_pos_all, size=min(n_eval_pos, len(val_pos_all)), replace=False)
        eval_neg = rng.choice(val_neg_all, size=min(n_eval_neg, len(val_neg_all)), replace=False)
        eval_sub_idx = np.sort(np.concatenate([eval_pos, eval_neg]))
        val_ds = val_full.select(eval_sub_idx)
        eval_pos_set = set(eval_pos.tolist())
        eval_neg_set = set(eval_neg.tolist())
        step(f"Eval subset (stratified, for early stop): {len(val_ds):,} rows "
             f"(pos_rate={len(eval_pos)/len(val_ds):.3f})")
    else:
        eval_pos_set, eval_neg_set = set(), set()
        step(f"Eval subset: using full val ({len(val_full):,} rows)")

    # --- 2) Quick subset (for monitoring) -- disjoint from eval subset ---
    remaining_pos = np.array([i for i in val_pos_all if i not in eval_pos_set])
    remaining_neg = np.array([i for i in val_neg_all if i not in eval_neg_set])
    quick_size = min(QUICK_EVAL_SIZE, len(remaining_pos) + len(remaining_neg))
    n_quick_pos = int(round(quick_size * pos_rate))
    n_quick_neg = quick_size - n_quick_pos
    quick_pos = rng.choice(remaining_pos, size=min(n_quick_pos, len(remaining_pos)), replace=False)
    quick_neg = rng.choice(remaining_neg, size=min(n_quick_neg, len(remaining_neg)), replace=False)
    quick_idx = np.sort(np.concatenate([quick_pos, quick_neg]))
    quick_val_ds = val_full.select(quick_idx)
    # Sanity: confirm disjoint
    overlap = (set(quick_idx.tolist()) & (eval_pos_set | eval_neg_set))
    assert not overlap, f"BUG: quick/eval subsets overlap by {len(overlap)} rows"
    step(f"Quick subset (stratified, for [QUICK] log): {len(quick_val_ds):,} rows "
         f"(pos_rate={len(quick_pos)/len(quick_val_ds):.3f}) "
         f"-- disjoint from eval subset")
    step(f"Full val ({len(val_full):,}) reserved for final OOF predict")

    # ── pos_weight from training fold (handle imbalance) ────────────────
    pos_w = compute_pos_weight(train_ds) if USE_POS_WEIGHT else None
    step(f"pos_weight = {pos_w:.4f}" if pos_w else "pos_weight = None")

    # ── Model ───────────────────────────────────────────────────────────
    model = MentalRoBERTaWithCustomHead(
        backbone=backbone, use_lora=True, num_classes=1, pos_weight=pos_w,
    )

    # ── Trainer ────────────────────────────────────────────────────────
    from transformers import (
        EarlyStoppingCallback, Trainer, TrainingArguments,
    )

    run_name = (f"fold_{args.fold}" if args.mode == "fold" else "full")
    out_dir  = CKPT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    args_tr = TrainingArguments(
        output_dir=str(out_dir),
        run_name=run_name,
        report_to=[],                                       # no W&B/HF Hub
        # batching + precision
        per_device_train_batch_size=HW["per_device_batch"],
        per_device_eval_batch_size= HW["per_device_batch"],  # = train batch (was *2, caused OOM)
        gradient_accumulation_steps=HW["grad_accum"],
        fp16=HW["fp16"],
        gradient_checkpointing=HW["gradient_checkpointing"],
        optim=HW["optim"],
        dataloader_num_workers=HW["dataloader_num_workers"],
        dataloader_pin_memory=HW.get("dataloader_pin_memory", False),
        # schedule
        num_train_epochs=args.epochs,
        learning_rate=LR_LORA,                              # default for non-head params
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        # eval / save
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST,
        greater_is_better=METRIC_GREATER_IS_BETTER,
        # logging
        logging_steps=QUICK_LOG_STEPS,
        disable_tqdm=True,                                  # subprocess capture spams tqdm
        seed=SEED,
        # group by length: batch samples of similar length together so padding
        # per-batch shrinks to the longest in that batch (not global max_len).
        # With max_len=512 but median ~150-200 tokens, this cuts wasted compute
        # on padding by ~25-35%. Trainer derives lengths from input_ids.
        group_by_length=True,
        # misc
        remove_unused_columns=False,
        # Kaggle has plenty of CPU RAM for the eval set; speed > memory
        eval_accumulation_steps=None,
    )

    # Discriminative LR via custom optimizer (LoRA: 5e-4, head: 1e-4).
    # Important: optimizer must reference model.parameters() BEFORE Trainer
    # wraps the model in DataParallel. The DP wrapper does NOT replace the
    # underlying parameter tensors; it only adds replication wrappers, so
    # parameter identity (and thus optimizer state) is preserved.
    param_groups = model.get_param_groups(
        lr_lora=LR_LORA, lr_head=LR_HEAD, weight_decay=WEIGHT_DECAY,
    )
    optimizer = torch.optim.AdamW(param_groups)

    from transformers import TrainerCallback

    class MetricPrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            parts = [f"step={state.global_step}"]
            for k, v in sorted(logs.items()):
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
            print("  [METRICS] " + "  ".join(parts), flush=True)

    # quick_val_ds was built above (stratified, disjoint from val_ds eval subset)

    # Holder so the callback can reach `trainer` after construction.
    trainer_ref: list = [None]

    class QuickEvalCallback(TrainerCallback):
        """Every QUICK_LOG_STEPS steps, run prediction on a small val subset
        and print F1 (macro/pos), recall (macro/pos), and eval loss. Cheap
        (~15s/call on 5K rows) compared to full eval at EVAL_STEPS cadence."""

        def __init__(self):
            self._last_logged = -1

        def on_step_end(self, args, state, control, **kwargs):
            gs = state.global_step
            if gs <= 0 or gs % QUICK_LOG_STEPS != 0 or gs == self._last_logged:
                return
            # Skip if this step coincides with the full eval (avoid duplicate work)
            if gs % args.eval_steps == 0:
                return
            self._last_logged = gs
            tr = trainer_ref[0]
            if tr is None:
                return
            # Free fragmented GPU cache before predict (OOM fix for DP broadcast)
            torch.cuda.empty_cache()
            pred = tr.predict(quick_val_ds, metric_key_prefix="quick")
            torch.cuda.empty_cache()  # also free after to release predict's peak
            m = pred.metrics or {}
            print(
                f"  [QUICK]   step={gs}  "
                f"loss={m.get('quick_loss', float('nan')):.4f}  "
                f"f1_macro={m.get('quick_f1_macro', float('nan')):.4f}  "
                f"f1_pos={m.get('quick_f1_pos', float('nan')):.4f}  "
                f"recall_pos={m.get('quick_recall_pos', float('nan')):.4f}  "
                f"recall_macro={m.get('quick_recall_macro', float('nan')):.4f}",
                flush=True,
            )

    # Trainer auto-detects multi-GPU and wraps in nn.DataParallel.
    # Confirmed: torch.cuda.device_count() determines this; we want all of them.
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tok),
        compute_metrics=compute_metrics_binary,
        optimizers=(optimizer, None),   # let Trainer build the scheduler
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            ),
            MetricPrinterCallback(),
            QuickEvalCallback(),
        ],
    )
    trainer_ref[0] = trainer

    # ── Train (with optional resume) ───────────────────────────────────
    t0 = time.perf_counter()
    resume_arg: bool | str | None = None
    if args.resume == "auto":
        resume_arg = True   # Trainer auto-finds latest checkpoint in out_dir
    elif args.resume:
        resume_arg = args.resume
    train_out = trainer.train(resume_from_checkpoint=resume_arg)
    train_time = time.perf_counter() - t0
    step(f"Training done in {train_time/60:.1f} min")

    # ── Final eval on subset (cheap; full-val metrics come from predict step) ─
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    step(f"VAL metrics (subset {len(val_ds):,}): "
         f"{ {k: round(v, 4) for k, v in val_metrics.items()} }")

    # ── Predict on splits & save ───────────────────────────────────────
    # CRITICAL: group_by_length reorders eval samples by length, so
    # pred.predictions comes back in length-sorted order while ds_pred["id"]
    # and ds_pred["labels"] are in original order -> p_text misaligned with
    # id/label (AUC collapses to ~0.5 even though the model is fine). Disable
    # it for prediction so the eval sampler is SequentialSampler (original
    # order). Training still uses group_by_length=True (harmless there).
    trainer.args.group_by_length = False

    def _predict_and_save(ds_pred, out_path: Path, name: str) -> dict:
        if len(ds_pred) == 0:
            step(f"  [{name}] empty -> skip")
            return {"n": 0}
        t1 = time.perf_counter()
        pred = trainer.predict(ds_pred)
        logits = pred.predictions
        if logits.ndim == 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))

        labels_arr = np.asarray(ds_pred["labels"], dtype=np.int8)
        out_df = pd.DataFrame({
            "id":     list(ds_pred["id"]),
            "p_text": probs.astype(np.float32),
            "label":  labels_arr,
        })

        # Sanity guard against the group_by_length misalignment bug: a working
        # model must give positives a higher mean p_text than negatives. If the
        # gap is ~0, p_text is shuffled relative to labels -> fail loudly rather
        # than silently writing a useless file.
        if labels_arr.min() != labels_arr.max():   # both classes present
            gap = probs[labels_arr == 1].mean() - probs[labels_arr == 0].mean()
            step(f"  [{name}] sanity: mean p_text(pos)-(neg) = {gap:+.4f}")
            if gap < 0.05:
                raise RuntimeError(
                    f"[{name}] p_text appears MISALIGNED with labels "
                    f"(pos-neg gap={gap:+.4f}, expected >0.05). "
                    f"Likely group_by_length reordered eval output. NOT saving."
                )

        out_df.to_parquet(out_path, engine="pyarrow",
                          compression="snappy", index=False)
        step(f"  [{name}] saved {out_path.name} "
             f"({len(out_df):,} rows, {time.perf_counter()-t1:.1f}s)")
        return {
            "n": int(len(out_df)),
            "metrics": {k: float(v) for k, v in pred.metrics.items()},
        }

    pred_meta: dict = {}
    if args.mode == "fold":
        # OOF: save predictions on the FULL validation slice (held-out fold),
        # not the subset used for early stopping.
        pred_meta["oof"]  = _predict_and_save(
            val_full, PRED_DIR / f"p_text_oof_fold{args.fold}.parquet",
            f"oof_fold{args.fold}",
        )
        # Also save predictions on test set (for later averaging across folds)
        pred_meta["test"] = _predict_and_save(
            test_ds, PRED_DIR / f"p_text_test_fold{args.fold}.parquet",
            f"test_fold{args.fold}",
        )
    else:
        pred_meta["val"]  = _predict_and_save(
            val_full, PRED_DIR / "p_text_val.parquet",  "val",
        )
        pred_meta["test"] = _predict_and_save(
            test_ds, PRED_DIR / "p_text_test.parquet", "test",
        )

    # ── Save run metadata ──────────────────────────────────────────────
    meta = {
        "mode":           args.mode,
        "fold":           args.fold,
        "subsample":      args.subsample,
        "profile":        PROFILE_NAME,
        "n_gpus":         n_gpus(),
        "epochs":         args.epochs,
        "train_time_sec": train_time,
        "tokenizer":      tok_name,
        "backbone":       bb_name,
        "pos_weight":     pos_w,
        "n_train":        len(train_ds),
        "n_val_subset":   len(val_ds),
        "n_val_full":     len(val_full),
        "n_test":         len(test_ds),
        "val_metrics":    {k: float(v) for k, v in val_metrics.items()},
        "predictions":    pred_meta,
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8",
    )

    section("Done.")
    return meta


def main() -> None:
    args = parse_args()
    if args.mode == "fold" and args.fold is None:
        raise SystemExit("--fold required when --mode=fold")
    train_one(args)


if __name__ == "__main__":
    main()
