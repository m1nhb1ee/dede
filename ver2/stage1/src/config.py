"""Stage 1 config: paths, model name, training hyperparameters.

Two hardware profiles are supported:
  - "kaggle"  : 2x T4 16GB (default; matches tech_plan section 5.5 updated)
  - "local"   : RTX 3050 6GB (debug / 100K subset only)

Override via env var STAGE1_PROFILE=kaggle|local.
"""

from __future__ import annotations

import os
from pathlib import Path

# Disable the noisy "TOKENIZERS_PARALLELISM" warning when datasets.map() is
# called from a forked subprocess. Safe because we set num_proc=1 in profiles.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREP_OUT     = PROJECT_ROOT / "ver2" / "preprocess" / "outputs"

# All paths can be overridden via env vars -- needed for Kaggle, where the
# inputs live at /kaggle/input/<dataset>/ and writes must go to /kaggle/working/.
def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw) if raw else default

POSTS_CLEAN_PARQUET = _env_path("STAGE1_POSTS_CLEAN", PREP_OUT / "posts_clean.parquet")
SPLITS_PARQUET      = _env_path("STAGE1_SPLITS",      PREP_OUT / "splits.parquet")

STAGE1_DIR    = Path(__file__).resolve().parent
OUT_DIR       = _env_path("STAGE1_OUT_DIR", STAGE1_DIR / "outputs")
CKPT_DIR      = OUT_DIR / "checkpoints"
PRED_DIR      = OUT_DIR / "predictions"
TOKEN_CACHE   = OUT_DIR / "token_cache"   # pre-tokenized HF Dataset cache
for d in (OUT_DIR, CKPT_DIR, PRED_DIR, TOKEN_CACHE):
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

# Prefer MentalRoBERTa (gated); fallback to roberta-base (same tokenizer vocab).
MODEL_FALLBACKS = [
    "mental/mental-roberta-base",
    "roberta-base",
]

# How many of the last hidden layers to weighted-average in the custom head.
# Per tech_plan section 5.3: 4 layers. Reduce to 2 if VRAM is tight.
LAYER_AGG_N = 4

# Multi-Sample Dropout K (output head)
MSD_K       = 5
MSD_P       = 0.5

# Hidden FFN expansion factor in residual block (768 -> 1536 -> 768)
FFN_EXPAND  = 2

# Dropout in the head's FFN block
HEAD_DROPOUT = 0.1

# Label smoothing for BCE loss
LABEL_SMOOTHING = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Tokenization (head+tail strategy, see tech_plan section 3.1)
# ─────────────────────────────────────────────────────────────────────────────

# Head fraction of body budget. Body tail gets (1 - HEAD_FRAC).
HEAD_FRAC = 0.35

# Inserted between head and tail when body is truncated, so the model sees
# a visible discontinuity marker.
TRUNC_MARKER = " ... "

# ─────────────────────────────────────────────────────────────────────────────
# LoRA (PEFT)
# ─────────────────────────────────────────────────────────────────────────────

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
LORA_TARGETS = ["query", "value"]   # attention only, per tech_plan 5.4

# ─────────────────────────────────────────────────────────────────────────────
# Hardware profiles
# ─────────────────────────────────────────────────────────────────────────────

PROFILES = {
    "kaggle": {
        "max_length":              512,
        "per_device_batch":        8,
        "grad_accum":              4,
        "fp16":                    True,
        "gradient_checkpointing":  True,   # OOM safety net (~20% slower, ~40% less VRAM)
        "optim":                   "adamw_torch",     # standard AdamW
        "dataloader_num_workers":  2,
        "dataloader_pin_memory":   True,    # Kaggle has plenty of RAM
        # datasets.map(): num_proc>1 + fast tokenizer can deadlock on Kaggle
        "tokenize_num_proc":       1,
    },
    "local": {
        # 3050 6GB debug: tiny everything, gradient_checkpointing on
        "max_length":              256,
        "per_device_batch":        4,
        "grad_accum":              8,
        "fp16":                    True,
        "gradient_checkpointing":  True,
        "optim":                   "adamw_bnb_8bit",
        "dataloader_num_workers":  0,
        "dataloader_pin_memory":   False,
        "tokenize_num_proc":       1,
    },
}

PROFILE_NAME = os.environ.get("STAGE1_PROFILE", "kaggle")
assert PROFILE_NAME in PROFILES, f"Unknown profile {PROFILE_NAME}"
HW = PROFILES[PROFILE_NAME]

# ─────────────────────────────────────────────────────────────────────────────
# Training schedule (shared across profiles)
# ─────────────────────────────────────────────────────────────────────────────

NUM_EPOCHS    = 1     # MentalRoBERTa pre-trained on domain -> 1 epoch usually enough
LR_LORA       = 5e-4    # LoRA adapters
LR_HEAD       = 1e-4    # Custom head (random init, smaller LR)
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.06
LR_SCHEDULER  = "linear"   # linear decay after warmup

# Eval / save cadence
# With 1.45M train + batch 64 effective => ~22K steps/epoch * 3 epochs = ~68K steps.
# eval_steps=5000 gives ~13 evaluations per fold => good for early stopping,
# total eval time ~30-50 min per fold (vs ~2.5h with eval_steps=2000).
EVAL_STEPS              = 1500   # detect convergence early; cheap because eval is on subset
SAVE_TOTAL_LIMIT        = 2
EARLY_STOPPING_PATIENCE = 2       # in eval cycles (= 2 * EVAL_STEPS without improvement)
EARLY_STOPPING_THRESHOLD = 0.001  # min loss decrease to count as "improvement"
QUICK_LOG_STEPS         = 100    # quick eval (F1/recall on small val subset) cadence
QUICK_EVAL_SIZE         = 5000   # subset size for quick eval -- ~20s per call
                                 # sampled from val_full \ eval_subset (DISJOINT)
EVAL_SUBSET_SIZE        = 60000  # stratified subset of val_ds for intermediate Trainer eval
                                 # (fast early-stop decisions; final predict still uses full val)
METRIC_FOR_BEST         = "f1_macro"  # early-stop when f1_macro stops improving
METRIC_GREATER_IS_BETTER = True

# Seed
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Class imbalance (computed from EDA: 19.4% positive)
# pos_weight for BCE = neg_count / pos_count
# Will be re-computed at runtime from the actual training fold.
# ─────────────────────────────────────────────────────────────────────────────

USE_POS_WEIGHT = True
