"""Stage 2 config: paths, feature set, LightGBM params, CV / tuning setup.

Subreddit-blind by construction: the feature list below is exactly the 23
meta features in meta_features.parquet (tech_plan 6.2) + p_text from Stage 1.
No subreddit (or proxy) appears anywhere.

All input/output paths can be overridden via env vars so a dry-run can point
at partial / temporary parquet files without editing this module.
"""

from __future__ import annotations

import os
from pathlib import Path

VER2_DIR    = Path(__file__).resolve().parents[2]   # .../DeDe/ver2
PREP_OUT    = VER2_DIR / "preprocess" / "outputs"
STAGE1_PRED = VER2_DIR / "stage1" / "outputs" / "predictions"


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw) if raw else default


# ── Inputs ───────────────────────────────────────────────────────────────────
META_FEATURES   = _env_path("STAGE2_META",      PREP_OUT / "meta_features.parquet")
SPLITS          = _env_path("STAGE2_SPLITS",    PREP_OUT / "splits.parquet")
# p_text per split: train <- OOF, test <- fold ensemble, val <- full-model (optional)
P_TEXT_OOF_TRAIN = _env_path("STAGE2_OOF_TRAIN", STAGE1_PRED / "p_text_oof_train.parquet")
P_TEXT_TEST      = _env_path("STAGE2_TEST",      STAGE1_PRED / "p_text_test_ensemble.parquet")
P_TEXT_VAL       = _env_path("STAGE2_VAL",       STAGE1_PRED / "p_text_val.parquet")

# ── Outputs ──────────────────────────────────────────────────────────────────
OUT_DIR    = _env_path("STAGE2_OUT_DIR", Path(__file__).resolve().parent.parent / "outputs")
MODELS_DIR = OUT_DIR / "models"
PRED_DIR   = OUT_DIR / "predictions"
for d in (OUT_DIR, MODELS_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

BEST_PARAMS_PATH = OUT_DIR / "best_params.json"
MODEL_PATH       = MODELS_DIR / "stage2_lgbm.pkl"

# ── Columns ──────────────────────────────────────────────────────────────────
ID_COL    = "id"
LABEL_COL = "label"

# Base columns that exist directly in meta_features.parquet (29 meta) + p_text.
BASE_FEATURE_COLS = [
    "p_text",                                              # Stage 1 OOF signal
    # length / structure
    "body_len_chars", "body_length_bucket", "title_len_chars", "body_to_title_ratio",
    # engagement
    "upvotes_log", "num_comments_log", "comments_per_upvote",
    # lexical / style (raw counts)
    "has_mh_keyword", "num_first_person", "num_negative_words",
    "num_exclamations", "num_questions", "num_caps_words", "num_ellipsis", "num_words",
    # Tier-2 psycholinguistic markers (raw)
    "num_absolutist", "num_second_person", "type_token_ratio", "avg_word_len",
    "num_sentences", "uppercase_ratio",
    # temporal (low-signal, kept for ablation)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_night_us_eastern",
    # boolean structure
    "has_title", "has_body",
]

# Length-normalized rates engineered in data.py (count / num_words, or derived).
# Decouple style signals from post length -- a smooth ratio a tree cannot encode
# in a single split.
ENGINEERED_COLS = [
    "first_person_rate", "negative_word_rate", "exclamation_rate", "question_rate",
    "caps_word_rate", "ellipsis_rate", "absolutist_rate", "second_person_rate",
    "avg_sentence_len",
]

FEATURE_COLS = BASE_FEATURE_COLS + ENGINEERED_COLS
# body_length_bucket is ordinal (0..7) and booleans map to 0/1, so all features
# are left numeric -- LightGBM splits handle them natively. No categorical list.

# ── CV / tuning ──────────────────────────────────────────────────────────────
N_FOLDS    = 3          # reuse the 3-fold OOF assignment from splits.parquet
SEED       = 42
N_TRIALS   = 50         # Optuna trials (tech_plan 6.4)

# ── LightGBM baseline (tech_plan 6.3) ────────────────────────────────────────
LGBM_BASE = {
    "objective":         "binary",
    "metric":            ["binary_logloss", "auc"],
    "boosting_type":     "gbdt",
    "num_leaves":        63,
    "max_depth":         -1,
    "learning_rate":     0.05,
    "n_estimators":      2000,      # capped by early stopping
    "min_child_samples": 100,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "subsample":         0.8,
    "subsample_freq":    1,         # subsample only takes effect with freq>0
    "colsample_bytree":  0.8,
    "class_weight":      "balanced",
    "random_state":      SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}

EARLY_STOPPING_ROUNDS = 100
