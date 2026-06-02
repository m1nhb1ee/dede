"""Preprocess config: paths, constants, splitting rules.

Phase 2 reads the raw parquet cached by EDA phase 1 (no re-parse of CSV).
All outputs land under ver2/preprocess/outputs/.
"""

from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDA_CACHE    = PROJECT_ROOT / "ver2" / "eda" / "outputs" / "cache" / "raw.parquet"

PREP_DIR     = Path(__file__).resolve().parent
OUT_DIR      = PREP_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSTS_CLEAN_PARQUET = OUT_DIR / "posts_clean.parquet"
SPLITS_PARQUET      = OUT_DIR / "splits.parquet"
META_FEAT_PARQUET   = OUT_DIR / "meta_features.parquet"
EVAL_SUBR_PARQUET   = OUT_DIR / "eval_subreddit.parquet"  # eval-only: id->subreddit, NEVER a model feature
STATS_JSON          = OUT_DIR / "stats.json"

# === Columns (mirror EDA config) ===
COL_ID, COL_SUBR, COL_TITLE, COL_BODY = "id", "subreddit", "title", "body"
COL_UPVOTES, COL_TIME, COL_NCMTS, COL_LABEL = "upvotes", "created_utc", "num_comments", "label"

REMOVED_TOKENS = {"[removed]", "[deleted]", "removed", "deleted",
                  "", "nan", "none", "null"}

# === Cleaning ===
URL_TOKEN  = " [URL] "
USER_TOKEN = " [USER] "
SUB_TOKEN  = " [SUB] "

# Dedup: only consider bodies >= this length (matches EDA threshold).
# Short / empty bodies are NOT treated as duplicates of each other.
DUP_BODY_MIN_LEN = 40

# Outlier clipping (matches tech_plan section 2.3)
CLIP_QUANTILE = 0.995

# === Splits ===
SEED         = 42
KFOLD_SEED   = 2024
N_FOLDS      = 3

# Time-based split boundaries (matches tech_plan section 4.1 post-EDA).
# Date range from EDA: 2008-01-26 -> 2022-12-31
TRAIN_MAX_YEAR = 2020   # train: <= 2020-12-31
VAL_YEAR       = 2021
TEST_YEAR      = 2022

# Random split ratios (secondary, for comparison)
RANDOM_VAL_PCT  = 0.10
RANDOM_TEST_PCT = 0.10

# === MH keyword set (shared with EDA section 06) ===
MH_KEYWORDS = {
    "depress", "depressed", "depression", "anxious", "anxiety", "panic",
    "hopeless", "worthless", "useless", "empty", "numb", "exhausted",
    "tired", "alone", "lonely", "isolat", "isolated",
    "suicide", "suicidal", "kill", "die", "death", "dead",
    "hurt", "hate", "broken", "lost", "pain", "crying", "tears",
    "therapy", "therapist", "medication", "meds", "pills",
    "trauma", "abuse", "abused", "ptsd",
}

FIRST_PERSON_WORDS = {"i", "me", "my", "mine", "myself", "im"}

NEGATIVE_WORDS = {
    "sad", "angry", "afraid", "scared", "fear", "lonely", "miserable",
    "horrible", "terrible", "awful", "bad", "worse", "worst",
    "hopeless", "helpless", "useless", "worthless", "stupid", "ugly",
    "fail", "failed", "failure", "wrong", "mistake", "regret",
    "sorry", "guilty", "shame", "ashamed", "hate",
}
