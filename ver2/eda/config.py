"""EDA config: paths, constants, sample sizes.

All scripts in this folder import from here so paths are defined once.
"""

from pathlib import Path

# === Project paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV      = PROJECT_ROOT / "eng dataset" / "dataset_raw" / "reddit_depression_dataset.csv"

# === EDA outputs ===
EDA_DIR     = Path(__file__).resolve().parent
OUT_DIR     = EDA_DIR / "outputs"
PLOTS_DIR   = OUT_DIR / "plots"
CACHE_DIR   = OUT_DIR / "cache"          # intermediate parquets + stats json
SUMMARY_CSV = OUT_DIR / "summary_stats.csv"
ISSUES_MD   = OUT_DIR / "data_issues.md"

# Parquet copy of the raw CSV — created once, reused by all later scripts
RAW_PARQUET = CACHE_DIR / "raw.parquet"

for d in (OUT_DIR, PLOTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# === Column names (single source of truth) ===
COL_ID       = "id"
COL_SUBR     = "subreddit"
COL_TITLE    = "title"
COL_BODY     = "body"
COL_UPVOTES  = "upvotes"
COL_TIME     = "created_utc"
COL_NCMTS    = "num_comments"
COL_LABEL    = "label"

TEXT_COLS    = [COL_TITLE, COL_BODY]
NUM_COLS     = [COL_UPVOTES, COL_NCMTS]
ALL_COLS     = [COL_ID, COL_SUBR, COL_TITLE, COL_BODY,
                COL_UPVOTES, COL_TIME, COL_NCMTS, COL_LABEL]

# === Constants ===
REMOVED_TOKENS = {"[removed]", "[deleted]", "removed", "deleted", "", "nan", "none"}

# Tokenizer model (matches tech_plan section 5).
# Note: mental/mental-roberta-base is a GATED HF repo (requires auth).
# Its tokenizer is identical to roberta-base (same BPE vocab) -- only the
# weights differ -- so we use roberta-base for EDA token-length stats.
TOKENIZER_NAME = "roberta-base"
TOKENIZER_FALLBACKS = ["mental/mental-roberta-base", "roberta-base"]

# Sample size for token-count analysis (full 2M tokenization is slow).
# 100K is statistically more than enough for length percentiles.
TOKEN_SAMPLE_SIZE = 100_000

# Random seed (matches tech_plan section 11.2)
SEED = 42

# Length thresholds we report in the issues file
TOKEN_THRESHOLDS = [128, 256, 384, 512]

# Rare-subreddit threshold (matches tech_plan section 2.3)
RARE_SUBR_MIN_COUNT = 50

# Plot style
PLOT_DPI    = 110
PLOT_FIGSZ  = (9, 5)
