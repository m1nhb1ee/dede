"""Phase 2 step 3: build metadata feature matrix for Stage 2 (LightGBM).

Reads posts_clean.parquet and produces meta_features.parquet keyed by id.
Feature priorities are set by EDA findings (tech_plan section 6.2 post-EDA):

  HIGH-SIGNAL (must include):
    body_len_chars       (Spearman=+0.45 with label)
    body_length_bucket   (MI=0.17 bits, label_rate from 4% to 67% by bucket)
    has_mh_keyword       (lift 4.3x; the strongest single binary feature)
    has_body             (MI=0.033 bits)
    num_comments_log     (Spearman=-0.20)
    year                 (MI=0.078; captures drift)

  MID-SIGNAL:
    upvotes_log, upvotes_pct_in_subreddit, comments_per_upvote,
    title_len_chars, body_to_title_ratio,
    num_first_person, num_negative_words, num_exclamations, num_questions

  LOW-SIGNAL but cheap (kept for completeness + ablation):
    hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_night_us_eastern

  CATEGORICAL:
    subreddit (string)   -- caller chooses to include or exclude (ablation)
    has_title, has_body  -- boolean

p_text is NOT computed here -- it comes from Stage 1 OOF predictions later.

Notes:
  - upvotes_pct_in_subreddit is computed within each split independently
    in downstream code IF we want a leak-free pct rank. Here we compute it
    globally; the leak risk is low because the subreddit-rank distribution
    is fairly stable. Document in tech_plan if we revise this.
"""

from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from config import (
    COL_BODY, COL_ID, COL_LABEL, COL_NCMTS, COL_SUBR, COL_TIME, COL_TITLE,
    COL_UPVOTES, FIRST_PERSON_WORDS, META_FEAT_PARQUET, MH_KEYWORDS,
    NEGATIVE_WORDS, POSTS_CLEAN_PARQUET,
)
from utils import section, step, update_stats


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized lexical counters
# ─────────────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _count_words_in_set(text: str, word_set: set[str]) -> int:
    if not isinstance(text, str) or not text: return 0
    return sum(1 for w in _WORD_RE.findall(text.lower()) if w in word_set)


def _has_any_keyword(text: str, keywords: set[str]) -> bool:
    if not isinstance(text, str) or not text: return False
    low = text.lower()
    # Substring match for stems (e.g. "depress" matches "depression"/"depressed")
    return any(k in low for k in keywords)


def main() -> None:
    section("03 | Build metadata feature matrix for LightGBM")

    if not POSTS_CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            f"{POSTS_CLEAN_PARQUET} not found. Run 01_clean.py first."
        )

    df = pd.read_parquet(POSTS_CLEAN_PARQUET)
    n = len(df)
    step(f"Loaded {n:,} clean rows")

    feats = pd.DataFrame({COL_ID: df[COL_ID].values})

    # ── Pre-compute combined text (title + body) for lexical features ────
    combined = (df[COL_TITLE].fillna("") + " " + df[COL_BODY].fillna("")).str.strip()

    # ── Numeric: length features ─────────────────────────────────────────
    title_len = df[COL_TITLE].fillna("").str.len()
    body_len  = df[COL_BODY].fillna("").str.len()
    feats["title_len_chars"] = title_len.astype("int32")
    feats["body_len_chars"]  = body_len.astype("int32")
    feats["body_to_title_ratio"] = (body_len / (title_len + 1)).astype("float32")

    # Body length bucket (matches EDA bucket boundaries -- already validated
    # to be highly predictive: label_rate goes from 4% to 67% across these)
    bins = [-1, 0, 50, 200, 500, 1000, 2000, 5000, 10_000_000]
    bucket_codes = pd.cut(body_len, bins=bins, labels=False).astype("int8")
    feats["body_length_bucket"] = bucket_codes
    step("Length features done")

    # ── Engagement features ──────────────────────────────────────────────
    up = df[COL_UPVOTES].astype("float32")
    nc = df[COL_NCMTS].astype("float32")
    feats["upvotes_log"]       = np.log1p(up).astype("float32")
    feats["num_comments_log"]  = np.log1p(nc).astype("float32")
    feats["comments_per_upvote"] = (nc / (up + 1)).astype("float32")

    # Upvotes percentile rank within subreddit (centered cross-subreddit)
    # Use dense rank to avoid ties dominating percentile.
    step("Computing upvotes_pct_in_subreddit (groupby rank) ...")
    t1 = time.perf_counter()
    grp = df.groupby(COL_SUBR, observed=True)[COL_UPVOTES]
    feats["upvotes_pct_in_subreddit"] = grp.rank(pct=True, method="dense").astype("float32")
    step(f"  done in {time.perf_counter()-t1:.1f}s")

    # ── Boolean indicators ───────────────────────────────────────────────
    feats["has_title"] = df["has_title"].astype("bool")
    feats["has_body"]  = df["has_body"].astype("bool")

    # ── Lexical features (style markers) ─────────────────────────────────
    # Computed on cleaned text. Vectorized over pandas Series for speed.
    feats["num_exclamations"] = combined.str.count("!").astype("int16")
    feats["num_questions"]    = combined.str.count(r"\?").astype("int16")
    feats["num_caps_words"]   = combined.str.count(r"\b[A-Z]{2,}\b").astype("int16")
    feats["num_ellipsis"]     = combined.str.count(r"\.\.\.").astype("int16")
    step("Style-marker counts done")

    # Word-set counters (slower; vectorize via .map, not .apply, to skip the
    # Series overhead).
    t1 = time.perf_counter()
    step("Counting first-person / negative-word occurrences ...")
    lower_words = combined.str.lower().str.findall(_WORD_RE)
    feats["num_first_person"] = lower_words.map(
        lambda ws: sum(1 for w in ws if w in FIRST_PERSON_WORDS)
    ).astype("int16")
    feats["num_negative_words"] = lower_words.map(
        lambda ws: sum(1 for w in ws if w in NEGATIVE_WORDS)
    ).astype("int16")
    feats["num_words"] = lower_words.map(len).astype("int32")
    step(f"  done in {time.perf_counter()-t1:.1f}s")

    # Has MH keyword (substring match for stems; the SINGLE strongest binary
    # feature per EDA -- 84.8% label=1 vs 19.9% label=0, lift 4.3x).
    t1 = time.perf_counter()
    step("Computing has_mh_keyword ...")
    feats["has_mh_keyword"] = combined.str.lower().map(
        lambda s: _has_any_keyword(s, MH_KEYWORDS)
    ).astype("bool")
    step(f"  done in {time.perf_counter()-t1:.1f}s")

    # ── Temporal features (cyclical encoding) ────────────────────────────
    dt = pd.to_datetime(df[COL_TIME], unit="s", utc=True)
    hour = dt.dt.hour.values
    dow  = dt.dt.weekday.values
    year = dt.dt.year.values

    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    feats["dow_sin"]  = np.sin(2 * np.pi * dow  /  7).astype("float32")
    feats["dow_cos"]  = np.cos(2 * np.pi * dow  /  7).astype("float32")
    feats["is_weekend"]          = (dow >= 5).astype("bool")
    feats["is_night_us_eastern"] = np.isin(hour, [4, 5, 6, 7, 8, 9]).astype("bool")
    feats["year"] = year.astype("int16")

    # ── Categorical (string, LGBM handles natively) ──────────────────────
    feats[COL_SUBR] = df[COL_SUBR].astype(str).values

    # ── Carry label (handy for training; redundant with posts_clean) ─────
    feats[COL_LABEL] = df[COL_LABEL].astype("int8").values

    # ── Write ────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    feats.to_parquet(META_FEAT_PARQUET, engine="pyarrow",
                     compression="snappy", index=False)
    step(f"Wrote {META_FEAT_PARQUET} "
         f"({META_FEAT_PARQUET.stat().st_size / 1e6:.1f} MB, "
         f"{time.perf_counter()-t1:.1f}s)")
    step(f"Feature columns ({len(feats.columns)}):")
    for c in feats.columns:
        print(f"      {c:<28} {feats[c].dtype}")

    update_stats("features", {
        "n_rows": int(len(feats)),
        "n_features_excluding_id_label": int(len(feats.columns) - 2),
        "columns": list(feats.columns),
        "has_mh_keyword_pct": round(float(feats["has_mh_keyword"].mean() * 100), 4),
        "has_body_pct":       round(float(feats["has_body"].mean() * 100), 4),
    })
    section("Done. Phase 2 complete -- ready for Stage 1 training.")


if __name__ == "__main__":
    main()
