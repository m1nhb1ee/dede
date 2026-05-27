"""Phase 2 step 1: full cleaning + body deduplication.

Reads the parquet cached by EDA phase 1 (no CSV re-parse).

Cleaning rules (tech_plan section 2):
  - Drop rows: null label, null id, duplicate id, null/invalid created_utc,
              both title AND body empty after cleaning.
  - Replace body in {[removed], [deleted], "", null} -> "" with has_body=False.
  - Apply text cleaning (URLs -> [URL], markdown stripped, etc.) to title + body.
  - Dedup by hash of (normalized_title + normalized_body) for bodies >= 40 chars.
    Keep FIRST occurrence (oldest by created_utc) to preserve temporal split.

Output:
  outputs/posts_clean.parquet  -- cleaned rows ready for splitting + features

Stats logged to outputs/stats.json["cleaning"].
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from config import (
    CLIP_QUANTILE, COL_BODY, COL_ID, COL_LABEL, COL_NCMTS, COL_SUBR,
    COL_TIME, COL_TITLE, COL_UPVOTES, EDA_CACHE, POSTS_CLEAN_PARQUET,
    REMOVED_TOKENS,
)
from utils import (
    hash_text, normalize_for_dedup, section, step, update_stats,
    vectorized_clean, vectorized_is_removed,
)


def main() -> None:
    section("01 | Clean + dedup")

    if not EDA_CACHE.exists():
        raise FileNotFoundError(
            f"{EDA_CACHE} not found. Run ver2/eda/01_load_convert.py first."
        )

    t0 = time.perf_counter()
    df = pd.read_parquet(EDA_CACHE)
    n0 = len(df)
    step(f"Loaded {n0:,} rows in {time.perf_counter()-t0:.1f}s")

    stats: dict = {"n_input": n0, "drops": {}, "modifications": {}}

    # ── Drop: null label, null id, duplicate id ──────────────────────────
    mask = df[COL_LABEL].notna() & df[COL_ID].notna()
    dropped = int((~mask).sum())
    df = df[mask].copy()
    stats["drops"]["null_label_or_id"] = dropped
    step(f"Dropped null label/id: {dropped:,}  (remaining {len(df):,})")

    n_before = len(df)
    df = df.drop_duplicates(subset=[COL_ID], keep="first")
    stats["drops"]["duplicate_id"] = n_before - len(df)
    step(f"Dropped duplicate id: {n_before - len(df):,}  (remaining {len(df):,})")

    # ── Drop: invalid created_utc ────────────────────────────────────────
    ts = pd.to_numeric(df[COL_TIME], errors="coerce")
    REDDIT_EPOCH = pd.Timestamp("2005-01-01", tz="UTC").timestamp()
    FUTURE_LIMIT = pd.Timestamp("2030-01-01", tz="UTC").timestamp()
    valid_ts = ts.notna() & (ts >= REDDIT_EPOCH) & (ts <= FUTURE_LIMIT)
    dropped = int((~valid_ts).sum())
    df = df[valid_ts].copy()
    df[COL_TIME] = ts[valid_ts].astype("int64")
    stats["drops"]["invalid_created_utc"] = dropped
    step(f"Dropped invalid created_utc: {dropped:,}  (remaining {len(df):,})")

    # ── Mark has_title / has_body BEFORE cleaning so we can keep title-only rows
    body_is_removed = vectorized_is_removed(df[COL_BODY], REMOVED_TOKENS)
    df["has_body"] = ~body_is_removed
    stats["modifications"]["body_marked_removed"] = int(body_is_removed.sum())
    step(f"Marked {int(body_is_removed.sum()):,} rows has_body=False")

    title_empty = vectorized_is_removed(df[COL_TITLE], REMOVED_TOKENS)
    df["has_title"] = ~title_empty
    stats["modifications"]["title_marked_empty"] = int(title_empty.sum())

    # ── Apply text cleaning ──────────────────────────────────────────────
    t1 = time.perf_counter()
    step("Cleaning titles ...")
    df[COL_TITLE] = vectorized_clean(df[COL_TITLE])
    step(f"  done in {time.perf_counter()-t1:.1f}s")

    t1 = time.perf_counter()
    step("Cleaning bodies (slower, longer text) ...")
    # Wipe removed/empty bodies to "" before cleaning to save work
    df.loc[body_is_removed, COL_BODY] = ""
    df[COL_BODY] = vectorized_clean(df[COL_BODY])
    step(f"  done in {time.perf_counter()-t1:.1f}s")

    # Re-check has_title / has_body AFTER cleaning (cleaning may empty a row)
    df["has_title"] = df[COL_TITLE].str.len() > 0
    df["has_body"]  = df[COL_BODY].str.len() > 0

    # ── Drop rows where BOTH title and body are empty post-cleaning ──────
    both_empty = (~df["has_title"]) & (~df["has_body"])
    dropped = int(both_empty.sum())
    df = df[~both_empty].copy()
    stats["drops"]["title_and_body_both_empty"] = dropped
    step(f"Dropped title+body both empty: {dropped:,}  (remaining {len(df):,})")

    # ── Numeric outlier clipping (matches tech_plan 2.3) ─────────────────
    for col in (COL_UPVOTES, COL_NCMTS):
        s = pd.to_numeric(df[col], errors="coerce")
        # Fill NaN with 0 (per tech_plan 2.1)
        n_null = int(s.isna().sum())
        s = s.fillna(0)
        # Clip negative (3 negative num_comments found in EDA) to 0
        n_neg = int((s < 0).sum())
        s = s.clip(lower=0)
        # Clip top outliers
        upper = float(s.quantile(CLIP_QUANTILE))
        n_clipped = int((s > upper).sum())
        s = s.clip(upper=upper)
        df[col] = s.astype("int64")
        stats["modifications"][f"{col}_null_filled"] = n_null
        stats["modifications"][f"{col}_neg_clipped_to_zero"] = n_neg
        stats["modifications"][f"{col}_top_clipped_at_p995"] = {
            "upper": upper, "n_clipped": n_clipped,
        }
    step(f"Clipped upvotes/num_comments at p{int(CLIP_QUANTILE*1000)/10}")

    # ── Normalize subreddit (lowercase strip) ────────────────────────────
    df[COL_SUBR] = df[COL_SUBR].astype(str).str.strip().str.lower()
    # Per EDA: only 24 unique subreddits, 19 are <50 rows totaling 63 rows.
    # Rare-bucketing skipped (tech_plan section 0a decision 5).

    # ── Body deduplication (HASH of normalized title+body) ───────────────
    # Important: dedup BEFORE split (per tech_plan section 4.1).
    # We dedup on title+body together because Reddit reposts often share both.
    step("Hashing for duplicate detection ...")
    t1 = time.perf_counter()
    # Only consider rows where body is long enough to be a meaningful dup signal.
    # Short / empty bodies are NOT treated as duplicates of each other.
    from config import DUP_BODY_MIN_LEN
    body_long_enough = df[COL_BODY].str.len() >= DUP_BODY_MIN_LEN

    # Build hash key (title + " || " + body, normalized)
    keys = (df.loc[body_long_enough, COL_TITLE].map(normalize_for_dedup) + " || " +
            df.loc[body_long_enough, COL_BODY].map(normalize_for_dedup))
    hashes = keys.map(hash_text)
    step(f"  hashed {len(hashes):,} long-body rows in {time.perf_counter()-t1:.1f}s")

    # Keep FIRST occurrence (oldest created_utc within each duplicate group).
    # This preserves the time ordering for downstream time-based split.
    dup_df = df.loc[body_long_enough, [COL_ID, COL_TIME]].copy()
    dup_df["_hash"] = hashes.values
    dup_df = dup_df.sort_values([COL_TIME, COL_ID])
    # Mark which rows to keep: first per group
    keep_mask_long = ~dup_df.duplicated(subset=["_hash"], keep="first")
    keep_ids_long  = set(dup_df.loc[keep_mask_long, COL_ID].tolist())

    short_ids = set(df.loc[~body_long_enough, COL_ID].tolist())
    keep_ids = short_ids | keep_ids_long
    n_before = len(df)
    df = df[df[COL_ID].isin(keep_ids)].copy()
    n_removed = n_before - len(df)
    stats["drops"]["body_duplicate_rows"] = n_removed
    stats["modifications"]["dedup_min_body_len"] = DUP_BODY_MIN_LEN
    step(f"Removed {n_removed:,} duplicate-body rows  (remaining {len(df):,})")

    # ── Final select + cast ──────────────────────────────────────────────
    df = df[[
        COL_ID, COL_SUBR, COL_TITLE, COL_BODY, "has_title", "has_body",
        COL_UPVOTES, COL_NCMTS, COL_TIME, COL_LABEL,
    ]].reset_index(drop=True)

    df[COL_LABEL] = df[COL_LABEL].astype("int8")
    df["has_title"] = df["has_title"].astype("bool")
    df["has_body"]  = df["has_body"].astype("bool")

    # ── Write ────────────────────────────────────────────────────────────
    step(f"Writing {POSTS_CLEAN_PARQUET}")
    t1 = time.perf_counter()
    df.to_parquet(POSTS_CLEAN_PARQUET, engine="pyarrow",
                  compression="snappy", index=False)
    step(f"  wrote {POSTS_CLEAN_PARQUET.stat().st_size / 1e6:.1f} MB "
         f"in {time.perf_counter()-t1:.1f}s")

    stats["n_output"]     = int(len(df))
    stats["overall_pct_dropped"] = round((n0 - len(df)) / n0 * 100, 4)
    stats["label_distribution_post_clean"] = {
        str(int(k)): int(v) for k, v in df[COL_LABEL].value_counts().items()
    }
    stats["subreddit_count_post_clean"] = int(df[COL_SUBR].nunique())
    update_stats("cleaning", stats)

    section(f"Done. {n0:,} -> {len(df):,} "
            f"({stats['overall_pct_dropped']}% dropped). "
            f"Run 02_split.py next.")


if __name__ == "__main__":
    main()
