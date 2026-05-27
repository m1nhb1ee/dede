"""Phase 2 step 2: train/val/test split + 5-fold OOF assignment.

Produces ONE parquet `splits.parquet` with columns:
    id            -- row id
    time_split    -- "train" | "val" | "test"   (primary, time-based)
    random_split  -- "train" | "val" | "test"   (secondary, stratified random)
    fold          -- 0..4 for rows where time_split=="train", else -1

Why two splits:
  - Time-based is the PRIMARY evaluation (tech_plan section 4.1 post-EDA).
    Train <= 2020, val = 2021, test = 2022. Justified by 0.73 label-rate
    drift across years.
  - Random stratified is SECONDARY -- we report both numbers so the
    over-estimate caused by ignoring drift is visible in the report.

K-fold OOF runs on the time-train set, stratified by label.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import (
    COL_ID, COL_LABEL, COL_TIME, KFOLD_SEED, N_FOLDS, POSTS_CLEAN_PARQUET,
    RANDOM_TEST_PCT, RANDOM_VAL_PCT, SEED, SPLITS_PARQUET, TEST_YEAR,
    TRAIN_MAX_YEAR, VAL_YEAR,
)
from utils import section, step, update_stats


def main() -> None:
    section(f"02 | Split (time-based primary + random secondary + {N_FOLDS}-fold OOF)")

    if not POSTS_CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            f"{POSTS_CLEAN_PARQUET} not found. Run 01_clean.py first."
        )

    df = pd.read_parquet(POSTS_CLEAN_PARQUET, columns=[COL_ID, COL_TIME, COL_LABEL])
    n = len(df)
    step(f"Loaded {n:,} clean rows")

    stats: dict = {"n_input": n}

    # ── Time-based split (PRIMARY) ───────────────────────────────────────
    # Boundaries: train <= 2020-12-31, val = 2021, test = 2022
    dt   = pd.to_datetime(df[COL_TIME], unit="s", utc=True)
    year = dt.dt.year

    time_split = pd.Series(index=df.index, dtype="object")
    time_split[year <= TRAIN_MAX_YEAR] = "train"
    time_split[year == VAL_YEAR]       = "val"
    time_split[year == TEST_YEAR]      = "test"
    # Any post-2022 (shouldn't be any per EDA) -> drop from splits
    time_split = time_split.fillna("unknown")

    ts_counts = time_split.value_counts().to_dict()
    step(f"Time split counts: {ts_counts}")
    stats["time_split"] = {
        "boundaries": {"train_max_year": TRAIN_MAX_YEAR,
                       "val_year": VAL_YEAR, "test_year": TEST_YEAR},
        "counts":     {str(k): int(v) for k, v in ts_counts.items()},
        "label_rate_per_split": {
            split: round(float(df.loc[time_split == split, COL_LABEL].mean()), 6)
            for split in ("train", "val", "test")
            if (time_split == split).any()
        },
    }
    for k, v in stats["time_split"]["label_rate_per_split"].items():
        print(f"      {k:>6} label_rate = {v:.4f}")

    # ── Random stratified split (SECONDARY) ──────────────────────────────
    # 80/10/10 stratified by label
    idx_all = np.arange(n)
    y_all   = df[COL_LABEL].values

    train_idx, temp_idx, _, y_temp = train_test_split(
        idx_all, y_all,
        test_size=(RANDOM_VAL_PCT + RANDOM_TEST_PCT),
        stratify=y_all, random_state=SEED,
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp,
        test_size=RANDOM_TEST_PCT / (RANDOM_VAL_PCT + RANDOM_TEST_PCT),
        stratify=y_temp, random_state=SEED,
    )
    random_split = np.array(["train"] * n, dtype=object)
    random_split[val_idx]  = "val"
    random_split[test_idx] = "test"
    random_split = pd.Series(random_split, index=df.index)

    rs_counts = random_split.value_counts().to_dict()
    step(f"Random split counts: {rs_counts}")
    stats["random_split"] = {
        "ratios":  {"val_pct": RANDOM_VAL_PCT, "test_pct": RANDOM_TEST_PCT},
        "counts":  {str(k): int(v) for k, v in rs_counts.items()},
        "label_rate_per_split": {
            split: round(float(df.loc[random_split == split, COL_LABEL].mean()), 6)
            for split in ("train", "val", "test")
        },
    }

    # ── 5-fold OOF on time-train (stratified by label) ───────────────────
    fold = np.full(n, -1, dtype=np.int8)
    train_mask = (time_split == "train").values
    train_indices = np.where(train_mask)[0]
    y_train       = df[COL_LABEL].values[train_mask]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=KFOLD_SEED)
    for k, (_, val_pos) in enumerate(skf.split(train_indices, y_train)):
        # val_pos are positions within train_indices; map back to global idx
        fold[train_indices[val_pos]] = k
    step(f"{N_FOLDS}-fold OOF assigned for {int(train_mask.sum()):,} train rows")

    fold_counts = {int(k): int((fold == k).sum()) for k in range(N_FOLDS)}
    stats["kfold"] = {
        "n_folds":              N_FOLDS,
        "seed":                 KFOLD_SEED,
        "rows_per_fold":        fold_counts,
        "label_rate_per_fold":  {
            int(k): round(float(df.loc[fold == k, COL_LABEL].mean()), 6)
            for k in range(N_FOLDS)
        },
    }

    # ── Compose + write ──────────────────────────────────────────────────
    out = pd.DataFrame({
        COL_ID:         df[COL_ID].values,
        "time_split":   time_split.values,
        "random_split": random_split.values,
        "fold":         fold,
    })
    t1 = time.perf_counter()
    out.to_parquet(SPLITS_PARQUET, engine="pyarrow",
                   compression="snappy", index=False)
    step(f"Wrote {SPLITS_PARQUET} "
         f"({SPLITS_PARQUET.stat().st_size / 1e6:.1f} MB, "
         f"{time.perf_counter()-t1:.1f}s)")

    update_stats("splitting", stats)
    section("Done. Run 03_features.py next.")


if __name__ == "__main__":
    main()
