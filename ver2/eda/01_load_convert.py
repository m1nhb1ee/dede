"""Convert the 1GB raw CSV to a parquet cache.

Why: pandas re-parses a 1GB CSV in ~60s; the parquet cache loads in ~5s.
All later EDA scripts read from the parquet copy.

Run once:
    python 01_load_convert.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from config import (
    ALL_COLS, COL_LABEL, COL_NCMTS, COL_SUBR, COL_TIME, COL_UPVOTES,
    RAW_CSV, RAW_PARQUET,
)
from utils import section, step, update_stats


# Memory-efficient dtypes. Strings stay as object (pandas StringDtype is slow on
# 2M rows for our use case); numerics become smallest int / float that fits.
READ_DTYPES = {
    COL_SUBR:    "category",
    COL_UPVOTES: "Int32",      # nullable int — some rows have missing upvotes
    COL_TIME:    "Int64",      # unix seconds; up to year 2262 fits in Int64
    COL_NCMTS:   "Int32",
    COL_LABEL:   "Int8",       # 0/1 (or small multi-class)
}


def main() -> None:
    section("01 | CSV -> Parquet cache")

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found at: {RAW_CSV}")

    step(f"Reading {RAW_CSV.name} ({RAW_CSV.stat().st_size / 1e9:.2f} GB)")
    t0 = time.perf_counter()

    # `low_memory=False` avoids dtype warnings on a single pass; OK with 24GB RAM.
    df = pd.read_csv(
        RAW_CSV,
        usecols=ALL_COLS,
        dtype=READ_DTYPES,
        low_memory=False,
    )
    step(f"Loaded {len(df):,} rows x {df.shape[1]} cols in {time.perf_counter()-t0:.1f}s")

    # Quick sanity peek before saving
    step("Dtypes after read:")
    for c in df.columns:
        print(f"      {c:<14} {str(df[c].dtype)}")

    step(f"Writing parquet -> {RAW_PARQUET}")
    t0 = time.perf_counter()
    df.to_parquet(RAW_PARQUET, engine="pyarrow", compression="snappy", index=False)
    step(f"Wrote {RAW_PARQUET.stat().st_size / 1e6:.1f} MB in {time.perf_counter()-t0:.1f}s")

    # Record into the shared stats JSON so the final report has provenance
    update_stats("dataset", {
        "csv_path":          str(RAW_CSV),
        "parquet_path":      str(RAW_PARQUET),
        "csv_size_bytes":    int(RAW_CSV.stat().st_size),
        "parquet_size_bytes": int(RAW_PARQUET.stat().st_size),
        "n_rows":            int(len(df)),
        "n_columns":         int(df.shape[1]),
        "columns":           list(df.columns),
    })
    section("Done. Run 02_basic_stats.py next.")


if __name__ == "__main__":
    main()
