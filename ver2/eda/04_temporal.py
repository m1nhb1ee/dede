"""Section 1.1 temporal analysis of created_utc.

- Validity of timestamps (range, nulls, out-of-range)
- Density by year and by month (drift detection)
- Density by hour-of-day UTC and day-of-week
  -> drives the temporal-feature decision in tech_plan section 3.2.C
"""

from __future__ import annotations

import pandas as pd

from config import COL_LABEL, COL_TIME
from utils import (
    load_raw, plot_bar, plot_line, section, step, update_stats,
)


def main() -> None:
    section("04 | Temporal analysis (created_utc)")

    df = load_raw(columns=[COL_TIME, COL_LABEL])
    n = len(df)
    step(f"Loaded {n:,} rows")

    ts = pd.to_numeric(df[COL_TIME], errors="coerce")
    n_null = int(ts.isna().sum())

    # Valid Reddit-era range: 2005 (Reddit launched) to "now-ish" 2030.
    REDDIT_EPOCH = pd.Timestamp("2005-01-01", tz="UTC").timestamp()
    FUTURE_LIMIT = pd.Timestamp("2030-01-01", tz="UTC").timestamp()
    invalid_mask = ts.notna() & ((ts < REDDIT_EPOCH) | (ts > FUTURE_LIMIT))
    n_invalid = int(invalid_mask.sum())

    valid = ts[ts.notna() & ~invalid_mask]
    dt = pd.to_datetime(valid, unit="s", utc=True)

    payload: dict = {
        "n_null":      n_null,
        "n_invalid":   n_invalid,
        "n_null_pct":    round(n_null / n * 100, 4),
        "n_invalid_pct": round(n_invalid / n * 100, 4),
        "min_date":      dt.min().isoformat() if not dt.empty else None,
        "max_date":      dt.max().isoformat() if not dt.empty else None,
    }
    step(f"Null timestamps:    {n_null:,} ({payload['n_null_pct']}%)")
    step(f"Invalid timestamps: {n_invalid:,} ({payload['n_invalid_pct']}%)")
    step(f"Date range: {payload['min_date']} -> {payload['max_date']}")

    if dt.empty:
        update_stats("temporal", payload)
        section("Done (no valid timestamps).")
        return

    # ── Year distribution ──────────────────────────────────────────────────
    year_counts = dt.dt.year.value_counts().sort_index()
    payload["by_year"] = {int(k): int(v) for k, v in year_counts.items()}
    plot_bar(
        labels=[str(y) for y in year_counts.index],
        values=year_counts.values.tolist(),
        name="posts_by_year",
        title="Posts by year (UTC)",
        rotation=0,
    )

    # ── Month-of-record (year-month) timeline for drift visualization ──────
    ym = dt.dt.to_period("M").value_counts().sort_index()
    payload["by_year_month_head"] = {str(k): int(v) for k, v in ym.head(6).items()}
    payload["by_year_month_tail"] = {str(k): int(v) for k, v in ym.tail(6).items()}
    plot_line(
        x=[str(p) for p in ym.index],
        y=ym.values.tolist(),
        name="posts_by_year_month",
        title="Posts per month (drift check)",
        xlabel="year-month",
        ylabel="post count",
    )

    # ── Hour-of-day UTC ────────────────────────────────────────────────────
    hour_counts = dt.dt.hour.value_counts().sort_index()
    payload["by_hour_utc"] = {int(k): int(v) for k, v in hour_counts.items()}
    plot_bar(
        labels=[f"{h:02d}" for h in hour_counts.index],
        values=hour_counts.values.tolist(),
        name="posts_by_hour_utc",
        title="Posts by hour-of-day (UTC)",
        rotation=0,
    )

    # ── Day-of-week ────────────────────────────────────────────────────────
    dow_counts = dt.dt.weekday.value_counts().sort_index()
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    payload["by_dow"] = {dow_labels[k]: int(v) for k, v in dow_counts.items()}
    plot_bar(
        labels=dow_labels,
        values=[int(dow_counts.get(i, 0)) for i in range(7)],
        name="posts_by_dow",
        title="Posts by day of week",
        rotation=0,
    )

    # ── Drift check: label rate by year ────────────────────────────────────
    if df[COL_LABEL].notna().any():
        labels_v = df.loc[ts.notna() & ~invalid_mask, COL_LABEL]
        years = dt.dt.year
        lbl_by_year = labels_v.groupby(years).agg(["mean", "size"])
        lbl_by_year = lbl_by_year[lbl_by_year["size"] >= 100]   # noise floor
        payload["label_rate_by_year"] = {
            int(y): {"label_rate": round(float(r["mean"]), 6), "n": int(r["size"])}
            for y, r in lbl_by_year.iterrows()
        }
        if not lbl_by_year.empty:
            plot_line(
                x=[str(y) for y in lbl_by_year.index],
                y=lbl_by_year["mean"].values.tolist(),
                name="label_rate_by_year",
                title="Label rate by year (drift check; only years with n>=100)",
                xlabel="year", ylabel="label rate",
            )
            # Drift severity
            yrs_rate = lbl_by_year["mean"].values
            drift = float(yrs_rate.max() - yrs_rate.min()) if len(yrs_rate) > 1 else 0.0
            payload["label_rate_drift_abs"] = round(drift, 6)
            step(f"Max - Min label rate across years: {drift:.4f}")

    update_stats("temporal", payload)
    section("Done. Run 05_label_rel.py next.")


if __name__ == "__main__":
    main()
