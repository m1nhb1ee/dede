"""Cross-feature interactions -- where the real EDA insights live.

Single-feature distributions only tell you what each variable looks like;
interactions tell you what the model can actually learn from combinations.

Outputs:
  - hour-of-day x label heatmap (does posting time amplify depression signal?)
  - day-of-week x hour heatmap of label rate (full temporal grid)
  - body-length-bucket x label rate (does long body imply depression?)
  - has_body x label (do empty-body posts behave differently?)
  - subreddit x year label rate stability check (drift inside subs)
  - mutual information between each top categorical/binned feature and label

These plots become "the figure" in the report -- they show where the
2-stage pipeline gets its lift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    COL_BODY, COL_LABEL, COL_NCMTS, COL_SUBR, COL_TIME, COL_UPVOTES,
)
from utils import (
    char_len, load_raw, plot_bar, plot_heatmap, section, step, update_stats,
)


def _mutual_info_binary(x: pd.Series, y: pd.Series) -> float:
    """Mutual information I(X; Y) in bits, X categorical, Y binary."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    if n == 0: return 0.0
    p_xy = df.groupby(["x", "y"]).size() / n
    p_x  = df.groupby("x").size() / n
    p_y  = df.groupby("y").size() / n
    mi = 0.0
    for (xv, yv), p in p_xy.items():
        denom = p_x[xv] * p_y[yv]
        if denom > 0 and p > 0:
            mi += p * np.log2(p / denom)
    return float(mi)


def main() -> None:
    section("08 | Cross-feature interactions")

    df = load_raw(columns=[
        COL_SUBR, COL_BODY, COL_UPVOTES, COL_NCMTS, COL_TIME, COL_LABEL,
    ])
    df = df.dropna(subset=[COL_LABEL])
    df[COL_LABEL] = df[COL_LABEL].astype(int)
    n = len(df)
    step(f"{n:,} rows with valid label")

    payload: dict = {}

    # ── Temporal features ──────────────────────────────────────────────────
    ts = pd.to_numeric(df[COL_TIME], errors="coerce")
    valid = ts.notna()
    dt = pd.to_datetime(ts[valid], unit="s", utc=True)
    df_v = df.loc[valid].copy()
    df_v["hour"] = dt.dt.hour.values
    df_v["dow"]  = dt.dt.weekday.values
    df_v["year"] = dt.dt.year.values

    # ── 1. Hour x DOW heatmap of label rate ────────────────────────────────
    grid = df_v.pivot_table(values=COL_LABEL, index="dow", columns="hour",
                            aggfunc="mean").reindex(index=range(7), columns=range(24))
    grid_n = df_v.pivot_table(values=COL_LABEL, index="dow", columns="hour",
                              aggfunc="size").reindex(index=range(7), columns=range(24)).fillna(0)
    # Mask cells with too few samples
    masked = grid.where(grid_n >= 50, other=np.nan)
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plot_heatmap(
        matrix=masked.values,
        row_labels=dow_labels,
        col_labels=[f"{h:02d}" for h in range(24)],
        name="heatmap_label_rate_dow_x_hour",
        title="Label rate by day-of-week x hour-of-day UTC (cells with n>=50)",
        cmap="coolwarm",
        annotate=False,
    )
    payload["hour_x_dow"] = {
        "max_label_rate":      float(np.nanmax(masked.values)) if np.isfinite(np.nanmax(masked.values)) else None,
        "min_label_rate":      float(np.nanmin(masked.values)) if np.isfinite(np.nanmin(masked.values)) else None,
        "max_minus_min":       float(np.nanmax(masked.values) - np.nanmin(masked.values))
                               if np.isfinite(np.nanmax(masked.values)) else None,
    }
    step(f"Hour x DOW label rate range: "
         f"{payload['hour_x_dow']['min_label_rate']} -> "
         f"{payload['hour_x_dow']['max_label_rate']}")

    # ── 2. Top-15 subreddits x year label-rate heatmap (stability) ─────────
    top_subs = df_v[COL_SUBR].value_counts().head(15).index.tolist()
    sub_yr = df_v[df_v[COL_SUBR].isin(top_subs)].pivot_table(
        values=COL_LABEL, index=COL_SUBR, columns="year", aggfunc="mean",
    )
    sub_yr_n = df_v[df_v[COL_SUBR].isin(top_subs)].pivot_table(
        values=COL_LABEL, index=COL_SUBR, columns="year", aggfunc="size",
    ).fillna(0)
    if not sub_yr.empty:
        sub_yr_masked = sub_yr.where(sub_yr_n >= 30, other=np.nan)
        plot_heatmap(
            matrix=sub_yr_masked.values,
            row_labels=[str(s)[:20] for s in sub_yr.index],
            col_labels=[str(c) for c in sub_yr.columns],
            name="heatmap_subreddit_x_year_label_rate",
            title="Label rate per (top-15 subreddit x year), cells with n>=30",
            cmap="coolwarm",
            annotate=True,
            fmt=".2f",
        )
        # Stability: max-min per subreddit
        stab = {}
        for sub in sub_yr_masked.index:
            row = sub_yr_masked.loc[sub].dropna()
            if len(row) >= 2:
                stab[str(sub)] = round(float(row.max() - row.min()), 4)
        payload["subreddit_year_drift_top15"] = stab

    # ── 3. Body-length bucket x label rate ─────────────────────────────────
    bl = char_len(df[COL_BODY])
    bins = [-1, 0, 50, 200, 500, 1000, 2000, 5000, 1_000_000]
    labs = ["empty", "1-50", "51-200", "201-500", "501-1k", "1k-2k", "2k-5k", "5k+"]
    bucket = pd.cut(bl, bins=bins, labels=labs)
    by_b = df.groupby(bucket, observed=False)[COL_LABEL].agg(["mean", "size"])
    payload["body_length_bucket_x_label"] = {
        str(idx): {"label_rate": round(float(r["mean"]), 6), "n": int(r["size"])}
        for idx, r in by_b.iterrows()
    }
    plot_bar(
        labels=[str(i) for i in by_b.index],
        values=by_b["mean"].tolist(),
        name="label_rate_by_body_length_bucket",
        title="Label rate by body-length bucket",
        rotation=0,
    )
    step("Label rate by body-length bucket:")
    for idx, r in by_b.iterrows():
        print(f"      {str(idx):<10}  n={int(r['size']):>8,}  label_rate={r['mean']:.4f}")

    # ── 4. has_body x label ────────────────────────────────────────────────
    has_body = (bl > 0)
    by_hb = df.groupby(has_body)[COL_LABEL].agg(["mean", "size"])
    payload["has_body_x_label"] = {
        str(bool(idx)): {"label_rate": round(float(r["mean"]), 6), "n": int(r["size"])}
        for idx, r in by_hb.iterrows()
    }

    # ── 5. Mutual information of categorical/binned features with label ────
    step("Computing mutual information with label (bits) ...")
    feats = {
        "subreddit (raw)":              df[COL_SUBR].astype(str),
        "hour_utc":                     df_v["hour"].reindex(df.index),
        "dow":                          df_v["dow"].reindex(df.index),
        "year":                         df_v["year"].reindex(df.index),
        "has_body":                     has_body,
        "body_length_bucket":           bucket,
        "upvotes_bucket":               pd.qcut(
                                            pd.to_numeric(df[COL_UPVOTES], errors="coerce"),
                                            q=10, duplicates="drop"
                                        ).astype("string"),
        "num_comments_bucket":          pd.qcut(
                                            pd.to_numeric(df[COL_NCMTS], errors="coerce"),
                                            q=10, duplicates="drop"
                                        ).astype("string"),
    }
    mi = {}
    for name, s in feats.items():
        mi[name] = round(_mutual_info_binary(s, df[COL_LABEL]), 6)
    mi_sorted = sorted(mi.items(), key=lambda r: r[1], reverse=True)
    payload["mutual_info_with_label_bits"] = dict(mi_sorted)
    step("Mutual information with label (bits):")
    for k, v in mi_sorted:
        print(f"      {k:<28} {v:.4f}")
    plot_bar(
        labels=[k for k, _ in mi_sorted],
        values=[v for _, v in mi_sorted],
        name="mutual_info_with_label",
        title="Mutual information I(feature; label) in bits "
              "-- relative discriminative power, ignoring direction.",
        horizontal=True,
    )

    update_stats("interactions", payload)
    section("Done. Run 09_samples.py next.")


if __name__ == "__main__":
    main()
