"""Section 1.2 of the tech plan: feature <-> label relationships.

This script answers the questions that decide whether the 2-stage architecture
is justified or whether subreddit alone leaks the label:

- Label rate per top-30 subreddit  (subreddit leakage check)
- Label rate per hour-of-day UTC and per day-of-week
- Numeric features (upvotes, num_comments, body_len) by label
- Pearson + Spearman correlations of numeric features with label
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    COL_BODY, COL_LABEL, COL_NCMTS, COL_SUBR, COL_TIME, COL_TITLE, COL_UPVOTES,
)
from utils import (
    char_len, load_raw, plot_bar, plot_box_by_group, plot_scatter,
    section, step, update_stats,
)


def main() -> None:
    section("05 | Feature <-> label relationships")

    df = load_raw(columns=[
        COL_SUBR, COL_TITLE, COL_BODY, COL_UPVOTES, COL_TIME, COL_NCMTS, COL_LABEL,
    ])
    df = df.dropna(subset=[COL_LABEL])
    df[COL_LABEL] = df[COL_LABEL].astype(int)
    n = len(df)
    step(f"Loaded {n:,} rows with valid label")

    payload: dict = {}

    # ── Label rate per subreddit (top by frequency) ────────────────────────
    grp = df.groupby(COL_SUBR, observed=True)[COL_LABEL]
    by_subr = pd.DataFrame({
        "n":          grp.size(),
        "label_rate": grp.mean(),
    }).sort_values("n", ascending=False)

    top30 = by_subr.head(30)
    payload["label_rate_top30_subreddits"] = {
        str(idx): {"n": int(row["n"]), "label_rate": round(float(row["label_rate"]), 6)}
        for idx, row in top30.iterrows()
    }

    # Leakage flag: subreddits where label_rate is >0.98 or <0.02 *and* n is large
    leakage_mask = (
        (by_subr["n"] >= 1000)
        & ((by_subr["label_rate"] >= 0.98) | (by_subr["label_rate"] <= 0.02))
    )
    leaky = by_subr[leakage_mask].sort_values("n", ascending=False)
    payload["leaky_subreddits"] = {
        str(idx): {"n": int(row["n"]), "label_rate": round(float(row["label_rate"]), 6)}
        for idx, row in leaky.head(50).iterrows()
    }
    payload["n_leaky_subreddits"] = int(len(leaky))
    step(f"Subreddits with extreme label rate (>=0.98 or <=0.02) and n>=1000: {len(leaky):,}")

    plot_bar(
        labels=[str(s) for s in top30.index],
        values=top30["label_rate"].values.tolist(),
        name="label_rate_top30_subreddits",
        title="Label rate (top-30 subreddits by post count)",
        horizontal=True,
    )

    # ── Subreddit scatter: n (log) vs label_rate -- find informative subs ──
    # Sweet spot = top-right or bottom-right: high n AND extreme label_rate
    subr_for_scatter = by_subr[by_subr["n"] >= 50].copy()
    if not subr_for_scatter.empty:
        top_labels = subr_for_scatter.nlargest(15, "n")
        annotate = [
            (float(row["n"]), float(row["label_rate"]), str(idx)[:18])
            for idx, row in top_labels.iterrows()
        ]
        plot_scatter(
            x=subr_for_scatter["n"], y=subr_for_scatter["label_rate"],
            name="scatter_subreddit_n_vs_label_rate",
            title="Per-subreddit: post count (log) vs label rate "
                  "-- top-right/bottom-right = strong signal",
            xlabel="post count (log)", ylabel="label rate",
            log_x=True, alpha=0.4,
            labels_to_annotate=annotate,
        )
        # Concentration: how many subreddits cover 50% / 80% / 95% of label=1 rows?
        ones_per_sub = (subr_for_scatter["n"] * subr_for_scatter["label_rate"]).sort_values(ascending=False)
        total_ones = float(ones_per_sub.sum())
        if total_ones > 0:
            cum = ones_per_sub.cumsum() / total_ones
            concentration = {
                "subs_covering_50pct_of_positives": int((cum < 0.5).sum() + 1),
                "subs_covering_80pct_of_positives": int((cum < 0.8).sum() + 1),
                "subs_covering_95pct_of_positives": int((cum < 0.95).sum() + 1),
            }
            payload["label1_concentration_in_subreddits"] = concentration
            step(f"Positives concentration: {concentration}")

    # ── Label rate by hour-of-day UTC ──────────────────────────────────────
    ts = pd.to_numeric(df[COL_TIME], errors="coerce")
    valid = ts.notna()
    dt = pd.to_datetime(ts[valid], unit="s", utc=True)
    labels_v = df.loc[valid, COL_LABEL]

    hour_grp = labels_v.groupby(dt.dt.hour).mean().reindex(range(24)).fillna(0)
    payload["label_rate_by_hour_utc"] = {int(k): round(float(v), 6) for k, v in hour_grp.items()}
    plot_bar(
        labels=[f"{h:02d}" for h in hour_grp.index],
        values=hour_grp.values.tolist(),
        name="label_rate_by_hour_utc",
        title="Label rate by hour-of-day (UTC)",
        rotation=0,
    )

    dow_grp = labels_v.groupby(dt.dt.weekday).mean().reindex(range(7)).fillna(0)
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    payload["label_rate_by_dow"] = {dow_labels[i]: round(float(dow_grp.iloc[i]), 6) for i in range(7)}
    plot_bar(
        labels=dow_labels,
        values=dow_grp.values.tolist(),
        name="label_rate_by_dow",
        title="Label rate by day of week",
        rotation=0,
    )

    # ── Numeric features vs label ──────────────────────────────────────────
    body_len = char_len(df[COL_BODY])
    title_len = char_len(df[COL_TITLE])

    bydf = pd.DataFrame({
        COL_LABEL:   df[COL_LABEL].values,
        COL_UPVOTES: pd.to_numeric(df[COL_UPVOTES], errors="coerce"),
        COL_NCMTS:   pd.to_numeric(df[COL_NCMTS], errors="coerce"),
        "body_len":  body_len.values,
        "title_len": title_len.values,
    })

    # Means by label class
    means_by_label = bydf.groupby(COL_LABEL).agg(["mean", "median"])
    payload["numeric_features_by_label"] = {
        str(lbl): {
            f"{col}_{stat}": round(float(means_by_label.loc[lbl, (col, stat)]), 4)
            for col in [COL_UPVOTES, COL_NCMTS, "body_len", "title_len"]
            for stat in ["mean", "median"]
            if not pd.isna(means_by_label.loc[lbl, (col, stat)])
        }
        for lbl in means_by_label.index
    }

    # Correlations with label (use Spearman for skewed Reddit numerics)
    corr_payload = {}
    for col in [COL_UPVOTES, COL_NCMTS, "body_len", "title_len"]:
        s = bydf[col].dropna()
        l = bydf.loc[s.index, COL_LABEL]
        if s.std(skipna=True) == 0 or l.std(skipna=True) == 0:
            corr_payload[col] = {"pearson": None, "spearman": None}
            continue
        pearson = float(np.corrcoef(s.values, l.values)[0, 1])
        spearman = float(pd.Series(s.values).rank().corr(pd.Series(l.values).rank()))
        corr_payload[col] = {
            "pearson":  round(pearson, 6),
            "spearman": round(spearman, 6),
        }
    payload["correlation_with_label"] = corr_payload
    step("Correlations with label (Spearman):")
    for col, v in corr_payload.items():
        sp = v["spearman"]
        sp_str = f"{sp:+.4f}" if sp is not None else "n/a"
        print(f"      {col:<14} {sp_str}")

    # Box plots: body_len vs label, upvotes vs label
    plot_box_by_group(
        df=pd.DataFrame({"body_len": body_len.values, COL_LABEL: df[COL_LABEL].values}),
        value_col="body_len", group_col=COL_LABEL,
        name="box_body_len_by_label",
        title="Body length (chars) by label (clipped p99)",
    )
    plot_box_by_group(
        df=pd.DataFrame({COL_UPVOTES: pd.to_numeric(df[COL_UPVOTES], errors="coerce").values,
                         COL_LABEL: df[COL_LABEL].values}),
        value_col=COL_UPVOTES, group_col=COL_LABEL,
        name="box_upvotes_by_label",
        title="Upvotes by label (clipped p99)",
    )

    update_stats("relationships", payload)
    section("Done. Run 06_report.py next.")


if __name__ == "__main__":
    main()
