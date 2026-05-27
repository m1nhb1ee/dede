"""Section 1.1 of the tech plan: sanity checks + per-column distributions.

- Row counts, duplicates, missing per column
- Label distribution (decides loss + class weighting)
- Subreddit distribution (decides rare bucket + cross-subreddit splits)
- Numeric distributions for upvotes / num_comments (skew, negatives, outliers)
"""

from __future__ import annotations

import pandas as pd

import numpy as np

from config import (
    COL_ID, COL_LABEL, COL_NCMTS, COL_SUBR, COL_UPVOTES, RARE_SUBR_MIN_COUNT,
    SEED,
)
from utils import (
    describe_numeric, load_raw, missing_summary, plot_bar, plot_hist,
    plot_kde_overlay, plot_scatter, section, step, update_stats, value_counts_dict,
)


def main() -> None:
    section("02 | Basic stats + per-column distributions")

    df = load_raw()
    n = len(df)
    step(f"Loaded {n:,} rows")

    payload: dict = {"n_rows": n}

    # ── Duplicates ──────────────────────────────────────────────────────────
    n_dup = int(df.duplicated(subset=[COL_ID]).sum())
    payload["duplicates_id"] = n_dup
    payload["duplicates_id_pct"] = round(n_dup / n * 100, 4)
    step(f"Duplicate ids: {n_dup:,} ({payload['duplicates_id_pct']}%)")

    # ── Missing per column ──────────────────────────────────────────────────
    miss = missing_summary(df)
    payload["missing_per_column"] = miss
    step("Missing per column:")
    for c, m in miss.items():
        print(f"      {c:<14} {m['missing']:>10,}  ({m['missing_pct']}%)")

    # ── Label distribution ──────────────────────────────────────────────────
    label_counts = value_counts_dict(df[COL_LABEL].dropna())
    n_label = sum(label_counts.values())
    label_pct = {k: round(v / n_label * 100, 4) for k, v in label_counts.items()}
    payload["label"] = {
        "n_unique":    len(label_counts),
        "counts":      label_counts,
        "percentages": label_pct,
        "is_binary":   set(label_counts.keys()).issubset({"0", "1"}),
    }
    step(f"Label classes: {label_counts}")

    plot_bar(
        labels=list(label_counts.keys()),
        values=list(label_counts.values()),
        name="label_distribution",
        title="Label distribution",
        rotation=0,
    )

    # ── Subreddit distribution ──────────────────────────────────────────────
    subr_counts = df[COL_SUBR].value_counts(dropna=False)
    n_subr = int(subr_counts.shape[0])
    n_rare = int((subr_counts < RARE_SUBR_MIN_COUNT).sum())
    rare_rows = int(subr_counts[subr_counts < RARE_SUBR_MIN_COUNT].sum())

    payload["subreddit"] = {
        "n_unique":               n_subr,
        "n_rare_below_threshold": n_rare,
        "rare_threshold":         RARE_SUBR_MIN_COUNT,
        "rare_rows_total":        rare_rows,
        "rare_rows_pct":          round(rare_rows / n * 100, 4),
        "top20":                  {str(k): int(v) for k, v in subr_counts.head(20).items()},
    }
    step(f"Unique subreddits: {n_subr:,} | rare (<{RARE_SUBR_MIN_COUNT}): {n_rare:,}")

    top20 = subr_counts.head(20)
    plot_bar(
        labels=[str(s) for s in top20.index],
        values=top20.values.tolist(),
        name="top20_subreddits",
        title="Top-20 subreddits by post count",
        horizontal=True,
    )

    # Long-tail visualization: count of subreddits at each frequency bucket
    bucket_edges = [0, 10, 50, 100, 500, 1000, 5000, 10_000, 100_000, 10_000_000]
    bucket_labels = [f"{a}-{b}" for a, b in zip(bucket_edges[:-1], bucket_edges[1:])]
    buckets = pd.cut(subr_counts, bins=bucket_edges, labels=bucket_labels)
    bucket_counts = buckets.value_counts().reindex(bucket_labels).fillna(0).astype(int)
    payload["subreddit"]["long_tail_buckets"] = bucket_counts.to_dict()
    plot_bar(
        labels=bucket_labels,
        values=bucket_counts.values.tolist(),
        name="subreddit_long_tail",
        title="Subreddit frequency buckets (how many subs have N posts)",
        rotation=30,
    )

    # ── Numeric distributions ───────────────────────────────────────────────
    for col in (COL_UPVOTES, COL_NCMTS):
        stats = describe_numeric(df[col])
        payload[col] = stats
        step(f"{col}: median={stats.get('median')}, p99={stats.get('p99')}, "
             f"max={stats.get('max')}, n_neg={stats.get('n_negative')}")

        plot_hist(df[col], name=f"hist_{col}",
                  title=f"{col} (clipped at p99)", clip_q=0.99)
        plot_hist(df[col], name=f"hist_{col}_log",
                  title=f"{col} (log-y, clipped at p99)",
                  clip_q=0.99, log_y=True)

    # ── Numeric distribution by label (insight: do classes differ?) ────────
    if df[COL_LABEL].notna().any():
        for col in (COL_UPVOTES, COL_NCMTS):
            by_lbl = {}
            for lbl in sorted(df[COL_LABEL].dropna().unique()):
                by_lbl[str(int(lbl))] = pd.to_numeric(
                    df.loc[df[COL_LABEL] == lbl, col], errors="coerce"
                ).dropna()
            plot_kde_overlay(
                by_lbl, name=f"kde_{col}_by_label",
                title=f"{col} distribution by label (log10)",
                xlabel=col, log_x=True,
            )

    # ── Engagement scatter (upvotes vs comments, colored by label) ─────────
    eng = df[[COL_UPVOTES, COL_NCMTS, COL_LABEL]].dropna()
    eng = eng.sample(n=min(50_000, len(eng)), random_state=SEED)
    eng_up = pd.to_numeric(eng[COL_UPVOTES], errors="coerce").clip(lower=1)
    eng_co = pd.to_numeric(eng[COL_NCMTS], errors="coerce").clip(lower=1)
    plot_scatter(
        x=eng_up, y=eng_co,
        color=eng[COL_LABEL].astype(float),
        name="scatter_engagement",
        title="Engagement: upvotes vs num_comments (color = label, sample 50K)",
        xlabel="upvotes (log)", ylabel="num_comments (log)",
        log_x=True, log_y=True, alpha=0.3,
    )

    # ── Engagement ratio: comments per upvote ─────────────────────────────
    ratio = (pd.to_numeric(df[COL_NCMTS], errors="coerce") /
             (pd.to_numeric(df[COL_UPVOTES], errors="coerce") + 1)).rename("comments_per_upvote")
    payload["comments_per_upvote"] = describe_numeric(ratio)
    if df[COL_LABEL].notna().any():
        by_lbl_ratio = {}
        for lbl in sorted(df[COL_LABEL].dropna().unique()):
            by_lbl_ratio[str(int(lbl))] = ratio[df[COL_LABEL] == lbl].dropna()
        plot_kde_overlay(
            by_lbl_ratio, name="kde_comments_per_upvote_by_label",
            title="comments_per_upvote by label (log10)",
            xlabel="comments_per_upvote", log_x=True,
        )

    update_stats("basic", payload)
    section("Done. Run 03_text_analysis.py next.")


if __name__ == "__main__":
    main()
