"""Final EDA report -- consolidates every script's findings into:

  outputs/summary_stats.csv  : flat key,value of every metric we computed
  outputs/data_issues.md     : insight-first narrative with recommended actions

The report is intentionally opinionated:
  - top of the file leads with KEY INSIGHTS the reader needs to act on
  - then a Recommended Actions section (cleaning + modeling decisions)
  - then a detailed section per analysis area for evidence
  - last: pointers to every CSV/PNG generated

No new computation happens here -- everything is read from stats.json.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    ISSUES_MD, OUT_DIR, PLOTS_DIR, SUMMARY_CSV, TOKEN_THRESHOLDS,
)
from utils import load_stats, section, step


# ─────────────────────────────────────────────────────────────────────────────
# summary_stats.csv  -- one row per metric, flattened key path
# ─────────────────────────────────────────────────────────────────────────────


def _flatten(prefix: str, val: Any, out: list[tuple[str, Any]]) -> None:
    if isinstance(val, dict):
        for k, v in val.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(val, list):
        for i, v in enumerate(val):
            _flatten(f"{prefix}[{i}]", v, out)
    else:
        out.append((prefix, val))


def write_summary_csv(stats: dict[str, Any]) -> None:
    rows: list[tuple[str, Any]] = []
    _flatten("", stats, rows)
    rows.sort(key=lambda r: r[0])
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)
    step(f"Wrote {SUMMARY_CSV.name}  ({len(rows):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for nested-dict access without crashing on missing keys
# ─────────────────────────────────────────────────────────────────────────────


def _safe(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt_pct(x) -> str:
    if x is None: return "n/a"
    return f"{float(x):.2f}%"


def _fmt_int(x) -> str:
    if x is None: return "n/a"
    try: return f"{int(x):,}"
    except Exception: return str(x)


# ─────────────────────────────────────────────────────────────────────────────
# Insight derivation -- compute a list of (severity, text) findings
# severity: "CRITICAL" | "WARNING" | "INFO" | "GOOD"
# ─────────────────────────────────────────────────────────────────────────────


def derive_insights(s: dict) -> list[tuple[str, str]]:
    insights: list[tuple[str, str]] = []

    # ── Language ──
    lang_en = _safe(s, "quality", "language_detection", "english_pct")
    detector = _safe(s, "quality", "language_detection", "detector", default="?")
    if lang_en is not None:
        if lang_en < 90:
            insights.append((
                "CRITICAL",
                f"Only **{lang_en:.1f}%** of posts detected as English "
                f"(detector={detector}). MentalRoBERTa was pretrained on English; "
                "non-English contamination will silently cap performance. "
                "**Action:** filter to English-only before training."
            ))
        elif lang_en < 98:
            insights.append((
                "WARNING",
                f"{lang_en:.1f}% English (detector={detector}). Some non-English "
                "contamination -- consider filtering or treat as noise."
            ))
        else:
            insights.append((
                "GOOD",
                f"{lang_en:.1f}% English -- dataset is monolingual as advertised."
            ))

    # ── Duplicates ──
    dup_pct = _safe(s, "quality", "body_duplicates", "duplicate_rows_pct")
    biggest = _safe(s, "quality", "body_duplicates", "biggest_group_size")
    if dup_pct is not None and dup_pct > 1.0:
        insights.append((
            "WARNING" if dup_pct < 5 else "CRITICAL",
            f"**{dup_pct:.2f}%** of posts are exact-body duplicates "
            f"(biggest group = {biggest} copies). These will leak across "
            "train/test splits if not deduplicated. **Action:** drop duplicates "
            "BEFORE splitting; do not just `drop_duplicates(['id'])`."
        ))
    elif dup_pct is not None:
        insights.append(("GOOD", f"Body duplicates negligible ({dup_pct:.2f}%)."))

    # ── Subreddit leakage ──
    n_leaky = _safe(s, "relationships", "n_leaky_subreddits", default=0)
    n_subs  = _safe(s, "basic", "subreddit", "n_unique", default=0)
    if n_leaky and n_subs:
        insights.append((
            "WARNING",
            f"**{n_leaky:,} subreddits** (of {n_subs:,}) have extreme label rate "
            "(>=98% or <=2%) with n>=1000. The `subreddit` feature alone can "
            "trivially predict label on these. **Action:** run an LGBM ablation "
            "WITHOUT subreddit to measure true text+metadata contribution."
        ))

    # ── Class imbalance ──
    lbl_pcts = list(_safe(s, "basic", "label", "percentages", default={}).values())
    if lbl_pcts:
        minority = min(lbl_pcts)
        if minority < 10:
            insights.append((
                "CRITICAL",
                f"Severe class imbalance -- minority class is **{minority:.1f}%**. "
                "Use focal loss or pos_weight in Stage 1; `class_weight='balanced'` "
                "in LGBM; report PR-AUC not just F1."
            ))
        elif minority < 30:
            insights.append((
                "WARNING",
                f"Moderate imbalance -- minority class {minority:.1f}%. "
                "Class weighting recommended."
            ))
        else:
            insights.append((
                "GOOD",
                f"Classes are roughly balanced (minority {minority:.1f}%)."
            ))

    # ── Token coverage ──
    cov = _safe(s, "text", "token_length_pair", "coverage_pct_at", default={})
    cov_256 = cov.get("256")
    cov_384 = cov.get("384")
    cov_512 = cov.get("512")
    if cov_256 is not None:
        if cov_256 >= 95:
            insights.append((
                "GOOD",
                f"`max_length=256` covers {cov_256:.1f}% of posts -- planned "
                "default is appropriate."
            ))
        elif cov_384 is not None and cov_384 >= 95:
            insights.append((
                "INFO",
                f"`max_length=256` only covers {cov_256:.1f}% of posts; "
                f"bump to 384 (covers {cov_384:.1f}%) if VRAM allows."
            ))
        elif cov_512 is not None and cov_512 >= 95:
            insights.append((
                "WARNING",
                f"`max_length=256` covers only {cov_256:.1f}%. Use 512 "
                f"({cov_512:.1f}% coverage) or head+tail truncation."
            ))
        else:
            insights.append((
                "WARNING",
                f"Even max_length=512 misses many posts. Consider Longformer "
                "or hierarchical splitting for long posts."
            ))

    # ── Text signal strength (does vocabulary actually distinguish classes?) ──
    lift = _safe(s, "text_signals", "lexical_lift_label1_over_label0", default={})
    mh_lift = _safe(lift, "has_mh_kw_pct", "lift_1_over_0")
    fp_lift = _safe(lift, "first_person_rate", "lift_1_over_0")
    if mh_lift is not None:
        if mh_lift >= 2.0:
            insights.append((
                "GOOD",
                f"Strong text signal: MH-keyword presence is **{mh_lift:.1f}x** "
                "more common in label=1 than label=0. Text model has clear "
                "signal to learn."
            ))
        elif mh_lift >= 1.3:
            insights.append((
                "INFO",
                f"Moderate text signal: MH-keyword lift = {mh_lift:.2f}x. "
                "Text model should help but not be the only signal."
            ))
        else:
            insights.append((
                "WARNING",
                f"Weak text signal: MH-keyword lift only {mh_lift:.2f}x. "
                "Labels may be subreddit-derived rather than content-derived "
                "-- text model may struggle."
            ))
    if fp_lift is not None and fp_lift >= 1.2:
        insights.append((
            "INFO",
            f"First-person pronoun use is {fp_lift:.2f}x more common in label=1 "
            "(matches clinical depression literature -- self-focus marker)."
        ))

    # ── Temporal drift ──
    drift = _safe(s, "temporal", "label_rate_drift_abs")
    if drift is not None and drift > 0.10:
        insights.append((
            "WARNING",
            f"Label rate drifts by **{drift:.2f}** between earliest and latest "
            "year. Time-based split (train on older, test on newer) will give "
            "more honest generalization estimate than random split."
        ))

    # ── Mutual information ranking ──
    mi = _safe(s, "interactions", "mutual_info_with_label_bits", default={})
    if mi:
        top_feat, top_mi = list(mi.items())[0]
        insights.append((
            "INFO",
            f"Most informative single feature for label is **{top_feat}** "
            f"(MI = {top_mi:.4f} bits). Sanity-check whether this matches "
            "expectations -- if it's `subreddit` by a large margin, leakage."
        ))

    # ── Hour x DOW interaction strength ──
    rng = _safe(s, "interactions", "hour_x_dow", "max_minus_min")
    if rng is not None and rng > 0.05:
        insights.append((
            "INFO",
            f"Hour-of-day x day-of-week label rate varies by {rng:.3f} across "
            "the grid -- temporal features (cyclical encoding) are worth "
            "including in LGBM."
        ))

    # ── Noise levels ──
    n_url_only = _safe(s, "quality", "noise", "url_only_posts", default=0)
    if n_url_only and n_url_only > 5000:
        insights.append((
            "INFO",
            f"{n_url_only:,} URL-only posts -- replace URLs with `[URL]` token "
            "during cleaning (already in tech_plan section 2.2)."
        ))

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────


SEVERITY_ICONS = {
    "CRITICAL": "[CRITICAL]",
    "WARNING":  "[WARNING]",
    "INFO":     "[INFO]",
    "GOOD":     "[OK]",
}


def write_issues_md(s: dict[str, Any]) -> None:
    md: list[str] = []
    md.append("# EDA Report -- Depression Detection Dataset\n")
    md.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")
    md.append("\n---\n\n")

    # ============================ HEADLINE INSIGHTS ============================
    insights = derive_insights(s)
    # Sort by severity rank: CRITICAL > WARNING > INFO > GOOD
    rank = {"CRITICAL": 0, "WARNING": 1, "INFO": 2, "GOOD": 3}
    insights.sort(key=lambda r: rank[r[0]])

    md.append("## TL;DR -- Headline Insights\n\n")
    if not insights:
        md.append("_No insights derived; check that stats.json was populated._\n")
    else:
        for sev, text in insights:
            md.append(f"- {SEVERITY_ICONS[sev]} {text}\n")
    md.append("\n---\n\n")

    # ============================ RECOMMENDED ACTIONS =========================
    md.append("## Recommended Actions (consolidated)\n\n")
    md.append("### Before training (data cleaning)\n")
    md.append("1. Drop rows where `label` is null or `id` is duplicated.\n")
    md.append("2. **Deduplicate by normalized body** (lowercased, whitespace-collapsed) "
              "BEFORE the train/val/test split -- not just by `id`.\n")
    md.append("3. Filter to English-only if non-English share is significant.\n")
    md.append("4. Mark `has_body=False` for empty / `[removed]` / `[deleted]` bodies; "
              "keep the row (title alone is usable). Drop rows with both title and body empty.\n")
    md.append("5. Replace URLs / `u/xxx` / `r/xxx` with sentinel tokens "
              "(`[URL]`, `[USER]`, `[SUB]`).\n")
    md.append("6. Bucket rare subreddits (n < 50) into `_rare_` BEFORE feeding LGBM.\n")
    md.append("7. Clip `upvotes` and `num_comments` at p99.5; store both raw and `log1p`.\n")

    md.append("\n### Stage 1 (MentalRoBERTa) decisions\n")
    cov = _safe(s, "text", "token_length_pair", "coverage_pct_at", default={})
    cov_256 = cov.get("256")
    if cov_256 is not None and cov_256 >= 95:
        md.append("- `max_length = 256` (covers >=95% of posts without truncation).\n")
    elif cov.get("384") is not None and cov["384"] >= 95:
        md.append("- `max_length = 384` (256 missed too much).\n")
    else:
        md.append("- `max_length = 512` or head+tail truncation strategy.\n")
    lbl_pcts = list(_safe(s, "basic", "label", "percentages", default={}).values())
    if lbl_pcts and min(lbl_pcts) < 30:
        md.append("- Use **Focal Loss** (gamma=2, alpha aligned to positive rate) "
                  "OR weighted BCE with `pos_weight = neg/pos`.\n")
    md.append("- Label smoothing 0.05 (helps with noisy Reddit labels).\n")
    md.append("- Apply the custom head from tech_plan section 5.3 "
              "(layer-avg + dual pool + residual FFN + multi-sample dropout).\n")

    md.append("\n### Stage 2 (LightGBM) decisions\n")
    md.append("- `subreddit` as native categorical feature (no one-hot).\n")
    md.append("- Cyclical encoding (sin/cos) for `hour_utc`, `dow`, `month`.\n")
    md.append("- Run **ablation without `subreddit`** to measure the true contribution "
              "of text + temporal + engagement features.\n")
    md.append("- `class_weight='balanced'` if imbalanced.\n")

    md.append("\n### Splitting\n")
    drift = _safe(s, "temporal", "label_rate_drift_abs")
    if drift is not None and drift > 0.10:
        md.append("- Strongly consider a **time-based split** (train on older years, "
                  "test on newer) given the label-rate drift over time.\n")
    md.append("- Stratify by `label` at minimum; ideally by `(label, subreddit_bucket)`.\n")
    md.append("- Apply deduplication BEFORE splitting.\n")

    md.append("\n---\n\n")

    # ============================ DETAILED EVIDENCE ===========================
    md.append("## Detailed evidence\n")

    # ── 1. Dataset provenance ──
    md.append("\n### 1. Dataset overview\n")
    ds = s.get("dataset", {})
    md.append(f"- Rows: **{_fmt_int(ds.get('n_rows'))}**\n")
    md.append(f"- Columns: {ds.get('n_columns', '?')}  "
              f"(`{', '.join(ds.get('columns', []))}`)\n")
    md.append(f"- CSV: `{ds.get('csv_path', '?')}`  "
              f"({(ds.get('csv_size_bytes', 0)/1e9):.2f} GB)\n")
    md.append(f"- Parquet cache: `{ds.get('parquet_path', '?')}`\n")

    # ── 2. Missingness + duplicates ──
    md.append("\n### 2. Missingness and ID duplicates\n")
    miss = _safe(s, "basic", "missing_per_column", default={})
    if miss:
        md.append("\n| column | missing | % |\n|---|---:|---:|\n")
        for c, m in miss.items():
            md.append(f"| {c} | {_fmt_int(m['missing'])} | {m['missing_pct']}% |\n")
    md.append(f"\n- Duplicate `id`: {_fmt_int(_safe(s, 'basic', 'duplicates_id'))} "
              f"({_safe(s, 'basic', 'duplicates_id_pct')}%)\n")

    # ── 3. Label distribution ──
    md.append("\n### 3. Label distribution\n")
    lbl = _safe(s, "basic", "label", default={})
    md.append(f"- Unique classes: {lbl.get('n_unique', '?')} | "
              f"binary: {lbl.get('is_binary', '?')}\n")
    for k, v in lbl.get("counts", {}).items():
        pct = lbl.get("percentages", {}).get(k, 0)
        md.append(f"- class **{k}** -> {_fmt_int(v)} ({pct}%)\n")

    # ── 4. Subreddit ──
    md.append("\n### 4. Subreddit\n")
    sub = _safe(s, "basic", "subreddit", default={})
    md.append(f"- Unique subreddits: {_fmt_int(sub.get('n_unique'))}\n")
    md.append(f"- Rare (<{sub.get('rare_threshold', '?')} posts): "
              f"{_fmt_int(sub.get('n_rare_below_threshold'))} subreddits, "
              f"{_fmt_int(sub.get('rare_rows_total'))} rows "
              f"({sub.get('rare_rows_pct', '?')}%)\n")
    conc = _safe(s, "relationships", "label1_concentration_in_subreddits", default={})
    if conc:
        md.append(f"- Positives concentration: top "
                  f"**{conc.get('subs_covering_50pct_of_positives')}** subs cover 50% of label=1 "
                  f"| top {conc.get('subs_covering_80pct_of_positives')} cover 80% "
                  f"| top {conc.get('subs_covering_95pct_of_positives')} cover 95%.\n")
    leaky = _safe(s, "relationships", "leaky_subreddits", default={})
    if leaky:
        md.append(f"- **Leaky subreddits** (label rate >=98% or <=2%, n>=1000): "
                  f"{len(leaky)} found. Top 10:\n\n")
        md.append("| subreddit | n | label_rate |\n|---|---:|---:|\n")
        for i, (sub_name, info) in enumerate(leaky.items()):
            if i >= 10: break
            md.append(f"| {sub_name} | {_fmt_int(info['n'])} | {info['label_rate']:.3f} |\n")

    # ── 5. Text length ──
    md.append("\n### 5. Text length and emptiness\n")
    em = _safe(s, "text", "emptiness", default={})
    md.append(f"- Body empty / [removed] / [deleted]: "
              f"{_fmt_int(em.get('body_empty_or_removed'))} "
              f"({em.get('body_empty_or_removed_pct', '?')}%)\n")
    md.append(f"- Title empty: {_fmt_int(em.get('title_empty'))} "
              f"({em.get('title_empty_pct', '?')}%)\n")
    cov = _safe(s, "text", "token_length_pair", "coverage_pct_at", default={})
    if cov:
        md.append("\n**Token coverage of (title, body) pairs** (MentalRoBERTa tokenizer):\n\n")
        md.append("| max_length | % posts fitting (no truncation) |\n|---:|---:|\n")
        for t in TOKEN_THRESHOLDS:
            md.append(f"| {t} | {cov.get(str(t), 'n/a')}% |\n")

    # ── 6. Numeric features ──
    md.append("\n### 6. Numeric features\n")
    for col in ("upvotes", "num_comments"):
        st = _safe(s, "basic", col, default={})
        if not st: continue
        md.append(f"- **{col}**: median={st.get('median')}, p95={st.get('p95')}, "
                  f"p99={st.get('p99')}, max={st.get('max')}, "
                  f"skew={st.get('skewness', 0):.2f}, neg={st.get('n_negative')}\n")

    # ── 7. Temporal ──
    md.append("\n### 7. Temporal\n")
    tm = s.get("temporal", {})
    md.append(f"- Date range: {tm.get('min_date', '?')} -> {tm.get('max_date', '?')}\n")
    md.append(f"- Null/invalid timestamps: {_fmt_int(tm.get('n_null'))} null, "
              f"{_fmt_int(tm.get('n_invalid'))} invalid\n")
    if "label_rate_drift_abs" in tm:
        md.append(f"- Label-rate drift across years (max-min): "
                  f"**{tm['label_rate_drift_abs']:.4f}**\n")

    # ── 8. Feature <-> label ──
    md.append("\n### 8. Feature <-> label correlations (Spearman)\n")
    corr = _safe(s, "relationships", "correlation_with_label", default={})
    if corr:
        md.append("\n| feature | Pearson | Spearman |\n|---|---:|---:|\n")
        for col, v in corr.items():
            p = v.get("pearson")
            sp = v.get("spearman")
            md.append(f"| {col} | "
                      f"{'n/a' if p is None else f'{p:+.4f}'} | "
                      f"{'n/a' if sp is None else f'{sp:+.4f}'} |\n")

    # ── 9. Text signals ──
    md.append("\n### 9. Text signals (lexical distinctiveness per class)\n")
    sig = _safe(s, "text_signals", "lexical_by_label", default={})
    if sig:
        md.append("\n| metric |")
        for lbl in sig: md.append(f" label={lbl} |")
        md.append("\n|---|" + "---:|" * len(sig) + "\n")
        for key in ("has_mh_kw_pct", "first_person_rate", "negation_rate",
                    "neg_affect_rate", "mean_words", "mean_exclaim",
                    "mean_question", "mean_caps_words"):
            row = [f"| {key} |"]
            for lbl in sig:
                row.append(f" {sig[lbl].get(key, 'n/a')} |")
            md.append("".join(row) + "\n")

    md.append("\n**Top-15 most distinctive unigrams for label=1:**\n\n")
    md.append("| term | z-score | n(label=1) | n(label=0) |\n|---|---:|---:|---:|\n")
    for r in (_safe(s, "text_signals", "top_distinctive_unigrams_label1_head",
                    default=[]) or [])[:15]:
        md.append(f"| `{r['term']}` | {r['z']:+.2f} | "
                  f"{_fmt_int(r['n1'])} | {_fmt_int(r['n0'])} |\n")

    md.append("\n**Top-15 most distinctive unigrams for label=0:**\n\n")
    md.append("| term | z-score | n(label=0) | n(label=1) |\n|---|---:|---:|---:|\n")
    for r in (_safe(s, "text_signals", "top_distinctive_unigrams_label0_head",
                    default=[]) or [])[:15]:
        md.append(f"| `{r['term']}` | {r['z']:+.2f} | "
                  f"{_fmt_int(r['n0'])} | {_fmt_int(r['n1'])} |\n")

    md.append("\n**Top-15 most distinctive bigrams for label=1:**\n\n")
    md.append("| bigram | z-score | n(label=1) | n(label=0) |\n|---|---:|---:|---:|\n")
    for r in (_safe(s, "text_signals", "top_distinctive_bigrams_label1_head",
                    default=[]) or [])[:15]:
        md.append(f"| `{r['term']}` | {r['z']:+.2f} | "
                  f"{_fmt_int(r['n1'])} | {_fmt_int(r['n0'])} |\n")

    # ── 10. Quality ──
    md.append("\n### 10. Data quality\n")
    lang = _safe(s, "quality", "language_detection", default={})
    if lang:
        md.append(f"- Language detection (detector=`{lang.get('detector')}`, "
                  f"sample n={_fmt_int(lang.get('sample_size'))}): "
                  f"**{lang.get('english_pct')}%** English, "
                  f"{lang.get('non_english_pct')}% non-English\n")
    dup = _safe(s, "quality", "body_duplicates", default={})
    if dup:
        md.append(f"- Exact body duplicates: {_fmt_int(dup.get('n_duplicate_rows_extra'))} "
                  f"rows ({dup.get('duplicate_rows_pct')}%) in "
                  f"{_fmt_int(dup.get('n_duplicate_groups'))} groups. "
                  f"Biggest group = {dup.get('biggest_group_size')} copies.\n")
    tb = _safe(s, "quality", "title_equals_body", default={})
    if tb:
        md.append(f"- Title == body: {_fmt_int(tb.get('n'))} ({tb.get('pct')}%)\n")
    noise = _safe(s, "quality", "noise", default={})
    if noise:
        md.append(f"- URL-only posts: {_fmt_int(noise.get('url_only_posts'))} "
                  f"({noise.get('url_only_pct')}%)\n")
        md.append(f"- Emoji-heavy posts: {_fmt_int(noise.get('emoji_heavy_posts'))} "
                  f"({noise.get('emoji_heavy_pct')}%)\n")
        md.append(f"- Extreme-long posts (>20K chars): "
                  f"{_fmt_int(noise.get('extreme_long_posts'))} "
                  f"({noise.get('extreme_long_pct')}%)\n")

    # ── 11. Interactions ──
    md.append("\n### 11. Cross-feature interactions\n")
    mi = _safe(s, "interactions", "mutual_info_with_label_bits", default={})
    if mi:
        md.append("\n**Mutual information I(feature; label), bits:**\n\n")
        md.append("| feature | MI (bits) |\n|---|---:|\n")
        for k, v in mi.items():
            md.append(f"| {k} | {v:.4f} |\n")
    bbk = _safe(s, "interactions", "body_length_bucket_x_label", default={})
    if bbk:
        md.append("\n**Label rate by body-length bucket:**\n\n")
        md.append("| bucket | n | label rate |\n|---|---:|---:|\n")
        for k, v in bbk.items():
            md.append(f"| {k} | {_fmt_int(v['n'])} | {v['label_rate']:.4f} |\n")

    # ── 12. Files ──
    md.append("\n### 12. Generated artifacts\n")
    md.append(f"- Flat metrics: `{SUMMARY_CSV.relative_to(OUT_DIR.parent)}`\n")
    md.append(f"- Plots: `{PLOTS_DIR.relative_to(OUT_DIR.parent)}/*.png` "
              f"({sum(1 for _ in PLOTS_DIR.glob('*.png'))} files)\n")
    md.append(f"- Distinctive terms CSVs: `outputs/cache/distinctive_*.csv`\n")
    md.append(f"- Duplicate samples: `outputs/cache/duplicate_body_samples.csv`\n")
    md.append(f"- Qualitative samples: `outputs/samples/*.md` "
              f"({sum(1 for _ in (OUT_DIR / 'samples').glob('*.md')) if (OUT_DIR / 'samples').exists() else 0} files)\n")

    md.append("\n---\n")
    md.append("\n_End of report. Open `outputs/samples/*.md` to eyeball labels manually._\n")

    ISSUES_MD.write_text("".join(md), encoding="utf-8")
    step(f"Wrote {ISSUES_MD.name}  ({len(md)} sections)")


# ─────────────────────────────────────────────────────────────────────────────
# entry
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    section("10 | Final report")
    stats = load_stats()
    if not stats:
        raise RuntimeError("stats.json is empty. Run 01..09 first.")

    write_summary_csv(stats)
    write_issues_md(stats)
    section(f"Done.\n"
            f"  - {SUMMARY_CSV}\n"
            f"  - {ISSUES_MD}\n"
            f"  - {PLOTS_DIR}/*.png\n"
            f"  - {OUT_DIR}/samples/*.md")


if __name__ == "__main__":
    main()
