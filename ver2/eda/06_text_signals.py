"""High-insight text analysis: vocabulary distinctiveness + lexical signals.

This is where we answer the most important question:
    "Is there actually any text signal that distinguishes label=1 from label=0?"

If the answer is NO, then Stage 1 (MentalRoBERTa) is doomed and the whole
pipeline collapses to whatever subreddit+metadata can do.

Outputs:
- Top distinctive unigrams and bigrams per class (log-odds with prior)
- Mental health keyword presence rate per class (depressed, suicide, ...)
- First-person pronoun rate per class (known depression marker)
- Negation / negative-affect word rate per class
- Style markers: ALL CAPS, question marks, exclamation marks, ellipses
- Saved CSVs of distinctive terms for the report

Runs on a 200K sample to keep counting fast.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from config import (
    CACHE_DIR, COL_BODY, COL_LABEL, COL_TITLE, SEED,
)
from utils import (
    load_raw, plot_bar, section, simple_words, step, update_stats,
)


# ─────────────────────────────────────────────────────────────────────────────
# Lexicons -- short, hand-curated; not exhaustive on purpose
# ─────────────────────────────────────────────────────────────────────────────

MH_KEYWORDS = {
    "depress", "depressed", "depression", "anxious", "anxiety", "panic",
    "hopeless", "worthless", "useless", "empty", "numb", "exhausted",
    "tired", "alone", "lonely", "isolat", "isolated",
    "suicide", "suicidal", "kill", "die", "death", "dead",
    "hurt", "hate", "broken", "lost", "pain", "crying", "tears",
    "therapy", "therapist", "medication", "meds", "pills",
    "trauma", "abuse", "abused", "ptsd",
}

NEGATION_WORDS = {
    "no", "not", "never", "none", "nothing", "nobody", "nowhere",
    "neither", "nor", "cant", "cannot", "wont", "dont", "doesnt",
    "didnt", "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
    "hadnt", "shouldnt", "wouldnt", "couldnt",
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "im"}

NEGATIVE_AFFECT = {
    "sad", "angry", "afraid", "scared", "fear", "lonely", "miserable",
    "horrible", "terrible", "awful", "bad", "worse", "worst",
    "hopeless", "helpless", "useless", "worthless", "stupid", "ugly",
    "fail", "failed", "failure", "wrong", "mistake", "regret",
    "sorry", "guilty", "shame", "ashamed", "hate", "angry",
}

ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "than", "too", "very", "s", "t", "just", "now", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "doing", "would", "should", "could", "ought",
    "it", "its", "this", "that", "these", "those", "they", "them", "their",
    "we", "us", "our", "you", "your", "he", "she", "his", "her", "him",
    "as", "so", "also", "like", "get", "got", "really", "much", "even",
    "still", "back", "way", "go", "going", "want", "know", "think",
    "going", "make", "made", "see", "say", "said", "thing", "things",
    "people", "time", "one", "two", "lot", "well", "yeah", "ok", "okay",
}

# Sample sizes -- vocabulary analysis is O(N), keep modest
VOCAB_SAMPLE = 200_000
TOP_K_TERMS = 60
MIN_TERM_FREQ = 25      # filter rare terms before scoring


# ─────────────────────────────────────────────────────────────────────────────
# Log-odds with informative Dirichlet prior (Monroe et al. 2008, simplified)
# ─────────────────────────────────────────────────────────────────────────────

def distinctive_terms(
    counts_a: Counter, counts_b: Counter, alpha: float = 0.01,
    min_freq: int = MIN_TERM_FREQ, top_k: int = TOP_K_TERMS,
) -> list[tuple[str, float, int, int]]:
    """Return terms most distinctive of group A vs group B.

    Score = log( p_a / (1 - p_a) ) - log( p_b / (1 - p_b) )
    with add-alpha smoothing (alpha = pseudo-count from a uniform prior).
    Positive score => more characteristic of A. Returns top_k sorted desc.
    """
    vocab = (set(counts_a) | set(counts_b))
    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())
    V = len(vocab)

    scored: list[tuple[str, float, int, int]] = []
    for w in vocab:
        ca = counts_a.get(w, 0)
        cb = counts_b.get(w, 0)
        if ca + cb < min_freq:
            continue
        # smoothed prob within class
        pa = (ca + alpha) / (n_a + alpha * V)
        pb = (cb + alpha) / (n_b + alpha * V)
        # log-odds (logit difference)
        score = math.log(pa / (1 - pa + 1e-12)) - math.log(pb / (1 - pb + 1e-12))
        # variance approx for normalization (Monroe et al. eq. 22)
        var = 1.0 / (ca + alpha) + 1.0 / (cb + alpha)
        z = score / math.sqrt(var)
        scored.append((w, z, ca, cb))

    scored.sort(key=lambda r: r[1], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Bigram extraction
# ─────────────────────────────────────────────────────────────────────────────

def bigrams_from_words(words: list[str]) -> list[str]:
    if len(words) < 2: return []
    return [f"{a}_{b}" for a, b in zip(words[:-1], words[1:])]


# ─────────────────────────────────────────────────────────────────────────────
# Per-row feature helpers (lightweight, vectorized where possible)
# ─────────────────────────────────────────────────────────────────────────────

def has_any_keyword(text: str, keywords: set[str]) -> bool:
    if not isinstance(text, str) or not text:
        return False
    low = text.lower()
    # substring match for stems like "depress" matching "depression"
    return any(k in low for k in keywords)


def main() -> None:
    section("06 | Text signals: distinctive vocab + MH keyword presence")

    df = load_raw(columns=[COL_TITLE, COL_BODY, COL_LABEL])
    df = df.dropna(subset=[COL_LABEL])
    df[COL_LABEL] = df[COL_LABEL].astype(int)
    df["text"] = (df[COL_TITLE].fillna("") + " . " + df[COL_BODY].fillna("")).str.strip()
    df = df[df["text"].str.len() > 0]
    n = len(df)
    step(f"{n:,} rows with valid label + non-empty text")

    payload: dict = {}

    # ── 1. Lexical signals per row (full data, vectorized strings) ─────────
    step("Computing per-row lexical signals (full data) ...")
    low = df["text"].str.lower()
    df["has_mh_kw"]    = low.apply(lambda s: has_any_keyword(s, MH_KEYWORDS))
    df["n_exclaim"]    = df["text"].str.count("!")
    df["n_question"]   = df["text"].str.count(r"\?")
    df["n_ellipsis"]   = df["text"].str.count(r"\.\.\.")
    df["n_caps_words"] = df["text"].str.count(r"\b[A-Z]{2,}\b")

    # Word-level counts (slower; sample for speed)
    sample_n = min(VOCAB_SAMPLE, n)
    sample = df.sample(n=sample_n, random_state=SEED).copy()
    step(f"Computing word-level signals on {sample_n:,} sample ...")
    word_lists = sample["text"].apply(simple_words)
    sample["n_words"]      = word_lists.apply(len)
    sample["n_first_pers"] = word_lists.apply(lambda ws: sum(1 for w in ws if w in FIRST_PERSON))
    sample["n_negation"]   = word_lists.apply(lambda ws: sum(1 for w in ws if w in NEGATION_WORDS))
    sample["n_neg_affect"] = word_lists.apply(lambda ws: sum(1 for w in ws if w in NEGATIVE_AFFECT))

    sample["first_pers_rate"] = sample["n_first_pers"] / sample["n_words"].clip(lower=1)
    sample["negation_rate"]   = sample["n_negation"]   / sample["n_words"].clip(lower=1)
    sample["neg_affect_rate"] = sample["n_neg_affect"] / sample["n_words"].clip(lower=1)

    # ── 2. Aggregate lexical signals per label ─────────────────────────────
    sig = {}
    for lbl in sorted(df[COL_LABEL].unique()):
        sub_full = df[df[COL_LABEL] == lbl]
        sub_samp = sample[sample[COL_LABEL] == lbl]
        sig[str(int(lbl))] = {
            "n_full":              int(len(sub_full)),
            "n_sample":            int(len(sub_samp)),
            "has_mh_kw_pct":       round(float(sub_full["has_mh_kw"].mean() * 100), 4),
            "mean_exclaim":        round(float(sub_full["n_exclaim"].mean()), 4),
            "mean_question":       round(float(sub_full["n_question"].mean()), 4),
            "mean_ellipsis":       round(float(sub_full["n_ellipsis"].mean()), 4),
            "mean_caps_words":     round(float(sub_full["n_caps_words"].mean()), 4),
            "first_person_rate":   round(float(sub_samp["first_pers_rate"].mean()), 6),
            "negation_rate":       round(float(sub_samp["negation_rate"].mean()), 6),
            "neg_affect_rate":     round(float(sub_samp["neg_affect_rate"].mean()), 6),
            "mean_words":          round(float(sub_samp["n_words"].mean()), 2),
        }
    payload["lexical_by_label"] = sig
    step("Lexical signal summary per label:")
    for lbl, s in sig.items():
        print(f"      label={lbl}  mh_kw={s['has_mh_kw_pct']:.1f}%  "
              f"first_pers={s['first_person_rate']:.4f}  "
              f"neg_affect={s['neg_affect_rate']:.4f}")

    # Lift table (label=1 / label=0) for each rate -- shows which signals discriminate
    if {"0", "1"}.issubset(sig.keys()):
        lift = {}
        for k in ("has_mh_kw_pct", "first_person_rate", "negation_rate",
                  "neg_affect_rate", "mean_exclaim", "mean_question",
                  "mean_ellipsis", "mean_caps_words", "mean_words"):
            v0 = sig["0"].get(k, 0.0)
            v1 = sig["1"].get(k, 0.0)
            ratio = (v1 / v0) if v0 > 0 else None
            lift[k] = {"label0": v0, "label1": v1,
                       "lift_1_over_0": round(ratio, 4) if ratio is not None else None}
        payload["lexical_lift_label1_over_label0"] = lift

        lift_vals = [(k, v["lift_1_over_0"]) for k, v in lift.items()
                     if v["lift_1_over_0"] is not None]
        lift_vals.sort(key=lambda r: r[1], reverse=True)
        plot_bar(
            labels=[r[0] for r in lift_vals],
            values=[r[1] for r in lift_vals],
            name="lexical_lift_label1_over_label0",
            title="Lexical signal lift: rate(label=1) / rate(label=0). "
                  ">1 = enriched in depression class.",
            horizontal=True,
        )

    # ── 3. Distinctive vocabulary per class (log-odds + bigrams) ───────────
    step("Counting unigrams + bigrams per label ...")
    counts_uni: dict[int, Counter] = {0: Counter(), 1: Counter()}
    counts_bi:  dict[int, Counter] = {0: Counter(), 1: Counter()}

    for lbl in (0, 1):
        if lbl not in df[COL_LABEL].unique(): continue
        sub_words = word_lists[sample[COL_LABEL] == lbl]
        for ws in sub_words:
            counts_uni[lbl].update(w for w in ws if w not in ENGLISH_STOPWORDS and len(w) > 2)
            counts_bi[lbl].update(bigrams_from_words(
                [w for w in ws if w not in ENGLISH_STOPWORDS and len(w) > 2]
            ))

    # Top distinctive unigrams for label=1 vs label=0 and vice versa
    if 0 in counts_uni and 1 in counts_uni:
        top_label1 = distinctive_terms(counts_uni[1], counts_uni[0])
        top_label0 = distinctive_terms(counts_uni[0], counts_uni[1])
        top_label1_bi = distinctive_terms(counts_bi[1], counts_bi[0])
        top_label0_bi = distinctive_terms(counts_bi[0], counts_bi[1])

        # Save full lists for the report
        for name, lst in [
            ("distinctive_unigrams_label1", top_label1),
            ("distinctive_unigrams_label0", top_label0),
            ("distinctive_bigrams_label1",  top_label1_bi),
            ("distinctive_bigrams_label0",  top_label0_bi),
        ]:
            pd.DataFrame(lst, columns=["term", "z_score", "count_a", "count_b"]).to_csv(
                CACHE_DIR / f"{name}.csv", index=False,
            )

        payload["top_distinctive_unigrams_label1_head"] = [
            {"term": t, "z": round(z, 3), "n1": ca, "n0": cb}
            for t, z, ca, cb in top_label1[:20]
        ]
        payload["top_distinctive_unigrams_label0_head"] = [
            {"term": t, "z": round(z, 3), "n0": ca, "n1": cb}
            for t, z, ca, cb in top_label0[:20]
        ]
        payload["top_distinctive_bigrams_label1_head"] = [
            {"term": t.replace("_", " "), "z": round(z, 3), "n1": ca, "n0": cb}
            for t, z, ca, cb in top_label1_bi[:20]
        ]

        step("Top distinctive unigrams for label=1:")
        for t, z, ca, cb in top_label1[:15]:
            print(f"      {t:<20} z={z:+.2f}  n1={ca:>6}  n0={cb:>6}")

        step("Top distinctive unigrams for label=0:")
        for t, z, ca, cb in top_label0[:15]:
            print(f"      {t:<20} z={z:+.2f}  n0={ca:>6}  n1={cb:>6}")

        # Bar plot of top-25 most distinctive label=1 terms
        plot_bar(
            labels=[t for t, _, _, _ in top_label1[:25]],
            values=[z for _, z, _, _ in top_label1[:25]],
            name="distinctive_terms_label1_top25",
            title="Top-25 unigrams enriched in label=1 (z-score, log-odds)",
            horizontal=True,
        )

    update_stats("text_signals", payload)
    section("Done. Run 07_quality_dupes.py next.")


if __name__ == "__main__":
    main()
