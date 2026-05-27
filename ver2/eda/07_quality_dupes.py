"""Data-quality checks that decide whether the dataset is usable as-is.

Two things can quietly destroy a 2M-row training run:

  1. Cross-split DUPLICATES -- the same body appearing in train AND test
     means we are evaluating partly on memorized data. Reddit reposts
     are real and common.

  2. Non-English posts in a dataset advertised as English. MentalRoBERTa
     was pretrained on English Reddit; a 5-10% non-English contamination
     can silently cap performance.

We also report:
  - All-URL posts, mostly-emoji posts, single-character spam
  - Title == body posts (lazy template)
  - Posts with extreme length outliers

Language detection uses langdetect if available, else a fast ASCII+
common-word heuristic. Both run on a sample (default 50K rows) for speed.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

import pandas as pd

from config import (
    CACHE_DIR, COL_BODY, COL_ID, COL_LABEL, COL_TITLE, SEED,
)
from utils import (
    load_raw, plot_bar, section, step, update_stats,
)


URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_HEAVY_RE = re.compile(r"[^\w\s.,!?'\"-]")  # rough "non-text" proxy

LANG_SAMPLE = 50_000
DUP_BODY_MIN_LEN = 40           # don't flag short generic bodies


# ─────────────────────────────────────────────────────────────────────────────
# Language detection -- prefer langdetect, fall back to heuristic
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_is_english(text: str) -> bool:
    """Cheap fallback: high ASCII ratio + presence of common English words."""
    if not isinstance(text, str) or len(text) < 10: return True
    common = {"the", "and", "to", "of", "a", "is", "in", "that", "it", "for",
              "i", "you", "my", "with", "this", "have", "be", "on", "are"}
    ascii_ratio = sum(c.isascii() for c in text) / max(len(text), 1)
    if ascii_ratio < 0.85: return False
    words = re.findall(r"[a-zA-Z']+", text.lower())[:50]
    if not words: return True
    hit = sum(1 for w in words if w in common)
    return hit / len(words) >= 0.05  # >= 5% common English words


def get_language_detector():
    """Return a (name, fn) tuple. fn(text) -> 2-letter lang code (or 'unk')."""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        def detect_fn(t: str) -> str:
            if not isinstance(t, str) or len(t.strip()) < 5: return "unk"
            try: return detect(t[:1000])
            except Exception: return "unk"
        return "langdetect", detect_fn
    except ImportError:
        def detect_fn(t: str) -> str:
            return "en" if _heuristic_is_english(t or "") else "non-en"
        return "heuristic", detect_fn


# ─────────────────────────────────────────────────────────────────────────────
# Hash helpers for fast duplicate detection
# ─────────────────────────────────────────────────────────────────────────────

def _norm_for_hash(s: str) -> str:
    """Normalize for duplicate detection: lowercase, collapse whitespace."""
    if not isinstance(s, str): return ""
    return re.sub(r"\s+", " ", s.lower()).strip()


def _hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def main() -> None:
    section("07 | Data quality: language detection + duplicates + noise")

    df = load_raw(columns=[COL_ID, COL_TITLE, COL_BODY, COL_LABEL])
    n = len(df)
    step(f"Loaded {n:,} rows")

    payload: dict = {}

    # ── 1. Duplicate bodies (exact, normalized) ────────────────────────────
    step("Hashing bodies for exact-duplicate detection ...")
    bodies = df[COL_BODY].fillna("").astype(str)
    long_enough = bodies.str.len() >= DUP_BODY_MIN_LEN

    hashes = bodies[long_enough].map(_norm_for_hash).map(_hash_text)
    hash_counts = hashes.value_counts()
    dup_hashes = hash_counts[hash_counts > 1]

    n_dup_rows = int(hash_counts[hash_counts > 1].sum() - len(dup_hashes))
    n_dup_groups = int(len(dup_hashes))
    payload["body_duplicates"] = {
        "min_body_len_considered":  DUP_BODY_MIN_LEN,
        "n_long_enough_bodies":     int(long_enough.sum()),
        "n_duplicate_groups":       n_dup_groups,
        "n_duplicate_rows_extra":   n_dup_rows,   # rows beyond the first in each group
        "duplicate_rows_pct":       round(n_dup_rows / n * 100, 4),
        "biggest_group_size":       int(dup_hashes.max()) if not dup_hashes.empty else 0,
        "top10_group_sizes":        [int(x) for x in dup_hashes.head(10).tolist()],
    }
    step(f"Exact body duplicates: {n_dup_rows:,} extra rows in "
         f"{n_dup_groups:,} groups ({payload['body_duplicates']['duplicate_rows_pct']}%)")

    # Save list of top duplicate groups for the report
    if not dup_hashes.empty:
        top_dup_hashes = dup_hashes.head(20).index.tolist()
        df_dup = df.loc[long_enough].assign(_h=hashes.values)
        sample_groups = []
        for h in top_dup_hashes:
            grp = df_dup[df_dup["_h"] == h].head(3)
            for _, r in grp.iterrows():
                sample_groups.append({
                    "group_hash": h[:10],
                    "group_size": int(dup_hashes[h]),
                    "id": str(r[COL_ID]),
                    "label": int(r[COL_LABEL]) if pd.notna(r[COL_LABEL]) else None,
                    "body_preview": str(r[COL_BODY])[:200],
                })
        pd.DataFrame(sample_groups).to_csv(
            CACHE_DIR / "duplicate_body_samples.csv", index=False,
        )

    # ── 2. Title == body (lazy / template posts) ───────────────────────────
    same_tb = (df[COL_TITLE].fillna("").str.strip().str.lower()
               == df[COL_BODY].fillna("").str.strip().str.lower())
    n_same = int(same_tb.sum())
    payload["title_equals_body"] = {
        "n":   n_same,
        "pct": round(n_same / n * 100, 4),
    }
    step(f"Title == body: {n_same:,} ({payload['title_equals_body']['pct']}%)")

    # ── 3. Language detection on a sample ──────────────────────────────────
    name, detect_fn = get_language_detector()
    sample_n = min(LANG_SAMPLE, n)
    sample = df.sample(n=sample_n, random_state=SEED).copy()
    step(f"Detecting language with `{name}` on {sample_n:,} sample ...")
    text = (sample[COL_TITLE].fillna("") + " " + sample[COL_BODY].fillna("")).str.strip()
    sample["lang"] = text.apply(detect_fn)
    lang_counts = sample["lang"].value_counts()
    payload["language_detection"] = {
        "detector":      name,
        "sample_size":   sample_n,
        "distribution":  {str(k): int(v) for k, v in lang_counts.items()},
        "english_pct":   round(float(lang_counts.get("en", 0)) / sample_n * 100, 4),
        "non_english_pct": round((1 - float(lang_counts.get("en", 0)) / sample_n) * 100, 4),
    }
    step(f"English %: {payload['language_detection']['english_pct']:.2f}  "
         f"(detector={name})")
    plot_bar(
        labels=[str(k) for k in lang_counts.head(15).index],
        values=lang_counts.head(15).values.tolist(),
        name="language_distribution",
        title=f"Detected language (top 15, sample n={sample_n:,}, detector={name})",
        horizontal=True,
    )

    # ── 4. Content-noise checks ────────────────────────────────────────────
    step("Counting URL-only / emoji-heavy / extreme-length posts ...")
    nonempty = bodies.str.len() > 0

    url_count    = bodies.str.findall(URL_RE).map(len)
    word_count   = bodies.str.findall(r"[a-zA-Z]{2,}").map(len)
    url_only     = nonempty & (word_count <= 2) & (url_count >= 1)

    emoji_count  = bodies.str.count(EMOJI_HEAVY_RE)
    emoji_ratio  = emoji_count / bodies.str.len().clip(lower=1)
    emoji_heavy  = nonempty & (emoji_ratio > 0.5) & (bodies.str.len() > 10)

    extreme_long = bodies.str.len() > 20_000     # raw chars; obvious outliers
    very_short   = nonempty & (bodies.str.len() < 5)

    payload["noise"] = {
        "url_only_posts":       int(url_only.sum()),
        "url_only_pct":         round(int(url_only.sum()) / n * 100, 4),
        "emoji_heavy_posts":    int(emoji_heavy.sum()),
        "emoji_heavy_pct":      round(int(emoji_heavy.sum()) / n * 100, 4),
        "extreme_long_posts":   int(extreme_long.sum()),
        "extreme_long_pct":     round(int(extreme_long.sum()) / n * 100, 4),
        "very_short_nonempty":  int(very_short.sum()),
        "very_short_pct":       round(int(very_short.sum()) / n * 100, 4),
    }
    step(f"URL-only: {payload['noise']['url_only_posts']:,} | "
         f"emoji-heavy: {payload['noise']['emoji_heavy_posts']:,} | "
         f"extreme-long: {payload['noise']['extreme_long_posts']:,}")

    update_stats("quality", payload)
    section("Done. Run 08_interactions.py next.")


if __name__ == "__main__":
    main()
