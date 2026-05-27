"""Section 1.1 text-length and emptiness analysis.

- Char-length distributions of title and body (full 2M rows, cheap)
- Token-length distributions using MentalRoBERTa tokenizer
  - sampled to TOKEN_SAMPLE_SIZE rows to keep runtime under a few minutes
- Percentage of empty / [removed] / [deleted] bodies
- Percentage of (title, body) pairs exceeding 128/256/384/512 tokens
  -> directly drives the max_length decision in tech_plan section 5.5
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    COL_BODY, COL_TITLE, REMOVED_TOKENS, SEED, TOKEN_SAMPLE_SIZE,
    TOKEN_THRESHOLDS, TOKENIZER_NAME,
)
from utils import (
    char_len, describe_numeric, is_removed_body, load_raw, plot_hist,
    section, step, update_stats,
)


def main() -> None:
    section("03 | Text analysis (char length + token length + emptiness)")

    df = load_raw(columns=[COL_TITLE, COL_BODY])
    n = len(df)
    step(f"Loaded {n:,} rows")

    payload: dict = {}

    # ── Removed / empty body share ─────────────────────────────────────────
    removed_mask = is_removed_body(df[COL_BODY], REMOVED_TOKENS)
    n_removed = int(removed_mask.sum())
    n_title_empty = int(df[COL_TITLE].fillna("").astype(str).str.strip().eq("").sum())

    payload["emptiness"] = {
        "body_empty_or_removed":     n_removed,
        "body_empty_or_removed_pct": round(n_removed / n * 100, 4),
        "title_empty":               n_title_empty,
        "title_empty_pct":           round(n_title_empty / n * 100, 4),
    }
    step(f"Body empty/[removed]/[deleted]: {n_removed:,} ({payload['emptiness']['body_empty_or_removed_pct']}%)")
    step(f"Title empty: {n_title_empty:,} ({payload['emptiness']['title_empty_pct']}%)")

    # ── Char-length distributions (cheap, full data) ───────────────────────
    title_chars = char_len(df[COL_TITLE])
    body_chars  = char_len(df[COL_BODY])

    payload["char_length"] = {
        "title": describe_numeric(title_chars.rename("title_chars")),
        "body":  describe_numeric(body_chars.rename("body_chars")),
    }
    step(f"Title chars  median={payload['char_length']['title']['median']:.0f}  "
         f"p95={payload['char_length']['title']['p95']:.0f}")
    step(f"Body chars   median={payload['char_length']['body']['median']:.0f}  "
         f"p95={payload['char_length']['body']['p95']:.0f}")

    plot_hist(title_chars, name="hist_title_chars",
              title="Title char length (clipped p99)")
    plot_hist(body_chars,  name="hist_body_chars",
              title="Body char length (clipped p99)")

    # ── Token-length distribution (sampled) ────────────────────────────────
    from transformers import AutoTokenizer
    from config import TOKENIZER_FALLBACKS
    tok = None
    used_name = None
    for name in TOKENIZER_FALLBACKS:
        try:
            step(f"Loading tokenizer: {name}")
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            used_name = name
            break
        except Exception as e:
            step(f"  {name} failed: {type(e).__name__}: {str(e)[:120]}")
    if tok is None:
        step("WARNING: no tokenizer loaded; skipping token analysis.")
        update_stats("text", payload)
        section("Done (token analysis skipped).")
        return
    payload["tokenizer_used"] = used_name

    sample_n = min(TOKEN_SAMPLE_SIZE, n)
    sample = df.sample(n=sample_n, random_state=SEED).reset_index(drop=True)
    step(f"Tokenizing {sample_n:,} sampled rows as (title, body) pairs ...")

    titles = sample[COL_TITLE].fillna("").astype(str).tolist()
    bodies = sample[COL_BODY].fillna("").astype(str).tolist()

    # Pair encoding mirrors the production pipeline (tech_plan section 5).
    # No truncation here — we want the *real* length distribution to decide
    # max_length. Padding is off so we only pay for actual tokens.
    encoded = tok(titles, bodies, truncation=False, padding=False,
                  add_special_tokens=True, return_attention_mask=False,
                  return_token_type_ids=False)
    token_lens = np.array([len(ids) for ids in encoded["input_ids"]], dtype=np.int32)

    token_stats = describe_numeric(pd.Series(token_lens, name="pair_tokens"))
    payload["token_length_pair"] = token_stats

    # Coverage at each threshold — what fraction of posts fit within max_length=T?
    coverage = {}
    for t in TOKEN_THRESHOLDS:
        pct_within = float((token_lens <= t).mean() * 100)
        coverage[str(t)] = round(pct_within, 4)
    payload["token_length_pair"]["coverage_pct_at"] = coverage
    step("Token coverage at thresholds:")
    for t, pct in coverage.items():
        print(f"      max_length={t:>4}  -> {pct:.2f}% posts fit (no truncation)")

    plot_hist(pd.Series(token_lens, name="tokens_pair"),
              name="hist_token_lens_pair",
              title=f"Token length of (title,body) pairs  (sample n={sample_n:,})",
              clip_q=0.99)

    # Title-only and body-only for separate visibility
    step("Tokenizing title-only and body-only (same sample) ...")
    title_lens = np.array(
        [len(ids) for ids in tok(titles, truncation=False, padding=False,
                                 add_special_tokens=False)["input_ids"]],
        dtype=np.int32,
    )
    body_lens = np.array(
        [len(ids) for ids in tok(bodies, truncation=False, padding=False,
                                 add_special_tokens=False)["input_ids"]],
        dtype=np.int32,
    )
    payload["token_length_title"] = describe_numeric(pd.Series(title_lens, name="title_tokens"))
    payload["token_length_body"]  = describe_numeric(pd.Series(body_lens,  name="body_tokens"))

    plot_hist(pd.Series(title_lens, name="title_tokens"),
              name="hist_token_lens_title",
              title=f"Title-only token length (sample n={sample_n:,})", clip_q=0.99)
    plot_hist(pd.Series(body_lens, name="body_tokens"),
              name="hist_token_lens_body",
              title=f"Body-only token length (sample n={sample_n:,})", clip_q=0.99)

    payload["token_sample_size"] = sample_n
    update_stats("text", payload)
    section("Done. Run 04_temporal.py next.")


if __name__ == "__main__":
    main()
