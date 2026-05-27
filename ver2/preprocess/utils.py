"""Shared preprocess helpers: text cleaning, dedup, IO, JSON stats."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import STATS_JSON, SUB_TOKEN, URL_TOKEN, USER_TOKEN


# ─────────────────────────────────────────────────────────────────────────────
# Stats accumulator (shared JSON across all phase-2 scripts)
# ─────────────────────────────────────────────────────────────────────────────


def load_stats() -> dict[str, Any]:
    if STATS_JSON.exists():
        return json.loads(STATS_JSON.read_text(encoding="utf-8"))
    return {}


def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    if isinstance(o, Path): return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def update_stats(section: str, payload: dict[str, Any]) -> None:
    stats = load_stats()
    stats[section] = payload
    STATS_JSON.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def section(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


def step(msg: str) -> None:
    print(f"  -> {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning -- compiled regexes for speed on 2M rows
# ─────────────────────────────────────────────────────────────────────────────

# Order matters: do markdown-link BEFORE URL (else URLs in [text](url) get
# replaced first and break the link regex).
_RE_MARKDOWN_LINK    = re.compile(r"\[([^\]]+)\]\(\s*https?://[^\)]+\)")
_RE_REF_LINK         = re.compile(r"\[([^\]]+)\]\[\d+\]")
_RE_URL              = re.compile(r"https?://\S+|www\.\S+")
_RE_USER_MENTION     = re.compile(r"(?:^|(?<=\s))/?u/[A-Za-z0-9_\-]+", flags=re.IGNORECASE)
_RE_SUB_MENTION      = re.compile(r"(?:^|(?<=\s))/?r/[A-Za-z0-9_\-]+", flags=re.IGNORECASE)
_RE_MD_BOLD_ITALIC   = re.compile(r"\*+([^*\n]+?)\*+")
_RE_MD_STRIKE        = re.compile(r"~~([^~\n]+?)~~")
_RE_MD_INLINE_CODE   = re.compile(r"`([^`\n]+?)`")
_RE_MD_HEADER        = re.compile(r"(?m)^#{1,6}\s*")
_RE_MD_BLOCKQUOTE    = re.compile(r"(?m)^>+\s?")
_RE_MD_HRULE         = re.compile(r"(?m)^[-*_]{3,}\s*$")
_RE_MULTI_NEWLINE    = re.compile(r"\n{3,}")
_RE_MULTI_SPACE      = re.compile(r"[ \t]{2,}")
_RE_ZERO_WIDTH       = re.compile(r"[​-‏‪-‮﻿]")
_RE_AMP_ENTITY       = re.compile(r"&\w+;")


def clean_text(s: str) -> str:
    """Apply tech_plan section 2.2 cleaning rules in correct order.

    Preserves: case, emoji, punctuation, censored words (f***).
    Removes/replaces: URLs, user/sub mentions, markdown formatting, HTML entities.
    """
    if not isinstance(s, str) or not s:
        return ""

    # 1. HTML entities (e.g. &amp; -> &). Cheap to do first.
    if "&" in s:
        s = html.unescape(s)

    # 2. Zero-width characters that break tokenization
    s = _RE_ZERO_WIDTH.sub("", s)

    # 3. Markdown links FIRST (preserve link text)
    s = _RE_MARKDOWN_LINK.sub(r"\1", s)
    s = _RE_REF_LINK.sub(r"\1", s)

    # 4. URLs / mentions -> sentinel tokens
    s = _RE_URL.sub(URL_TOKEN, s)
    s = _RE_USER_MENTION.sub(USER_TOKEN, s)
    s = _RE_SUB_MENTION.sub(SUB_TOKEN, s)

    # 5. Markdown formatting (strip markers, keep content)
    s = _RE_MD_BOLD_ITALIC.sub(r"\1", s)
    s = _RE_MD_STRIKE.sub(r"\1", s)
    s = _RE_MD_INLINE_CODE.sub(r"\1", s)
    s = _RE_MD_HEADER.sub("", s)
    s = _RE_MD_BLOCKQUOTE.sub("", s)
    s = _RE_MD_HRULE.sub("", s)

    # 6. Stray HTML entities the unescape didn't catch (numeric etc.)
    s = _RE_AMP_ENTITY.sub(" ", s)

    # 7. Whitespace normalization
    s = _RE_MULTI_NEWLINE.sub("\n\n", s)
    s = _RE_MULTI_SPACE.sub(" ", s)
    s = s.strip()
    return s


def is_removed(s, removed_set) -> bool:
    """True if string is null / [removed] / [deleted] / empty after strip."""
    if not isinstance(s, str): return True
    return s.strip().lower() in removed_set


# Vectorized version for pandas (much faster than .apply on 2M rows)
def vectorized_clean(series: pd.Series) -> pd.Series:
    """Apply clean_text to a Series. Pandas .map is faster than .apply here."""
    return series.fillna("").astype(str).map(clean_text)


def vectorized_is_removed(series: pd.Series, removed_set: set[str]) -> pd.Series:
    """Boolean mask: row is empty / [removed] / [deleted]."""
    return series.fillna("").astype(str).str.strip().str.lower().isin(removed_set)


# ─────────────────────────────────────────────────────────────────────────────
# Dedup -- normalize then hash; only flag groups whose length >= min_len
# ─────────────────────────────────────────────────────────────────────────────

_RE_DEDUP_WHITESPACE = re.compile(r"\s+")


def normalize_for_dedup(s: str) -> str:
    """Lowercase + collapse all whitespace runs. Cheap stable hash key."""
    if not isinstance(s, str) or not s: return ""
    return _RE_DEDUP_WHITESPACE.sub(" ", s.lower()).strip()


def hash_text(s: str) -> str:
    """MD5 first 16 chars -- 64-bit collision space, plenty for 2M rows."""
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()[:16]
