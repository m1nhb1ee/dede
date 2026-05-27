"""Stage 1 dataset + head+tail tokenization.

Per tech_plan section 3.1 / Phase 3 discussion:
  - title is NEVER truncated.
  - body is truncated head+tail when (title + body + specials) exceeds max_length.
    A " ... " marker is inserted between the head and tail so the model can
    see the discontinuity.
  - 96.6% of posts fit max_length=512 without any truncation (per EDA).
  - The remaining ~3.4% (longest, mostly label=1 per EDA) get smart truncation.

Implementation strategy:
  - Pre-tokenize the entire dataset ONCE using `datasets.Dataset.map(...)`,
    cached on disk under stage1/outputs/token_cache. All folds reuse the cache.
  - Each row stores `input_ids` (variable-length list of ints), the collator
    pads at batch time.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence

try:
    from ver2.stage1.src.config import (
        HEAD_FRAC, HW, POSTS_CLEAN_PARQUET, SPLITS_PARQUET, TOKEN_CACHE,
        TRUNC_MARKER,
    )
    from ver2.stage1.src.utils import step
except ImportError:  # flat layout (e.g., Kaggle /kaggle/working/stage1/)
    from config import (
        HEAD_FRAC, HW, POSTS_CLEAN_PARQUET, SPLITS_PARQUET, TOKEN_CACHE,
        TRUNC_MARKER,
    )
    from utils import step


# ─────────────────────────────────────────────────────────────────────────────
# Head + Tail tokenization
# ─────────────────────────────────────────────────────────────────────────────


def encode_head_tail(
    title: str,
    body: str,
    tokenizer,
    max_length: int,
    head_frac: float = HEAD_FRAC,
    trunc_marker: str = TRUNC_MARKER,
) -> dict:
    """Tokenize (title, body) as a RoBERTa pair, applying head+tail truncation
    to the body if the total length exceeds max_length.

    Layout: [CLS] title_tokens [SEP] body_tokens [SEP]
    If body is too long: body_tokens = head_tokens + marker_tokens + tail_tokens

    Returns: {"input_ids": list[int], "attention_mask": list[int]}
    """
    title = title or ""
    body  = body  or ""

    # Tokenize each piece WITHOUT special tokens; we add them manually so we
    # can interleave the truncation marker correctly.
    title_ids = tokenizer(title, add_special_tokens=False)["input_ids"]
    body_ids  = tokenizer(body,  add_special_tokens=False)["input_ids"]
    marker_ids = tokenizer(trunc_marker, add_special_tokens=False)["input_ids"]

    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id

    # RoBERTa pair encoding: [CLS] A [SEP] [SEP] B [SEP]  (uses 2 SEPs between)
    # We follow the standard HF roberta layout.
    n_specials = 4   # CLS + SEP + SEP + SEP
    # Budget for actual content (title + body):
    content_budget = max_length - n_specials

    # If title alone is absurdly long, hard-truncate the title FIRST so we
    # leave some room for body. (Extremely rare per EDA: title p95 = 25 tokens.)
    max_title = max(8, content_budget // 2)
    if len(title_ids) > max_title:
        title_ids = title_ids[:max_title]

    body_budget = content_budget - len(title_ids)

    if len(body_ids) <= body_budget:
        # No truncation needed
        final_body = body_ids
    elif body_budget <= len(marker_ids) + 4:
        # No room for head+tail; just head-truncate body
        final_body = body_ids[:max(0, body_budget)]
    else:
        # Head + Tail
        usable = body_budget - len(marker_ids)
        head_len = max(1, int(usable * head_frac))
        tail_len = max(1, usable - head_len)
        final_body = (body_ids[:head_len]
                      + marker_ids
                      + body_ids[-tail_len:])

    input_ids = ([cls]
                 + title_ids
                 + [sep, sep]
                 + final_body
                 + [sep])

    # Hard cap (paranoia)
    input_ids = input_ids[:max_length]
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────


def build_or_load_tokenized(
    tokenizer,
    tokenizer_name: str,
    force_rebuild: bool = False,
) -> Dataset:
    """Load posts_clean + splits, tokenize once (head+tail), cache to disk.

    The cache key includes the tokenizer name and max_length so different
    configurations produce different caches.
    """
    cache_dir = TOKEN_CACHE / f"{tokenizer_name.replace('/', '_')}_ml{HW['max_length']}"
    if cache_dir.exists() and not force_rebuild:
        step(f"Loading cached tokenized dataset: {cache_dir}")
        return Dataset.load_from_disk(str(cache_dir))

    step(f"Building tokenized dataset (will cache to {cache_dir})")
    t0 = time.perf_counter()

    # Read parquet + join splits
    posts  = pd.read_parquet(POSTS_CLEAN_PARQUET,
                             columns=["id", "title", "body", "label",
                                      "has_title", "has_body", "subreddit",
                                      "created_utc"])
    splits = pd.read_parquet(SPLITS_PARQUET)
    df = posts.merge(splits, on="id", how="inner")
    step(f"  merged: {len(df):,} rows  ({time.perf_counter()-t0:.1f}s)")

    # Build HF Dataset
    ds = Dataset.from_pandas(df, preserve_index=False)

    max_length = HW["max_length"]

    def _tokenize_batch(batch: dict) -> dict:
        out_ids:   list[list[int]] = []
        out_masks: list[list[int]] = []
        for title, body in zip(batch["title"], batch["body"]):
            enc = encode_head_tail(title, body, tokenizer, max_length=max_length)
            out_ids.append(enc["input_ids"])
            out_masks.append(enc["attention_mask"])
        return {"input_ids": out_ids, "attention_mask": out_masks}

    t1 = time.perf_counter()
    # num_proc=1: fast tokenizer is already multi-threaded internally; spawning
    # multiple Python processes on top of it tends to deadlock on Kaggle.
    ds = ds.map(
        _tokenize_batch,
        batched=True, batch_size=2048,
        remove_columns=["title", "body"],     # drop raw text to save space
        num_proc=HW.get("tokenize_num_proc", 1),
        desc="head+tail tokenize",
    )
    step(f"  tokenized in {time.perf_counter()-t1:.1f}s")

    # Cast label column to float32 (BCE expects float targets)
    ds = ds.map(lambda x: {"labels": float(x["label"])},
                remove_columns=["label"])

    ds.save_to_disk(str(cache_dir))
    step(f"  cached: {cache_dir}")
    return ds


def make_fold_splits(ds: Dataset, fold: int | None) -> tuple[Dataset, Dataset, Dataset]:
    """Select train/val rows for a given 5-fold OOF setup.

    If `fold` is an int 0..N-1 (N = number of folds in splits.parquet):
        train = time_split == "train" AND fold != k
        val   = time_split == "train" AND fold == k
        test  = time_split == "test"
    If `fold` is None (full-train mode):
        train = time_split == "train"
        val   = time_split == "val"
        test  = time_split == "test"
    """
    is_train_split = np.array(ds["time_split"]) == "train"
    is_val_split   = np.array(ds["time_split"]) == "val"
    is_test_split  = np.array(ds["time_split"]) == "test"

    if fold is None:
        train_mask = is_train_split
        val_mask   = is_val_split
        test_mask  = is_test_split
    else:
        fold_arr = np.array(ds["fold"], dtype=np.int8)
        train_mask = is_train_split & (fold_arr != fold)
        val_mask   = is_train_split & (fold_arr == fold)
        test_mask  = is_test_split

    train = ds.select(np.where(train_mask)[0])
    val   = ds.select(np.where(val_mask)[0])
    test  = ds.select(np.where(test_mask)[0])
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Collator: dynamic padding of input_ids and attention_mask
# ─────────────────────────────────────────────────────────────────────────────


class PadCollator:
    """Dynamic batch padding for input_ids and attention_mask.

    Pads to the longest sequence in the batch (not to max_length). RoBERTa
    pad token is usually 1; we pull it from the tokenizer at construction
    time so it's correct for any model in MODEL_FALLBACKS.
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = 1   # roberta default

    def __call__(self, features: list[dict]) -> dict:
        ids   = [torch.tensor(f["input_ids"],      dtype=torch.long) for f in features]
        masks = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.float32)

        ids_padded   = pad_sequence(ids,   batch_first=True, padding_value=self.pad_id)
        masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
        return {
            "input_ids":      ids_padded,
            "attention_mask": masks_padded,
            "labels":         labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Class-weight helper (for BCE pos_weight)
# ─────────────────────────────────────────────────────────────────────────────


def compute_pos_weight(train_ds: Dataset) -> float:
    labels = np.asarray(train_ds["labels"], dtype=np.float32)
    n_pos = float((labels > 0.5).sum())
    n_neg = float((labels <= 0.5).sum())
    if n_pos == 0: return 1.0
    return n_neg / n_pos
