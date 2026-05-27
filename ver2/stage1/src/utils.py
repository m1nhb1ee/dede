"""Stage 1 helpers: seed, device, tokenizer loading, metric, logging."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

try:
    from ver2.stage1.src.config import MODEL_FALLBACKS, SEED
except ImportError:  # flat layout (e.g., Kaggle /kaggle/working/stage1/)
    from config import MODEL_FALLBACKS, SEED


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def n_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def load_tokenizer(local_dir: str | None = None):
    """Load the first available tokenizer from MODEL_FALLBACKS.

    If `local_dir` is provided (Kaggle path to pre-downloaded weights), tries
    that first.
    """
    from transformers import AutoTokenizer

    candidates: list[str] = []
    if local_dir and Path(local_dir).exists():
        candidates.append(local_dir)
    candidates.extend(MODEL_FALLBACKS)

    last_err: Exception | None = None
    for name in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            print(f"[tokenizer] loaded: {name}")
            return tok, name
        except Exception as e:
            last_err = e
            print(f"[tokenizer] {name} failed: {type(e).__name__}")
    raise RuntimeError(f"All tokenizer candidates failed. Last error: {last_err}")


def load_backbone(local_dir: str | None = None):
    """Load the first available backbone from MODEL_FALLBACKS."""
    from transformers import AutoModel

    candidates: list[str] = []
    if local_dir and Path(local_dir).exists():
        candidates.append(local_dir)
    candidates.extend(MODEL_FALLBACKS)

    last_err: Exception | None = None
    for name in candidates:
        try:
            m = AutoModel.from_pretrained(name)
            print(f"[backbone] loaded: {name}  hidden={m.config.hidden_size}")
            return m, name
        except Exception as e:
            last_err = e
            print(f"[backbone] {name} failed: {type(e).__name__}")
    raise RuntimeError(f"All backbone candidates failed. Last error: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics_binary(eval_pred) -> dict[str, float]:
    """For HuggingFace Trainer: predictions are raw logits."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score, recall_score,
        roc_auc_score,
    )

    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    # Squeeze single-logit output (binary head produces [B, 1])
    if logits.ndim == 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    out = {
        "accuracy":  float(accuracy_score(labels, preds)),
        "f1_macro":  float(f1_score(labels, preds, average="macro")),
        "f1_pos":    float(f1_score(labels, preds, average="binary", pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
        out["pr_auc"]  = float(average_precision_score(labels, probs))
    except ValueError:
        out["roc_auc"] = 0.0
        out["pr_auc"]  = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def section(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}", flush=True)


def step(msg: str) -> None:
    print(f"  -> {msg}", flush=True)
