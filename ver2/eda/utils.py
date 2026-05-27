"""Shared EDA helpers: IO, stats, plotting, stats accumulator."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CACHE_DIR, PLOTS_DIR, PLOT_DPI, PLOT_FIGSZ, RAW_PARQUET

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Stats accumulator — each script appends to a shared JSON, report.py reads it
# ─────────────────────────────────────────────────────────────────────────────

STATS_JSON = CACHE_DIR / "stats.json"


def load_stats() -> dict[str, Any]:
    if STATS_JSON.exists():
        return json.loads(STATS_JSON.read_text(encoding="utf-8"))
    return {}


def save_stats(stats: dict[str, Any]) -> None:
    STATS_JSON.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def update_stats(section: str, payload: dict[str, Any]) -> None:
    """Merge a new section into the shared stats JSON without clobbering siblings."""
    stats = load_stats()
    stats[section] = payload
    save_stats(stats)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_raw(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the parquet cache produced by 01_load_convert.py."""
    if not RAW_PARQUET.exists():
        raise FileNotFoundError(
            f"{RAW_PARQUET} not found. Run 01_load_convert.py first."
        )
    df = pd.read_parquet(RAW_PARQUET, columns=columns)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────


def describe_numeric(s: pd.Series) -> dict[str, float]:
    """Robust numeric summary including high percentiles."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"count": 0}
    q = s.quantile([0.5, 0.75, 0.90, 0.95, 0.99, 0.995]).to_dict()
    return {
        "count":    int(s.shape[0]),
        "mean":     float(s.mean()),
        "std":      float(s.std()),
        "min":      float(s.min()),
        "max":      float(s.max()),
        "median":   float(q[0.5]),
        "p75":      float(q[0.75]),
        "p90":      float(q[0.90]),
        "p95":      float(q[0.95]),
        "p99":      float(q[0.99]),
        "p995":     float(q[0.995]),
        "skewness": float(s.skew()),
        "n_negative": int((s < 0).sum()),
        "n_zero":     int((s == 0).sum()),
    }


def value_counts_dict(s: pd.Series, top: int | None = None) -> dict[str, int]:
    vc = s.value_counts(dropna=False)
    if top is not None:
        vc = vc.head(top)
    return {str(k): int(v) for k, v in vc.items()}


def missing_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    n = len(df)
    out = {}
    for c in df.columns:
        nm = int(df[c].isna().sum())
        out[c] = {"missing": nm, "missing_pct": round(nm / n * 100, 4) if n else 0.0}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


def _save_fig(fig: plt.Figure, name: str) -> Path:
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hist(
    s: pd.Series,
    name: str,
    title: str,
    bins: int = 60,
    log_x: bool = False,
    log_y: bool = False,
    clip_q: float | None = 0.99,
) -> Path:
    """Histogram with sensible defaults for heavy-tailed Reddit data."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if clip_q is not None:
        upper = s.quantile(clip_q)
        s = s[s <= upper]

    fig, ax = plt.subplots(figsize=PLOT_FIGSZ)
    ax.hist(s, bins=bins, edgecolor="black", linewidth=0.3)
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(s.name or "value")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    return _save_fig(fig, name)


def plot_bar(
    labels: list[str],
    values: list[float],
    name: str,
    title: str,
    rotation: int = 45,
    horizontal: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=PLOT_FIGSZ)
    if horizontal:
        ax.barh(labels, values)
        ax.invert_yaxis()
    else:
        ax.bar(labels, values)
        plt.xticks(rotation=rotation, ha="right")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="x" if horizontal else "y")
    return _save_fig(fig, name)


def plot_line(
    x: list[Any],
    y: list[float],
    name: str,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
) -> Path:
    fig, ax = plt.subplots(figsize=PLOT_FIGSZ)
    ax.plot(x, y, marker="o", markersize=3, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    return _save_fig(fig, name)


def plot_box_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    name: str,
    title: str,
    clip_q: float | None = 0.99,
) -> Path:
    data = df[[value_col, group_col]].dropna()
    if clip_q is not None:
        upper = data[value_col].quantile(clip_q)
        data = data[data[value_col] <= upper]
    groups = sorted(data[group_col].unique())
    arrays = [data.loc[data[group_col] == g, value_col].values for g in groups]
    fig, ax = plt.subplots(figsize=PLOT_FIGSZ)
    ax.boxplot(arrays, labels=[str(g) for g in groups], showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    ax.grid(alpha=0.25)
    return _save_fig(fig, name)


def plot_scatter(
    x: pd.Series,
    y: pd.Series,
    name: str,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    color: pd.Series | None = None,
    size: pd.Series | None = None,
    log_x: bool = False,
    log_y: bool = False,
    alpha: float = 0.5,
    labels_to_annotate: list[tuple[float, float, str]] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=PLOT_FIGSZ)
    sc = ax.scatter(
        x, y,
        c=color if color is not None else None,
        s=size if size is not None else 20,
        alpha=alpha,
        cmap="coolwarm" if color is not None else None,
        edgecolors="none",
    )
    if color is not None:
        plt.colorbar(sc, ax=ax)
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    if labels_to_annotate:
        for xi, yi, txt in labels_to_annotate:
            ax.annotate(txt, (xi, yi), fontsize=7, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    return _save_fig(fig, name)


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    name: str,
    title: str,
    cmap: str = "coolwarm",
    fmt: str = ".2f",
    annotate: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.6),
                                    max(4, len(row_labels) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    if annotate and matrix.size <= 400:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt),
                        ha="center", va="center", fontsize=7,
                        color="black" if abs(matrix[i, j]) < 0.6 else "white")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    return _save_fig(fig, name)


def plot_kde_overlay(
    series_by_label: dict[str, pd.Series],
    name: str,
    title: str,
    xlabel: str = "",
    clip_q: float | None = 0.99,
    log_x: bool = False,
) -> Path:
    """Overlay distributions (one curve per label) using histograms-as-density."""
    fig, ax = plt.subplots(figsize=PLOT_FIGSZ)
    upper = max((s.quantile(clip_q) for s in series_by_label.values()),
                default=None) if clip_q else None
    for lbl, s in series_by_label.items():
        s = pd.to_numeric(s, errors="coerce").dropna()
        if upper is not None: s = s[s <= upper]
        if log_x: s = s[s > 0]; s = np.log10(s)
        ax.hist(s, bins=60, density=True, alpha=0.45, label=f"label={lbl}")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel + (" (log10)" if log_x else ""))
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    return _save_fig(fig, name)


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight word tokenization (no model dependency)
# ─────────────────────────────────────────────────────────────────────────────

import re

_WORD_RE = re.compile(r"[a-zA-Z']+")


def simple_words(text: str, lower: bool = True) -> list[str]:
    """Split text into alphabetic word-tokens. Cheap; for vocab stats."""
    if not isinstance(text, str): return []
    s = text.lower() if lower else text
    return _WORD_RE.findall(s)


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────


def is_removed_body(s: pd.Series, removed_set: set[str]) -> pd.Series:
    """Boolean mask: body is null / removed / deleted / empty."""
    return s.fillna("").astype(str).str.strip().str.lower().isin(removed_set)


def char_len(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.len()


# ─────────────────────────────────────────────────────────────────────────────
# Logging-lite (no need to pull in `logging` for a few prints)
# ─────────────────────────────────────────────────────────────────────────────


def section(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


def step(msg: str) -> None:
    print(f"  -> {msg}", flush=True)
