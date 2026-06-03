"""Quick eval of Stage 1 fold predictions: metrics + confusion matrices.

Reads p_text_oof_fold{k}.parquet (held-out OOF) and p_text_test_fold{k}.parquet
(fixed 2022 test) for the folds available, computes full-set metrics at the
default 0.5 threshold AND the F1-optimal threshold, and saves a 2x2 grid of
confusion matrices to outputs/predictions/confusion_matrices.png.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, accuracy_score,
)

PRED_DIR = Path("ver2/stage1/outputs/predictions")
FOLDS = [0, 1]


def best_threshold(y, p):
    """Threshold maximizing macro-F1 (scan 0.05..0.95)."""
    ts = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y, p >= t, average="macro", zero_division=0) for t in ts]
    i = int(np.argmax(f1s))
    return float(ts[i]), float(f1s[i])


def metrics_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {
        "acc":          accuracy_score(y, yhat),
        "f1_macro":     f1_score(y, yhat, average="macro", zero_division=0),
        "f1_pos":       f1_score(y, yhat, zero_division=0),
        "prec_pos":     precision_score(y, yhat, zero_division=0),
        "recall_pos":   recall_score(y, yhat, zero_division=0),
        "recall_macro": recall_score(y, yhat, average="macro", zero_division=0),
        "roc_auc":      roc_auc_score(y, p),
        "pr_auc":       average_precision_score(y, p),
    }


def fmt(m):
    return (f"acc={m['acc']:.4f}  f1_macro={m['f1_macro']:.4f}  "
            f"f1_pos={m['f1_pos']:.4f}  P_pos={m['prec_pos']:.4f}  "
            f"R_pos={m['recall_pos']:.4f}  roc_auc={m['roc_auc']:.4f}  "
            f"pr_auc={m['pr_auc']:.4f}")


panels = []   # (title, cm, subtitle)

for k in FOLDS:
    for kind, fname in [("OOF (val held-out, 605K)", f"p_text_oof_fold{k}.parquet"),
                        ("TEST (2022, 213K)",       f"p_text_test_fold{k}.parquet")]:
        fp = PRED_DIR / fname
        if not fp.exists():
            print(f"[skip] {fname} not found")
            continue
        df = pd.read_parquet(fp)
        y, p = df["label"].to_numpy(), df["p_text"].to_numpy()

        thr_opt, f1_opt = best_threshold(y, p)
        m05  = metrics_at(y, p, 0.5)
        mopt = metrics_at(y, p, thr_opt)

        print(f"\n===== FOLD {k} | {kind} =====")
        print(f"  n={len(y):,}  label_rate={y.mean():.4f}  "
              f"p_text mean={p.mean():.3f}")
        print(f"  @0.50  {fmt(m05)}")
        print(f"  @{thr_opt:.2f}* {fmt(mopt)}")

        # Confusion matrix at the F1-optimal threshold (more honest given the
        # undersampling-induced positive shift in p_text).
        cm = confusion_matrix(y, (p >= thr_opt).astype(int))
        panels.append((
            f"Fold {k} — {kind.split(' (')[0]}",
            cm,
            f"thr={thr_opt:.2f}  f1_macro={mopt['f1_macro']:.3f}  "
            f"R_pos={mopt['recall_pos']:.3f}",
        ))

# ── Plot grid ────────────────────────────────────────────────────────────
n = len(panels)
ncols = 2
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.2 * nrows))
axes = np.array(axes).reshape(-1)

for ax, (title, cm, sub) in zip(axes, panels):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["0 normal", "1 depr"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["0 normal", "1 depr"])
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            cnt = cm[i, j]
            ax.text(j, i, f"{cnt:,}\n({cnt/total:.1%})",
                    ha="center", va="center",
                    color="white" if cnt > cm.max() * 0.5 else "black",
                    fontsize=10)
    ax.text(0.5, -0.28, sub, transform=ax.transAxes, ha="center", fontsize=9,
            color="dimgray")

for ax in axes[n:]:
    ax.axis("off")

fig.suptitle("Stage 1 — Confusion Matrices (F1-optimal threshold)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = PRED_DIR / "confusion_matrices.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved {out}")
