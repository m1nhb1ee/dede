"""Report figures for both stages -> ver2/report_figures/*.png

Reads only the test-set predictions (+ OOF for reference) and the Stage 2
feature-importance CSV. Re-run after retraining Stage 2 to refresh the
Stage 2 panels; Stage 1 panels are stable.

    python ver2/report_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

ROOT      = Path(__file__).resolve().parent          # ver2/
S1_PRED   = ROOT / "stage1" / "outputs" / "predictions"
S2_OUT    = ROOT / "stage2" / "outputs"
FIG_DIR   = ROOT / "report_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

POS, NEG = "#c0392b", "#2980b9"


# ── helpers ──────────────────────────────────────────────────────────────────
def best_threshold(y, p):
    ts = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y, p >= t, average="macro", zero_division=0) for t in ts]
    i = int(np.argmax(f1s))
    return float(ts[i]), float(f1s[i])


def metrics_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    return dict(
        acc=accuracy_score(y, yhat),
        f1_macro=f1_score(y, yhat, average="macro", zero_division=0),
        f1_pos=f1_score(y, yhat, zero_division=0),
        prec=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0),
        roc_auc=roc_auc_score(y, p),
        pr_auc=average_precision_score(y, p),
    )


def _cm_panel(ax, cm, title, sub):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["0 normal", "1 depr"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["0 normal", "1 depr"])
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            c = cm[i, j]
            ax.text(j, i, f"{c:,}\n({c/total:.1%})", ha="center", va="center",
                    color="white" if c > cm.max() * 0.5 else "black", fontsize=10)
    ax.text(0.5, -0.30, sub, transform=ax.transAxes, ha="center",
            fontsize=9, color="dimgray")


def confusion_fig(y, p, stage: str, fname: str):
    thr_opt, _ = best_threshold(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, thr, tag in [(axes[0], 0.50, "default"),
                         (axes[1], thr_opt, "F1-optimal")]:
        m = metrics_at(y, p, thr)
        cm = confusion_matrix(y, (p >= thr).astype(int))
        _cm_panel(ax, cm, f"{stage} — thr={thr:.2f} ({tag})",
                  f"f1_macro={m['f1_macro']:.3f}  f1_pos={m['f1_pos']:.3f}  "
                  f"P={m['prec']:.3f}  R={m['recall']:.3f}")
    fig.suptitle(f"{stage} — Confusion Matrix (test 2022)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}  (thr_opt={thr_opt:.2f})")


def roc_pr_fig(y, p, stage: str, fname: str):
    fpr, tpr, _ = roc_curve(y, p); auc = roc_auc_score(y, p)
    prec, rec, _ = precision_recall_curve(y, p); ap = average_precision_score(y, p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    a1.plot(fpr, tpr, color=POS, lw=2, label=f"AUC = {auc:.4f}")
    a1.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    a1.set_xlabel("False Positive Rate"); a1.set_ylabel("True Positive Rate")
    a1.set_title("ROC curve"); a1.legend(loc="lower right")
    a2.plot(rec, prec, color=NEG, lw=2, label=f"AP = {ap:.4f}")
    a2.axhline(y.mean(), ls="--", color="gray", lw=1,
               label=f"baseline = {y.mean():.3f}")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
    a2.set_title("Precision–Recall curve"); a2.legend(loc="lower left")
    fig.suptitle(f"{stage} — ROC & PR (test 2022)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def calibration_fig(y, p, stage: str, fname: str):
    frac, mean_pred = calibration_curve(y, p, n_bins=15, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
    ax.plot(mean_pred, frac, "o-", color=POS, lw=2, label=stage)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title(f"{stage} — Calibration (test 2022)", fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def score_dist_fig(y, p, score_name: str, stage: str, fname: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 51)
    ax.hist(p[y == 0], bins=bins, alpha=0.6, color=NEG, label="label 0 (normal)",
            density=True)
    ax.hist(p[y == 1], bins=bins, alpha=0.6, color=POS, label="label 1 (depr)",
            density=True)
    ax.set_xlabel(score_name); ax.set_ylabel("density")
    ax.set_title(f"{stage} — {score_name} distribution by label (test 2022)",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def importance_fig(fname: str):
    df = pd.read_csv(S2_OUT / "feature_importance.csv")
    if "gain_norm" not in df.columns:
        df["gain_norm"] = df["gain"] / df["gain"].sum()
    df = df.sort_values("gain_norm", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(df["feature"], df["gain_norm"], color="#16a085")
    for yi, v in enumerate(df["gain_norm"]):
        if v > 0:
            ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_xlabel("Normalized gain  (gain / Σgain,  Σ = 1)")
    ax.set_title("Stage 2 — LightGBM Feature Importance (normalized)",
                 fontweight="bold")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}  (p_text gain_norm={df['gain_norm'].max():.3f})")


def ablation_fig(f1_A, f1_B, f1_C, fp_A, fp_B, fp_C, fname: str):
    labels = ["A: text only\n(p_text)", "B: meta only\n(no p_text)",
              "C: full ensemble\n(p_text + meta)"]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    b1 = ax.bar(x - w/2, [f1_A, f1_B, f1_C], w, color="#34495e", label="f1_macro")
    b2 = ax.bar(x + w/2, [fp_A, fp_B, fp_C], w, color="#e67e22", label="f1_pos")
    for b in (*b1, *b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004,
                f"{b.get_height():.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("score"); ax.set_ylim(0.78, 1.0)
    ax.set_title("Ablation A/B/C — test 2022 (subreddit-blind, thr 0.5)",
                 fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ── load test predictions ────────────────────────────────────────────────────
def main() -> None:
    s2 = pd.read_parquet(S2_OUT / "predictions" / "p_final_test.parquet")  # id,p_final,label
    s1 = pd.read_parquet(S1_PRED / "p_text_test_ensemble.parquet")          # id,p_text
    s1 = s1.merge(s2[["id", "label"]], on="id", how="inner")

    y1, p1 = s1["label"].to_numpy(), s1["p_text"].to_numpy()
    y2, p2 = s2["label"].to_numpy(), s2["p_final"].to_numpy()
    print(f"Stage1 test n={len(y1):,}  Stage2 test n={len(y2):,}  "
          f"pos_rate={y2.mean():.4f}")

    print("Stage 1 (text) figures:")
    confusion_fig(y1, p1, "Stage 1 (text)", "s1_confusion.png")
    roc_pr_fig(y1, p1, "Stage 1 (text)", "s1_roc_pr.png")
    calibration_fig(y1, p1, "Stage 1 (text)", "s1_calibration.png")
    score_dist_fig(y1, p1, "p_text", "Stage 1 (text)", "s1_score_dist.png")

    print("Stage 2 (ensemble) figures:")
    importance_fig("s2_feature_importance.png")
    confusion_fig(y2, p2, "Stage 2 (ensemble)", "s2_confusion.png")
    roc_pr_fig(y2, p2, "Stage 2 (ensemble)", "s2_roc_pr.png")
    calibration_fig(y2, p2, "Stage 2 (ensemble)", "s2_calibration.png")
    score_dist_fig(y2, p2, "p_final", "Stage 2 (ensemble)", "s2_score_dist.png")

    # Ablation: A = p_text@0.5 (live), C = p_final@0.5 (live), B = meta-only run.
    f1_A = f1_score(y1, p1 >= 0.5, average="macro")
    fp_A = f1_score(y1, p1 >= 0.5)
    f1_C = f1_score(y2, p2 >= 0.5, average="macro")
    fp_C = f1_score(y2, p2 >= 0.5)
    f1_B, fp_B = 0.8805, 0.8263   # from stage2 meta-only ablation (no p_text)
    print("Ablation figure:")
    ablation_fig(f1_A, f1_B, f1_C, fp_A, fp_B, fp_C, "ablation_abc.png")

    print(f"\nAll figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
