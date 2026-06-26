"""Stage 2 before/after comparison report.

Compares the legacy baseline (class_weight=balanced) against the recalibrated
models (raw = no class_weight, calibrated = +isotonic) on the TEST split, and
writes a self-contained report:

    outputs/report/comparison_report.md      metrics tables + confusion matrices
    outputs/report/confusion_matrices.png     CMs @0.5 and @best-f1 threshold
    outputs/report/reliability_test.png       calibration curves (copied here)

Metrics cover BOTH views:
  - probability quality (log loss, Brier, ECE)  -> the thing we optimized
  - thresholded labels (acc/prec/rec/f1 + confusion matrix) @0.5 and @best-f1

    python report_compare.py
"""

from __future__ import annotations

import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)

try:
    from ver2.stage2.src.config import (
        MODEL_CALIBRATED_PATH, MODEL_PATH, MODEL_RAW_PATH, OUT_DIR,
    )
    from ver2.stage2.src.data import build_dataset, get_xy, split_frame
except ImportError:  # flat layout
    from config import (
        MODEL_CALIBRATED_PATH, MODEL_PATH, MODEL_RAW_PATH, OUT_DIR,
    )
    from data import build_dataset, get_xy, split_frame


REPORT_DIR = OUT_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("baseline (before)", MODEL_PATH),
    ("raw (after)",       MODEL_RAW_PATH),
    ("calibrated (after)", MODEL_CALIBRATED_PATH),
]


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out += abs(y[m].mean() - p[m].mean()) * m.mean()
    return float(out)


def best_macro_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold (0.01..0.99) that maximizes f1_macro."""
    grid = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y, (p >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(f1s))])


def prob_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier":    brier_score_loss(y, p),
        "ECE":      ece(y, p),
        "roc_auc":  roc_auc_score(y, p),
        "mean_p":   float(p.mean()),
    }


def label_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    return {
        "threshold": thr,
        "accuracy":  accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall":    recall_score(y, pred, zero_division=0),
        "f1_pos":    f1_score(y, pred, pos_label=1, zero_division=0),
        "f1_macro":  f1_score(y, pred, average="macro"),
    }


def md_table(headers: list[str], rows: list[list]) -> str:
    line = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def cm_block(name: str, y: np.ndarray, p: np.ndarray, thr: float) -> str:
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return (f"**{name}** @thr={thr:.2f}\n\n"
            + md_table(["actual \\ pred", "neg (0)", "pos (1)"],
                       [["neg (0)", tn, fp], ["pos (1)", fn, tp]]))


def main() -> None:
    models = [(n, joblib.load(path)) for n, path in MODELS if path.exists()]
    if not models:
        raise SystemExit("No Stage 2 models found. Run train_final.py first.")

    test = split_frame(build_dataset(), "test")
    X, y_ser = get_xy(test)
    y = y_ser.to_numpy().astype(int)
    base_rate = float(y.mean())
    probs = {name: m.predict_proba(X)[:, 1].astype(np.float64) for name, m in models}

    # ── metric tables ────────────────────────────────────────────────────────
    prob_rows, half_rows, best_rows, thr_best = [], [], [], {}
    for name in probs:
        p = probs[name]
        pm = prob_metrics(y, p)
        prob_rows.append([name] + [f"{pm[k]:.4f}" for k in
                                   ("log_loss", "brier", "ECE", "roc_auc", "mean_p")])
        lm = label_metrics(y, p, 0.5)
        half_rows.append([name] + [f"{lm[k]:.4f}" for k in
                                   ("accuracy", "precision", "recall", "f1_pos", "f1_macro")])
        t = best_macro_threshold(y, p)
        thr_best[name] = t
        bm = label_metrics(y, p, t)
        best_rows.append([name, f"{t:.2f}"] + [f"{bm[k]:.4f}" for k in
                          ("accuracy", "precision", "recall", "f1_pos", "f1_macro")])

    # ── confusion-matrix figure (2 rows: @0.5, @best-f1 ; cols = models) ──────
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    axes = np.atleast_2d(axes)
    for col, (name, _) in enumerate(models):
        for row, thr in enumerate((0.5, thr_best[name])):
            pred = (probs[name] >= thr).astype(int)
            cm = confusion_matrix(y, pred, labels=[0, 1])
            ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(
                ax=axes[row, col], colorbar=False, cmap="Blues", values_format="d")
            tag = "0.50" if row == 0 else f"{thr:.2f}*"
            axes[row, col].set_title(f"{name}\n@thr={tag}", fontsize=10)
    fig.suptitle("Stage 2 confusion matrices — before vs after  (* = best-f1 thr)")
    fig.tight_layout()
    cm_path = REPORT_DIR / "confusion_matrices.png"
    fig.savefig(cm_path, dpi=130)
    print(f"Saved {cm_path}")

    # copy reliability curve next to the report if it exists
    rel_src = OUT_DIR / "reliability_test.png"
    if rel_src.exists():
        shutil.copy(rel_src, REPORT_DIR / "reliability_test.png")

    # ── markdown report ──────────────────────────────────────────────────────
    md = []
    md.append("# Stage 2 — Before/After Comparison\n")
    md.append(f"TEST split: **{len(y):,}** rows · base rate (actual positive) = "
              f"**{base_rate:.4f}**\n")
    md.append("- **before** = `stage2_lgbm.pkl` (class_weight=balanced, f1-tuned)")
    md.append("- **after**  = `stage2_lgbm_raw.pkl` (no class_weight) and "
              "`stage2_lgbm_calibrated.pkl` (+ isotonic, canonical)\n")

    md.append("## 1. Probability quality (what we optimized)\n")
    md.append("Lower log loss / Brier / ECE = better. `mean_p` should sit near the "
              "base rate; the baseline over-predicts.\n")
    md.append(md_table(["model", "log_loss", "brier", "ECE", "roc_auc", "mean_p"],
                       prob_rows) + "\n")

    md.append("## 2. Thresholded labels @ 0.50\n")
    md.append(md_table(["model", "accuracy", "precision", "recall", "f1_pos", "f1_macro"],
                       half_rows) + "\n")
    md.append("> Note: after dropping class_weight the probability scale shrank, so "
              "recall @0.5 drops — that is expected, not a regression. See §3 for a "
              "fair like-for-like at each model's best threshold.\n")

    md.append("## 3. Thresholded labels @ best-f1-macro threshold\n")
    md.append(md_table(["model", "thr*", "accuracy", "precision", "recall", "f1_pos", "f1_macro"],
                       best_rows) + "\n")

    md.append("## 4. Confusion matrices\n")
    md.append("![confusion matrices](confusion_matrices.png)\n")
    for name in probs:
        md.append(cm_block(name, y, probs[name], 0.5) + "\n")
        md.append(cm_block(name, y, probs[name], thr_best[name]) + "\n")

    if (REPORT_DIR / "reliability_test.png").exists():
        md.append("## 5. Reliability (calibration) curve\n")
        md.append("![reliability](reliability_test.png)\n")

    report_path = REPORT_DIR / "comparison_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {report_path}")

    # also echo the core tables to stdout
    print("\n# Probability quality")
    print(md_table(["model", "log_loss", "brier", "ECE", "roc_auc", "mean_p"], prob_rows))
    print("\n# Labels @0.5")
    print(md_table(["model", "acc", "prec", "rec", "f1_pos", "f1_macro"], half_rows))
    print("\n# Labels @best-f1 thr")
    print(md_table(["model", "thr*", "acc", "prec", "rec", "f1_pos", "f1_macro"], best_rows))


if __name__ == "__main__":
    main()
