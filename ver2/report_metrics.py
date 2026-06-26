"""Project-wide metrics aggregation.

Collects every available model/fold/split into one long-form CSV plus a
formatted markdown report:

  Stage 1 (text model, MentalRoBERTa+LoRA):
    - per-fold OOF  (held-out validation, p_text_oof_fold{k})
    - per-fold TEST (2022, p_text_test_fold{k})
    - TEST ensemble (mean of fold models, p_text_test_ensemble)
    - OOF train aggregate (all train rows, p_text_oof_train)
  Stage 2 (LightGBM meta-model, baseline -> production):
    - TEST (p_final_test)

For every entry we report metrics @0.50 and @best-f1-macro threshold, plus the
confusion matrix. Per-fold groups also get mean +/- std.

    python ver2/report_metrics.py

Outputs -> ver2/report/all_metrics.csv  and  ver2/report/PROJECT_METRICS.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent          # .../DeDe/ver2
S1_PRED = ROOT / "stage1" / "outputs" / "predictions"
S2_PRED = ROOT / "stage2" / "outputs" / "predictions"
OUT_DIR = ROOT / "report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_KEYS = ["acc", "f1_macro", "f1_pos", "prec_pos", "recall_pos",
               "roc_auc", "pr_auc"]


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    ts = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, p >= t, average="macro", zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f1s))])


def metrics_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "thr":        round(thr, 3),
        "acc":        accuracy_score(y, yhat),
        "f1_macro":   f1_score(y, yhat, average="macro", zero_division=0),
        "f1_pos":     f1_score(y, yhat, zero_division=0),
        "prec_pos":   precision_score(y, yhat, zero_division=0),
        "recall_pos": recall_score(y, yhat, zero_division=0),
        "roc_auc":    roc_auc_score(y, p) if y.min() != y.max() else float("nan"),
        "pr_auc":     average_precision_score(y, p) if y.min() != y.max() else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def load_yp(path: Path, score_col: str, label_from: Path | None = None):
    """Return (y, p) from a predictions parquet; merge labels from another file
    when the score file has no label column (the test ensemble)."""
    df = pd.read_parquet(path)
    if "label" not in df.columns:
        lab = pd.read_parquet(label_from, columns=["id", "label"])
        df = df.merge(lab, on="id", how="inner")
    return df["label"].to_numpy().astype(int), df[score_col].to_numpy()


# ── collect every (stage, model, split, fold) entry ──────────────────────────
entries = []   # (stage, model, split, fold, y, p)

for k in (0, 1, 2):
    f_oof = S1_PRED / f"p_text_oof_fold{k}.parquet"
    if f_oof.exists():
        y, p = load_yp(f_oof, "p_text")
        entries.append(("Stage1", "text", "OOF-val", str(k), y, p))
    f_te = S1_PRED / f"p_text_test_fold{k}.parquet"
    if f_te.exists():
        y, p = load_yp(f_te, "p_text")
        entries.append(("Stage1", "text", "TEST", str(k), y, p))

f_ens = S1_PRED / "p_text_test_ensemble.parquet"
if f_ens.exists():
    y, p = load_yp(f_ens, "p_text", label_from=S1_PRED / "p_text_test_fold0.parquet")
    entries.append(("Stage1", "text-ensemble", "TEST", "ens", y, p))

f_ooft = S1_PRED / "p_text_oof_train.parquet"
if f_ooft.exists():
    y, p = load_yp(f_ooft, "p_text")
    entries.append(("Stage1", "text-OOF", "TRAIN-oof", "all", y, p))

f_s2 = S2_PRED / "p_final_test.parquet"
if f_s2.exists():
    y, p = load_yp(f_s2, "p_final")
    entries.append(("Stage2", "lgbm-baseline", "TEST", "-", y, p))


# ── build long-form rows ─────────────────────────────────────────────────────
rows = []
for stage, model, split, fold, y, p in entries:
    print(f"computing {stage}/{model}/{split}/fold={fold}  (n={len(y):,})")
    for mode, thr in (("@0.50", 0.5), ("@best-f1", best_threshold(y, p))):
        m = metrics_at(y, p, thr)
        rows.append({
            "stage": stage, "model": model, "split": split, "fold": fold,
            "mode": mode, "n": len(y), "pos_rate": round(float(y.mean()), 4),
            **{k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in m.items()},
        })

allm = pd.DataFrame(rows)
csv_path = OUT_DIR / "all_metrics.csv"
allm.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}  ({len(allm)} rows)")


# ── markdown report ──────────────────────────────────────────────────────────
def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, r in df.iterrows():
        body.append("| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join([head, sep] + body)


def fold_agg(df: pd.DataFrame, keys: list[str]) -> str:
    """mean +/- std line for a per-fold group."""
    parts = []
    for k in keys:
        parts.append(f"{k}={df[k].mean():.4f}±{df[k].std(ddof=0):.4f}")
    return "  ".join(parts)


show = ["stage", "model", "split", "fold", "mode", "n", "pos_rate",
        "thr", "acc", "f1_macro", "f1_pos", "prec_pos", "recall_pos",
        "roc_auc", "pr_auc", "TN", "FP", "FN", "TP"]

md = ["# DeDe — Project Metrics Report\n"]
md.append("Generated by `ver2/report_metrics.py`. Threshold modes: **@0.50** "
          "(default) and **@best-f1** (macro-F1-optimal, scanned 0.05–0.95).\n")
md.append("- Stage 1 = MentalRoBERTa + LoRA text model (per-fold + ensemble).")
md.append("- Stage 2 = LightGBM meta-model, **baseline** (production).")
md.append("- OOF-val base rate ≈ 0.198 (full corpus); TEST (2022) base rate ≈ 0.260.\n")

md.append("## Chú thích metrics\n")
md.append("Quy ước: lớp **dương (1)** = trầm cảm, lớp **âm (0)** = bình thường. "
          "Confusion matrix: **TP** = dương đoán đúng, **TN** = âm đoán đúng, "
          "**FP** = âm bị đoán nhầm thành dương (báo động giả), "
          "**FN** = dương bị bỏ sót (nguy hiểm nhất trong sàng lọc trầm cảm).\n")
md.append("\n".join([
    "| Metric | Ý nghĩa | Công thức |",
    "| --- | --- | --- |",
    "| **n** | số mẫu trong tập | — |",
    "| **pos_rate** | tỉ lệ lớp dương (base rate) | (TP+FN)/n |",
    "| **thr** | ngưỡng quyết định: p ≥ thr → đoán dương | — |",
    "| **acc** (accuracy) | tỉ lệ đoán đúng tổng thể; dễ gây ảo khi lệch lớp | (TP+TN)/n |",
    "| **prec_pos** (precision) | trong các ca *đoán* dương, bao nhiêu % thật sự dương → ít báo nhầm | TP/(TP+FP) |",
    "| **recall_pos** (recall/sensitivity) | trong các ca *thật sự* dương, bắt được bao nhiêu % → ít bỏ sót | TP/(TP+FN) |",
    "| **f1_pos** | F1 của riêng lớp dương = trung bình điều hòa precision & recall | 2·P·R/(P+R) |",
    "| **f1_macro** | trung bình F1 của cả 2 lớp (không trọng số) → đánh giá cân bằng khi lệch lớp | (F1_pos+F1_neg)/2 |",
    "| **roc_auc** | khả năng phân biệt 2 lớp ở mọi ngưỡng (0.5=ngẫu nhiên, 1=hoàn hảo) | AUC của ROC |",
    "| **pr_auc** | average precision; phù hợp khi lớp dương hiếm | AUC của Precision–Recall |",
]) + "\n")
md.append("**Chế độ ngưỡng:** `@0.50` = ngưỡng mặc định; `@best-f1` = ngưỡng tối đa hóa "
          "f1_macro (quét 0.05–0.95).  \n"
          "**OOF** (out-of-fold) = dự đoán trên dữ liệu mà fold đó *không* được train → "
          "ước lượng trung thực, không rò rỉ.  \n"
          "**Sàng lọc trầm cảm** ưu tiên **recall_pos** cao (giảm FN — bỏ sót ca dương).\n")

md.append("## 1. Stage 1 — text model\n")
s1 = allm[allm["stage"] == "Stage1"]
md.append(md_table(s1, show) + "\n")

# per-fold aggregates
md.append("### Stage 1 fold aggregates (mean ± std)\n")
for split in ("OOF-val", "TEST"):
    grp = allm[(allm["model"] == "text") & (allm["split"] == split) & (allm["mode"] == "@best-f1")]
    if len(grp):
        md.append(f"- **{split} @best-f1** ({len(grp)} folds): "
                  + fold_agg(grp, ["f1_macro", "f1_pos", "recall_pos", "roc_auc"]))
    grp05 = allm[(allm["model"] == "text") & (allm["split"] == split) & (allm["mode"] == "@0.50")]
    if len(grp05):
        md.append(f"- **{split} @0.50** ({len(grp05)} folds): "
                  + fold_agg(grp05, ["f1_macro", "f1_pos", "recall_pos", "roc_auc"]))
md.append("")

md.append("## 2. Stage 2 — LightGBM meta-model (baseline / production)\n")
s2 = allm[allm["stage"] == "Stage2"]
md.append(md_table(s2, show) + "\n")

md.append("## 3. Headline (production) — TEST 2022\n")
ens = allm[(allm["model"] == "text-ensemble") & (allm["mode"] == "@best-f1")]
s2b = allm[(allm["stage"] == "Stage2") & (allm["mode"] == "@0.50")]
if len(ens):
    r = ens.iloc[0]
    md.append(f"- **Stage 1 ensemble (text only)** @best-f1: f1_macro={r.f1_macro:.4f}, "
              f"f1_pos={r.f1_pos:.4f}, recall_pos={r.recall_pos:.4f}, roc_auc={r.roc_auc:.4f}")
if len(s2b):
    r = s2b.iloc[0]
    md.append(f"- **Stage 2 ensemble (full, baseline)** @0.50: f1_macro={r.f1_macro:.4f}, "
              f"f1_pos={r.f1_pos:.4f}, recall_pos={r.recall_pos:.4f}, roc_auc={r.roc_auc:.4f}")
md.append("\n> Calibration experiment (reverted; reference only): see "
          "`stage2/outputs/report/comparison_report.md`.\n")

report_path = OUT_DIR / "PROJECT_METRICS.md"
report_path.write_text("\n".join(md), encoding="utf-8")
print(f"Saved {report_path}")
