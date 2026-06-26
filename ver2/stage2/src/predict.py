"""Stage 2 final prediction: score val + test with the trained LightGBM.

    python predict.py            # writes p_final_test.parquet (+ val if available)

For splits that carry labels, prints accuracy / f1_macro / f1_pos / AUC so you
can sanity-check the meta-model against Stage 1's p_text alone.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

try:
    from ver2.stage2.src.config import ID_COL, LABEL_COL, MODEL_PATH, PRED_DIR
    from ver2.stage2.src.data import build_dataset, get_xy, split_frame
except ImportError:  # flat layout
    from config import ID_COL, LABEL_COL, MODEL_PATH, PRED_DIR
    from data import build_dataset, get_xy, split_frame


def _report(name: str, y, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    m = {
        "n":         int(len(y)),
        "accuracy":  float(accuracy_score(y, pred)),
        "f1_macro":  float(f1_score(y, pred, average="macro")),
        "f1_pos":    float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall":    float(recall_score(y, pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y, prob)) if y.min() != y.max() else 0.0,
    }
    print(f"  [{name}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in m.items()))
    return m


def predict_split(model, df, split: str):
    part = split_frame(df, split)
    if len(part) == 0:
        print(f"  [{split}] empty -> skip")
        return
    X, y = get_xy(part)
    prob = model.predict_proba(X)[:, 1].astype(np.float32)
    out = pd.DataFrame({ID_COL: part[ID_COL].values, "p_final": prob,
                        LABEL_COL: y.values})
    out_path = PRED_DIR / f"p_final_{split}.parquet"
    out.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    _report(split, y, prob)
    print(f"  [{split}] wrote {out_path}  ({len(out):,} rows)")


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}. Run train_final.py first.")
    model = joblib.load(MODEL_PATH)
    df = build_dataset()
    for split in ("val", "test"):
        predict_split(model, df, split)


if __name__ == "__main__":
    main()
