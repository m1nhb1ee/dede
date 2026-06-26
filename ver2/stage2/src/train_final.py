"""Stage 2 final training: fit LightGBM on full TRAIN with best params.

Early stopping uses a small stratified holdout carved from train (so it does
not depend on the full-model val p_text being ready). Saves the model plus a
feature-importance table + plot.

    python train_final.py            # uses best_params.json if present, else baseline
"""

from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from ver2.stage2.src.config import (
        BEST_PARAMS_PATH, EARLY_STOPPING_ROUNDS, FEATURE_COLS, LGBM_BASE,
        MODEL_PATH, OUT_DIR, SEED,
    )
    from ver2.stage2.src.data import build_dataset, get_xy, split_frame
except ImportError:  # flat layout
    from config import (
        BEST_PARAMS_PATH, EARLY_STOPPING_ROUNDS, FEATURE_COLS, LGBM_BASE,
        MODEL_PATH, OUT_DIR, SEED,
    )
    from data import build_dataset, get_xy, split_frame


def load_params() -> dict:
    if BEST_PARAMS_PATH.exists():
        params = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        print(f"Using tuned params from {BEST_PARAMS_PATH.name}")
        return params
    print("No best_params.json -> using baseline LGBM_BASE")
    return dict(LGBM_BASE)


def main() -> None:
    df = build_dataset()
    train_df = split_frame(df, "train")
    X, y = get_xy(train_df)

    # Stratified holdout just for early stopping.
    Xtr, Xes, ytr, yes = train_test_split(
        X, y, test_size=0.05, stratify=y, random_state=SEED,
    )
    print(f"Fit on {len(Xtr):,} rows, early-stop on {len(Xes):,} holdout")

    model = lgb.LGBMClassifier(**load_params())
    model.fit(
        Xtr, ytr,
        eval_set=[(Xes, yes)], eval_metric=["binary_logloss", "auc"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                   lgb.log_evaluation(100)],
    )
    print(f"Best iteration: {model.best_iteration_}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

    # Feature importance (gain).
    imp = (pd.DataFrame({"feature": FEATURE_COLS,
                         "gain": model.booster_.feature_importance("gain")})
           .sort_values("gain", ascending=False).reset_index(drop=True))
    imp_path = OUT_DIR / "feature_importance.csv"
    imp.to_csv(imp_path, index=False)
    print(f"Saved importance -> {imp_path}")
    print(imp.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(imp["feature"][::-1], imp["gain"][::-1], color="steelblue")
    ax.set_xlabel("gain"); ax.set_title("Stage 2 LightGBM feature importance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=130)
    print(f"Saved plot -> {OUT_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
