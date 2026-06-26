"""Stage 2 hyperparameter tuning: Optuna + 3-fold CV on TRAIN.

Reuses the fold assignment from splits.parquet (same 3 folds as Stage 1 OOF)
so CV is consistent across stages. Objective = mean f1_macro @ 0.5 over folds.

    python tune.py                 # 50 trials (config.N_TRIALS)
    python tune.py --n_trials 20   # quick
"""

from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import f1_score

try:
    from ver2.stage2.src.config import (
        BEST_PARAMS_PATH, EARLY_STOPPING_ROUNDS, LGBM_BASE, N_FOLDS, N_TRIALS, SEED,
    )
    from ver2.stage2.src.data import build_dataset, get_xy, split_frame
except ImportError:  # flat layout
    from config import (
        BEST_PARAMS_PATH, EARLY_STOPPING_ROUNDS, LGBM_BASE, N_FOLDS, N_TRIALS, SEED,
    )
    from data import build_dataset, get_xy, split_frame


def _cv_f1(params: dict, train_df) -> float:
    """Mean f1_macro across the N_FOLDS held-out folds."""
    scores = []
    for k in range(N_FOLDS):
        tr = train_df[train_df["fold"] != k]
        va = train_df[train_df["fold"] == k]
        Xtr, ytr = get_xy(tr)
        Xva, yva = get_xy(va)
        model = lgb.LGBMClassifier(**params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        pred = (model.predict_proba(Xva)[:, 1] >= 0.5).astype(int)
        scores.append(f1_score(yva, pred, average="macro"))
    return float(np.mean(scores))


def make_objective(train_df):
    def objective(trial: optuna.Trial) -> float:
        params = dict(LGBM_BASE)
        params.update({
            "num_leaves":        trial.suggest_categorical("num_leaves", [31, 63, 127, 255]),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_categorical("min_child_samples", [50, 100, 200, 500]),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "colsample_bytree":  trial.suggest_categorical("colsample_bytree", [0.6, 0.8, 1.0]),
            "subsample":         trial.suggest_categorical("subsample", [0.6, 0.8, 1.0]),
        })
        return _cv_f1(params, train_df)
    return objective


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2 LightGBM tuning")
    p.add_argument("--n_trials", type=int, default=N_TRIALS)
    args = p.parse_args()

    df = build_dataset()
    train_df = split_frame(df, "train")
    print(f"Tuning on {len(train_df):,} train rows, {N_FOLDS}-fold CV, "
          f"{args.n_trials} trials")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(),
    )

    def _persist_best(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Checkpoint best params after every improving trial so an interrupted
        # run still leaves a usable best_params.json.
        if study.best_trial.number == trial.number:
            best = dict(LGBM_BASE)
            best.update(study.best_params)
            BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2), encoding="utf-8")

    study.optimize(make_objective(train_df), n_trials=args.n_trials,
                   show_progress_bar=True, callbacks=[_persist_best])

    best = dict(LGBM_BASE)
    best.update(study.best_params)
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"\nBest f1_macro (CV): {study.best_value:.4f}")
    print(f"Best params -> {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    main()
