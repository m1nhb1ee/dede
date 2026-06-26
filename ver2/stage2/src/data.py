"""Stage 2 dataset assembly: meta_features + splits + Stage 1 p_text.

p_text is sourced per split so there is no leakage:
    train rows <- p_text_oof_train.parquet      (OOF: model never saw the row)
    test  rows <- p_text_test_ensemble.parquet  (mean over fold models)
    val   rows <- p_text_val.parquet            (full model; optional)

Rows whose p_text is unavailable (e.g. a dry-run before all folds finished)
are dropped from TRAIN with a warning, so the pipeline still runs end-to-end
on whatever OOF coverage exists.
"""

from __future__ import annotations

import pandas as pd

try:
    from ver2.stage2.src.config import (
        ENGINEERED_COLS, FEATURE_COLS, ID_COL, LABEL_COL, META_FEATURES,
        P_TEXT_OOF_TRAIN, P_TEXT_TEST, P_TEXT_VAL, SPLITS,
    )
except ImportError:  # flat layout
    from config import (
        ENGINEERED_COLS, FEATURE_COLS, ID_COL, LABEL_COL, META_FEATURES,
        P_TEXT_OOF_TRAIN, P_TEXT_TEST, P_TEXT_VAL, SPLITS,
    )


def _p_text_map() -> pd.Series:
    """Union of id->p_text across train(OOF) / test / val sources.

    Splits are id-disjoint, so a plain concat keyed by id is unambiguous.
    Missing optional files (val, or OOF before fold 0 lands) are skipped.
    """
    frames = []
    for path in (P_TEXT_OOF_TRAIN, P_TEXT_TEST, P_TEXT_VAL):
        if path.exists():
            df = pd.read_parquet(path, columns=[ID_COL, "p_text"])
            frames.append(df)
            print(f"  p_text <- {path.name}  ({len(df):,} rows)")
        else:
            print(f"  [skip] {path.name} not found")
    if not frames:
        raise FileNotFoundError("No p_text source parquet found (need at least OOF or test).")
    allp = pd.concat(frames, ignore_index=True)
    dup = int(allp[ID_COL].duplicated().sum())
    if dup:
        print(f"  [WARN] {dup:,} duplicate ids across p_text sources; keeping first")
        allp = allp.drop_duplicates(ID_COL, keep="first")
    return allp.set_index(ID_COL)["p_text"]


def _engineer(df: pd.DataFrame) -> None:
    """Add length-normalized rate features in place (ENGINEERED_COLS).

    Rates divide a count by num_words; avg_sentence_len by num_sentences. Both
    denominators are clipped at 1 to avoid div-by-zero on empty posts.
    """
    w = df["num_words"].clip(lower=1).astype("float32")
    df["first_person_rate"]  = (df["num_first_person"]   / w).astype("float32")
    df["negative_word_rate"] = (df["num_negative_words"] / w).astype("float32")
    df["exclamation_rate"]   = (df["num_exclamations"]   / w).astype("float32")
    df["question_rate"]      = (df["num_questions"]      / w).astype("float32")
    df["caps_word_rate"]     = (df["num_caps_words"]     / w).astype("float32")
    df["ellipsis_rate"]      = (df["num_ellipsis"]       / w).astype("float32")
    df["absolutist_rate"]    = (df["num_absolutist"]     / w).astype("float32")
    df["second_person_rate"] = (df["num_second_person"]  / w).astype("float32")
    s = df["num_sentences"].clip(lower=1).astype("float32")
    df["avg_sentence_len"]   = (df["num_words"] / s).astype("float32")


def build_dataset() -> pd.DataFrame:
    """Return one dataframe: id, FEATURE_COLS, label, time_split, fold.

    p_text is attached from the per-split sources. Rows missing p_text are
    dropped (warned). All other meta features come from meta_features.parquet.
    """
    print("Building Stage 2 dataset...")
    meta   = pd.read_parquet(META_FEATURES)
    splits = pd.read_parquet(SPLITS, columns=[ID_COL, "time_split", "fold"])
    df = meta.merge(splits, on=ID_COL, how="inner")
    print(f"  meta+splits: {len(df):,} rows")

    df["p_text"] = df[ID_COL].map(_p_text_map()).astype("float32")

    n_missing = int(df["p_text"].isna().sum())
    if n_missing:
        by_split = df[df["p_text"].isna()]["time_split"].value_counts().to_dict()
        print(f"  [WARN] dropping {n_missing:,} rows with no p_text {by_split}")
        df = df[df["p_text"].notna()].reset_index(drop=True)

    _engineer(df)

    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Feature columns absent from data: {missing_cols}")
    print(f"  ready: {len(df):,} rows  | {len(FEATURE_COLS)} features "
          f"({len(ENGINEERED_COLS)} engineered)")
    return df


def split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["time_split"] == split].reset_index(drop=True)


def get_xy(df: pd.DataFrame):
    """Return (X[FEATURE_COLS], y[label]) with booleans cast to int for LGBM."""
    X = df[FEATURE_COLS].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype("int8")
    y = df[LABEL_COL].astype("int8")
    return X, y
