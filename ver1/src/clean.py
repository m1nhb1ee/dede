import pandas as pd
import re
from pathlib import Path

RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_MULTI_SPACE   = re.compile(r"[ \t]{2,}")
RE_MD_LINK       = re.compile(r"\[([^\]]+)\]\(http[^\)]+\)")
RE_URL           = re.compile(r"http\S+")
RE_WHITESPACE    = re.compile(r"\s+")

def clean(input_path: str):
    df = pd.read_csv(input_path)
    print(f"Loaded: {len(df):,} rows | columns: {list(df.columns)}")
    df.drop(columns=[c for c in df.columns if c not in ["text", "class", "label"]], inplace=True, errors='ignore')
    # Chuẩn hoá tên cột
    df = df.rename(columns={'class': 'label'}, errors='ignore')

    # Clean text
    t = df["text"].fillna("").astype(str).str.strip()
    t = t.str.replace(RE_MULTI_NEWLINE, "\n\n", regex=True)
    t = t.str.replace(RE_MULTI_SPACE,   " ",    regex=True)
    t = t.str.replace(RE_MD_LINK,       r"\1",  regex=True)
    t = t.str.replace(RE_URL,           "",     regex=True)
    t = t.str.replace(RE_WHITESPACE,    " ",    regex=True).str.strip()
    df["text"] = t

    # Bỏ row rỗng hoặc quá ngắn
    df = df[df["text"].str.split().str.len() >= 5]
    df = df[df["label"].isin([0, 1])]

    # Lọc duplicate
    before     = len(df)
    duplicates = df[df.duplicated(subset=["text"], keep="first")]
    print(f"Duplicates — label 0: {(duplicates['label']==0).sum():,} | label 1: {(duplicates['label']==1).sum():,}")
    df = df.drop_duplicates(subset=["text"], keep="first")
    print(f"Removed duplicates: {before - len(df):,} rows")

    out_path = str(Path(input_path).parent / (Path(input_path).stem + "_cleaned.csv"))
    df[["text", "label"]].to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"Done: {len(df):,} rows → {out_path}")
    print(f"Label: {df['label'].value_counts().to_dict()}")

if __name__ == "__main__":
    clean(r"C:\Users\LOQ\Downloads\data_vietnamese.csv")