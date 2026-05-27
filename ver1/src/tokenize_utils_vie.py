from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

MAX_SEQ_LEN    = 512
MAX_BODY_LEN   = 509   
LENGTH_SCALE   = 0.1
MAX_BODY_WORDS = 1000
ENCODE_BATCH   = 512

SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
PAD_TOKEN      = "[PAD]"
CLS_TOKEN      = "[CLS]"
SEP_TOKEN      = "[SEP]"

def train_bpe_tokenizer(
    df: pd.DataFrame,
    output_dir: str,
    vocab_size: int = 16000,
    min_frequency: int = 2,
) -> ByteLevelBPETokenizer:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.txt"
    print(f"[tokenizer] Ghi corpus → {corpus_path}")
    df["body"].fillna("").astype(str).to_csv(corpus_path, index=False, header=False)

    print(f"[tokenizer] Train BPE vocab_size={vocab_size} ...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(corpus_path)],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(str(output_dir))
    print(f"[tokenizer] Saved → {output_dir}/vocab.json + merges.txt")
    return tokenizer


def load_tokenizer(tokenizer_dir: str) -> ByteLevelBPETokenizer:
    d = Path(tokenizer_dir)
    tokenizer = ByteLevelBPETokenizer(
        vocab=str(d / "vocab.json"),
        merges=str(d / "merges.txt"),
    )
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer


def compute_length_feat(word_count: int) -> float:
    raw = min(word_count / MAX_BODY_WORDS, 1.0)
    return float(raw * LENGTH_SCALE)


def adaptive_truncate(
    body_tokens: list[int],
    max_len: int = MAX_BODY_LEN,
    label: int | None = None,
) -> list[int]:
    if len(body_tokens) <= max_len:
        return body_tokens
    head_ratio = 0.50 if label != 0 else 0.67
    head_len   = int(max_len * head_ratio)
    tail_len   = max_len - head_len
    return body_tokens[:head_len] + body_tokens[-tail_len:]


def tokenize_sample(
    body: str,
    tokenizer: ByteLevelBPETokenizer,
    label: int | None = None,
) -> dict:
    """
    Format: [CLS] [SEP] <body_tokens> [SEP] [PAD]...
              seg0  seg0   seg1...              seg0
    """
    vocab  = tokenizer.get_vocab()
    PAD_ID = vocab[PAD_TOKEN]
    CLS_ID = vocab[CLS_TOKEN]
    SEP_ID = vocab[SEP_TOKEN]

    body_ids = adaptive_truncate(
        tokenizer.encode(str(body)).ids, MAX_BODY_LEN, label=label
    )

    seq     = [CLS_ID, SEP_ID] + body_ids + [SEP_ID]
    seq_len = len(seq)
    pad_len = MAX_SEQ_LEN - seq_len

    segment_ids = (
        [0, 0]                        +   # [CLS] [SEP]
        [1] * len(body_ids)           +   # body
        [1]                           +   # trailing [SEP]
        [0] * pad_len                     # padding
    )

    return {
        "input_ids":      np.array(seq + [PAD_ID] * pad_len, dtype=np.int32),
        "attention_mask": np.array([1] * seq_len + [0] * pad_len, dtype=np.int8),
        "segment_ids":    np.array(segment_ids, dtype=np.int8),
        "length_feat":    np.float32(compute_length_feat(len(str(body).split()))),
        "label":          label,
    }


def _encode_split(
    split_df: pd.DataFrame,
    split_name: str,
    tokenizer: ByteLevelBPETokenizer,
    output_dir: Path,
) -> None:
    n = len(split_df)
    print(f"\n[encode:{split_name}] {n} samples | batch_size={ENCODE_BATCH}")

    bodies  = split_df["body"].fillna("").astype(str).tolist()
    labels  = split_df["label"].tolist()
    all_ids = split_df["id"].values if "id" in split_df.columns else np.arange(n)

    vocab  = tokenizer.get_vocab()
    PAD_ID = vocab[PAD_TOKEN]
    CLS_ID = vocab[CLS_TOKEN]
    SEP_ID = vocab[SEP_TOKEN]

    out_input_ids      = np.zeros((n, MAX_SEQ_LEN), dtype=np.int32)
    out_attention_mask = np.zeros((n, MAX_SEQ_LEN), dtype=np.int8)
    out_segment_ids    = np.zeros((n, MAX_SEQ_LEN), dtype=np.int8)
    out_length_feat    = np.zeros(n,                dtype=np.float32)
    out_labels         = np.zeros(n,                dtype=np.int8)

    for start in range(0, n, ENCODE_BATCH):
        end      = min(start + ENCODE_BATCH, n)
        if start % (ENCODE_BATCH * 10) == 0:
            print(f"  {start}/{n}")

        body_encs = tokenizer.encode_batch(bodies[start:end])

        for j, i in enumerate(range(start, end)):
            label    = int(labels[i])
            body_ids = adaptive_truncate(body_encs[j].ids, MAX_BODY_LEN, label=label)

            # [CLS] [SEP] body... [SEP]
            seq     = [CLS_ID, SEP_ID] + body_ids + [SEP_ID]
            seq_len = len(seq)
            pad_len = MAX_SEQ_LEN - seq_len

            out_input_ids[i]      = seq + [PAD_ID] * pad_len
            out_attention_mask[i] = [1] * seq_len + [0] * pad_len
            out_segment_ids[i]    = (
                [0, 0]                  +   # [CLS] [SEP]
                [1] * len(body_ids)     +   # body
                [1]                     +   # trailing [SEP]
                [0] * pad_len               # padding
            )
            out_length_feat[i]    = compute_length_feat(len(bodies[i].split()))
            out_labels[i]         = label

    out_path = output_dir / f"{split_name}.npz"
    np.savez_compressed(
        out_path,
        input_ids      = out_input_ids,
        attention_mask = out_attention_mask,
        segment_ids    = out_segment_ids,
        length_feat    = out_length_feat,
        labels         = out_labels,
        ids            = all_ids,
    )
    print(f"  {n}/{n}")
    print(f"  Saved → {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")


def _load_and_normalize(csv_path: str) -> pd.DataFrame:
    """Load CSV và chuẩn hoá tên cột về 'body' + 'label'."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"text": "body", "class": "label"}, errors="ignore")
    df = df.drop(columns=["timeutc", "time_utc", "timeUTC", "title"], errors="ignore")
    for col in ("body", "label"):
        if col not in df.columns:
            raise ValueError(
                f"[{csv_path}] thiếu column '{col}'. "
                f"Các cột hiện có: {list(df.columns)}"
            )
    df["label"] = df["label"].astype(int)
    return df


def run_preprocessing(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    output_dir: str,
    tokenizer_dir: str,
    do_train_tokenizer: bool = False,
    vocab_size: int = 16000,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sep = "=" * 55

    print(f"\n{sep}")

    splits = {}
    for name, path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        print(f"[load] {name}: {path}")
        df = _load_and_normalize(path)
        print(f"       {len(df):,} samples | "
              f"label 0: {(df['label']==0).sum():,} | "
              f"label 1: {(df['label']==1).sum():,}")
        splits[name] = df

    print(f"\n[tokenizer] ...")
    if do_train_tokenizer:
        tokenizer = train_bpe_tokenizer(splits["train"], tokenizer_dir, vocab_size=vocab_size)
    else:
        print(f"  Load từ: {tokenizer_dir}")
        tokenizer = load_tokenizer(tokenizer_dir)

    print(f"\n[encode] format: [CLS][SEP][Body][SEP]")
    for name, df in splits.items():
        _encode_split(df, name, tokenizer, output_dir)

    print(f"\n{sep}")
    print(f"Output: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name:<25} {f.stat().st_size/1024/1024:>6.1f} MB")
    print(sep)


def tokenize_for_inference(
    body: str,
    tokenizer: ByteLevelBPETokenizer,
) -> dict:
    """Tokenize một sample để inference (không cần label, không cần title)."""
    sample = tokenize_sample(body=body, tokenizer=tokenizer, label=None)
    sample.pop("label")
    return sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",           type=str, required=True,
                        help="Đường dẫn file train.csv")
    parser.add_argument("--val",             type=str, required=True,
                        help="Đường dẫn file val.csv")
    parser.add_argument("--test",            type=str, required=True,
                        help="Đường dẫn file test.csv")
    parser.add_argument("--output",          type=str, default="./data_processed",
                        help="Thư mục lưu file .npz")
    parser.add_argument("--tokenizer_dir",   type=str, default="./tokenizer",
                        help="Thư mục chứa vocab.json + merges.txt")
    parser.add_argument("--train_tokenizer", action="store_true",
                        help="Train BPE tokenizer mới từ corpus train")
    parser.add_argument("--vocab_size",      type=int, default=16000)
    args = parser.parse_args()

    run_preprocessing(
        train_csv          = args.train,
        val_csv            = args.val,
        test_csv           = args.test,
        output_dir         = args.output,
        tokenizer_dir      = args.tokenizer_dir,
        do_train_tokenizer = args.train_tokenizer,
        vocab_size         = args.vocab_size,
    )