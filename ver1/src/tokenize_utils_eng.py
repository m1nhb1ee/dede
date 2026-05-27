from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from sklearn.model_selection import train_test_split

MAX_SEQ_LEN    = 512
MAX_TITLE_LEN  = 32
MAX_BODY_LEN   = 477
LENGTH_SCALE   = 0.1
MAX_BODY_WORDS = 1000
SPLIT_RATIOS   = (0.75, 0.15, 0.10)
ENCODE_BATCH   = 512    # số bài encode mỗi lần

SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
PAD_TOKEN      = "[PAD]"
CLS_TOKEN      = "[CLS]"
SEP_TOKEN      = "[SEP]"

def train_bpe_tokenizer(
    df: pd.DataFrame,
    output_dir: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> ByteLevelBPETokenizer:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.txt"
    print(f"[tokenizer] Ghi corpus → {corpus_path}")
    texts = df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
    texts.to_csv(corpus_path, index=False, header=False)

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
    title: str,
    body: str,
    tokenizer: ByteLevelBPETokenizer,
    label: int | None = None,
) -> dict:
    vocab  = tokenizer.get_vocab()
    PAD_ID = vocab[PAD_TOKEN]
    CLS_ID = vocab[CLS_TOKEN]
    SEP_ID = vocab[SEP_TOKEN]

    title_ids = tokenizer.encode(str(title)).ids[:MAX_TITLE_LEN]
    body_ids  = adaptive_truncate(
        tokenizer.encode(str(body)).ids, MAX_BODY_LEN, label=label
    )

    seq     = [CLS_ID] + title_ids + [SEP_ID] + body_ids + [SEP_ID]
    seq_len = len(seq)
    pad_len = MAX_SEQ_LEN - seq_len

    return {
        "input_ids":      np.array(seq + [PAD_ID] * pad_len,     dtype=np.int32),
        "attention_mask": np.array([1] * seq_len + [0] * pad_len, dtype=np.int8),
        "segment_ids":    np.array(
            [0] * (1 + len(title_ids) + 1) +
            [1] * (len(body_ids) + 1)       +
            [0] * pad_len,                    dtype=np.int8),
        "length_feat":    np.float32(compute_length_feat(len(str(body).split()))),
        "label":          label,
    }

def split_dataset(df: pd.DataFrame, seed: int = 42):
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS

    train_df, temp_df = train_test_split(
        df, test_size=1 - train_ratio,
        stratify=df["label"], random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_ratio / (val_ratio + test_ratio),
        stratify=temp_df["label"], random_state=seed,
    )

    total = len(df)
    print(f"\n[split] Total : {total}")
    print(f"        Train : {len(train_df):>6}  ({len(train_df)/total*100:.1f}%)")
    print(f"        Val   : {len(val_df):>6}  ({len(val_df)/total*100:.1f}%)")
    print(f"        Test  : {len(test_df):>6}  ({len(test_df)/total*100:.1f}%)")
    for name, sdf in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        d = sdf["label"].value_counts(normalize=True)
        print(f"        {name} label → 0: {d.get(0,0):.2%} | 1: {d.get(1,0):.2%}")

    return train_df, val_df, test_df

def _encode_split(
    split_df: pd.DataFrame,
    split_name: str,
    tokenizer: ByteLevelBPETokenizer,
    output_dir: Path,
) -> None:
    n = len(split_df)
    print(f"\n[encode:{split_name}] {n} samples | batch_size={ENCODE_BATCH}")

    titles  = split_df["title"].fillna("").astype(str).tolist()
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
        end = min(start + ENCODE_BATCH, n)
        if start % (ENCODE_BATCH * 10) == 0:
            print(f"  {start}/{n}")

        title_encs = tokenizer.encode_batch(titles[start:end])
        body_encs  = tokenizer.encode_batch(bodies[start:end])

        for j, i in enumerate(range(start, end)):
            label     = int(labels[i])
            title_ids = title_encs[j].ids[:MAX_TITLE_LEN]
            body_ids  = adaptive_truncate(body_encs[j].ids, MAX_BODY_LEN, label=label)

            seq     = [CLS_ID] + title_ids + [SEP_ID] + body_ids + [SEP_ID]
            seq_len = len(seq)
            pad_len = MAX_SEQ_LEN - seq_len

            out_input_ids[i]      = seq + [PAD_ID] * pad_len
            out_attention_mask[i] = [1] * seq_len + [0] * pad_len
            out_segment_ids[i]    = (
                [0] * (1 + len(title_ids) + 1) +
                [1] * (len(body_ids) + 1)       +
                [0] * pad_len
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
    print(f"  {start + (end - start)}/{n}")
    print(f"  Saved → {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")

def run_preprocessing(
    input_csv: str,
    output_dir: str,
    tokenizer_dir: str,
    do_train_tokenizer: bool = False,
    vocab_size: int = 32000,
    seed: int = 42,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sep = "=" * 55

    print(f"\n{sep}")
    print(f"[1/4] Load CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    for col in ("title", "body", "label"):
        if col not in df.columns:
            raise ValueError(f"CSV thiếu column: '{col}'")
    df = df.drop(columns=["timeutc", "time_utc", "timeUTC"], errors="ignore")
    df["label"] = df["label"].astype(int)
    print(f"      Columns: {list(df.columns)}")
    print(f"      Shape  : {df.shape}")
    print(f"      Labels : {df['label'].value_counts().to_dict()}")

    print(f"\n[2/4] Split 75 / 15 / 10 ...")
    train_df, val_df, test_df = split_dataset(df, seed=seed)

    print(f"\n[3/4] Tokenizer ...")
    if do_train_tokenizer:
        tokenizer = train_bpe_tokenizer(df, tokenizer_dir, vocab_size=vocab_size)
    else:
        print(f"      Load từ: {tokenizer_dir}")
        tokenizer = load_tokenizer(tokenizer_dir)

    print(f"\n[4/4] Encode ...")
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        _encode_split(sdf, name, tokenizer, output_dir)

    print(f"\n{sep}")
    print(f"Output: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name:<25} {f.stat().st_size/1024/1024:>6.1f} MB")
    print(sep)

def tokenize_for_inference(
    title: str,
    body: str,
    tokenizer: ByteLevelBPETokenizer,
) -> dict:

    sample = tokenize_sample(title=title, body=body, tokenizer=tokenizer, label=None)
    sample.pop("label")
    return sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",           type=str, required=True)
    parser.add_argument("--output_dir",      type=str, default="./data_processed")
    parser.add_argument("--tokenizer_dir",   type=str, default="./tokenizer")
    parser.add_argument("--train_tokenizer", action="store_true")
    parser.add_argument("--vocab_size",      type=int, default=32000)
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    run_preprocessing(
        input_csv          = args.input,
        output_dir         = args.output_dir,
        tokenizer_dir      = args.tokenizer_dir,
        do_train_tokenizer = args.train_tokenizer,
        vocab_size         = args.vocab_size,
        seed               = args.seed,
    )