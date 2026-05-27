from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from safetensors.torch import load_file
from ver1.src.structure_vie import DepressionDetector

MODEL_PATH     = "C:\\Job\\Depression Detect\\checkpoints\\inference\\best.safetensors"
TOKENIZER_PATH = "C:\\Job\\Depression Detect\\vie dataset\\tokenizer"
INPUT_FILE     = "C:\\Job\\Depression Detect\\vie dataset\\dataset_raw\\test.csv"  # .txt hoac .csv
THRESHOLD      = 0.5
MC_SAMPLES     = 30
USE_CKPT       = False
MAX_SEQ_LEN    = 512

# CSV config
CSV_TEXT_COL   = "text"     
CSV_LABEL_COL  = "label"    
CSV_ENCODING   = "utf-8-sig"


def load_model(model_path: str, device: torch.device) -> nn.Module:
    model = DepressionDetector(use_checkpoint=USE_CKPT).to(device)
    path  = Path(model_path)
    if path.suffix == ".safetensors":
        state = load_file(model_path, device=str(device))
        model.load_state_dict(state)
    else:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        key  = "model" if "model" in ckpt else None
        model.load_state_dict(ckpt[key] if key else ckpt)
    print(f"[model] Loaded <- {path.name}")
    return model


def compute_length_feat(word_count: int, max_words: int = 500) -> float:
    return min(word_count / max_words, 1.0)


def tokenize(text: str, tokenizer) -> dict:
    encoded = tokenizer.encode(text)
    ids     = encoded.ids[:MAX_SEQ_LEN - 3]  

    cls_id = tokenizer.token_to_id("[CLS]") or 1
    sep_id = tokenizer.token_to_id("[SEP]") or 2
    pad_id = tokenizer.token_to_id("[PAD]") or 0

    seq     = [cls_id, sep_id] + ids + [sep_id]
    seq_len = len(seq)
    pad_len = MAX_SEQ_LEN - seq_len

    input_ids      = seq + [pad_id] * pad_len
    attention_mask = [1] * seq_len + [0] * pad_len
    segment_ids    = [0, 0] + [1] * (seq_len - 2) + [0] * pad_len
    length_feat    = compute_length_feat(len(text.split()))

    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0),
        "segment_ids":    torch.tensor(segment_ids,    dtype=torch.long).unsqueeze(0),
        "length_feat":    torch.tensor([length_feat],  dtype=torch.float32),
    }


def predict_mc(model: nn.Module, inputs: dict, device: torch.device) -> tuple[float, float]:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            with autocast(device_type=device.type):
                logit = model(
                    inputs["input_ids"].to(device),
                    inputs["attention_mask"].to(device),
                    inputs["segment_ids"].to(device),
                    inputs["length_feat"].to(device),
                )
            preds.append(logit.float().clamp(1e-6, 1 - 1e-6).item())

    model.eval()
    return float(np.mean(preds)), float(np.std(preds))


def truncate_text(text: str, head: int = 20, tail: int = 20) -> str:
    text = text.strip()
    if len(text) <= head + tail + 5:
        return text
    return f"{text[:head]}...{text[-tail:]}"


def label_text(label: int) -> str:
    return "Depression" if label == 1 else "Neutral"


def correct_text(pred: int, true: int) -> str:
    return "Correct" if pred == true else "Incorrect"


def print_result(idx: int, text: str, prob: float, std: float, true_label: int | None = None):
    pct        = prob * 100
    confidence = max(0.0, 1 - 2 * std) * 100
    pred_label = 1 if prob >= THRESHOLD else 0
    preview    = truncate_text(text)

    print(f"\n-- Post {idx:>3} " + "-"*40)
    print(f"  Text        : {preview}")
    print(f"  Predict     : {label_text(pred_label)}")
    print(f"  Probability : {pct:.1f}%")
    print(f"  Confidence  : {confidence:.1f}%")

    if true_label is not None:
        print(f"  Ground Truth: {label_text(true_label)}  [{correct_text(pred_label, true_label)}]")


def compute_metrics(preds: list[int], labels: list[int]) -> dict:
    p = np.array(preds)
    l = np.array(labels)
    tp = ((p == 1) & (l == 1)).sum()
    fp = ((p == 1) & (l == 0)).sum()
    fn = ((p == 0) & (l == 1)).sum()
    tn = ((p == 0) & (l == 0)).sum()
    acc       = (tp + tn) / len(l)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "acc": float(acc), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def load_txt(path: str) -> tuple[list[str], list[None]]:
    with open(path, encoding="utf-8") as f:
        posts = [line.strip() for line in f if line.strip()]
    return posts, [None] * len(posts)


def load_csv(path: str) -> tuple[list[str], list[int | None]]:
    posts, labels = [], []
    with open(path, encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)

        if CSV_TEXT_COL not in reader.fieldnames:
            raise ValueError(
                f"Khong tim thay cot '{CSV_TEXT_COL}' trong CSV.\n"
                f"Cac cot hien co: {reader.fieldnames}"
            )

        has_label = CSV_LABEL_COL in reader.fieldnames
        if has_label:
            print(f"[csv] Tim thay cot label '{CSV_LABEL_COL}' -> se tinh metrics")
        else:
            print(f"[csv] Khong co cot label -> chi du doan")

        for row in reader:
            text = row[CSV_TEXT_COL].strip()
            if not text:
                continue
            posts.append(text)
            if has_label:
                try:
                    labels.append(int(row[CSV_LABEL_COL]))
                except (ValueError, KeyError):
                    labels.append(None)
            else:
                labels.append(None)

    return posts, labels


def process_file(file_path: str, tokenizer, model: nn.Module, device: torch.device):
    path = Path(file_path)
    if not path.exists():
        print(f"[error] Khong tim thay file: {file_path}")
        return

    ext = path.suffix.lower()
    if ext == ".txt":
        posts, labels = load_txt(file_path)
    elif ext == ".csv":
        posts, labels = load_csv(file_path)
    else:
        print(f"[error] Chi ho tro .txt va .csv, nhan duoc: {ext}")
        return

    total       = len(posts)
    has_label   = any(l is not None for l in labels)
    all_preds   = []
    all_labels  = []
    n_depression = 0

    print(f"\n[file] {path.name}  --  {total} posts")
    print("=" * 55)

    for i, (text, true_label) in enumerate(zip(posts, labels), 1):
        inputs     = tokenize(text, tokenizer)
        prob, std  = predict_mc(model, inputs, device)
        pred_label = 1 if prob >= THRESHOLD else 0

        print_result(i, text, prob, std, true_label)

        if pred_label == 1:
            n_depression += 1
        all_preds.append(pred_label)
        if true_label is not None:
            all_labels.append(true_label)

    # Tong ket
    print(f"\n{'='*55}")
    print(f"  Tong posts   : {total}")
    print(f"  Tram cam     : {n_depression}  ({n_depression/total*100:.1f}%)")
    print(f"  Binh thuong  : {total-n_depression}  ({(total-n_depression)/total*100:.1f}%)")

    # Metrics neu co label
    if has_label and len(all_labels) == total:
        m = compute_metrics(all_preds, all_labels)
        print(f"\n  --- Metrics ---")
        print(f"  Accuracy     : {m['acc']*100:.2f}%")
        print(f"  F1           : {m['f1']:.4f}")
        print(f"  Precision    : {m['precision']:.4f}")
        print(f"  Recall       : {m['recall']:.4f}")
        print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

    print(f"{'='*55}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Load tokenizer
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(Path(TOKENIZER_PATH) / "tokenizer.json"))
        print(f"[tokenizer] Loaded <- {TOKENIZER_PATH}")
    except Exception as e:
        print(f"[error] Khong load duoc tokenizer: {e}")
        return

    # Load model
    model = load_model(MODEL_PATH, device)
    model.eval()

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = INPUT_FILE

    process_file(file_path, tokenizer, model, device)


if __name__ == "__main__":
    main()