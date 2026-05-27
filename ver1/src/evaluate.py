from __future__ import annotations
import torch
import numpy as np
from safetensors.torch import load_file
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from ver1.src.structure_eng import DepressionDetector

WEIGHTS_PATH = r"C:\Job\Depression Detect\checkpoints\inference\hudd_eng.safetensors"
TEST_NPZ     = r"C:\Job\Depression Detect\eng dataset\dataset_tokenized\test.npz"
THRESHOLD    = 0.50
BATCH_SIZE   = 128
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weights_path):
    model = DepressionDetector(use_checkpoint=False).to(DEVICE)
    path  = str(weights_path)

    if path.endswith(".safetensors"):
        model.load_state_dict(load_file(path, device=str(DEVICE)))
        print(f"[model] loaded safetensors ← {path}")

    elif path.endswith(".pt"):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            # Full state checkpoint (resume)
            model.load_state_dict(ckpt["model"])
            print(f"[model] loaded full-state .pt  epoch={ckpt.get('epoch','?')}  val_f1={ckpt.get('val_f1','?')} ← {path}")
        else:
            # Weights-only .pt
            model.load_state_dict(ckpt)
            print(f"[model] loaded weights-only .pt ← {path}")

    else:
        raise ValueError(f"Không nhận diện được định dạng file: {path}")

    model.eval()
    return model


def load_test_data(npz_path):
    print(f"[data]  load ← {npz_path}")
    data = np.load(npz_path)
    input_ids      = torch.from_numpy(data["input_ids"].astype(np.int32)).long()
    attention_mask = torch.from_numpy(data["attention_mask"].astype(np.int8)).long()
    segment_ids    = torch.from_numpy(data["segment_ids"].astype(np.int8)).long()
    length_feat    = torch.from_numpy(data["length_feat"].astype(np.float32))
    labels         = torch.from_numpy(data["labels"].astype(np.int8)).float()

    n = len(labels)
    print(f"  samples : {n:,}")
    print(f"  label 0 : {(labels==0).sum().item():,}  ({(labels==0).float().mean()*100:.1f}%)")
    print(f"  label 1 : {(labels==1).sum().item():,}  ({(labels==1).float().mean()*100:.1f}%)\n")
    return input_ids, attention_mask, segment_ids, length_feat, labels


def evaluate(model, input_ids, attention_mask, segment_ids, length_feat, labels):
    n = len(labels)
    all_probs = []

    for start in tqdm(range(0, n, BATCH_SIZE), desc="evaluating"):
        end = min(start + BATCH_SIZE, n)
        with torch.no_grad():
            with torch.amp.autocast(device_type=DEVICE.type):
                probs = model(
                    input_ids[start:end].to(DEVICE),
                    attention_mask[start:end].to(DEVICE),
                    segment_ids[start:end].to(DEVICE),
                    length_feat[start:end].to(DEVICE),
                ).cpu().float().numpy()
        all_probs.append(probs)

    all_probs  = np.concatenate(all_probs)
    all_labels = labels.numpy().astype(int)
    all_preds  = (all_probs >= THRESHOLD).astype(int)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()

    acc         = (tp + tn) / n
    precision   = tp / (tp + fp + 1e-8)
    recall      = tp / (tp + fn + 1e-8)
    f1          = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    auc_roc     = roc_auc_score(all_labels, all_probs)
    auc_pr      = average_precision_score(all_labels, all_probs)

    sep = "─" * 40
    print(f"\n{sep}")
    print(f"  kết quả  ({n:,} samples, threshold={THRESHOLD})")
    print(sep)
    print(f"  accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  precision   : {precision:.4f}")
    print(f"  recall      : {recall:.4f}")
    print(f"  specificity : {specificity:.4f}")
    print(f"  f1          : {f1:.4f}")
    print(f"  auc-roc     : {auc_roc:.4f}")
    print(f"  auc-pr      : {auc_pr:.4f}")
    print(f"\n  confusion matrix:")
    print(f"               pred 0    pred 1")
    print(f"  actual 0   {tn:>8,}  {fp:>8,}")
    print(f"  actual 1   {fn:>8,}  {tp:>8,}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['neutral', 'depression'])}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",       type=str,   default=TEST_NPZ,     help="đường dẫn file test.npz")
    parser.add_argument("--weights",    type=str,   default=WEIGHTS_PATH, help="đường dẫn file .pt hoặc .safetensors")
    parser.add_argument("--threshold",  type=float, default=THRESHOLD)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    args = parser.parse_args()

    THRESHOLD  = args.threshold
    BATCH_SIZE = args.batch_size

    model = load_model(args.weights)
    input_ids, attention_mask, segment_ids, length_feat, labels = load_test_data(args.test)

    print(torch.cuda.is_available())
    print(DEVICE)

    evaluate(model, input_ids, attention_mask, segment_ids, length_feat, labels)