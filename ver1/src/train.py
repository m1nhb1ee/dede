from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from safetensors.torch import save_file
from tqdm import tqdm
from ver1.src.structure_vie import DepressionDetector

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

CFG = {
    # Paths
    "train_npz":       "C:\\Job\\Depression Detect\\vie dataset\\dataset_tokenized\\train.npz",
    "val_npz":         "C:\\Job\\Depression Detect\\vie dataset\\dataset_tokenized\\val.npz",
    "checkpoint_dir":  "C:\\Job\\Depression Detect\\checkpoints",

    # Training
    "max_epochs":      50,
    "batch_size":      32,
    "grad_accum":      4,
    "lr":              1e-5, # 5e-5 cho phase 1 và 1e-5 cho phase 2
    "warmup_ratio":    0.0, # 0.05 cho phase 1 và 0.0 cho phase 2
    "weight_decay":    0.05,
    "max_grad_norm":   1.0,

    # Checkpoint
    "save_every_epoch": True, 

    # Loss 
    "focal_alpha":     0.85, # 0.5 cho phase 1 và 0.75 cho phase 2 
    "focal_gamma":     2.0,
    "label_smooth":    0.05,

    # Model
    "vocab_size":      16000,
    "max_seq_len":     512,
    "d_model":         512,
    "n_heads":         8,
    "n_layers":        4,
    "d_ff":            2048,
    "dropout":         0.3,
    "use_checkpoint":  True,   

    # Inference
    "threshold":       0.5,
}

class DepressionDataset(Dataset):
    def __init__(self, npz_path: str):
        print(f"[dataset] Load {npz_path} ...")
        data = np.load(npz_path)

        self.input_ids      = torch.from_numpy(data["input_ids"].astype(np.int32)).long()
        self.attention_mask = torch.from_numpy(data["attention_mask"].astype(np.int8)).long()
        self.segment_ids    = torch.from_numpy(data["segment_ids"].astype(np.int8)).long()
        self.length_feat    = torch.from_numpy(data["length_feat"].astype(np.float32))
        self.labels         = torch.from_numpy(data["labels"].astype(np.int8)).float()

        print(f"  Samples : {len(self.labels):,}")
        print(f"  Label 0 : {(self.labels==0).sum().item():,}  ({(self.labels==0).float().mean()*100:.1f}%)")
        print(f"  Label 1 : {(self.labels==1).sum().item():,}  ({(self.labels==1).float().mean()*100:.1f}%)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "segment_ids":    self.segment_ids[idx],
            "length_feat":    self.length_feat[idx],
            "label":          self.labels[idx],
        }


class FocalBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = CFG["focal_alpha"]
        self.gamma = CFG["focal_gamma"]
        self.eps   = CFG["label_smooth"]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.clamp(1e-6, 1 - 1e-6)
        targets_smooth = targets * (1 - self.eps) + self.eps / 2
        bce            = F.binary_cross_entropy(logits, targets_smooth, reduction="none")
        p_t            = logits * targets + (1 - logits) * (1 - targets)
        focal_weight   = (1 - p_t) ** self.gamma
        alpha_t        = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_weight * bce).mean()


def get_scheduler(optimizer, total_steps: int):
    warmup_steps = int(total_steps * CFG["warmup_ratio"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    binary    = (preds >= CFG["threshold"]).astype(int)
    tp = ((binary == 1) & (labels == 1)).sum()
    fp = ((binary == 1) & (labels == 0)).sum()
    fn = ((binary == 0) & (labels == 1)).sum()
    tn = ((binary == 0) & (labels == 0)).sum()

    acc       = (tp + tn) / len(labels)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "acc":       float(acc),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
    }



def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch) -> dict:
    model.train()

    total_loss = 0.0
    all_preds  = []
    all_labels = []
    n_batches  = len(loader)

    optimizer.zero_grad()

    bar = tqdm(enumerate(loader), total=n_batches,
               desc=f"Epoch {epoch}", ncols=100, unit="batch")

    for step, batch in bar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        segment_ids    = batch["segment_ids"].to(device)
        length_feat    = batch["length_feat"].to(device)
        labels         = batch["label"].to(device)

        with autocast(device_type="cuda"):
            logits = model(input_ids, attention_mask, segment_ids, length_feat)

        logits = logits.float().clamp(1e-6, 1 - 1e-6)
        loss = criterion(logits.float(), labels.float()) / CFG["grad_accum"]
        scaler.scale(loss).backward()

        if (step + 1) % CFG["grad_accum"] == 0 or (step + 1) == n_batches:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * CFG["grad_accum"]
        all_preds.append(logits.detach().cpu().float().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        avg_loss = total_loss / (step + 1)
        lr_now = max(scheduler.get_last_lr())
        bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}")

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics    = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / n_batches
    return metrics

@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        segment_ids    = batch["segment_ids"].to(device)
        length_feat    = batch["length_feat"].to(device)
        labels         = batch["label"].to(device)

        with autocast(device_type="cuda"):
            logits = model(input_ids, attention_mask, segment_ids, length_feat)

        logits = logits.float().clamp(1e-6, 1 - 1e-6)
        loss = criterion(logits.float(), labels.float())
        total_loss += loss.item()
        all_preds.append(logits.cpu().float().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics    = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics

def _full_state(model, optimizer, scheduler, scaler, epoch, val_f1) -> dict:
    """Đóng gói toàn bộ state để resume."""
    return {
        "epoch":     epoch,
        "val_f1":    val_f1,
        "cfg":       CFG,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
    }


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_f1, pt_path: Path):
    """Lưu full state → .pt (dùng để resume)."""
    torch.save(_full_state(model, optimizer, scheduler, scaler, epoch, val_f1), pt_path)
    print(f"  [ckpt] Saved {pt_path}")


def save_inference(model, safetensors_path: Path):
    """Lưu weights only → .safetensors (dùng để inference)."""
    safetensors_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), safetensors_path)
    print(f"  [ckpt] Saved {safetensors_path}")


def load_checkpoint(pt_path: Path, model, optimizer, scheduler, scaler, device):
    """Load full state từ .pt để resume training."""
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"  [resume] epoch={ckpt['epoch']}  val_f1={ckpt['val_f1']:.4f}  ← {pt_path}")
    return ckpt["epoch"], ckpt["val_f1"]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"{'='*55}\n")

    train_set = DepressionDataset(CFG["train_npz"])
    val_set   = DepressionDataset(CFG["val_npz"])

    train_loader = DataLoader(
        train_set, batch_size=CFG["batch_size"],
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=CFG["batch_size"] * 2,
        shuffle=False, num_workers=0, pin_memory=False,
    )

    model     = DepressionDetector(use_checkpoint=CFG["use_checkpoint"]).to(device)
    criterion = FocalBCELoss()
    print(f"Parameters: {model.count_parameters():,}")

# # -----------------PHASE 1------------------
    # decay    = [p for n, p in model.named_parameters()
    #             if p.requires_grad and not any(nd in n for nd in ["bias", "norm"])]
    # no_decay = [p for n, p in model.named_parameters()
    #             if p.requires_grad and     any(nd in n for nd in ["bias", "norm"])]
    # optimizer = torch.optim.AdamW([
    #     {"params": decay,    "weight_decay": CFG["weight_decay"]},
    #     {"params": no_decay, "weight_decay": 0.0},
    # ], lr=CFG["lr"])

# ------------------PHASE 2------------------
    def make_param_groups(module, lr, wd):
        decay    = [p for n, p in module.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in ["bias", "norm"])]
        no_decay = [p for n, p in module.named_parameters()
                    if p.requires_grad and     any(nd in n for nd in ["bias", "norm"])]
        return [
            {"params": decay,    "lr": lr, "weight_decay": wd},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]

    param_groups = (
        make_param_groups(model.embeddings, lr=1e-5,  wd=0.0)  +
        make_param_groups(model.encoder,    lr=5e-5,  wd=0.0)  +
        make_param_groups(model.dual_pool,  lr=1e-4,  wd=0.01) +
        make_param_groups(model.gate,       lr=1e-4,  wd=0.01) +
        make_param_groups(model.classifier, lr=1e-4,  wd=0.01)
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
# ---------------------------------------
    total_steps = (len(train_loader) // CFG["grad_accum"]) * CFG["max_epochs"]
    scheduler   = get_scheduler(optimizer, total_steps)
    scaler      = GradScaler()

    checkpoint_dir = Path(CFG["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    inference_dir  = checkpoint_dir / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)

    last_pt   = checkpoint_dir / "last.pt"
    best_pt   = checkpoint_dir / "best.pt"
    best_safe = inference_dir  / "best.safetensors"
    last_safe = inference_dir  / "last.safetensors"

    start_epoch = 1
    best_val_f1 = 0.0
    resume_from = Path(args.checkpoint) if args.checkpoint else last_pt

    # ---------------PHASE 1------------------

    # if args.resume:
    #     if resume_from.exists():
    #         start_epoch, best_val_f1 = load_checkpoint(
    #             resume_from, model, optimizer, scheduler, scaler, device
    #         )
    #         start_epoch += 1
    #     else:
    #         print(f"  [warn] Checkpoint không tìm thấy: {resume_from}  → train từ đầu")

    # ---------------PHASE 2------------------
    if args.resume:
        if resume_from.exists():
            ckpt = torch.load(resume_from, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            start_epoch = ckpt["epoch"] + 1       
            best_val_f1 = ckpt["val_f1"]          
            print(f"  [stage2] Loaded weights  val_f1={ckpt['val_f1']:.4f}  epoch={ckpt['epoch']}")
        else:
            print(f"  [warn] Checkpoint không tìm thấy: {resume_from}  → train từ đầu")
    # ---------------------------------------

    eff = CFG["batch_size"] * CFG["grad_accum"]
    print(f"\nTrain  : {len(train_set):,} | Val: {len(val_set):,}")
    print(f"Epochs : {CFG['max_epochs']} | Batch: {CFG['batch_size']} × {CFG['grad_accum']} = {eff}")
    print(f"Steps  : {total_steps:,}")
    print(f"{'='*55}\n")

    for epoch in range(start_epoch, CFG["max_epochs"] + 1):
        print(f"\n── Epoch {epoch}/{CFG['max_epochs']} {'─'*35}")

        train_m = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )
        print(f"\n  [Train] loss={train_m['loss']:.4f}  acc={train_m['acc']:.4f}"
              f"  f1={train_m['f1']:.4f}  recall={train_m['recall']:.4f}")

        val_m = validate(model, val_loader, criterion, device)
        print(f"  [Val]   loss={val_m['loss']:.4f}  acc={val_m['acc']:.4f}"
              f"  f1={val_m['f1']:.4f}  recall={val_m['recall']:.4f}"
              f"  precision={val_m['precision']:.4f}")

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_m["f1"], last_pt)
        save_inference(model, last_safe)

        if CFG["save_every_epoch"]:
            epoch_pt = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_m["f1"], epoch_pt)

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_m["f1"], best_pt)
            save_inference(model, best_safe)
            print(f"  ★ New best val F1: {best_val_f1:.4f}")

    print(f"\n{'='*55}")
    print(f"Done!  Best val F1 = {best_val_f1:.4f}")
    print(f"  Full state  → {best_pt}")
    print(f"  Inference   → {best_safe}")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",     action="store_true",
                        help="Resume từ checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Đường dẫn checkpoint cụ thể (mặc định: checkpoints/last.pt)")
    args = parser.parse_args()
    main(args)