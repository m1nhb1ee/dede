from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from safetensors.torch import save_file
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision  = "tf32"

CFG = {
    "train_npz":        "/kaggle/input/datasets/minhngtrb1e/depression-detect-via-reddit-post-eng/dataset_tokenized/train.npz",
    "val_npz":          "/kaggle/input/datasets/minhngtrb1e/depression-detect-via-reddit-post-eng/dataset_tokenized/val.npz",
    "checkpoint_dir":   "/kaggle/working/checkpoints",
    "max_epochs":       15,
    "batch_size":       128,     # 64 per gpu × 2
    "grad_accum":       1,
    "lr":               4e-4,
    "warmup_ratio":     0.05,
    "weight_decay":     0.01,
    "max_grad_norm":    1.0,
    "save_every_epoch": True,
    "focal_alpha":      0.75,
    "focal_gamma":      2.0,
    "label_smooth":     0.025,
    "vocab_size":       32000,
    "max_seq_len":      512,
    "d_model":          512,
    "n_heads":          8,
    "n_layers":         6,
    "d_ff":             2048,
    "dropout":          0.1,
    "use_checkpoint":   False,   # tắt vì kaggle có đủ vram
    "num_workers":      4,
    "pin_memory":       True,
    "threshold":        0.5,
}

VOCAB_SIZE  = CFG["vocab_size"]
MAX_SEQ_LEN = CFG["max_seq_len"]
D_MODEL     = CFG["d_model"]
N_HEADS     = CFG["n_heads"]
N_LAYERS    = CFG["n_layers"]
D_FF        = CFG["d_ff"]
DROPOUT     = CFG["dropout"]
N_SEGMENTS  = 2


class DepressionEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb   = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.segment_emb = nn.Embedding(N_SEGMENTS, D_MODEL)
        self.layer_norm  = nn.LayerNorm(D_MODEL)
        self.dropout     = nn.Dropout(DROPOUT)
        self.register_buffer("pos_enc", self._build_sinusoidal(MAX_SEQ_LEN, D_MODEL))

    @staticmethod
    def _build_sinusoidal(max_len, d_model):
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, input_ids, segment_ids):
        return self.dropout(self.layer_norm(
            self.token_emb(input_ids)
            + self.pos_enc[:, :input_ids.size(1)]
            + self.segment_emb(segment_ids)
        ))


class RelativePositionBias(nn.Module):
    def __init__(self, n_heads, max_dist=127):
        super().__init__()
        self.max_dist = max_dist
        self.bias     = nn.Embedding(2 * max_dist + 1, n_heads)

    def forward(self, seq_len, device):
        pos  = torch.arange(seq_len, device=device)
        diff = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(-self.max_dist, self.max_dist) + self.max_dist
        return self.bias(diff).permute(2, 0, 1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads  = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        self.scale    = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(D_MODEL, D_MODEL * 3, bias=False)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.rel_bias = RelativePositionBias(N_HEADS)
        self.dropout  = nn.Dropout(DROPOUT)

    def forward(self, x, attn_mask):
        B, L, _ = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        def reshape(t):
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores + self.rel_bias(L, x.device)
        scores = scores.masked_fill((attn_mask == 0).unsqueeze(1).unsqueeze(2), float("-inf"))

        out = (self.dropout(F.softmax(scores, dim=-1)) @ v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, L, D_MODEL))


class GatedFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj  = nn.Linear(D_MODEL, D_FF, bias=False)
        self.value_proj = nn.Linear(D_MODEL, D_FF, bias=False)
        self.out_proj   = nn.Linear(D_FF, D_MODEL, bias=False)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.out_proj(F.silu(self.gate_proj(x)) * self.value_proj(x)))


class TransformerEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1    = nn.LayerNorm(D_MODEL)
        self.norm2    = nn.LayerNorm(D_MODEL)
        self.attn     = MultiHeadSelfAttention()
        self.ffn      = GatedFFN()
        self.dropout  = nn.Dropout(DROPOUT)

    def forward(self, x, attn_mask):
        x = x + self.dropout(self.attn(self.norm1(x), attn_mask))  
        x = x + self.dropout(self.ffn(self.norm2(x)))               
        return x


class DualPooling(nn.Module):
    # vectorized hoàn toàn, không vòng for, chạy thuần gpu
    def __init__(self):
        super().__init__()
        self.q_proj       = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.title_k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.title_v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.body_k_proj  = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.body_v_proj  = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.title_norm   = nn.LayerNorm(D_MODEL)
        self.body_norm    = nn.LayerNorm(D_MODEL)
        self.scale        = D_MODEL ** -0.5

    def _pool(self, cls_vec, hidden, k_proj, v_proj, pad_mask, norm):
        Q      = self.q_proj(cls_vec).unsqueeze(1)
        scores = (Q @ k_proj(hidden).transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
        return norm((F.softmax(scores, dim=-1) @ v_proj(hidden)).squeeze(1))

    def forward(self, x, segment_ids, attn_mask):
        cls_vec = x[:, 0, :]

        title_valid       = (segment_ids == 0) & (attn_mask == 1)
        title_valid[:, 0] = False  # loại cls token
        body_valid        = (segment_ids == 1) & (attn_mask == 1)

        title_vec = self._pool(cls_vec, x, self.title_k_proj, self.title_v_proj, ~title_valid, self.title_norm)
        body_vec  = self._pool(cls_vec, x, self.body_k_proj,  self.body_v_proj,  ~body_valid,  self.body_norm)
        return title_vec, body_vec


class LearnedGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_linear = nn.Linear(D_MODEL * 2, 1)
        nn.init.constant_(self.gate_linear.bias, -1.0)

    def forward(self, title_vec, body_vec):
        gate = torch.sigmoid(self.gate_linear(torch.cat([title_vec, body_vec], dim=-1)))
        return gate * title_vec + (1 - gate) * body_vec


class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(D_MODEL + 1, 256)
        self.fc2     = nn.Linear(256, 1)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, fused, length_feat):
        x = torch.cat([fused, length_feat], dim=-1)
        return torch.sigmoid(self.fc2(self.dropout(F.gelu(self.fc1(x))))).squeeze(-1)


class DepressionDetector(nn.Module):
    def __init__(self, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.embeddings = DepressionEmbeddings()
        self.encoder    = nn.ModuleList([TransformerEncoderBlock() for _ in range(N_LAYERS)])
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.dual_pool  = DualPooling()
        self.gate       = LearnedGate()
        self.classifier = ClassifierHead()

    def forward(self, input_ids, attention_mask, segment_ids, length_feat):
        x = self.embeddings(input_ids, segment_ids)

        for block in self.encoder:
            if self.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask)

        x = self.final_norm(x)
        title_vec, body_vec = self.dual_pool(x, segment_ids, attention_mask)
        fused = self.gate(title_vec, body_vec)

        if length_feat.dim() == 1:
            length_feat = length_feat.unsqueeze(-1)
        return self.classifier(fused, length_feat)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


class DepressionDataset(Dataset):
    def __init__(self, npz_path):
        print(f"[dataset] load {npz_path} ...")
        data = np.load(npz_path)
        self.input_ids      = torch.from_numpy(data["input_ids"].astype(np.int32)).long()
        self.attention_mask = torch.from_numpy(data["attention_mask"].astype(np.int8)).long()
        self.segment_ids    = torch.from_numpy(data["segment_ids"].astype(np.int8)).long()
        self.length_feat    = torch.from_numpy(data["length_feat"].astype(np.float32))
        self.labels         = torch.from_numpy(data["labels"].astype(np.int8)).float()
        print(f"  samples : {len(self.labels):,}")
        print(f"  label 0 : {(self.labels==0).sum().item():,}  ({(self.labels==0).float().mean()*100:.1f}%)")
        print(f"  label 1 : {(self.labels==1).sum().item():,}  ({(self.labels==1).float().mean()*100:.1f}%)")

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

    def forward(self, logits, targets):
        logits = logits.clamp(1e-6, 1 - 1e-6)
        targets_smooth = targets * (1 - self.eps) + self.eps / 2
        bce          = F.binary_cross_entropy(logits, targets_smooth, reduction="none")
        p_t          = logits * targets + (1 - logits) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t      = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_weight * bce).mean()


def get_scheduler(optimizer, total_steps):
    warmup_steps = int(total_steps * CFG["warmup_ratio"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_metrics(preds, labels):
    binary = (preds >= CFG["threshold"]).astype(int)
    tp = ((binary == 1) & (labels == 1)).sum()
    fp = ((binary == 1) & (labels == 0)).sum()
    fn = ((binary == 0) & (labels == 1)).sum()
    tn = ((binary == 0) & (labels == 0)).sum()
    acc       = (tp + tn) / len(labels)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return {"acc": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    n_batches = len(loader)
    optimizer.zero_grad()

    bar = tqdm(enumerate(loader), total=n_batches, desc=f"  train {epoch}", ncols=110, unit="batch")
    for step, batch in bar:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        segment_ids    = batch["segment_ids"].to(device, non_blocking=True)
        length_feat    = batch["length_feat"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        with autocast(device_type="cuda"):
            logits = model(input_ids, attention_mask, segment_ids, length_feat)

        loss = criterion(logits.float(), labels.float()) / CFG["grad_accum"]
        scaler.scale(loss).backward()

        if ((step + 1) % CFG["grad_accum"] == 0) or ((step + 1) == n_batches):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(unwrap(model).parameters(), CFG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * CFG["grad_accum"]
        all_preds.append(logits.detach().cpu().float().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        bar.set_postfix(loss=f"{total_loss/(step+1):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics    = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / n_batches
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  val  ", ncols=110, unit="batch"):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        segment_ids    = batch["segment_ids"].to(device, non_blocking=True)
        length_feat    = batch["length_feat"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        with autocast(device_type="cuda"):
            logits = model(input_ids, attention_mask, segment_ids, length_feat)

        total_loss += criterion(logits.float(), labels.float()).item()
        all_preds.append(logits.cpu().float().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics    = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_f1, pt_path):
    torch.save({
        "epoch":     epoch,
        "val_f1":    val_f1,
        "cfg":       CFG,
        "model":     unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
    }, pt_path)
    print(f"  [ckpt] saved {pt_path}")


def save_inference(model, safetensors_path):
    Path(safetensors_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(unwrap(model).state_dict(), safetensors_path)
    print(f"  [ckpt] saved {safetensors_path}")


def load_checkpoint(pt_path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"  [resume] epoch={ckpt['epoch']}  val_f1={ckpt['val_f1']:.4f}  ← {pt_path}")
    return ckpt["epoch"], ckpt["val_f1"]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    print(f"\n{'='*55}")
    print(f"device : {device}  |  gpus: {n_gpus}")
    for i in range(n_gpus):
        print(f"  gpu {i}: {torch.cuda.get_device_name(i)}"
              f"  ({torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} gb)")
    print(f"{'='*55}\n")

    train_set = DepressionDataset(CFG["train_npz"])
    val_set   = DepressionDataset(CFG["val_npz"])

    train_loader = DataLoader(
        train_set, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"],
        prefetch_factor=2, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=CFG["batch_size"] * 2, shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"],
        persistent_workers=True,
    )

    model = DepressionDetector(use_checkpoint=CFG["use_checkpoint"]).to(device)
    if n_gpus > 1:
        print(f"  → dataparallel trên {n_gpus} gpus")
        model = nn.DataParallel(model)

    criterion = FocalBCELoss()
    print(f"parameters: {unwrap(model).count_parameters():,}")

    decay    = [p for n, p in unwrap(model).named_parameters()
                if p.requires_grad and not any(nd in n for nd in ["bias", "norm"])]
    no_decay = [p for n, p in unwrap(model).named_parameters()
                if p.requires_grad and     any(nd in n for nd in ["bias", "norm"])]
    optimizer = torch.optim.AdamW([
        {"params": decay,    "weight_decay": CFG["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=CFG["lr"])

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
    if args.resume:
        if resume_from.exists():
            start_epoch, best_val_f1 = load_checkpoint(
                resume_from, model, optimizer, scheduler, scaler, device
            )
            start_epoch += 1
        else:
            print(f"  [warn] không tìm thấy: {resume_from} → train từ đầu")

    eff = CFG["batch_size"] * CFG["grad_accum"]
    print(f"train  : {len(train_set):,} | val: {len(val_set):,}")
    print(f"epochs : {CFG['max_epochs']} | batch: {CFG['batch_size']} × {CFG['grad_accum']} = {eff}")
    print(f"steps  : {total_steps:,}")
    print(f"{'='*55}\n")

    for epoch in range(start_epoch, CFG["max_epochs"] + 1):
        print(f"\n── epoch {epoch}/{CFG['max_epochs']} {'─'*35}")

        train_m = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch)
        print(f"  [train] loss={train_m['loss']:.4f}  acc={train_m['acc']:.4f}"
              f"  f1={train_m['f1']:.4f}  recall={train_m['recall']:.4f}")

        val_m = validate(model, val_loader, criterion, device)
        print(f"  [val]   loss={val_m['loss']:.4f}  acc={val_m['acc']:.4f}"
              f"  f1={val_m['f1']:.4f}  recall={val_m['recall']:.4f}"
              f"  precision={val_m['precision']:.4f}")

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_m["f1"], last_pt)
        save_inference(model, last_safe)

        if CFG["save_every_epoch"]:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_m["f1"],
                            checkpoint_dir / f"epoch_{epoch:03d}.pt")

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_m["f1"], best_pt)
            save_inference(model, best_safe)
            print(f"  ★ new best val f1: {best_val_f1:.4f}")

    print(f"\n{'='*55}")
    print(f"done!  best val f1 = {best_val_f1:.4f}")
    print(f"  full state  → {best_pt}")
    print(f"  inference   → {best_safe}")
    print(f"{'='*55}")


if __name__ == "__main__":
    import sys
    is_notebook = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules

    if is_notebook:
        class Args:
            resume     = False  
            checkpoint = None   
        args = Args()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--resume",     action="store_true")
        parser.add_argument("--checkpoint", type=str, default=None)
        args = parser.parse_args()

    main(args)