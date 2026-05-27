"""Stage 1 model: MentalRoBERTa backbone + LoRA + Custom Classification Head.

The custom head is the "Best Option" design from tech_plan section 5.3:

  1. Layer-wise weighted average of the last N hidden layers
       H_agg = gamma * sum_i softmax(w)_i * H_(-i)
  2. Dual pooling: [CLS] vector + masked mean vector  (concat -> [B, 2H])
  3. Fusion projection: LayerNorm -> Linear(2H, H) -> GELU -> Dropout
  4. Residual FFN block (Transformer-style, pre-LayerNorm, expand-contract,
     zero-init output for identity-start)
  5. Multi-Sample Dropout output head (K shared-weight dropout passes
     averaged into final logits during training; single pass at eval)

Loss: BCEWithLogitsLoss with pos_weight (computed from fold) + soft labels
(label smoothing 0.05).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ver2.stage1.src.config import (
        FFN_EXPAND, HEAD_DROPOUT, LABEL_SMOOTHING, LAYER_AGG_N,
        LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGETS, MSD_K, MSD_P,
    )
except ImportError:  # flat layout (e.g., Kaggle /kaggle/working/stage1/)
    from config import (
        FFN_EXPAND, HEAD_DROPOUT, LABEL_SMOOTHING, LAYER_AGG_N,
        LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGETS, MSD_K, MSD_P,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Custom classification head
# ─────────────────────────────────────────────────────────────────────────────


class CustomHead(nn.Module):
    """Layer-avg + Dual pool + Residual FFN + Multi-Sample Dropout."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_layers_to_avg: int = LAYER_AGG_N,
        ffn_expand: int = FFN_EXPAND,
        dropout: float = HEAD_DROPOUT,
        msd_k: int = MSD_K,
        msd_p: float = MSD_P,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_layers_to_avg = num_layers_to_avg
        self.hidden_size = hidden_size
        self.msd_k = msd_k
        self.msd_p = msd_p

        # ─ (1) Layer aggregation parameters (4 weights + gamma) ─
        self.layer_weights = nn.Parameter(torch.ones(num_layers_to_avg))
        self.layer_gamma   = nn.Parameter(torch.ones(1))

        # ─ (3) Fusion projection (concat[CLS, mean] -> hidden) ─
        self.fusion_norm = nn.LayerNorm(2 * hidden_size)
        self.fusion_proj = nn.Linear(2 * hidden_size, hidden_size)

        # ─ (4) Residual FFN block ─
        ffn_inner = hidden_size * ffn_expand
        self.block_norm   = nn.LayerNorm(hidden_size)
        self.block_up     = nn.Linear(hidden_size, ffn_inner)
        self.block_down   = nn.Linear(ffn_inner, hidden_size)
        self.dropout      = nn.Dropout(dropout)

        # Zero-init the residual output so the block starts as identity.
        nn.init.zeros_(self.block_down.weight)
        nn.init.zeros_(self.block_down.bias)

        # ─ (5) MSD output ─
        self.out_norm   = nn.LayerNorm(hidden_size)
        self.out_linear = nn.Linear(hidden_size, num_classes)

    def _layer_aggregate(self, hidden_states: tuple) -> torch.Tensor:
        """hidden_states is a tuple of (n_layers + 1) tensors [B, L, H].
        We take the last N (excluding embeddings layer 0) and weighted-sum.
        """
        layers = hidden_states[-self.num_layers_to_avg:]
        # Stack: [N, B, L, H]
        stacked = torch.stack(layers, dim=0)
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        h_agg = (stacked * weights).sum(dim=0) * self.layer_gamma
        return h_agg

    @staticmethod
    def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: [B, L, H]; mask: [B, L]
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def forward(self, hidden_states: tuple,
                attention_mask: torch.Tensor) -> torch.Tensor:
        # (1) Layer-weighted average
        h = self._layer_aggregate(hidden_states)   # [B, L, H]

        # (2) Dual pool
        cls_vec  = h[:, 0, :]                       # [B, H]
        mean_vec = self._masked_mean(h, attention_mask)
        pooled   = torch.cat([cls_vec, mean_vec], dim=-1)   # [B, 2H]

        # (3) Fusion: norm -> down-project to H
        x = self.fusion_norm(pooled)
        x = self.fusion_proj(x)                     # [B, H]

        # (4) Residual FFN block (pre-LN)
        residual = x
        h_ff = self.block_norm(x)
        h_ff = F.gelu(self.block_up(h_ff))
        h_ff = self.dropout(h_ff)
        h_ff = self.block_down(h_ff)
        h_ff = self.dropout(h_ff)
        x = residual + h_ff                          # zero-init makes start ~= residual

        # (5) Multi-Sample Dropout output
        x = self.out_norm(x)
        if self.training and self.msd_k > 1:
            logits_sum = 0.0
            for _ in range(self.msd_k):
                logits_sum = logits_sum + self.out_linear(F.dropout(x, p=self.msd_p, training=True))
            logits = logits_sum / self.msd_k
        else:
            logits = self.out_linear(F.dropout(x, p=self.msd_p, training=False))
        return logits      # [B, num_classes]


# ─────────────────────────────────────────────────────────────────────────────
# Full model wrapper
# ─────────────────────────────────────────────────────────────────────────────


class MentalRoBERTaWithCustomHead(nn.Module):
    """Backbone (LoRA-adapted) + CustomHead + BCE loss with label smoothing.

    Designed for HuggingFace Trainer: forward returns either logits or
    a dict {"loss", "logits"} when labels are provided.
    """

    def __init__(
        self,
        backbone,
        use_lora: bool = True,
        num_classes: int = 1,
        pos_weight: Optional[float] = None,
        label_smoothing: float = LABEL_SMOOTHING,
    ):
        super().__init__()
        # Apply LoRA to the backbone (PEFT)
        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=LORA_R, lora_alpha=LORA_ALPHA,
                target_modules=LORA_TARGETS,
                lora_dropout=LORA_DROPOUT, bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.backbone = get_peft_model(backbone, lora_cfg)
            # Print trainable summary so the user can verify LoRA wrapped
            try:
                self.backbone.print_trainable_parameters()
            except Exception:
                pass
        else:
            self.backbone = backbone

        hidden_size = backbone.config.hidden_size
        self.head = CustomHead(hidden_size=hidden_size, num_classes=num_classes)

        self.num_classes      = num_classes
        self.label_smoothing  = label_smoothing
        self.pos_weight_value = pos_weight
        # Register pos_weight as buffer so it moves with .to(device)
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32)
            )
        else:
            self.pos_weight = None

    # ── HuggingFace conventional gradient_checkpointing toggles ──────────
    def gradient_checkpointing_enable(self, **kwargs):
        # With LoRA the backbone params are frozen, so input embeddings produce
        # no grad and checkpointing breaks the graph. enable_input_require_grads
        # forces embeddings to retain grad so LoRA adapters get gradients.
        kwargs.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         Optional[torch.Tensor] = None,
    ) -> dict:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = self.head(out.hidden_states, attention_mask)
        # Squeeze to [B] for binary
        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        result = {"logits": logits}
        if labels is not None:
            # Soft labels (label smoothing 0.05): y* = y*(1-ε) + 0.5*ε
            ls = self.label_smoothing
            soft = labels.float() * (1 - ls) + 0.5 * ls

            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight if self.pos_weight is not None else None
            )
            loss = loss_fn(logits, soft)
            result["loss"] = loss
        return result

    # ── Parameter groups for discriminative LR (LoRA vs head) ────────────
    def get_param_groups(self, lr_lora: float, lr_head: float, weight_decay: float):
        lora_params, head_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("head."):
                head_params.append(p)
            else:
                lora_params.append(p)
        return [
            {"params": lora_params, "lr": lr_lora, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ]
