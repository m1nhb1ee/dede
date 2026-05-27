import torch
import torch.nn as nn
import torch.nn.functional as F
import math


VOCAB_SIZE  = 16_000
MAX_SEQ_LEN = 512
D_MODEL     = 512
N_HEADS     = 8
N_LAYERS    = 4
D_FF        = 2_048
DROPOUT     = 0.3
N_SEGMENTS  = 2


class DepressionEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb   = nn.Embedding(VOCAB_SIZE,  D_MODEL, padding_idx=0)
        self.segment_emb = nn.Embedding(N_SEGMENTS,  D_MODEL)
        self.layer_norm  = nn.LayerNorm(D_MODEL)
        self.dropout     = nn.Dropout(DROPOUT)
        self.register_buffer("pos_enc", self._build_sinusoidal(MAX_SEQ_LEN, D_MODEL))

    @staticmethod
    def _build_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        x = (
            self.token_emb(input_ids)
            + self.pos_enc[:, :input_ids.size(1)]
            + self.segment_emb(segment_ids)
        )
        return self.dropout(self.layer_norm(x))


class RelativePositionBias(nn.Module):
    def __init__(self, n_heads: int, max_dist: int = 127):
        super().__init__()
        self.max_dist = max_dist
        self.bias     = nn.Embedding(2 * max_dist + 1, n_heads)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pos  = torch.arange(seq_len, device=device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        diff = diff.clamp(-self.max_dist, self.max_dist) + self.max_dist
        bias = self.bias(diff)
        return bias.permute(2, 0, 1)


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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape(t):
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores + self.rel_bias(L, x.device)

        pad_mask = (attn_mask == 0).unsqueeze(1).unsqueeze(2)
        scores   = scores.masked_fill(pad_mask, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out  = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, L, D_MODEL)
        return self.out_proj(out)


class GatedFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj  = nn.Linear(D_MODEL, D_FF, bias=False)
        self.value_proj = nn.Linear(D_MODEL, D_FF, bias=False)
        self.out_proj   = nn.Linear(D_FF, D_MODEL, bias=False)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate  = F.silu(self.gate_proj(x))
        value = self.value_proj(x)
        return self.dropout(self.out_proj(gate * value))


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
    """
    Không còn title tokens.
    Format: [CLS][SEP][Body tokens][SEP]

    - cls_vec  : x[:, 0, :]  — đại diện global (segment 0)
    - body_vec : attention pooling trên các token segment==1
    """
    def __init__(self):
        super().__init__()
        self.q_proj      = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.body_k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.body_v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.cls_norm    = nn.LayerNorm(D_MODEL)
        self.body_norm   = nn.LayerNorm(D_MODEL)
        self.scale       = D_MODEL ** -0.5

    def _pool(
        self,
        cls_vec:  torch.Tensor,
        hidden:   torch.Tensor,
        k_proj:   nn.Linear,
        v_proj:   nn.Linear,
        pad_mask: torch.Tensor,
        norm:     nn.LayerNorm,
    ) -> torch.Tensor:
        Q = self.q_proj(cls_vec).unsqueeze(1)
        K = k_proj(hidden)
        V = v_proj(hidden)

        scores = (Q @ K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
        attn   = F.softmax(scores, dim=-1)
        pooled = (attn @ V).squeeze(1)
        return norm(pooled)

    def forward(
        self,
        x:           torch.Tensor,
        segment_ids: torch.Tensor,
        attn_mask:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # cls_vec: CLS token tại vị trí 0
        cls_vec = self.cls_norm(x[:, 0, :])

        # body_vec: attention pool trên segment==1 tokens
        body_pad_mask = ~((segment_ids == 1) & (attn_mask == 1))
        body_vec = self._pool(
            x[:, 0, :], x,
            self.body_k_proj, self.body_v_proj,
            body_pad_mask, self.body_norm,
        )

        return cls_vec, body_vec


class LearnedGate(nn.Module):
    """Gate giữa cls_vec (global) và body_vec (attended body)."""
    def __init__(self):
        super().__init__()
        self.gate_linear = nn.Linear(D_MODEL * 2, 1)
        nn.init.constant_(self.gate_linear.bias, -1.0)

    def forward(self, cls_vec: torch.Tensor, body_vec: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([cls_vec, body_vec], dim=-1)
        gate     = torch.sigmoid(self.gate_linear(combined))
        return gate * cls_vec + (1 - gate) * body_vec


class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(D_MODEL + 1, 256)
        self.fc2     = nn.Linear(256, 1)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, fused: torch.Tensor, length_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fused, length_feat], dim=-1)
        x = self.dropout(F.gelu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x)).squeeze(-1)
        return x


class DepressionDetector(nn.Module):
    """
    Full pipeline — format [CLS][SEP][Body][SEP]:
      Layer 1   : DepressionEmbeddings
      Layer 2–5 : TransformerEncoderBlock x N_LAYERS (gradient checkpointing)
      Layer 6   : Final LayerNorm
      Layer 7   : DualPooling → (cls_vec, body_vec)
      Layer 8   : LearnedGate
      Layer 9   : ClassifierHead
    """
    def __init__(self, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.embeddings = DepressionEmbeddings()
        self.encoder    = nn.ModuleList([
            TransformerEncoderBlock() for _ in range(N_LAYERS)
        ])
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.dual_pool  = DualPooling()
        self.gate       = LearnedGate()
        self.classifier = ClassifierHead()

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids:    torch.Tensor,
        length_feat:    torch.Tensor,
    ) -> torch.Tensor:

        x = self.embeddings(input_ids, segment_ids)

        for block in self.encoder:
            if self.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask)

        x = self.final_norm(x)
        cls_vec, body_vec = self.dual_pool(x, segment_ids, attention_mask)

        fused = self.gate(cls_vec, body_vec)
        if length_feat.dim() == 1:
            length_feat = length_feat.unsqueeze(-1)
        return self.classifier(fused, length_feat)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DepressionDetector(use_checkpoint=True).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    B = 4
    input_ids      = torch.randint(0, VOCAB_SIZE, (B, MAX_SEQ_LEN)).to(device)
    attention_mask = torch.ones(B, MAX_SEQ_LEN, dtype=torch.long).to(device)
    segment_ids    = torch.ones(B, MAX_SEQ_LEN, dtype=torch.long).to(device)
    segment_ids[:, :2] = 0
    length_feat    = torch.rand(B).to(device) * 0.1

    with torch.no_grad():
        out = model(input_ids, attention_mask, segment_ids, length_feat)

    print(f"Output shape : {out.shape}")
    print(f"Output values: {out}")
    print("OK")