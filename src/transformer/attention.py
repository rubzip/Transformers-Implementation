import torch
from torch import nn


class AttentionLayer(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, d_model: int, d_k: int, d_v: int, masking: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_k_sqrt = d_k**0.5

        self.w_q = nn.Linear(d_model, d_k)
        self.w_k = nn.Linear(d_model, d_k)
        self.w_v = nn.Linear(d_model, d_v)

        self.softmax = nn.Softmax(dim=-1)
        self.masking = masking

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        attention = (query @ key.transpose(-2, -1)) / self.d_k_sqrt
        if self.masking:
            seq_len = attention.size(-1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=attention.device), diagonal=1)
            attention = attention.masked_fill(mask.bool(), float('-inf'))

        score = self.softmax(attention)
        context = score @ value
        return context


class MultiHeadAttention(nn.Module):
    """MultiHead Attention"""

    def __init__(self, d_model: int, h: int, masking: bool = False):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_model // h

        self.w_heads = nn.ModuleList([
            AttentionLayer(d_model, self.d_k, self.d_v, masking) for _ in range(h)
        ])
        self.w_o = nn.Linear(h * self.d_v, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads = [head(x) for head in self.w_heads]
        x_head = torch.cat(heads, dim=-1)
        context = self.w_o(x_head)
        return context
