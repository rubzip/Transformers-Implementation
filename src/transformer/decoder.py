import torch
from torch import nn

from .attention import MultiHeadAttention
from .position_wise_fnn import PositionWiseFFN
from .add_and_norm import AddAndNorm

class DecoderLayer(nn.Module):
    """Transformer's Decoder Layer"""

    def __init__(self, d_model: int, d_ff: int, h: int):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model=d_model, h=h)
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, h=h)
        self.feed_forward = PositionWiseFFN(d_model=d_model, d_ff=d_ff)

        self.add_norm_1, self.add_norm_2, self.add_norm_3 = [
            AddAndNorm(d_model) for _ in range(3)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att_out = self.masked_multi_head_attention(x)
        x = self.add_norm_1(x, att_out)

        att_out = self.multi_head_attention(x)
        x = self.add_norm_2(x, att_out)

        ff_out = self.feed_forward(x)
        x = self.add_norm_3(x, ff_out)
        return x


class Decoder(nn.Module):
    """Transformer's Decoder Model"""

    def __init__(self, d_model: int, d_ff: int, h: int, n: int = 6):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, d_ff=d_ff, h=h) for _ in range(n)]
        )

    def forward(
        self, out_emb: torch.Tensor, pos_emb: torch.Tensor, enc_emb: torch.Tensor
    ) -> torch.Tensor:
        x = out_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        return x
