import torch
from torch import nn

from .attention import MultiHeadAttention
from .position_wise_fnn import PositionWiseFFN
from .add_and_norm import AddAndNorm

class EncoderLayer(nn.Module):
    """Transformer's Encoder Layer"""

    def __init__(self, d_model: int, d_ff: int, h: int):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, h=h)
        self.feed_forward = PositionWiseFFN(d_model=d_model, d_ff=d_ff)
        self.add_norm_1 = AddAndNorm(d_model)
        self.add_norm_2 = AddAndNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att_out = self.multi_head_attention(x)
        x = self.add_norm_1(x, att_out)

        ff_out = self.feed_forward(x)
        x = self.add_norm_2(x, ff_out)
        return x


class Encoder(nn.Module):
    """Transformer's Encoder Model"""

    def __init__(self, d_model: int, d_ff: int, h: int, n: int = 6):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, d_ff=d_ff, h=h) for _ in range(n)]
        )

    def forward(self, in_emb: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        x = in_emb + pos_emb
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x
