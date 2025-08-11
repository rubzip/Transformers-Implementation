import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.encoder = Encoder(d_model=d_model, d_ff=d_ff, h=h, n=n)
        self.decoder = Decoder(d_model=d_model, d_ff=d_ff, h=h, n=n)
        self.pos_encoder = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pos_encoder(x)
        y = self.encoder(x, pe)
