import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, h: int, n: int = 6, max_seq_len: int = 512):
        super().__init__()

        self.encoder = Encoder(d_model=d_model, d_ff=d_ff, h=h, n=n)
        self.decoder = Decoder(d_model=d_model, d_ff=d_ff, h=h, n=n)
        self.pos_encoder = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_model)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_pe = self.pos_encoder(src)
        tgt_pe = self.pos_encoder(tgt)

        enc_out = self.encoder(src, src_pe)
        dec_out = self.decoder(tgt, tgt_pe, enc_out)
        return dec_out
