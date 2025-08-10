import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, divisor: float = 10_000.):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(divisor)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe.unsqueeze(0)

    def forward(self, pos: int) -> torch.Tensor:
        return self.pe[:, pos, :]
