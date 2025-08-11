import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int, divisor: float = 10000.0):
        super().__init__()
        self.seq_len = max_seq_len
        self.d_model = d_model

        self.pe = torch.zeros(1, max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(divisor)) / d_model)
        )
        self.pe[:, :, 0::2] = torch.sin(position * div_term)
        self.pe[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1), :]
