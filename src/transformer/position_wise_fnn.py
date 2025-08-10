import torch
from torch import nn


class PositionWiseFFN(nn.Module):
    """Position-wise Feed-Forward Networks, works as two convolutions with kernel size 1."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.relu(self.fc1(x))
        y2 = self.fc2(y1)
        return y2
