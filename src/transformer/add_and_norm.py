import torch
from torch import nn


class AddAndNorm(nn.Module):
    """Add & Norm Layer"""

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.norm(x + y)
