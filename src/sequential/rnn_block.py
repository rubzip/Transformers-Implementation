import torch
from torch import nn
from .sequential_block import SequentialBlock


class RNNBlock(SequentialBlock):
    """Recurrent Neural Network Block"""

    def __init__(
        self, x_dim: int, h_dim: int, act_fn: nn.Module = nn.ReLU(), bias: bool = True
    ):
        super().__init__(uses_c=False)
        self.w = nn.Linear(x_dim, h_dim, bias=bias)
        self.u = nn.Linear(h_dim, h_dim, bias=False)  # As we are adding w(x) + u(h) only 1 bias is needed (bc they are added)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        h = self.act_fn(self.w(x) + self.u(h_prev))
        return h
