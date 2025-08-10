from abc import abstractmethod
import torch
from torch import nn


class SequentialBlock(nn.Module):
    """Sequential Block Interface"""

    def __init__(self, uses_c: bool = False):
        super().__init__()
        self._uses_c = uses_c

    @property
    def uses_c(self) -> bool:
        return self._uses_c

    @abstractmethod
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor=None):
        pass
