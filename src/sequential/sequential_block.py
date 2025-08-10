from abc import ABC, abstractmethod
from torch import nn


class SequentialBlock(nn.Module, ABC):
    """Sequential Block Interface"""

    def __init__(self, uses_c: bool = False):
        super().__init__()
        self._uses_c = uses_c

    @property
    def uses_c(self) -> bool:
        return self._uses_c

    @abstractmethod
    def forward(self, x, h_prev, c_prev=None):
        pass
