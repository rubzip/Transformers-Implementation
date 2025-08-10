from abc import ABC, abstractmethod
from torch import nn
import torch

from .sequential_block import SequentialBlock


class SequentialModel(nn.Module, ABC):
    def __init__(
        self, x_dim: int, h_dim: int, out_dim: int, block_class: SequentialBlock
    ):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.seq_block = block_class(x_dim, h_dim)
        self.out_layer = nn.Linear(h_dim, out_dim)

    @abstractmethod
    def forward(self, x):
        pass


class SequentialManyToMany(SequentialModel):
    def __init__(self, x_dim, h_dim, out_dim, block_class):
        super().__init__(x_dim, h_dim, out_dim, block_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.h_dim)
        c = torch.zeros(batch, self.h_dim) if self.seq_block.uses_c else None

        outputs = []
        for t in range(seq_len):
            if self.seq_block.uses_c:
                h, c = self.seq_block(x[:, t, :], h, c)
            else:
                h = self.seq_block(x[:, t, :], h)
            out = self.out_layer(h)
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class SequentialManyToOne(SequentialModel):
    def __init__(self, x_dim, h_dim, out_dim, block_class):
        super().__init__(x_dim, h_dim, out_dim, block_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.h_dim)
        c = torch.zeros(batch, self.h_dim) if self.seq_block.uses_c else None

        for t in range(seq_len):
            if self.seq_block.uses_c:
                h, c = self.seq_block(x[:, t, :], h, c)
            else:
                h = self.seq_block(x[:, t, :], h)
        out = self.out_layer(h)
        return out
