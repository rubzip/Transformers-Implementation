from torch import nn
from .sequential_block import SequentialBlock


class RNNBlock(SequentialBlock):
    def __init__(
        self, x_dim: int, h_dim: int, act_fn: nn.Module = nn.ReLU(), bias: bool = True
    ):
        super().__init__(uses_c=False)
        self.context_layer = nn.Linear(x_dim, h_dim, bias=bias)
        self.state_layer = nn.Linear(h_dim, h_dim, bias=False)  # As we are adding f(x) + g(h) only 1 bias is needed (bc they are added)
        self.act_fn = act_fn

    def forward(self, x, h_prev):
        x_forward = self.context_layer(x)
        h_forward = self.state_layer(h_prev)
        h = self.act_fn(x_forward + h_forward)
        return h
