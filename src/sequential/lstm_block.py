import torch
from torch import nn
from .sequential_block import SequentialBlock


class LSTMBlock(SequentialBlock):
    """Long Short-Term Memory Block"""

    def __init__(self, x_dim: int, h_dim: int, bias: bool = True):
        super().__init__(uses_c=True)
        self.w_f = nn.Linear(x_dim, h_dim, bias=bias)
        self.w_i = nn.Linear(x_dim, h_dim, bias=bias)
        self.w_o = nn.Linear(x_dim, h_dim, bias=bias)
        self.w_c = nn.Linear(x_dim, h_dim, bias=bias)

        self.u_f = nn.Linear(h_dim, h_dim, bias=False)
        self.u_i = nn.Linear(h_dim, h_dim, bias=False)
        self.u_o = nn.Linear(h_dim, h_dim, bias=False)
        self.u_c = nn.Linear(h_dim, h_dim, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f_t = self.sigmoid(self.w_f(x_t) + self.u_f(h_prev))
        i_t = self.sigmoid(self.w_i(x_t) + self.u_i(h_prev))
        o_t = self.sigmoid(self.w_o(x_t) + self.u_o(h_prev))
        c_t_candidate = self.tanh(self.w_c(x_t) + self.u_c(h_prev))

        c_t = f_t * c_prev + i_t * c_t_candidate
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t
