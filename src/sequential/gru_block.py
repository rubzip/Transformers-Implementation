from torch import nn
from .sequential_block import SequentialBlock


class GRUBlock(SequentialBlock):
    def __init__(self, x_dim: int, h_dim: int, bias: bool = True):
        super().__init__(uses_c=False)
        self.w_z = nn.Linear(x_dim, h_dim, bias=bias)
        self.w_r = nn.Linear(x_dim, h_dim, bias=bias)
        self.w_h = nn.Linear(x_dim, h_dim, bias=bias)

        self.u_z = nn.Linear(h_dim, h_dim, bias=False)
        self.u_r = nn.Linear(h_dim, h_dim, bias=False)
        self.u_h = nn.Linear(h_dim, h_dim, bias=False)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, h_prev):
        z_t = self.sigmoid(self.w_z(x_t) + self.u_z(h_prev))
        r_t = self.sigmoid(self.w_r(x_t) + self.u_r(h_prev))
        h_t_candidate = self.tanh(self.w_h(x_t) + self.u_h(r_t * h_prev))
        h_t = (1 - z_t) * h_prev + z_t * h_t_candidate
        return h_t
