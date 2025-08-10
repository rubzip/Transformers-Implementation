from torch import nn
from .sequential_block import SequentialBlock


class LSTMBlock(SequentialBlock):
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

    def forward(self, x_t, h_prev, c_prev):
        forget_gate = self.sigmoid(self.w_f(x_t) + self.u_f(h_prev))
        input_gate = self.sigmoid(self.w_i(x_t) + self.u_i(h_prev))
        output_gate = self.sigmoid(self.w_o(x_t) + self.u_o(h_prev))
        cell_candidate = self.tanh(self.w_c(x_t) + self.u_c(h_prev))

        cell_state = forget_gate * c_prev + input_gate * cell_candidate
        hidden_state = output_gate * self.tanh(cell_state)

        return hidden_state, cell_state
