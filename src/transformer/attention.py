from torch import nn

class AttentionLayer(nn.Module):
    # Scaled Dot-Product Attention Implementation in PyTorch
    def __init__(self, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_k_sqrt = d_k ** 0.5

        self.q_layer = nn.Linear(d_model, d_k)
        self.k_layer = nn.Linear(d_model, d_k)
        self.v_layer = nn.Linear(d_model, d_v)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.q_layer(x)
        key = self.k_layer(x)
        value = self.v_layer(x)

        score = self.softmax(
            (query @ key.transpose(-2, -1)) / self.d_k_sqrt
        )
        context = score @ value
        return context
