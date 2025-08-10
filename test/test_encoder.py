import torch
from src.transformer.encoder import Encoder

batch = 32
n_tokens = 20

d_model = 512
d_ff = 2_048
h = 8
n = 6

def test_encoder():
    x_emb = torch.randn((batch, n_tokens, d_model))
    p_emb = torch.randn((batch, n_tokens, d_model))
    encoder = Encoder(d_model=d_model, d_ff=d_ff, h=h, n=n)

    out = encoder(x_emb, p_emb)

    exp_shape = (batch, n_tokens, d_model)
    assert out.shape == exp_shape, f"Unexpected shape: got {out.shape}. Expected: {exp_shape}"
