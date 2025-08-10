import torch
from src.transformer.encoder import Encoder

BATCH = 32
N_TOKENS = 20

D_MODEL = 512
D_FF = 2_048
H = 8
N = 6


def test_encoder():
    x_emb = torch.randn((BATCH, N_TOKENS, D_MODEL))
    p_emb = torch.randn((BATCH, N_TOKENS, D_MODEL))
    encoder = Encoder(d_model=D_MODEL, d_ff=D_FF, h=H, n=N)

    out = encoder(x_emb, p_emb)

    exp_shape = (BATCH, N_TOKENS, D_MODEL)
    assert (
        out.shape == exp_shape
    ), f"Unexpected shape: got {out.shape}. Expected: {exp_shape}"
