import torch
from src.transformer.attention import AttentionLayer, MultiHeadAttention

BATCH = 32
N_TOKENS = 20
D_MODEL = 1024
D_K = 128
D_V = 256

H = 8


def test_attention():
    emb = torch.randn((BATCH, N_TOKENS, D_MODEL))
    attention = AttentionLayer(d_model=D_MODEL, d_k=D_K, d_v=D_V)

    out = attention(emb)

    exp_shape = (BATCH, N_TOKENS, D_V)
    assert (
        out.shape == exp_shape
    ), f"Unexpected shape: got {out.shape}. Expected: {exp_shape}"


def test_multihead_attention():
    emb = torch.randn((BATCH, N_TOKENS, D_MODEL))
    multihead_attention = MultiHeadAttention(d_model=D_MODEL, h=H)

    out = multihead_attention(emb)

    exp_shape = (BATCH, N_TOKENS, D_MODEL)
    assert (
        out.shape == exp_shape
    ), f"Unexpected shape: got {out.shape}. Expected: {exp_shape}"
