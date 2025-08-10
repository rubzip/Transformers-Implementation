import torch
from src.transformer.attention import AttentionLayer, MultiHeadAttention

batch = 32
n_tokens = 20
d_model = 1024
d_k = 128
d_v = 256

h = 8

def test_attention():
    emb = torch.randn((batch, n_tokens, d_model))
    attention = AttentionLayer(d_model=d_model, d_k=d_k, d_v=d_v)

    out = attention(emb)

    exp_shape = (batch, n_tokens, d_v)
    assert out.shape == exp_shape, f"Unexpected shape: got {out.shape}. Expected: {exp_shape}"

def test_multihead_attention():
    emb = torch.randn((batch, n_tokens, d_model))
    multihead_attention = MultiHeadAttention(d_model=d_model, h=h)

    out = multihead_attention(emb)
    
    exp_shape = (batch, n_tokens, d_model)
    assert out.shape == exp_shape, f"Unexpected shape: got {out.shape}. Expected: {exp_shape}"
