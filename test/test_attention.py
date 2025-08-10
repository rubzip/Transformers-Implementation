import torch
from src.transformer.attention import AttentionLayer


def test_attention_shape():
    batch = 32
    n_tokens = 20
    d_model = 1024
    d_k = 128
    d_v = 256

    sentences_embedding = torch.randn((batch, n_tokens, d_model))
    attention = AttentionLayer(d_model=d_model, d_k=d_k, d_v=d_v)

    with torch.no_grad():
        out = attention(sentences_embedding)

    assert out.shape == (
        batch,
        n_tokens,
        d_v,
    ), f"Unexpected shape: got {out.shape}. Expected: {(batch, n_tokens, d_v)}"
