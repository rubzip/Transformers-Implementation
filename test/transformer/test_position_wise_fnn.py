import torch
from src.transformer.position_wise_fnn import PositionWiseFFN


BATCH = 4
SEQ_LEN = 10
D_MODEL = 16
D_FF = 2048

pwffn = PositionWiseFFN(d_model=D_MODEL, d_ff=D_FF)
x = torch.zeros(BATCH, SEQ_LEN, D_MODEL)


def test_position_wise_fnn_shape():
    out = pwffn(x)
    assert out.shape == x.shape, "Invalid Shape"


def test_position_wise_ffn_no_nan():
    out = pwffn(x)
    assert not torch.isnan(out).any(), "Output contains NaN values"


def test_position_wise_ffn_gradient():
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    out = pwffn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient computed for input"
    assert not torch.isnan(x.grad).any(), "Gradient contains NaNs"
