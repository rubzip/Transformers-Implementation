import pytest
import torch
from src.sequential import (
    SequentialBlock,
    RNNBlock,
    LSTMBlock,
    GRUBlock,
    SequentialManyToMany,
    SequentialManyToOne,
)


batch = 32
t_dim = 12
x_dim = 100
h_dim = 20
out_dim = 10


@pytest.mark.parametrize("block_class", [RNNBlock, LSTMBlock, GRUBlock])
def test_many_to_many(block_class: SequentialBlock):
    model = SequentialManyToMany(
        x_dim=x_dim, h_dim=h_dim, out_dim=out_dim, block_class=block_class
    )
    sample = torch.randn((batch, t_dim, x_dim))
    with torch.no_grad():
        out = model(sample)
    expected_shape = (batch, t_dim, out_dim)
    assert (
        out.shape == expected_shape
    ), f"Output shape {out.shape} does not match expected {expected_shape}"


@pytest.mark.parametrize("block_class", [RNNBlock, LSTMBlock, GRUBlock])
def test_many_to_one(block_class: SequentialBlock):
    model = SequentialManyToOne(
        x_dim=x_dim, h_dim=h_dim, out_dim=out_dim, block_class=block_class
    )
    sample = torch.randn((batch, t_dim, x_dim))
    with torch.no_grad():
        out = model(sample)
    expected_shape = (batch, out_dim)
    assert (
        out.shape == expected_shape
    ), f"Output shape {out.shape} does not match expected {expected_shape}"
