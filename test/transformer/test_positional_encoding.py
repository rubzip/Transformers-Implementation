import math
import torch
from src.transformer.positional_encoding import PositionalEncoding


BATCH = 4
SEQ_LEN = 10
D_MODEL = 16
pe = PositionalEncoding(max_seq_len=SEQ_LEN, d_model=D_MODEL)
x = torch.zeros(BATCH, SEQ_LEN, D_MODEL)


def test_positional_encoding_shape():
    out = pe(x)
    expected_shape = (1, SEQ_LEN, D_MODEL)
    assert (
        out.shape == expected_shape
    ), f"Unexpected shape: {out.shape}. Expected: {expected_shape}"


def test_positional_encoding_deterministic():
    out1 = pe(x)
    out2 = pe(x)
    assert torch.allclose(
        out1, out2
    ), "PositionalEncoding output should be deterministic"


def test_positional_encoding_values():
    x_batch_1 = torch.zeros(1, SEQ_LEN, D_MODEL)
    out = pe(x_batch_1)

    expected_out = torch.zeros(1, SEQ_LEN, D_MODEL)
    for pos in range(SEQ_LEN):
        for i in range(0, D_MODEL, 2):
            div_term = math.exp(-math.log(10000.0) * i / D_MODEL)
            expected_out[:, pos, i] = math.sin(pos * div_term)
            if i + 1 < D_MODEL:
                expected_out[:, pos, i + 1] = math.cos(pos * div_term)

    assert torch.allclose(
        out, expected_out, atol=1e-6
    ), "Positional encoding values differ from expected"
