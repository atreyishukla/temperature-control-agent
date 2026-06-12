import torch
import pytest
from lstm_model import BuildingLSTM


def test_output_shape_single():
    model = BuildingLSTM()
    x = torch.randn(1, 24, 8)
    out = model(x)
    assert out.shape == (1, 2), f"Expected (1,2), got {out.shape}"


def test_output_shape_batch():
    model = BuildingLSTM()
    x = torch.randn(64, 24, 8)
    out = model(x)
    assert out.shape == (64, 2)


def test_output_is_float32():
    model = BuildingLSTM()
    x = torch.randn(4, 24, 8)
    out = model(x)
    assert out.dtype == torch.float32


def test_gradients_flow():
    model = BuildingLSTM()
    x = torch.randn(4, 24, 8)
    loss = model(x).sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_eval_mode_no_dropout_change():
    """In eval mode, two identical forward passes should give identical output."""
    model = BuildingLSTM()
    model.eval()
    x = torch.randn(4, 24, 8)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)
