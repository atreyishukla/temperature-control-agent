import os
import numpy as np
import torch
import pytest
from train_lstm import train
from lstm_model import BuildingLSTM


def _make_tiny_data(tmp_path):
    """Return a DataFrame with 60 rows — fast enough for tests."""
    import pandas as pd
    np.random.seed(0)
    n = 60
    data = {
        'T_outside': np.random.randn(n),
        'T_inside':  np.random.randn(n),
        'T_floor':   np.random.randn(n),
        'SR_direct': np.abs(np.random.randn(n)),
        'fan_on':    np.random.randint(0, 2, n).astype(float),
        'heater_on': np.random.randint(0, 2, n).astype(float),
    }
    return pd.DataFrame(data)


def test_train_returns_model(tmp_path):
    df = _make_tiny_data(tmp_path)
    model = train(
        df_train=df, df_val=df,
        model_path=str(tmp_path / 'lstm_test.pt'),
        scaler_path=str(tmp_path / 'scaler_test.pkl'),
        max_epochs=2,
        patience=10,
    )
    assert isinstance(model, BuildingLSTM)


def test_train_saves_model_file(tmp_path):
    df = _make_tiny_data(tmp_path)
    model_path = str(tmp_path / 'lstm_test.pt')
    train(
        df_train=df, df_val=df,
        model_path=model_path,
        scaler_path=str(tmp_path / 'scaler_test.pkl'),
        max_epochs=2,
        patience=10,
    )
    assert os.path.exists(model_path)


def test_train_val_loss_decreases(tmp_path):
    """Val loss at epoch 5 should be <= epoch 1 on a learnable synthetic dataset."""
    import pandas as pd
    np.random.seed(42)
    n = 200
    t = np.linspace(0, 4 * np.pi, n)
    df = pd.DataFrame({
        'T_outside': np.sin(t),
        'T_inside':  np.sin(t + 0.1),
        'T_floor':   np.sin(t + 0.2),
        'SR_direct': np.abs(np.cos(t)),
        'fan_on':    (np.sin(t) > 0).astype(float),
        'heater_on': (np.cos(t) > 0).astype(float),
    })
    losses = []
    train(
        df_train=df, df_val=df,
        model_path=str(tmp_path / 'lstm_test.pt'),
        scaler_path=str(tmp_path / 'scaler_test.pkl'),
        max_epochs=5,
        patience=10,
        loss_log=losses,
    )
    assert losses[-1] <= losses[0], "Val loss never decreased over 5 epochs"
