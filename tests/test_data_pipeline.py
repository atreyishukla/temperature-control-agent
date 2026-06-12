import numpy as np
import pytest
from data_pipeline import load_data, split_scale, make_sequences, save_scaler, load_scaler

DATA_PATH = 'data/Concrete_floor_results.xlsx'


def test_load_returns_8_features():
    df = load_data(DATA_PATH)
    assert list(df.columns) == ['T_outside', 'T_inside', 'T_floor', 'SR_direct',
                                 'fan_on', 'heater_on', 'hour_sin', 'hour_cos']
    assert len(df) == 8760


def test_actions_are_binary():
    df = load_data(DATA_PATH)
    assert set(df['fan_on'].unique()).issubset({0.0, 1.0})
    assert set(df['heater_on'].unique()).issubset({0.0, 1.0})


def test_no_missing_values():
    df = load_data(DATA_PATH)
    assert df.isnull().sum().sum() == 0


def test_split_sizes():
    df = load_data(DATA_PATH)
    train, val, test, _ = split_scale(df)
    assert len(train) == 6132
    assert len(val) == 1184
    assert len(test) == 1444
    assert len(train) + len(val) + len(test) == 8760


def test_scaler_fitted_on_train_only():
    df = load_data(DATA_PATH)
    train, val, test, scaler = split_scale(df)
    for col in ['T_outside', 'T_inside', 'T_floor', 'SR_direct']:
        assert abs(train[col].mean()) < 0.05, f"{col} mean not ~0"
        assert abs(train[col].std() - 1.0) < 0.05, f"{col} std not ~1"


def test_binary_actions_not_scaled():
    df = load_data(DATA_PATH)
    train, val, test, scaler = split_scale(df)
    assert set(train['fan_on'].unique()).issubset({0.0, 1.0})
    assert set(train['heater_on'].unique()).issubset({0.0, 1.0})


def test_make_sequences_shapes():
    df = load_data(DATA_PATH)
    train, val, test, _ = split_scale(df)
    X, y = make_sequences(train)
    assert X.shape == (6108, 24, 8)
    assert y.shape == (6108, 2)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_sequence_target_is_delta_T_inside_T_floor():
    df = load_data(DATA_PATH)
    train, _, _, _ = split_scale(df)
    X, y = make_sequences(train)
    arr = train.values.astype('float32')
    expected_delta = arr[24, 1:3] - arr[23, 1:3]
    np.testing.assert_array_almost_equal(y[0], expected_delta)


def test_save_and_load_scaler(tmp_path):
    df = load_data(DATA_PATH)
    _, _, _, scaler = split_scale(df)
    path = str(tmp_path / 'scaler.pkl')
    save_scaler(scaler, path)
    loaded = load_scaler(path)
    np.testing.assert_array_almost_equal(scaler.mean_, loaded.mean_)
