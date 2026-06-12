import csv
import os
import pickle
import time

import numpy as np
import pytest
import torch

from lstm_model import BuildingLSTM
from retrain import (
    LSTM_INTERVAL_H,
    PPO_INTERVAL_H,
    MIN_NEW_ROWS,
    fine_tune_lstm,
    maybe_retrain,
    _load_stamps,
    _save_stamps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_experience(path: str, n_rows: int) -> None:
    """Write n_rows of fake experience data."""
    import math
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['T_outside', 'T_inside', 'T_floor', 'SR_direct',
                         'fan_on', 'heater_on', 'hour_sin', 'hour_cos'])
        rng = np.random.default_rng(0)
        for i in range(n_rows):
            hour = i % 24
            writer.writerow([
                rng.uniform(-10, 30),
                rng.uniform(15, 25),
                rng.uniform(15, 25),
                rng.uniform(0, 800),
                rng.integers(0, 2),
                rng.integers(0, 2),
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
            ])


def _write_scaler(path: str) -> None:
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    rng = np.random.default_rng(1)
    df  = pd.DataFrame({
        'T_outside': rng.uniform(-10, 30, 200),
        'T_inside':  rng.uniform(15, 25, 200),
        'T_floor':   rng.uniform(15, 25, 200),
        'SR_direct': rng.uniform(0, 800, 200),
    })
    sc = StandardScaler().fit(df)
    with open(path, 'wb') as fh:
        pickle.dump(sc, fh)


def _write_lstm(path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(BuildingLSTM().state_dict(), path)


# ---------------------------------------------------------------------------
# Stamp helpers
# ---------------------------------------------------------------------------

def test_stamps_round_trip(tmp_path):
    path   = str(tmp_path / 'stamps.pkl')
    stamps = {'lstm_last': 123.0, 'ppo_last': 456.0}
    _save_stamps(stamps, path)
    loaded = _load_stamps(path)
    assert loaded == stamps


def test_load_stamps_defaults_when_missing(tmp_path):
    path   = str(tmp_path / 'nonexistent.pkl')
    stamps = _load_stamps(path)
    assert stamps['lstm_last'] == 0.0
    assert stamps['ppo_last']  == 0.0


# ---------------------------------------------------------------------------
# fine_tune_lstm — skip conditions
# ---------------------------------------------------------------------------

def test_fine_tune_skips_when_no_experience_file(tmp_path):
    ran = fine_tune_lstm(
        experience_path=str(tmp_path / 'missing.csv'),
        lstm_path=str(tmp_path / 'lstm.pt'),
        scaler_path=str(tmp_path / 'scaler.pkl'),
    )
    assert ran is False


def test_fine_tune_skips_when_too_few_rows(tmp_path):
    exp_path    = str(tmp_path / 'exp.csv')
    lstm_path   = str(tmp_path / 'lstm.pt')
    scaler_path = str(tmp_path / 'scaler.pkl')

    _write_experience(exp_path, n_rows=5)   # < MIN_NEW_ROWS
    _write_scaler(scaler_path)
    _write_lstm(lstm_path)

    ran = fine_tune_lstm(
        experience_path=exp_path,
        lstm_path=lstm_path,
        scaler_path=scaler_path,
    )
    assert ran is False


# ---------------------------------------------------------------------------
# fine_tune_lstm — runs and saves updated weights
# ---------------------------------------------------------------------------

def test_fine_tune_runs_with_sufficient_data(tmp_path):
    exp_path    = str(tmp_path / 'exp.csv')
    lstm_path   = str(tmp_path / 'lstm.pt')
    scaler_path = str(tmp_path / 'scaler.pkl')

    _write_experience(exp_path, n_rows=MIN_NEW_ROWS + 10)
    _write_scaler(scaler_path)
    _write_lstm(lstm_path)

    ran = fine_tune_lstm(
        experience_path=exp_path,
        lstm_path=lstm_path,
        scaler_path=scaler_path,
        epochs=1,
    )
    assert ran is True


def test_fine_tune_updates_weights(tmp_path):
    exp_path    = str(tmp_path / 'exp.csv')
    lstm_path   = str(tmp_path / 'lstm.pt')
    scaler_path = str(tmp_path / 'scaler.pkl')

    _write_experience(exp_path, n_rows=MIN_NEW_ROWS + 10)
    _write_scaler(scaler_path)
    _write_lstm(lstm_path)

    weights_before = torch.load(lstm_path, weights_only=True)

    fine_tune_lstm(
        experience_path=exp_path,
        lstm_path=lstm_path,
        scaler_path=scaler_path,
        epochs=2,
    )

    weights_after = torch.load(lstm_path, weights_only=True)
    changed = any(
        not torch.equal(weights_before[k], weights_after[k])
        for k in weights_before
    )
    assert changed, "Expected LSTM weights to change after fine-tuning"


# ---------------------------------------------------------------------------
# maybe_retrain — interval gating
# ---------------------------------------------------------------------------

def test_maybe_retrain_skips_both_when_intervals_not_elapsed(tmp_path):
    stamp_path = str(tmp_path / 'stamps.pkl')
    now        = time.time()
    _save_stamps({'lstm_last': now - 1, 'ppo_last': now - 1}, stamp_path)

    result = maybe_retrain(
        experience_path=str(tmp_path / 'missing.csv'),
        lstm_path=str(tmp_path / 'lstm.pt'),
        scaler_path=str(tmp_path / 'scaler.pkl'),
        ppo_path=str(tmp_path / 'ppo'),
        stamp_path=stamp_path,
        now=now,
    )
    assert result == {'lstm_ran': False, 'ppo_ran': False}


def test_maybe_retrain_runs_lstm_after_interval(tmp_path):
    exp_path    = str(tmp_path / 'exp.csv')
    lstm_path   = str(tmp_path / 'lstm.pt')
    scaler_path = str(tmp_path / 'scaler.pkl')
    stamp_path  = str(tmp_path / 'stamps.pkl')

    _write_experience(exp_path, n_rows=MIN_NEW_ROWS + 10)
    _write_scaler(scaler_path)
    _write_lstm(lstm_path)

    now = time.time()
    _save_stamps({
        'lstm_last': now - (LSTM_INTERVAL_H + 1) * 3600,
        'ppo_last':  now,
    }, stamp_path)

    result = maybe_retrain(
        experience_path=exp_path,
        lstm_path=lstm_path,
        scaler_path=scaler_path,
        ppo_path=str(tmp_path / 'ppo'),
        stamp_path=stamp_path,
        now=now,
        epochs=1,
    )
    assert result['lstm_ran'] is True
    assert result['ppo_ran']  is False


def test_maybe_retrain_updates_lstm_stamp(tmp_path):
    exp_path    = str(tmp_path / 'exp.csv')
    lstm_path   = str(tmp_path / 'lstm.pt')
    scaler_path = str(tmp_path / 'scaler.pkl')
    stamp_path  = str(tmp_path / 'stamps.pkl')

    _write_experience(exp_path, n_rows=MIN_NEW_ROWS + 10)
    _write_scaler(scaler_path)
    _write_lstm(lstm_path)

    now = time.time()
    _save_stamps({
        'lstm_last': now - (LSTM_INTERVAL_H + 1) * 3600,
        'ppo_last':  now,
    }, stamp_path)

    maybe_retrain(
        experience_path=exp_path,
        lstm_path=lstm_path,
        scaler_path=scaler_path,
        ppo_path=str(tmp_path / 'ppo'),
        stamp_path=stamp_path,
        now=now,
        epochs=1,
    )

    stamps = _load_stamps(stamp_path)
    assert abs(stamps['lstm_last'] - now) < 1.0
