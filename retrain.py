"""
Online continual learning.

  LSTM  — fine-tuned every 24 h on new rows from experience.csv
  PPO   — retrained from scratch every 7 days inside the updated LSTM sim

Run once per hour from cron or a scheduler; the functions check elapsed time
themselves so extra calls are no-ops until the threshold is passed.
"""

import os
import time
import pickle
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import make_sequences
from lstm_model import BuildingLSTM
from train_ppo import train as train_ppo

log = logging.getLogger(__name__)

LSTM_PATH      = 'models/lstm_best.pt'
SCALER_PATH    = 'models/scaler.pkl'
PPO_PATH       = 'models/ppo_hvac'
EXPERIENCE_PATH = 'logs/experience.csv'
STAMP_PATH     = 'logs/retrain_stamps.pkl'

LSTM_INTERVAL_H = 24
PPO_INTERVAL_H  = 24 * 7
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LR     = 1e-4
MIN_NEW_ROWS     = 50   # need at least SEQ_LEN+1 rows to form one sequence


# ---------------------------------------------------------------------------
# Stamp helpers
# ---------------------------------------------------------------------------

def _load_stamps(path: str = STAMP_PATH) -> dict:
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    return {'lstm_last': 0.0, 'ppo_last': 0.0}


def _save_stamps(stamps: dict, path: str = STAMP_PATH) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(stamps, fh)


# ---------------------------------------------------------------------------
# LSTM fine-tune
# ---------------------------------------------------------------------------

def fine_tune_lstm(
    experience_path: str = EXPERIENCE_PATH,
    lstm_path:       str = LSTM_PATH,
    scaler_path:     str = SCALER_PATH,
    epochs:          int = FINE_TUNE_EPOCHS,
    lr:              float = FINE_TUNE_LR,
    min_new_rows:    int = MIN_NEW_ROWS,
) -> bool:
    """
    Fine-tune the LSTM on rows from experience.csv.

    Returns True if fine-tuning ran, False if skipped (not enough data).
    """
    if not os.path.exists(experience_path):
        log.info('fine_tune_lstm: no experience file, skipping')
        return False

    df = pd.read_csv(experience_path)
    if len(df) < min_new_rows:
        log.info('fine_tune_lstm: only %d rows, need %d, skipping', len(df), min_new_rows)
        return False

    with open(scaler_path, 'rb') as fh:
        scaler = pickle.load(fh)

    scale_cols = ['T_outside', 'T_inside', 'T_floor', 'SR_direct']
    df[scale_cols] = scaler.transform(df[scale_cols])

    if 'hour_sin' not in df.columns:
        df['hour_sin'] = 0.0
        df['hour_cos'] = 1.0

    X, y = make_sequences(df)
    if len(X) == 0:
        log.info('fine_tune_lstm: not enough rows to form sequences, skipping')
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = BuildingLSTM().to(device)
    model.load_state_dict(torch.load(lstm_path, weights_only=True))
    model.train()

    loader  = DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y)),
        batch_size=32, shuffle=True,
    )
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        log.info('fine_tune_lstm epoch %d/%d done', epoch + 1, epochs)

    torch.save(model.state_dict(), lstm_path)
    log.info('fine_tune_lstm: saved updated weights to %s', lstm_path)
    return True


# ---------------------------------------------------------------------------
# Main retrain entry point
# ---------------------------------------------------------------------------

def maybe_retrain(
    experience_path:   str   = EXPERIENCE_PATH,
    lstm_path:         str   = LSTM_PATH,
    scaler_path:       str   = SCALER_PATH,
    ppo_path:          str   = PPO_PATH,
    stamp_path:        str   = STAMP_PATH,
    now:               float = None,
    epochs:            int   = FINE_TUNE_EPOCHS,
) -> dict:
    """
    Run LSTM fine-tune and/or PPO retrain if their intervals have elapsed.

    Returns a dict: {'lstm_ran': bool, 'ppo_ran': bool}.
    """
    if now is None:
        now = time.time()

    stamps        = _load_stamps(stamp_path)
    lstm_elapsed  = (now - stamps['lstm_last']) / 3600
    ppo_elapsed   = (now - stamps['ppo_last'])  / 3600

    result = {'lstm_ran': False, 'ppo_ran': False}

    if lstm_elapsed >= LSTM_INTERVAL_H:
        ran = fine_tune_lstm(
            experience_path=experience_path,
            lstm_path=lstm_path,
            scaler_path=scaler_path,
            epochs=epochs,
        )
        if ran:
            stamps['lstm_last'] = now
            result['lstm_ran']  = True

    if ppo_elapsed >= PPO_INTERVAL_H:
        log.info('maybe_retrain: retraining PPO')
        train_ppo(lstm_path=lstm_path, ppo_path=ppo_path)
        stamps['ppo_last'] = now
        result['ppo_ran']  = True

    _save_stamps(stamps, stamp_path)
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    result = maybe_retrain()
    print(f'Retrain result: {result}')
