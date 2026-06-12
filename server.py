import collections
import csv
import os
import pickle

import numpy as np
import torch
from flask import Flask, jsonify, request

from hvac_env import ACTION_MAP
from lstm_model import BuildingLSTM
from mpc import MPCSolver

LSTM_PATH  = 'models/lstm_best.pt'
SCALER_PATH = 'models/scaler.pkl'
PPO_PATH   = 'models/ppo_hvac.zip'
LOG_PATH   = 'logs/experience.csv'
MPC_WEIGHT = 0.7
WINDOW_LEN = 24
SCALE_COLS = ['T_outside', 'T_inside', 'T_floor', 'SR_direct']
LOG_FIELDS = ['T_outside', 'T_inside', 'T_floor', 'SR_direct', 'fan_on', 'heater_on']

app = Flask(__name__)

_lstm       = None
_ppo        = None
_mpc        = None
_scaler     = None
_window_buf = None   # deque(maxlen=WINDOW_LEN) of (6,) normalised rows


def _load_models(
    lstm_path:   str = LSTM_PATH,
    scaler_path: str = SCALER_PATH,
    ppo_path:    str = PPO_PATH,
) -> None:
    global _lstm, _ppo, _mpc, _scaler, _window_buf
    from stable_baselines3 import PPO as SB3PPO

    _lstm = BuildingLSTM()
    _lstm.load_state_dict(torch.load(lstm_path, weights_only=True))
    _lstm.eval()

    with open(scaler_path, 'rb') as fh:
        _scaler = pickle.load(fh)

    t_mean = float(_scaler.mean_[1])
    t_std  = float(_scaler.scale_[1])

    _mpc = MPCSolver(lstm=_lstm, t_inside_mean=t_mean, t_inside_std=t_std)
    _ppo = SB3PPO.load(ppo_path)

    _window_buf = collections.deque(maxlen=WINDOW_LEN)


def _normalise_row(body: dict, fan_on: int = 0, heater_on: int = 0) -> np.ndarray:
    raw    = np.array([[body['T_outside'], body['T_inside'],
                        body['T_floor'],   body['SR_direct']]], dtype=np.float32)
    scaled = _scaler.transform(raw)[0]   # (4,)
    return np.array([scaled[0], scaled[1], scaled[2], scaled[3],
                     float(fan_on), float(heater_on)], dtype=np.float32)


def _get_window() -> np.ndarray:
    buf = list(_window_buf)
    if not buf:
        buf = [np.zeros(6, dtype=np.float32)]
    while len(buf) < WINDOW_LEN:
        buf.insert(0, buf[0])
    return np.array(buf, dtype=np.float32)   # (24, 6)


def _ppo_obs(window: np.ndarray) -> np.ndarray:
    return np.concatenate([
        window[:, 1],        # (24,) T_inside history
        window[-1, 0:1],     # (1,)  T_outside now
        window[-1, 3:4],     # (1,)  SR now
    ]).astype(np.float32)    # (26,)


@app.route('/predict', methods=['POST'])
def predict():
    body     = request.get_json(force=True)
    norm_row = _normalise_row(body)
    _window_buf.append(norm_row)
    window = _get_window()

    if np.random.random() < MPC_WEIGHT:
        action = _mpc.solve(window)
        source = 'mpc'
    else:
        action, _ = _ppo.predict(_ppo_obs(window), deterministic=True)
        action     = int(action)
        source     = 'ppo'

    fan_on, heater_on = ACTION_MAP[action]
    _window_buf[-1] = _normalise_row(body, fan_on=fan_on, heater_on=heater_on)

    return jsonify({
        'fan_on':    fan_on,
        'heater_on': heater_on,
        'action':    action,
        'source':    source,
    })


@app.route('/log', methods=['POST'])
def log():
    body = request.get_json(force=True)
    os.makedirs(os.path.dirname(LOG_PATH) or '.', exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: body[k] for k in LOG_FIELDS})
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    _load_models()
    app.run(host='0.0.0.0', port=5000)
