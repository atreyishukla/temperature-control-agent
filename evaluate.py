"""
Offline evaluation on the held-out test split (rows 7316-8760).

For each hourly window we ask four strategies to pick an action, feed that
action into the LSTM to get the predicted next T_inside, then compute the
reward.  No real sensors needed — the historical data drives everything.

Usage:
    python evaluate.py
"""

import pickle
import numpy as np
import torch
from tabulate import tabulate          # pip install tabulate  (or we print manually)

from data_pipeline import load_data, split_scale, make_sequences
from lstm_model import BuildingLSTM
from mpc import MPCSolver
from reward import compute_reward
from hvac_env import ACTION_MAP

try:
    from stable_baselines3 import PPO as SB3PPO
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False

DATA_PATH   = 'data/Concrete_floor_results.xlsx'
LSTM_PATH   = 'models/lstm_best.pt'
SCALER_PATH = 'models/scaler.pkl'
PPO_PATH    = 'models/ppo_hvac.zip'

FEATURE_COLS = ['T_outside', 'T_inside', 'T_floor', 'SR_direct', 'fan_on', 'heater_on']


def _ppo_obs(window: np.ndarray) -> np.ndarray:
    return np.concatenate([window[:, 1], window[-1, 0:1], window[-1, 3:4]]).astype(np.float32)


def evaluate(n_steps: int = 500) -> None:
    # ------------------------------------------------------------------ load
    print('Loading data and models...')
    df = load_data(DATA_PATH)
    df_train, df_val, df_test, scaler = split_scale(df)

    with open(SCALER_PATH, 'rb') as fh:
        scaler = pickle.load(fh)

    lstm = BuildingLSTM()
    lstm.load_state_dict(torch.load(LSTM_PATH, weights_only=True))
    lstm.eval()

    t_mean = float(scaler.mean_[1])
    t_std  = float(scaler.scale_[1])

    mpc = MPCSolver(lstm=lstm, t_inside_mean=t_mean, t_inside_std=t_std,
                    n_candidates=64, horizon=4)

    ppo = None
    if _HAS_SB3:
        try:
            ppo = SB3PPO.load(PPO_PATH)
        except Exception:
            print('  (PPO model not found, skipping PPO column)')

    # --------------------------------------------------- build test windows
    X_test, _ = make_sequences(df_test)
    n_steps   = min(n_steps, len(X_test))
    indices   = np.linspace(0, len(X_test) - 1, n_steps, dtype=int)

    # -------------------------------------------------------- historical actions
    # The last row of each window has the actual fan_on / heater_on used
    # We map (fan_on, heater_on) → action index for lookup
    inv_action = {v: k for k, v in ACTION_MAP.items()}

    strategies = ['historical', 'mpc', 'always_off', 'always_heat']
    if ppo:
        strategies.insert(2, 'ppo')

    records = {s: [] for s in strategies}
    t_records = {s: [] for s in strategies}   # real T_inside per step

    print(f'Evaluating {n_steps} windows from test split...')
    for i, idx in enumerate(indices):
        if i % 50 == 0:
            print(f'  {i}/{n_steps}', end='\r', flush=True)
        window = X_test[idx]   # (24, 6) normalised

        def _step_reward(action: int) -> tuple[float, float]:
            fan_on, heater_on = ACTION_MAP[action]
            inp = window.copy()
            inp[-1, 4] = float(fan_on)
            inp[-1, 5] = float(heater_on)
            x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                delta = lstm(x).squeeze(0).cpu().numpy()
            new_t_inside_norm = float(window[-1, 1]) + float(delta[0])
            T_real = new_t_inside_norm * t_std + t_mean
            r = compute_reward(T_real, fan_on, heater_on)
            return r, T_real

        # historical
        hist_fan   = int(round(window[-1, 4]))
        hist_heat  = int(round(window[-1, 5]))
        hist_action = inv_action.get((hist_fan, hist_heat), 0)
        r, T = _step_reward(hist_action)
        records['historical'].append(r)
        t_records['historical'].append(T)

        # mpc
        r, T = _step_reward(mpc.solve(window))
        records['mpc'].append(r)
        t_records['mpc'].append(T)

        # ppo
        if ppo:
            obs = _ppo_obs(window)
            act, _ = ppo.predict(obs, deterministic=True)
            r, T = _step_reward(int(act))
            records['ppo'].append(r)
            t_records['ppo'].append(T)

        # baselines
        r, T = _step_reward(0)   # both off
        records['always_off'].append(r)
        t_records['always_off'].append(T)

        r, T = _step_reward(2)   # heater only
        records['always_heat'].append(r)
        t_records['always_heat'].append(T)

    print(f'  {n_steps}/{n_steps}')

    # ----------------------------------------------------------- report
    rows = []
    for s in strategies:
        R = np.array(records[s])
        T = np.array(t_records[s])
        in_comfort = np.mean((T >= 18.0) & (T <= 24.0)) * 100
        rows.append([
            s,
            f'{R.mean():.2f}',
            f'{R.std():.2f}',
            f'{in_comfort:.1f}%',
            f'{T.mean():.1f}°C',
            f'{T.min():.1f}°C',
            f'{T.max():.1f}°C',
        ])

    headers = ['Strategy', 'Mean reward', 'Std', 'In comfort', 'Mean T_inside', 'Min T', 'Max T']
    print()
    try:
        print(tabulate(rows, headers=headers, tablefmt='rounded_outline'))
    except Exception:
        # fallback if tabulate not installed
        print(f'{"Strategy":<14} {"Mean reward":>11} {"Std":>8} {"In comfort":>10} {"Mean T":>10} {"Min T":>7} {"Max T":>7}')
        print('-' * 70)
        for r in rows:
            print(f'{r[0]:<14} {r[1]:>11} {r[2]:>8} {r[3]:>10} {r[4]:>10} {r[5]:>7} {r[6]:>7}')

    print(f'\nEvaluated on {n_steps} windows from the test split '
          f'({len(df_test)} total rows, rows 7316–8760 of original data).')


if __name__ == '__main__':
    evaluate()
