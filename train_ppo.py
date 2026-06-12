import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from data_pipeline import load_data, split_scale, make_sequences
from lstm_model import BuildingLSTM
from hvac_env import HVACEnv

DATA_PATH   = 'data/Concrete_floor_results.xlsx'
LSTM_PATH   = 'models/lstm_best.pt'
PPO_PATH    = 'models/ppo_hvac'


def train(
    lstm=None,
    train_sequences=None,
    t_inside_mean=None,
    t_inside_std=None,
    data_path:       str = DATA_PATH,
    lstm_path:       str = LSTM_PATH,
    ppo_path:        str = PPO_PATH,
    total_timesteps: int = 500_000,
    n_envs:          int = 4,
    n_steps:         int = 2048,
) -> PPO:
    """
    Train PPO inside the frozen LSTM simulator.

    Pass lstm/train_sequences/t_inside_mean/t_inside_std directly (fast, for
    tests), or omit them to load everything from disk (for production).
    """
    os.makedirs(os.path.dirname(ppo_path) or '.', exist_ok=True)

    if lstm is None:
        _lstm = BuildingLSTM()
        _lstm.load_state_dict(torch.load(lstm_path, weights_only=True))
        _lstm.eval()

        df = load_data(data_path)
        df_train, _, _, scaler = split_scale(df)
        X_train = make_sequences(df_train)[0]
        t_mean  = float(scaler.mean_[1])
        t_std   = float(scaler.scale_[1])
    else:
        _lstm   = lstm
        X_train = train_sequences
        t_mean  = t_inside_mean
        t_std   = t_inside_std

    def _make_env():
        return HVACEnv(lstm=_lstm, train_sequences=X_train,
                       t_inside_mean=t_mean, t_inside_std=t_std,
                       device='cpu')

    vec_env = make_vec_env(_make_env, n_envs=n_envs)

    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(ppo_path)
    return model


if __name__ == '__main__':
    model = train()
    print(f'PPO training complete. Model saved to {PPO_PATH}')
