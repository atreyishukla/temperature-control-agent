import os
import torch
import numpy as np
import pytest
from stable_baselines3 import PPO
from lstm_model import BuildingLSTM
from hvac_env import HVACEnv
from train_ppo import train


def _fake_inputs():
    lstm = BuildingLSTM()
    lstm.eval()
    seqs = np.random.randn(50, 24, 6).astype(np.float32)
    return lstm, seqs


def test_train_saves_model(tmp_path):
    lstm, seqs = _fake_inputs()
    ppo_path = str(tmp_path / 'ppo_test')
    train(lstm=lstm, train_sequences=seqs,
          t_inside_mean=20.0, t_inside_std=10.0,
          ppo_path=ppo_path, total_timesteps=512, n_envs=1, n_steps=128)
    assert os.path.exists(ppo_path + '.zip')


def test_train_returns_ppo_instance(tmp_path):
    lstm, seqs = _fake_inputs()
    model = train(lstm=lstm, train_sequences=seqs,
                  t_inside_mean=20.0, t_inside_std=10.0,
                  ppo_path=str(tmp_path / 'ppo_test'),
                  total_timesteps=512, n_envs=1, n_steps=128)
    assert isinstance(model, PPO)


def test_trained_model_predicts_valid_action(tmp_path):
    lstm, seqs = _fake_inputs()
    ppo_path = str(tmp_path / 'ppo_test')
    train(lstm=lstm, train_sequences=seqs,
          t_inside_mean=20.0, t_inside_std=10.0,
          ppo_path=ppo_path, total_timesteps=512, n_envs=1, n_steps=128)

    model = PPO.load(ppo_path)
    env = HVACEnv(lstm=lstm, train_sequences=seqs,
                  t_inside_mean=20.0, t_inside_std=10.0)
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert int(action) in range(4)


def test_lstm_weights_unchanged_after_ppo_train(tmp_path):
    lstm, seqs = _fake_inputs()
    params_before = {n: p.clone() for n, p in lstm.named_parameters()}

    train(lstm=lstm, train_sequences=seqs,
          t_inside_mean=20.0, t_inside_std=10.0,
          ppo_path=str(tmp_path / 'ppo_test'),
          total_timesteps=512, n_envs=1, n_steps=128)

    for n, p in lstm.named_parameters():
        assert torch.allclose(params_before[n], p), f"LSTM weight {n} changed during PPO!"
