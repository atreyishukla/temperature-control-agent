import numpy as np
import torch
import pytest
from hvac_env import HVACEnv
from lstm_model import BuildingLSTM


def _make_env():
    """Make an env with a randomly-initialised LSTM and 50 fake sequences."""
    lstm = BuildingLSTM()
    lstm.eval()
    seqs = np.random.randn(50, 24, 8).astype(np.float32)
    env = HVACEnv(lstm=lstm, train_sequences=seqs,
                  t_inside_mean=20.0, t_inside_std=10.0)
    return env


def test_reset_obs_shape():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (26,), f"Expected (26,), got {obs.shape}"
    assert obs.dtype == np.float32


def test_step_returns_correct_types():
    env = _make_env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (26,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_episode_terminates_at_8():
    env = _make_env()
    env.reset(seed=0)
    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == 8


def test_all_4_actions_valid():
    env = _make_env()
    for action in range(4):
        env.reset(seed=action)
        obs, reward, _, _, _ = env.step(action)
        assert obs.shape == (26,)
        assert np.isfinite(reward)


def test_lstm_not_modified_by_step():
    """Frozen LSTM weights must not change during env stepping."""
    lstm = BuildingLSTM()
    lstm.eval()
    seqs = np.random.randn(10, 24, 8).astype(np.float32)
    env  = HVACEnv(lstm=lstm, train_sequences=seqs,
                   t_inside_mean=20.0, t_inside_std=10.0)

    params_before = {n: p.clone() for n, p in lstm.named_parameters()}
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())
    for n, p in lstm.named_parameters():
        assert torch.allclose(params_before[n], p), f"LSTM weight {n} changed!"


def test_observation_space_matches_gym_spec():
    env = _make_env()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_action_space_is_discrete_4():
    env = _make_env()
    assert env.action_space.n == 4
