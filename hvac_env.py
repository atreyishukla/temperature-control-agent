import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from data_pipeline import SEQ_LEN
from lstm_model import BuildingLSTM
from reward import compute_reward

EPISODE_LEN = 8  # 8 steps — keeps LSTM close to training distribution

# Maps Discrete(4) integer → (fan_on, heater_on)
ACTION_MAP = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}


class HVACEnv(gym.Env):
    """
    Gymnasium environment for HVAC control.

    Observation: Box(26,) = [last 24h T_inside (normalised), T_outside_now, SR_now]
    Action:      Discrete(4) = {both off, fan only, heater only, both on}
    Physics:     Frozen BuildingLSTM — weights never updated inside this env.

    t_inside_mean / t_inside_std: from StandardScaler fitted on training data
    (scaler.mean_[1] and scaler.scale_[1]). Used to convert normalised LSTM
    output back to real °C for the reward function.
    """

    metadata = {}

    def __init__(
        self,
        lstm:            BuildingLSTM,
        train_sequences: np.ndarray,
        t_inside_mean:   float,
        t_inside_std:    float,
        device:          str = 'cpu',
        min_t_inside_c:  float = 10.0,
    ):
        super().__init__()
        self.lstm          = lstm
        # Filter to windows where the current T_inside >= min_t_inside_c (real °C).
        # Prevents episodes starting in physically irrecoverable cold states.
        min_norm = (min_t_inside_c - t_inside_mean) / t_inside_std
        mask = train_sequences[:, -1, 1] >= min_norm
        self.seqs = train_sequences[mask]   # (N, 24, 6), normalised
        self.t_inside_mean = t_inside_mean
        self.t_inside_std  = t_inside_std
        self.device        = device

        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(SEQ_LEN + 2,), dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

        self._window = None
        self._steps  = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx          = self.np_random.integers(0, len(self.seqs))
        self._window = self.seqs[idx].copy()   # (24, 6)
        self._steps  = 0
        return self._obs(), {}

    def step(self, action: int):
        fan_on, heater_on = ACTION_MAP[action]

        # Inject current action into last row of window before predicting
        lstm_input        = self._window.copy()
        lstm_input[-1, 4] = float(fan_on)
        lstm_input[-1, 5] = float(heater_on)

        # Frozen LSTM forward pass — no gradients, no weight updates
        x = torch.tensor(lstm_input, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            delta = self.lstm(x).squeeze(0).cpu().numpy()   # [ΔT_inside_norm, ΔT_floor_norm]

        # Apply delta to current normalised values
        new_t_inside = float(self._window[-1, 1]) + float(delta[0])
        new_t_floor  = float(self._window[-1, 2]) + float(delta[1])

        T_inside_real = new_t_inside * self.t_inside_std + self.t_inside_mean
        reward        = float(compute_reward(T_inside_real, fan_on, heater_on))

        # Slide window forward: inherit T_outside and SR from last row
        new_row    = self._window[-1].copy()
        new_row[1] = new_t_inside
        new_row[2] = new_t_floor
        new_row[4] = float(fan_on)
        new_row[5] = float(heater_on)
        self._window = np.vstack([self._window[1:], new_row])

        self._steps += 1
        terminated   = self._steps >= EPISODE_LEN
        return self._obs(), reward, terminated, False, {}

    def _obs(self) -> np.ndarray:
        t_inside_hist = self._window[:, 1]     # (24,) last 24h T_inside
        t_outside_now = self._window[-1, 0:1]  # (1,)
        sr_now        = self._window[-1, 3:4]  # (1,)
        return np.concatenate([t_inside_hist, t_outside_now, sr_now]).astype(np.float32)
