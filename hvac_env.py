import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from data_pipeline import SEQ_LEN
from lstm_model import BuildingLSTM
from reward import compute_reward

EPISODE_LEN = 24  # 24 steps — one full day; long enough for PPO to learn sustained-heating strategies

# Maps Discrete(4) integer → (fan_on, heater_on)
ACTION_MAP = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}


class HVACEnv(gym.Env):
    """
    Gymnasium environment for HVAC control.

    Observation: Box(35,) = [24h T_inside, 6h T_outside trend, SR, prev_fan, prev_heater, hour_sin, hour_cos]
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
        self.t_inside_mean = t_inside_mean
        self.t_inside_std  = t_inside_std
        self.device        = device

        # Keep the full sequence array so step() can look up real T_outside/SR/hour.
        # Only start episodes that have EPISODE_LEN future rows available.
        self._all_seqs = train_sequences  # (M, SEQ_LEN, 8), normalised
        min_norm  = (min_t_inside_c - t_inside_mean) / t_inside_std
        max_start = len(train_sequences) - EPISODE_LEN - 1
        mask      = train_sequences[:, -1, 1] >= min_norm
        mask[max_start:] = False
        self._valid_idx = np.where(mask)[0]   # indices into _all_seqs
        self.seqs = train_sequences[mask]     # kept for backward compatibility

        # 24 T_inside + 6 T_outside trend + SR_now + prev_fan + prev_heater + hour_sin + hour_cos
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(35,), dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

        self._window = None
        self._steps  = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        i                  = self.np_random.integers(0, len(self._valid_idx))
        self._orig_seq_idx = int(self._valid_idx[i])
        self._window       = self._all_seqs[self._orig_seq_idx].copy()
        self._steps        = 0
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

        # Slide window forward
        new_row    = self._window[-1].copy()
        new_row[1] = new_t_inside
        new_row[2] = new_t_floor
        new_row[4] = float(fan_on)
        new_row[5] = float(heater_on)

        # Pull real T_outside, SR, and hour from the next historical row.
        # _all_seqs[orig + steps + 1, -1] is the row exactly one step ahead
        # of our current position in the original dataset — always within the
        # LSTM's training distribution (no extrapolation divergence).
        ref = self._all_seqs[self._orig_seq_idx + self._steps + 1, -1]
        new_row[0] = ref[0]   # T_outside
        new_row[3] = ref[3]   # SR_direct
        new_row[6] = ref[6]   # hour_sin
        new_row[7] = ref[7]   # hour_cos

        self._window = np.vstack([self._window[1:], new_row])

        self._steps += 1
        terminated   = self._steps >= EPISODE_LEN
        return self._obs(), reward, terminated, False, {}

    def _obs(self) -> np.ndarray:
        t_inside_hist  = self._window[:, 1]       # (24,) last 24h T_inside
        t_outside_6h   = self._window[-6:, 0]     # (6,)  T_outside trend
        sr_now         = self._window[-1, 3:4]    # (1,)
        prev_fan       = self._window[-1, 4:5]    # (1,)
        prev_heater    = self._window[-1, 5:6]    # (1,)
        hour_sin       = self._window[-1, 6:7]    # (1,)
        hour_cos       = self._window[-1, 7:8]    # (1,)
        return np.concatenate([t_inside_hist, t_outside_6h, sr_now,
                                prev_fan, prev_heater, hour_sin, hour_cos]).astype(np.float32)
