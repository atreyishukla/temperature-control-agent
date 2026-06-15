import numpy as np
import torch
from lstm_model import BuildingLSTM
from reward import compute_reward

HORIZON = 24
N_CANDIDATES = 1024
GAMMA = 0.95
CEM_ITERATIONS = 3
CEM_ELITE_FRAC = 0.1   # top 10% guide the next sample

ACTION_MAP = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}


class MPCSolver:
    """
    Random-shooting MPC over a frozen LSTM world model.

    Samples N candidate 24-step action sequences, batch-rolls them through
    the LSTM, scores each by discounted cumulative reward, and returns the
    first action of the best sequence (receding-horizon MPC).
    """

    def __init__(
        self,
        lstm:          BuildingLSTM,
        t_inside_mean: float,
        t_inside_std:  float,
        device:        str   = 'cpu',
        horizon:       int   = HORIZON,
        n_candidates:  int   = N_CANDIDATES,
        gamma:         float = GAMMA,
        cem_iterations: int  = CEM_ITERATIONS,
        cem_elite_frac: float = CEM_ELITE_FRAC,
    ):
        self.lstm           = lstm
        self.t_inside_mean  = t_inside_mean
        self.t_inside_std   = t_inside_std
        self.device         = device
        self.horizon        = horizon
        self.n_candidates   = n_candidates
        self.gamma          = gamma
        self.cem_iterations = cem_iterations
        self.n_elite        = max(1, int(n_candidates * cem_elite_frac))

    def solve(self, window: np.ndarray) -> int:
        """
        Return the best first action (0-3) for the given 24-step window.

        Uses Cross-Entropy Method: iteratively refine action probabilities
        toward high-reward regions instead of pure random shooting.

        window: (24, 8) normalised feature array — same layout as LSTM input.
        """
        N, H = self.n_candidates, self.horizon
        # Uniform prior: each of 4 actions equally likely at each horizon step
        probs = np.full((H, 4), 0.25)

        actions = self._sample(probs)
        for _ in range(self.cem_iterations):
            scores  = self._rollout(window, actions)
            elite   = np.argsort(scores)[-self.n_elite:]
            # Re-estimate action probabilities from elite sequences
            counts  = np.zeros((H, 4), dtype=np.float64)
            for t in range(H):
                for a in actions[elite, t]:
                    counts[t, a] += 1
            probs   = counts / self.n_elite
            actions = self._sample(probs)

        scores = self._rollout(window, actions)
        best   = int(np.argmax(scores))
        return int(actions[best, 0])

    def _sample(self, probs: np.ndarray) -> np.ndarray:
        """Sample n_candidates action sequences from per-step action probabilities."""
        H = probs.shape[0]
        actions = np.empty((self.n_candidates, H), dtype=np.int32)
        for t in range(H):
            actions[:, t] = np.random.choice(4, size=self.n_candidates, p=probs[t])
        return actions

    def _rollout(self, window: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Batch-roll N candidate sequences through the LSTM.

        window:  (24, 8)
        actions: (N, H) integer action indices
        Returns: (N,) discounted cumulative reward
        """
        N = actions.shape[0]
        H = actions.shape[1]

        windows      = np.tile(window[np.newaxis], (N, 1, 1))  # (N, 24, 6)
        total_reward = np.zeros(N, dtype=np.float64)
        discount     = 1.0

        for h in range(H):
            fan_arr  = np.array([ACTION_MAP[a][0] for a in actions[:, h]], dtype=np.float32)
            heat_arr = np.array([ACTION_MAP[a][1] for a in actions[:, h]], dtype=np.float32)

            lstm_input        = windows.copy()
            lstm_input[:, -1, 4] = fan_arr
            lstm_input[:, -1, 5] = heat_arr

            x = torch.tensor(lstm_input, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                delta = self.lstm(x).cpu().numpy()  # (N, 2) — ΔT_inside, ΔT_floor

            new_t_inside  = windows[:, -1, 1] + delta[:, 0]
            new_t_floor   = windows[:, -1, 2] + delta[:, 1]
            T_inside_real = new_t_inside * self.t_inside_std + self.t_inside_mean
            step_rewards  = np.array([
                compute_reward(float(T_inside_real[i]), int(fan_arr[i]), int(heat_arr[i]))
                for i in range(N)
            ])
            total_reward += discount * step_rewards
            discount     *= self.gamma

            new_rows       = windows[:, -1, :].copy()
            new_rows[:, 1] = new_t_inside
            new_rows[:, 2] = new_t_floor
            new_rows[:, 4] = fan_arr
            new_rows[:, 5] = heat_arr
            windows = np.concatenate(
                [windows[:, 1:, :], new_rows[:, np.newaxis, :]], axis=1
            )

        return total_reward
