import numpy as np
import torch
import pytest
from mpc import MPCSolver


class _ConstLSTM:
    """Mock LSTM that always predicts a fixed (T_inside_norm, T_floor_norm)."""

    def __init__(self, t_inside_norm: float = 0.1, t_floor_norm: float = 0.0):
        self.t_inside_norm = t_inside_norm
        self.t_floor_norm  = t_floor_norm

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        out = torch.tensor(
            [[self.t_inside_norm, self.t_floor_norm]] * N,
            dtype=torch.float32,
        )
        return out


def _make_solver(lstm=None, t_inside_norm=0.1):
    if lstm is None:
        lstm = _ConstLSTM(t_inside_norm=t_inside_norm)
    # t_inside_mean=20, t_std=10 → normalised 0.1 → real 21°C (comfort zone)
    return MPCSolver(
        lstm=lstm,
        t_inside_mean=20.0,
        t_inside_std=10.0,
        horizon=4,
        n_candidates=64,
        gamma=0.95,
    )


def _make_window() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((24, 6)).astype(np.float32)


# ---------------------------------------------------------------------------
# Shape / type contracts
# ---------------------------------------------------------------------------

def test_solve_returns_valid_action():
    solver = _make_solver()
    action = solver.solve(_make_window())
    assert action in {0, 1, 2, 3}, f"Expected action in 0-3, got {action}"


def test_solve_returns_python_int():
    solver = _make_solver()
    action = solver.solve(_make_window())
    assert isinstance(action, int)


def test_rollout_returns_correct_shape():
    solver = _make_solver()
    window  = _make_window()
    actions = np.random.randint(0, 4, size=(64, 4))
    scores  = solver._rollout(window, actions)
    assert scores.shape == (64,), f"Expected (64,), got {scores.shape}"


def test_rollout_scores_are_finite():
    solver = _make_solver()
    window  = _make_window()
    actions = np.random.randint(0, 4, size=(32, 4))
    scores  = solver._rollout(window, actions)
    assert np.all(np.isfinite(scores))


# ---------------------------------------------------------------------------
# Policy correctness (using controlled mock)
# ---------------------------------------------------------------------------

def test_prefers_low_energy_in_comfort_zone():
    """
    When LSTM always predicts 21°C (comfort), both-off (action 0) should win
    because it gets r_comfort=+2 with zero energy cost — best possible reward.
    """
    solver = _make_solver(t_inside_norm=0.1)   # 21°C

    # Force all candidates to be tested: generate one of each action in first step
    # by running with enough candidates and checking the chosen action is 0
    np.random.seed(7)
    counts = {a: 0 for a in range(4)}
    for _ in range(20):
        counts[solver.solve(_make_window())] += 1
    # Action 0 (both off, energy cost 0) must win most often in comfort zone
    assert counts[0] > counts[3], "Expected action 0 (both off) to dominate in comfort zone"


def test_prefers_heater_when_very_cold():
    """
    When LSTM predicts T_inside = 5°C (cold_dev=13, well below 18°C),
    turning on the heater avoids the inaction penalty and must beat both-off.
    """
    # 5°C → (5 - 20) / 10 = -1.5 normalised
    solver = _make_solver(t_inside_norm=-1.5)

    np.random.seed(42)
    heater_on_count = 0
    heater_off_count = 0
    for _ in range(30):
        a = solver.solve(_make_window())
        fan, heat = (a in {1, 3}), (a in {2, 3})
        if heat:
            heater_on_count += 1
        else:
            heater_off_count += 1
    assert heater_on_count > heater_off_count, (
        "Expected heater-on actions to dominate when T_inside is very cold"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_reproducible_with_same_seed():
    solver = _make_solver()
    window = _make_window()
    np.random.seed(0)
    a1 = solver.solve(window)
    np.random.seed(0)
    a2 = solver.solve(window)
    assert a1 == a2, "Same seed must produce same action"


# ---------------------------------------------------------------------------
# Window is not mutated
# ---------------------------------------------------------------------------

def test_window_not_mutated_by_solve():
    solver = _make_solver()
    window = _make_window()
    before = window.copy()
    solver.solve(window)
    np.testing.assert_array_equal(window, before, err_msg="solve() must not mutate the input window")


# ---------------------------------------------------------------------------
# Custom horizon / n_candidates
# ---------------------------------------------------------------------------

def test_custom_horizon_and_candidates():
    solver = MPCSolver(
        lstm=_ConstLSTM(),
        t_inside_mean=20.0,
        t_inside_std=10.0,
        horizon=8,
        n_candidates=16,
        gamma=0.99,
    )
    action = solver.solve(_make_window())
    assert action in {0, 1, 2, 3}
