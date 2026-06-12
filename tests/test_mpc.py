import numpy as np
import torch
import pytest
from mpc import MPCSolver


class _ConstLSTM:
    """Mock LSTM that always returns a fixed (ΔT_inside_norm, ΔT_floor_norm)."""

    def __init__(self, delta_t_inside: float = 0.0, delta_t_floor: float = 0.0):
        self.delta_t_inside = delta_t_inside
        self.delta_t_floor  = delta_t_floor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        return torch.tensor(
            [[self.delta_t_inside, self.delta_t_floor]] * N,
            dtype=torch.float32,
        )


def _make_solver(lstm=None):
    if lstm is None:
        lstm = _ConstLSTM()
    return MPCSolver(
        lstm=lstm,
        t_inside_mean=20.0,
        t_inside_std=10.0,
        horizon=4,
        n_candidates=64,
        gamma=0.95,
    )


def _make_window(t_inside_norm: float = 0.0) -> np.ndarray:
    """Return a (24, 6) window with a known T_inside in the last row."""
    rng = np.random.default_rng(42)
    w = rng.standard_normal((24, 6)).astype(np.float32)
    w[-1, 1] = t_inside_norm  # set current T_inside to a known normalised value
    return w


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
    Window has T_inside=21°C (comfort zone). LSTM returns delta=0 so temperature
    stays at 21°C. Both-off (action 0) should dominate — highest reward, zero cost.
    """
    # 21°C → (21 - 20) / 10 = 0.1 normalised
    solver = _make_solver(lstm=_ConstLSTM(delta_t_inside=0.0))
    window = _make_window(t_inside_norm=0.1)   # current T_inside = 21°C

    np.random.seed(7)
    counts = {a: 0 for a in range(4)}
    for _ in range(20):
        counts[solver.solve(window)] += 1
    assert counts[0] > counts[3], "Expected action 0 (both off) to dominate in comfort zone"


def test_prefers_heater_when_very_cold():
    """
    Window has T_inside=5°C. LSTM returns delta=0 so temperature stays at 5°C.
    Heater-on avoids the inaction penalty and must dominate.
    """
    # 5°C → (5 - 20) / 10 = -1.5 normalised
    solver = _make_solver(lstm=_ConstLSTM(delta_t_inside=0.0))
    window = _make_window(t_inside_norm=-1.5)  # current T_inside = 5°C

    np.random.seed(42)
    heater_on_count = 0
    heater_off_count = 0
    for _ in range(30):
        a = solver.solve(window)
        heat = a in {2, 3}
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
