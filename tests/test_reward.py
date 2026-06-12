import pytest
from reward import compute_reward


def test_comfort_zone_nothing_on():
    # T=21, both off → r_comfort=+2, r_inaction=0, r_energy=0
    assert compute_reward(21.0, 0, 0) == pytest.approx(2.0)


def test_comfort_zone_fan_on():
    # T=23, fan on → r_comfort=+2, r_energy=-0.05
    assert compute_reward(23.0, 1, 0) == pytest.approx(1.95)


def test_cold_2deg_no_inaction():
    # T=16, cold_dev=2, heater off → r_comfort=-(4*3)=-12, no inaction (dev<3)
    assert compute_reward(16.0, 0, 0) == pytest.approx(-12.0)


def test_cold_5deg_heater_off():
    # T=13, cold_dev=5, heater off → r_comfort=-75, r_inaction=-50 → -125
    assert compute_reward(13.0, 0, 0) == pytest.approx(-125.0)


def test_cold_5deg_heater_on():
    # T=13, cold_dev=5, heater on → r_comfort=-75, r_inaction=0, r_energy=-0.10 → -75.10
    assert compute_reward(13.0, 0, 1) == pytest.approx(-75.10)


def test_hot_16deg_fan_off():
    # T=40, hot_dev=16, fan off → r_comfort=-256, r_inaction=-64 → -320
    assert compute_reward(40.0, 0, 0) == pytest.approx(-320.0)


def test_hot_16deg_fan_on():
    # T=40, hot_dev=16, fan on → r_comfort=-256, r_inaction=0, r_energy=-0.05 → -256.05
    assert compute_reward(40.0, 1, 0) == pytest.approx(-256.05)


def test_wrong_action_heating_when_hot():
    # T=30, hot_dev=6, fan=0, heater=1 → r_comfort=-36, r_inaction=-24, r_energy=-0.10 → -60.10
    assert compute_reward(30.0, 0, 1) == pytest.approx(-60.10)


def test_inaction_only_fires_above_3deg_cold():
    # T=14.5, cold_dev=3.5, heater off → inaction fires
    r = compute_reward(14.5, 0, 0)
    cold_dev = 3.5
    expected = -(cold_dev**2)*3 + (-10*cold_dev)
    assert r == pytest.approx(expected)


def test_inaction_does_not_fire_below_3deg():
    # T=15.5, cold_dev=2.5, heater off → no inaction
    r = compute_reward(15.5, 0, 0)
    cold_dev = 2.5
    expected = -(cold_dev**2)*3
    assert r == pytest.approx(expected)
