import pytest
from reward import compute_reward


def test_comfort_zone_nothing_on():
    # T=21, both off → r_comfort=+2, r_inaction=0, r_warming=0, r_energy=0
    assert compute_reward(21.0, 0, 0) == pytest.approx(2.0)


def test_comfort_zone_fan_on():
    # T=23, fan on → r_comfort=+2, r_energy=-0.05
    assert compute_reward(23.0, 1, 0) == pytest.approx(1.95)


def test_cold_2deg_heater_off_inaction_fires():
    # T=16, cold_dev=2 > 1 threshold, heater off → inaction fires
    # r_comfort=-(4)*2=-8, r_inaction=-5*2=-10, r_warming=0, r_energy=0
    assert compute_reward(16.0, 0, 0) == pytest.approx(-18.0)


def test_cold_2deg_heater_on_warming_bonus():
    # T=16, cold_dev=2, heater on → no inaction, +warming bonus
    # r_comfort=-8, r_inaction=0, r_warming=+1, r_energy=-0.10
    assert compute_reward(16.0, 0, 1) == pytest.approx(-7.10)


def test_cold_5deg_heater_off():
    # T=13, cold_dev=5, heater off → r_comfort=-50, r_inaction=-25
    assert compute_reward(13.0, 0, 0) == pytest.approx(-75.0)


def test_cold_5deg_heater_on():
    # T=13, cold_dev=5, heater on → r_comfort=-50, r_inaction=0, r_warming=+1, r_energy=-0.10
    assert compute_reward(13.0, 0, 1) == pytest.approx(-49.10)


def test_hot_16deg_fan_off():
    # T=40, hot_dev=16, fan off → r_comfort=-(256)*2=-512, r_inaction=-5*16=-80
    assert compute_reward(40.0, 0, 0) == pytest.approx(-592.0)


def test_hot_16deg_fan_on():
    # T=40, hot_dev=16, fan on → r_comfort=-512, r_inaction=0, r_energy=-0.05
    assert compute_reward(40.0, 1, 0) == pytest.approx(-512.05)


def test_wrong_action_heating_when_hot():
    # T=30, hot_dev=6, fan=0, heater=1 → r_comfort=-(36)*2=-72, r_inaction=-5*6=-30, r_energy=-0.10
    assert compute_reward(30.0, 0, 1) == pytest.approx(-102.10)


def test_inaction_fires_above_1deg_cold():
    # T=16.5, cold_dev=1.5 > 1.0 → inaction fires
    # r_comfort=-(2.25)*2=-4.5, r_inaction=-5*1.5=-7.5
    assert compute_reward(16.5, 0, 0) == pytest.approx(-12.0)


def test_inaction_does_not_fire_at_or_below_1deg():
    # T=17.5, cold_dev=0.5 ≤ 1.0 → no inaction
    # r_comfort=-(0.25)*2=-0.5
    assert compute_reward(17.5, 0, 0) == pytest.approx(-0.5)


def test_warming_bonus_only_when_heater_on_and_cold():
    # T=21 (comfort), heater on → no warming bonus (not cold)
    # r_comfort=+2, r_energy=-0.10
    assert compute_reward(21.0, 0, 1) == pytest.approx(1.90)


def test_no_warming_bonus_when_heater_off():
    # T=16, heater off → no warming bonus
    # r_comfort=-8, r_inaction=-10
    assert compute_reward(16.0, 0, 0) == pytest.approx(-18.0)
