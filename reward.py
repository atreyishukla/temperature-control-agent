def compute_reward(T_inside: float, fan_on: int, heater_on: int) -> float:
    """
    Reward for one timestep.

    Comfort zone: [18, 24]°C.
    Cold penalty is 3x hot — Edmonton building reached -8°C inside.
    Inaction penalty fires when deviation > 3°C and the corrective device is off.
    Energy cost is a tiebreaker only.
    """
    cold_dev = max(0.0, 18.0 - T_inside)
    hot_dev  = max(0.0, T_inside - 24.0)

    if cold_dev > 0:
        r_comfort = -(cold_dev ** 2) * 3.0
    elif hot_dev > 0:
        r_comfort = -(hot_dev ** 2) * 1.0
    else:
        r_comfort = 2.0

    if cold_dev > 3.0 and heater_on == 0:
        r_inaction = -10.0 * cold_dev
    elif hot_dev > 3.0 and fan_on == 0:
        r_inaction = -4.0 * hot_dev
    else:
        r_inaction = 0.0

    r_energy = -(0.05 * fan_on + 0.10 * heater_on)

    return r_comfort + r_inaction + r_energy
