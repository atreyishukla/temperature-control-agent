def compute_reward(T_inside: float, fan_on: int, heater_on: int,
                   t_low: float = 18.0, t_high: float = 24.0) -> float:
    """
    Reward for one timestep.

    Comfort zone: [t_low, t_high] (default 18–24°C).
    Penalties are symmetric for cold and hot deviations.
    Inaction penalty fires when deviation > 3°C and the corrective device is off.
    Energy cost is a tiebreaker only.
    """
    cold_dev = max(0.0, t_low  - T_inside)
    hot_dev  = max(0.0, T_inside - t_high)

    if cold_dev > 0:
        r_comfort = -(cold_dev ** 2) * 2.0
    elif hot_dev > 0:
        r_comfort = -(hot_dev  ** 2) * 2.0
    else:
        r_comfort = 2.0

    if cold_dev > 1.0 and heater_on == 0:
        r_inaction = -5.0 * cold_dev
    elif hot_dev > 3.0 and fan_on == 0:
        r_inaction = -5.0 * hot_dev
    else:
        r_inaction = 0.0

    r_warming = 1.0 if (heater_on == 1 and cold_dev > 0) else 0.0

    r_energy = -(0.05 * fan_on + 0.10 * heater_on)

    return r_comfort + r_inaction + r_warming + r_energy
