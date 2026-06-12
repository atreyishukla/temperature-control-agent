# LSTM + PPO HVAC Control System — Design Spec

**Date:** 2026-06-11  
**Project:** Concrete floor HVAC control, Edmonton building  
**Option:** A — PyTorch throughout + Flask HTTP server  
**Status:** Approved, ready for implementation

---

## Overview

A three-component system that controls a concrete-floor HVAC system by predicting building temperatures and learning an optimal control policy. Outputs discrete fan on/off and heater on/off commands every hour.

**Three components (must train in order):**

1. **LSTM World Model** — supervised learning on historical data; predicts T_inside and T_floor one hour ahead given the last 24h of sensor readings + actions
2. **PPO Agent** — RL policy trained inside a Gymnasium env that uses the frozen LSTM as its physics simulator; learns reactive control
3. **MPC Optimizer** — random shooting over 1024 candidate 24-hour schedules at runtime; no separate training needed

**Runtime:** Sensors → MPC (70%) + PPO (30%) blend → fan/heater command → log to experience.csv → periodic retrain

---

## Dataset

**Source:** `data/Concrete_floor_results.xlsx`, Results sheet, 8760 rows (hourly, full year)

**Raw columns:** `Date_time, T_outside, T_inside, T_floor, SR_direct_outside, Cooling_power, Heating_power`

**Derived columns:**
- `fan_on = (Cooling_power > 0)` → binary
- `heater_on = (Heating_power > 0)` → binary

**Feature set (6 features):** `T_outside, T_inside, T_floor, SR_direct_outside, fan_on, heater_on`

**Key stats:**
- T_outside: −28.5 to 29.7°C (Edmonton)
- T_inside: −7.86 to 51.13°C (polycarbonate facade, high solar gain)
- T_floor: −8.46 to 56.94°C (concrete thermal mass)
- Cooling active: 205 hours; Heating active: 192 hours
- Zero missing values

---

## Phase 1 — LSTM World Model

### Data pipeline

1. Load Results sheet, `header=1` (double-header Excel file)
2. Binarise: `fan_on = (Cooling_power > 0)`, `heater_on = (Heating_power > 0)`
3. Time-based split — no shuffling:
   - Train: rows 0–6131 (70%, ~8.5 months)
   - Val: rows 6132–7315 (15%)
   - Test: rows 7316–8759 (15%)
4. Fit `StandardScaler` on training set only → save as `models/scaler.pkl`
   - Scales: T_outside, T_inside, T_floor, SR_direct_outside
   - fan_on and heater_on stay as 0/1 (already bounded)
5. Sliding window (length=24):
   - `X[t]`: features[t−24:t], shape (24, 6)
   - `y[t]`: [T_inside[t], T_floor[t]], shape (2,), normalised
   - Result: 6108 train / 1294 val / 1444 test sequences

### Architecture

```
Input (batch, 24, 6)
  → LSTM layer 1  (hidden=128, returns sequence)
  → Dropout 0.2
  → LSTM layer 2  (hidden=128, returns last hidden state only)
  → FC 128→64  (ReLU)
  → FC 64→2    (linear)
Output: [T_inside_next, T_floor_next]  (normalised)
```

### Training config

| Parameter | Value |
|-----------|-------|
| Loss | MSE on normalised targets |
| Optimiser | Adam, lr=1e-3 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping patience | 15 |

**Saved:** `models/lstm_best.pt` (lowest val loss), `models/scaler.pkl`

### MPC autoregressive rollout

```python
window = last_24h_real_data.copy()  # (24, 6), normalised
for step in range(24):
    T_next = lstm(window)           # [T_inside, T_floor], normalised
    new_row = [T_outside_forecast[step], T_next[0], T_next[1],
               SR_forecast[step], fan_schedule[step], heater_schedule[step]]
    window = slide(window, new_row) # drop oldest, append new
```

Errors compound over 24 steps — this is expected. The LSTM only needs to rank schedules, not predict perfectly.

---

## Phase 2 — PPO Agent

### Gymnasium environment

**Observation space:** `Box(shape=(26,), dtype=float32)`
- 24 values: last 24h T_inside (normalised)
- 1 value: current T_outside (normalised)
- 1 value: current SR_direct (normalised)

**Action space:** `Discrete(4)`
- 0 → fan=0, heater=0
- 1 → fan=1, heater=0
- 2 → fan=0, heater=1
- 3 → fan=1, heater=1

**Episode structure:**
- `reset()`: sample random start hour from training data, load real 24h window
- `step(action)`: decode action → run frozen LSTM → slide window → compute reward → return obs
- Episode length: 168 steps (7 simulated days)
- `done=True` after 168 steps

**LSTM is frozen during PPO training** — `lstm.eval()` + `torch.no_grad()` inside `step()`. The physics model must not drift.

### Reward function

```python
cold_dev = max(0, 18 - T_inside)   # °C below comfort band
hot_dev  = max(0, T_inside - 24)   # °C above comfort band

# Comfort: quadratic, asymmetric (cold 3× hot — Edmonton building hit -8°C inside)
r_comfort = +2.0                  if T_inside in [18, 24]
          = -(cold_dev**2) * 3.0  if cold
          = -(hot_dev**2)  * 1.0  if hot

# Inaction: fires when deviation > 3°C AND relevant device is off
r_inaction = -10 * cold_dev  if cold_dev > 3 and heater_on == 0
           = -4  * hot_dev   if hot_dev  > 3 and fan_on   == 0
           = 0               otherwise

# Energy: tiebreaker only (~0.5% of a 3°C comfort penalty)
r_energy = -(0.05 * fan_on + 0.10 * heater_on)

r(t) = r_comfort + r_inaction + r_energy
```

**Key examples:**
- T=21°C, nothing on → r = +2.0 (best case)
- T=13°C, heater=0 → r = −125.0 (inaction fires, agent never wants this)
- T=13°C, heater=1 → r = −75.1 (50pt gap → heater learned)

### Training config (Stable-Baselines3 PPO)

```python
PPO(
    policy="MlpPolicy",       # FC(64,tanh) → FC(64,tanh) → actor/critic heads
    total_timesteps=500_000,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.01,            # entropy bonus prevents collapse to single action
)
```

**Saved:** `models/ppo_hvac.zip`

---

## Phase 3 — MPC (Random Shooting)

No separate training. Uses frozen LSTM at inference time.

### Algorithm

```
Input: current_window (24, 6), weather_forecast (24, 2)

Step 1 — Sample 1024 random schedules
  schedules = np.random.randint(0, 2, size=(1024, 24, 2))  # [fan, heater]

Step 2 — Simulate all 1024 schedules (batched LSTM forward pass)
  for t in range(24):
      lstm_input = build_batch(windows, schedules[:, t])  # (1024, 24, 6)
      T_next = lstm(lstm_input)                           # (1024, 2)
      rewards[:] += gamma**t * reward(T_next[:, 0], schedules[:, t])
      windows = slide_batch(windows, T_next, weather[t], schedules[:, t])

Step 3 — Pick best first action
  best_k = argmax(total_reward)
  mpc_action = schedules[best_k, 0]   # [fan_on, heater_on] for this hour
```

**Performance:** ~80–150ms on CPU (batched across 1024). No GPU required.

**Receding horizon:** Only the first step of the winning schedule is executed. MPC re-runs next hour with fresh sensor data.

---

## Runtime Loop (every hour)

```python
# 1. Get current sensor reading from Node-RED (POST /predict)
obs = build_obs(sensor_data, rolling_window)

# 2. MPC: simulate 1024 schedules, pick best first action
mpc_fan, mpc_heater = mpc.solve(rolling_window, weather_forecast)

# 3. PPO: forward pass
ppo_integer = ppo.predict(obs)
ppo_fan, ppo_heater = decode(ppo_integer)

# 4. Blend 70% MPC + 30% PPO (per device)
fan    = 1 if (0.7*mpc_fan    + 0.3*ppo_fan)    >= 0.5 else 0
heater = 1 if (0.7*mpc_heater + 0.3*ppo_heater) >= 0.5 else 0

# 5. Send command, log experience
return {"fan_on": fan, "heater_on": heater}
```

---

## Flask Server

**Two endpoints:**

`POST /predict`
- Input: `{ T_outside, T_inside, T_floor, SR_direct, timestamp }`
- Server maintains rolling 24h window in memory
- Returns: `{ fan_on: 0|1, heater_on: 0|1 }`

`POST /log`
- Input: `{ ...same fields + T_inside_actual (measured next hour) }`
- Appends one row to `experience.csv`
- Returns: `{ ok: true, rows_logged: N }`

---

## Online Learning Loop

| Trigger | Action |
|---------|--------|
| Every hour | Node-RED calls `/log` with actual outcome → appended to `experience.csv` |
| Every 24h (or 168 new rows) | `retrain.py` fine-tunes LSTM on new rows (10 epochs, lr=1e-4) → server hot-reloads weights |
| Every 7 days | `retrain.py` retrains PPO inside updated LSTM sim → server hot-reloads policy |

**Order:** LSTM retrain always runs before PPO retrain. PPO always trains inside the latest LSTM.

---

## File Structure

```
lstm_ppo_agent/
  data/
    Concrete_floor_results.xlsx
  models/
    lstm_best.pt        ← trained LSTM weights
    scaler.pkl          ← fitted StandardScaler
    ppo_hvac.zip        ← trained PPO policy
  docs/
    superpowers/specs/
      2026-06-11-lstm-ppo-hvac-design.md   ← this file
  train_lstm.py         ← Phase 1: supervised LSTM training
  train_ppo.py          ← Phase 2: RL training in LSTM sim
  mpc.py                ← random shooting MPC
  server.py             ← Flask HTTP server (/predict + /log)
  retrain.py            ← scheduled fine-tuning (LSTM + PPO)
  experience.csv        ← real sensor log for continual learning
```

---

## Future: Option D (Node-RED native inference)

Export path for later:
1. Export LSTM: PyTorch → ONNX → TFJS (`lstm/model.json` + `lstm.weights.bin`)
2. Export PPO policy: PyTorch → ONNX → TFJS (`policy/model.json` + `policy.weights.bin`)
3. Export scaler: `scaler.json` (mean/std arrays)
4. Node-RED custom node loads all three at startup — no Python process at runtime
5. Background Python `retrain.py` periodically overwrites exported files → Node-RED hot-reloads
