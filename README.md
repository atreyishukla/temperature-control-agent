# Temperature Control Agent

An HVAC control system for a concrete floor radiant heating building. Uses an LSTM world model trained on historical sensor data, a PPO reinforcement learning agent, and a Model Predictive Control (MPC) optimizer to decide when to run the heater and cooling fan each hour.

**Result:** MPC achieves **57% comfort zone time** vs **11.8%** with the original historical control strategy — a 4.8× improvement.

---

## How It Works

```
Sensor readings (hourly)
        │
        ▼
  Flask server (/predict)
        │
   90% MPC  +  10% PPO
        │
  LSTM world model
  (predicts ΔT next hour)
        │
   Best action → fan_on / heater_on
```

**LSTM world model** predicts `ΔT_inside` and `ΔT_floor` for the next hour given the last 24 hours of sensor history. Trained with two physical constraints baked in:
- Heater effect is always ≥ 0 (heater only warms — softplus constraint)
- Fan effect is always ≤ 0 (fan is cooling power — −softplus constraint)
- Current actions are zeroed out before the LSTM sees the window, preventing spurious correlations between action and temperature level

**MPC** generates 256 random 4-step action sequences, rolls each through the LSTM, picks the sequence with the highest discounted reward.

**PPO** is trained inside the frozen LSTM simulator for 500k timesteps and runs as a fallback (10% of requests).

**Online retraining:** LSTM fine-tunes every 24 h on new sensor readings logged via `/log`. PPO retrains every 7 days.

---

## Comfort Zone & Reward

Target: **18–24°C**

| Condition | Penalty |
|---|---|
| T in 18–24°C | +2.0 |
| T < 18°C | −(18−T)² × 3 |
| T > 24°C | −(T−24)² |
| T < 15°C and heater off | −10 × (15−T) extra |
| T > 27°C and fan off | −4 × (T−27) extra |
| Fan on | −0.05 |
| Heater on | −0.10 |

Cold is penalised 3× harder than hot, reflecting the Edmonton climate (building reached −8°C inside historically).

---

## Evaluation Results

| Strategy | In comfort | Mean T | Mean reward |
|---|---|---|---|
| Historical (human operator) | 11.8% | 10.2°C | −427 |
| Always off | 11.0% | 9.8°C | −450 |
| Always heat | 32.4% | 17.6°C | −103 |
| **MPC (this system)** | **57.0%** | **17.5°C** | **−48** |
| PPO (this system) | 39.0% | 15.8°C | −99 |

---

## Project Structure

```
├── data_pipeline.py      # Load Excel, split train/val/test, make sequences
├── lstm_model.py         # 2-layer LSTM with causal action separation
├── train_lstm.py         # Supervised training with early stopping
├── hvac_env.py           # Gymnasium environment wrapping the LSTM
├── train_ppo.py          # PPO training via Stable-Baselines3
├── mpc.py                # Random-shooting MPC over frozen LSTM
├── reward.py             # Comfort + energy reward function
├── server.py             # Flask API: POST /predict and POST /log
├── retrain.py            # Online LSTM fine-tune + PPO retrain scheduler
├── evaluate.py           # Offline evaluation on held-out test split
├── seed_experience.py    # Bootstrap experience.csv from historical data
├── tests/                # Pytest suite (72 tests)
├── nodered_manual_input.json   # Node-RED flow with manual temperature inputs
├── nodered_simple_test.json    # Node-RED flow with random test data
└── nodered_flow.json           # Node-RED production flow (hourly)
```

---

## Setup

**Requirements:** Python 3.10+, the Excel dataset (`data/Concrete_floor_results.xlsx`)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Training

```bash
# 1. Train LSTM world model (~5 min on CPU)
python train_lstm.py

# 2. Train PPO agent inside LSTM simulator (~10 min on CPU)
python train_ppo.py

# 3. Evaluate on held-out test split
python evaluate.py
```

Trained weights are saved to `models/lstm_best.pt`, `models/ppo_hvac.zip`, `models/scaler.pkl`.

---

## Running the Server

```bash
python server.py
# Listening on http://0.0.0.0:5001
```

**POST /predict** — returns fan/heater decision

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"T_outside": 3.0, "T_inside": 10.0, "T_floor": 10.0, "SR_direct": 0.0}'
```

```json
{"action": 2, "fan_on": 0, "heater_on": 1, "source": "mpc"}
```

**POST /log** — log a sensor reading to `logs/experience.csv` for online retraining

```bash
curl -X POST http://localhost:5001/log \
  -H "Content-Type: application/json" \
  -d '{"T_outside": 3.0, "T_inside": 10.0, "T_floor": 10.0, "SR_direct": 0.0, "fan_on": 0, "heater_on": 1}'
```

---

## Node-RED Integration

Import any of the three flow files via the Node-RED hamburger menu → Import.

- **nodered_manual_input.json** — 5 preset temperature scenarios + a custom input node you can edit
- **nodered_simple_test.json** — inject node → random sensor data → `/predict` → debug
- **nodered_flow.json** — production flow that polls sensors every hour

The server must be running on port 5001 before deploying flows.

---

## Online Retraining

Run `retrain.py` once per hour from cron or a Node-RED inject node:

```bash
python retrain.py
```

- LSTM fine-tunes every 24 h if ≥ 50 new rows exist in `logs/experience.csv`
- PPO retrains from scratch every 7 days inside the updated LSTM simulator

---

## Tests

```bash
pytest --tb=short -q
# 72 passed
```

---

## Features

| Column | Description |
|---|---|
| T_outside | Outdoor temperature (°C) |
| T_inside | Indoor air temperature (°C) |
| T_floor | Concrete floor surface temperature (°C) |
| SR_direct | Direct solar radiation (W/m²) |
| fan_on | Cooling fan active (0/1) |
| heater_on | Floor heater active (0/1) |
| hour_sin / hour_cos | Time of day encoded as sin/cos cycle |
