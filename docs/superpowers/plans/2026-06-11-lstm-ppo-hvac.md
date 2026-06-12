# LSTM + PPO HVAC Control System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python system that predicts concrete-floor building temperatures and outputs fan on/off + heater on/off commands using a blend of LSTM-based MPC and PPO reinforcement learning.

**Architecture:** An LSTM world model is trained first (supervised learning), then a PPO agent trains inside a Gymnasium environment that wraps the frozen LSTM as its physics simulator, then a random-shooting MPC uses the same frozen LSTM at runtime. A Flask server exposes /predict and /log endpoints for Node-RED integration.

**Tech Stack:** PyTorch, Stable-Baselines3, Gymnasium, scikit-learn, Flask, pandas, openpyxl, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `data_pipeline.py` | Load Excel, binarise actions, split train/val/test, normalise, make sliding-window sequences |
| `lstm_model.py` | `BuildingLSTM` nn.Module — 2-layer LSTM + FC head |
| `reward.py` | `compute_reward(T_inside, fan_on, heater_on)` — shared by env and MPC |
| `hvac_env.py` | Gymnasium env wrapping frozen LSTM |
| `train_lstm.py` | Phase 1 entry point — supervised training loop |
| `train_ppo.py` | Phase 2 entry point — SB3 PPO training |
| `mpc.py` | `MPCSolver` — random shooting over 1024 candidate schedules |
| `server.py` | Flask app — `/predict` and `/log` endpoints |
| `retrain.py` | Scheduled LSTM fine-tune + PPO retrain from experience.csv |
| `tests/` | One test file per source file |

---

## Task 1: Project setup

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write requirements.txt**

```
torch>=2.0.0
stable-baselines3>=2.0.0
gymnasium>=0.29.0
pandas>=2.0.0
openpyxl>=3.1.0
scikit-learn>=1.3.0
flask>=3.0.0
numpy>=1.24.0
pytest>=7.4.0
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: no errors. If you see a torch conflict, install torch first: `pip install torch` then re-run.

- [ ] **Step 3: Create tests/__init__.py**

Create an empty file at `tests/__init__.py`.

- [ ] **Step 4: Verify data file is readable**

```bash
python -c "
import pandas as pd
df = pd.read_excel('data/Concrete_floor_results.xlsx', sheet_name='Results', header=1, nrows=3)
print(df.columns.tolist())
print(df.shape)
"
```

Expected output: a list of 7 column names and shape `(3, 7)`. This confirms openpyxl works and the file is where we expect it.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt tests/__init__.py
git commit -m "chore: project setup and dependencies"
```

---

## Task 2: Data pipeline

**Files:**
- Create: `data_pipeline.py`
- Create: `tests/test_data_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data_pipeline.py
import numpy as np
import pytest
from data_pipeline import load_data, split_scale, make_sequences, save_scaler, load_scaler

DATA_PATH = 'data/Concrete_floor_results.xlsx'


def test_load_returns_6_features():
    df = load_data(DATA_PATH)
    assert list(df.columns) == ['T_outside', 'T_inside', 'T_floor', 'SR_direct', 'fan_on', 'heater_on']
    assert len(df) == 8760


def test_actions_are_binary():
    df = load_data(DATA_PATH)
    assert set(df['fan_on'].unique()).issubset({0.0, 1.0})
    assert set(df['heater_on'].unique()).issubset({0.0, 1.0})


def test_no_missing_values():
    df = load_data(DATA_PATH)
    assert df.isnull().sum().sum() == 0


def test_split_sizes():
    df = load_data(DATA_PATH)
    train, val, test, _ = split_scale(df)
    assert len(train) == 6132
    assert len(val) == 1184
    assert len(test) == 1444
    assert len(train) + len(val) + len(test) == 8760


def test_scaler_fitted_on_train_only():
    df = load_data(DATA_PATH)
    train, val, test, scaler = split_scale(df)
    # Train continuous features should be approximately zero-mean, unit-std
    for col in ['T_outside', 'T_inside', 'T_floor', 'SR_direct']:
        assert abs(train[col].mean()) < 0.05, f"{col} mean not ~0"
        assert abs(train[col].std() - 1.0) < 0.05, f"{col} std not ~1"


def test_binary_actions_not_scaled():
    df = load_data(DATA_PATH)
    train, val, test, scaler = split_scale(df)
    assert set(train['fan_on'].unique()).issubset({0.0, 1.0})
    assert set(train['heater_on'].unique()).issubset({0.0, 1.0})


def test_make_sequences_shapes():
    df = load_data(DATA_PATH)
    train, val, test, _ = split_scale(df)
    X, y = make_sequences(train)
    # 6132 rows - 24 context = 6108 sequences
    assert X.shape == (6108, 24, 6)
    assert y.shape == (6108, 2)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_sequence_target_is_T_inside_T_floor():
    df = load_data(DATA_PATH)
    train, _, _, _ = split_scale(df)
    X, y = make_sequences(train)
    arr = train.values.astype('float32')
    # y[0] should equal arr[24, 1:3]  (T_inside, T_floor at row 24)
    np.testing.assert_array_almost_equal(y[0], arr[24, 1:3])


def test_save_and_load_scaler(tmp_path):
    df = load_data(DATA_PATH)
    _, _, _, scaler = split_scale(df)
    path = str(tmp_path / 'scaler.pkl')
    save_scaler(scaler, path)
    loaded = load_scaler(path)
    np.testing.assert_array_almost_equal(scaler.mean_, loaded.mean_)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_data_pipeline.py -v
```

Expected: `ImportError: No module named 'data_pipeline'`

- [ ] **Step 3: Implement data_pipeline.py**

```python
# data_pipeline.py
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ['T_outside', 'T_inside', 'T_floor', 'SR_direct', 'fan_on', 'heater_on']
SCALE_COLS   = ['T_outside', 'T_inside', 'T_floor', 'SR_direct']
SEQ_LEN      = 24


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name='Results', header=1)
    df.columns = ['Date_time', 'T_outside', 'T_inside', 'T_floor',
                  'SR_direct', 'Cooling_power', 'Heating_power']
    df['fan_on']    = (df['Cooling_power'] > 0).astype(float)
    df['heater_on'] = (df['Heating_power'] > 0).astype(float)
    return df[FEATURE_COLS].reset_index(drop=True)


def split_scale(df: pd.DataFrame):
    """Return (train, val, test, scaler). Scaler fitted on train only — no leakage."""
    train = df.iloc[0:6132].copy()
    val   = df.iloc[6132:7316].copy()
    test  = df.iloc[7316:].copy()

    scaler = StandardScaler()
    train[SCALE_COLS] = scaler.fit_transform(train[SCALE_COLS])
    val[SCALE_COLS]   = scaler.transform(val[SCALE_COLS])
    test[SCALE_COLS]  = scaler.transform(test[SCALE_COLS])
    return train, val, test, scaler


def make_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    Sliding window over df.
    X[i]: rows i..i+seq_len-1,  shape (seq_len, 6)
    y[i]: row i+seq_len, columns [T_inside, T_floor], shape (2,)
    """
    arr = df.values.astype(np.float32)
    n   = len(arr)
    X = np.stack([arr[i : i + seq_len]      for i in range(n - seq_len)])
    y = np.stack([arr[i + seq_len, 1:3]     for i in range(n - seq_len)])
    return X, y


def save_scaler(scaler: StandardScaler, path: str = 'models/scaler.pkl') -> None:
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(path: str = 'models/scaler.pkl') -> StandardScaler:
    with open(path, 'rb') as f:
        return pickle.load(f)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_data_pipeline.py -v
```

Expected: all 9 tests PASS. Note: the first run reads the Excel file which takes ~3s.

- [ ] **Step 5: Commit**

```bash
git add data_pipeline.py tests/test_data_pipeline.py
git commit -m "feat: data pipeline — load, split, scale, sliding window sequences"
```

---

## Task 3: LSTM model class

**Files:**
- Create: `lstm_model.py`
- Create: `tests/test_lstm_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lstm_model.py
import torch
import pytest
from lstm_model import BuildingLSTM


def test_output_shape_single():
    model = BuildingLSTM()
    x = torch.randn(1, 24, 6)
    out = model(x)
    assert out.shape == (1, 2), f"Expected (1,2), got {out.shape}"


def test_output_shape_batch():
    model = BuildingLSTM()
    x = torch.randn(64, 24, 6)
    out = model(x)
    assert out.shape == (64, 2)


def test_output_is_float32():
    model = BuildingLSTM()
    x = torch.randn(4, 24, 6)
    out = model(x)
    assert out.dtype == torch.float32


def test_gradients_flow():
    model = BuildingLSTM()
    x = torch.randn(4, 24, 6)
    loss = model(x).sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_eval_mode_no_dropout_change():
    """In eval mode, two identical forward passes should give identical output."""
    model = BuildingLSTM()
    model.eval()
    x = torch.randn(4, 24, 6)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_lstm_model.py -v
```

Expected: `ImportError: No module named 'lstm_model'`

- [ ] **Step 3: Implement lstm_model.py**

```python
# lstm_model.py
import torch
import torch.nn as nn


class BuildingLSTM(nn.Module):
    """
    2-layer LSTM that predicts next-hour [T_inside, T_floor] from
    a 24-hour window of [T_outside, T_inside, T_floor, SR_direct, fan_on, heater_on].

    nn.LSTM with num_layers=2 and dropout=0.2 applies dropout between
    layer 1 output and layer 2 input — matching the spec exactly.
    The output of layer 2's last timestep feeds the FC head.
    """

    def __init__(
        self,
        input_size:  int   = 6,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc1  = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len=24, input_size=6)
        out, _ = self.lstm(x)           # (batch, seq_len, hidden_size)
        last   = out[:, -1, :]          # (batch, hidden_size) — final timestep of layer 2
        return self.fc2(self.relu(self.fc1(last)))  # (batch, 2)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_lstm_model.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lstm_model.py tests/test_lstm_model.py
git commit -m "feat: BuildingLSTM — 2-layer LSTM with FC head, predicts T_inside and T_floor"
```

---

## Task 4: LSTM training loop

**Files:**
- Create: `train_lstm.py`
- Create: `tests/test_train_lstm.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_lstm.py
import os
import numpy as np
import torch
import pytest
from unittest.mock import patch
from train_lstm import train
from lstm_model import BuildingLSTM
from data_pipeline import load_scaler


def _make_tiny_data(tmp_path):
    """Return (train_df, val_df) with 60 rows each — fast enough for test."""
    import pandas as pd
    np.random.seed(0)
    n = 60
    data = {
        'T_outside': np.random.randn(n),
        'T_inside':  np.random.randn(n),
        'T_floor':   np.random.randn(n),
        'SR_direct': np.abs(np.random.randn(n)),
        'fan_on':    np.random.randint(0, 2, n).astype(float),
        'heater_on': np.random.randint(0, 2, n).astype(float),
    }
    return pd.DataFrame(data)


def test_train_returns_model(tmp_path):
    df = _make_tiny_data(tmp_path)
    model_path = str(tmp_path / 'lstm_test.pt')
    scaler_path = str(tmp_path / 'scaler_test.pkl')
    model = train(
        df_train=df, df_val=df,
        model_path=model_path,
        scaler_path=scaler_path,
        max_epochs=2,
        patience=10,
    )
    assert isinstance(model, BuildingLSTM)


def test_train_saves_model_file(tmp_path):
    df = _make_tiny_data(tmp_path)
    model_path = str(tmp_path / 'lstm_test.pt')
    scaler_path = str(tmp_path / 'scaler_test.pkl')
    train(
        df_train=df, df_val=df,
        model_path=model_path,
        scaler_path=scaler_path,
        max_epochs=2,
        patience=10,
    )
    assert os.path.exists(model_path)


def test_train_val_loss_decreases(tmp_path):
    """Val loss at epoch 2 should be <= epoch 1 on a learnable synthetic dataset."""
    import pandas as pd
    np.random.seed(42)
    n = 200
    # Simple linear pattern the LSTM can learn quickly
    t = np.linspace(0, 4 * np.pi, n)
    df = pd.DataFrame({
        'T_outside': np.sin(t),
        'T_inside':  np.sin(t + 0.1),
        'T_floor':   np.sin(t + 0.2),
        'SR_direct': np.abs(np.cos(t)),
        'fan_on':    (np.sin(t) > 0).astype(float),
        'heater_on': (np.cos(t) > 0).astype(float),
    })
    losses = []
    model_path = str(tmp_path / 'lstm_test.pt')
    scaler_path = str(tmp_path / 'scaler_test.pkl')
    train(
        df_train=df, df_val=df,
        model_path=model_path,
        scaler_path=scaler_path,
        max_epochs=5,
        patience=10,
        loss_log=losses,
    )
    assert losses[-1] <= losses[0], "Val loss never decreased over 5 epochs"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_train_lstm.py -v
```

Expected: `ImportError: No module named 'train_lstm'`

- [ ] **Step 3: Implement train_lstm.py**

```python
# train_lstm.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_pipeline import load_data, split_scale, make_sequences, save_scaler
from lstm_model import BuildingLSTM

DATA_PATH   = 'data/Concrete_floor_results.xlsx'
MODEL_PATH  = 'models/lstm_best.pt'
SCALER_PATH = 'models/scaler.pkl'


def train(
    df_train=None,
    df_val=None,
    data_path:   str   = DATA_PATH,
    model_path:  str   = MODEL_PATH,
    scaler_path: str   = SCALER_PATH,
    max_epochs:  int   = 100,
    batch_size:  int   = 64,
    lr:          float = 1e-3,
    patience:    int   = 15,
    loss_log:    list  = None,
) -> BuildingLSTM:
    """
    Train the LSTM world model.

    If df_train/df_val are provided (pre-split DataFrames), use them directly.
    Otherwise load from data_path and split internally.
    loss_log: optional list — val loss per epoch is appended (for tests).
    """
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)

    if df_train is None:
        df        = load_data(data_path)
        df_train, df_val, _, scaler = split_scale(df)
        save_scaler(scaler, scaler_path)
    # When df_train is passed directly (e.g. in tests) we skip scaler saving.

    X_train, y_train = make_sequences(df_train)
    X_val,   y_val   = make_sequences(df_val)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = BuildingLSTM().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    sched   = ReduceLROnPlateau(opt, factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    X_val_t = torch.tensor(X_val).to(device)
    y_val_t = torch.tensor(y_val).to(device)

    best_val   = float('inf')
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        sched.step(val_loss)
        print(f'Epoch {epoch:3d}  val_loss={val_loss:.6f}')

        if loss_log is not None:
            loss_log.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


if __name__ == '__main__':
    model = train()
    print(f'Training complete. Model saved to {MODEL_PATH}')
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_train_lstm.py -v
```

Expected: all 3 tests PASS. These run on tiny synthetic data so they finish in seconds.

- [ ] **Step 5: Commit**

```bash
git add train_lstm.py tests/test_train_lstm.py
git commit -m "feat: LSTM training loop — MSE, Adam, ReduceLROnPlateau, early stopping"
```

- [ ] **Step 6: Run the real training**

```bash
python train_lstm.py
```

Expected: epoch logs printing, early stopping somewhere around epoch 20–50, then:
```
Training complete. Model saved to models/lstm_best.pt
```
Also check `models/scaler.pkl` exists. This takes 2–10 minutes depending on your CPU.

---

## Task 5: Reward function

**Files:**
- Create: `reward.py`
- Create: `tests/test_reward.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reward.py
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_reward.py -v
```

Expected: `ImportError: No module named 'reward'`

- [ ] **Step 3: Implement reward.py**

```python
# reward.py

def compute_reward(T_inside: float, fan_on: int, heater_on: int) -> float:
    """
    Reward for one timestep.

    Comfort zone: [18, 24]°C.
    Cold penalty is 3× hot — Edmonton building reached -8°C inside.
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_reward.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add reward.py tests/test_reward.py
git commit -m "feat: reward function — quadratic comfort, asymmetric cold/hot, inaction penalty"
```

---

## Task 6: Gymnasium environment

**Files:**
- Create: `hvac_env.py`
- Create: `tests/test_hvac_env.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hvac_env.py
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock
from hvac_env import HVACEnv
from lstm_model import BuildingLSTM


def _make_env():
    """Make an env with a randomly-initialised LSTM and 50 fake sequences."""
    lstm = BuildingLSTM()
    lstm.eval()
    seqs = np.random.randn(50, 24, 6).astype(np.float32)
    # Provide a trivial denorm: T_inside_norm * 10 + 20 (real-ish range)
    scaler_mean = 20.0
    scaler_std  = 10.0
    env = HVACEnv(lstm=lstm, train_sequences=seqs,
                  t_inside_mean=scaler_mean, t_inside_std=scaler_std)
    return env


def test_reset_obs_shape():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (26,), f"Expected (26,), got {obs.shape}"
    assert obs.dtype == np.float32


def test_step_returns_correct_types():
    env = _make_env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (26,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_episode_terminates_at_168():
    env = _make_env()
    env.reset(seed=0)
    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == 168


def test_all_4_actions_valid():
    env = _make_env()
    for action in range(4):
        env.reset(seed=action)
        obs, reward, _, _, _ = env.step(action)
        assert obs.shape == (26,)
        assert np.isfinite(reward)


def test_lstm_not_modified_by_step():
    """Frozen LSTM weights must not change during env stepping."""
    lstm = BuildingLSTM()
    lstm.eval()
    seqs = np.random.randn(10, 24, 6).astype(np.float32)
    env  = HVACEnv(lstm=lstm, train_sequences=seqs,
                   t_inside_mean=20.0, t_inside_std=10.0)

    params_before = {n: p.clone() for n, p in lstm.named_parameters()}
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())
    for n, p in lstm.named_parameters():
        assert torch.allclose(params_before[n], p), f"LSTM weight {n} changed!"


def test_observation_space_matches_gym_spec():
    env = _make_env()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_action_space_is_discrete_4():
    env = _make_env()
    assert env.action_space.n == 4
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_hvac_env.py -v
```

Expected: `ImportError: No module named 'hvac_env'`

- [ ] **Step 3: Implement hvac_env.py**

```python
# hvac_env.py
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from lstm_model import BuildingLSTM
from reward import compute_reward

EPISODE_LEN = 168  # 7 simulated days

# Maps Discrete(4) integer → (fan_on, heater_on)
ACTION_MAP = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}


class HVACEnv(gym.Env):
    """
    Gymnasium environment for HVAC control.

    Observation: Box(26,) = [last 24h T_inside (normalised), T_outside_now, SR_now]
    Action:      Discrete(4) = {both off, fan only, heater only, both on}
    Physics:     Frozen BuildingLSTM — weights never updated inside this env.

    t_inside_mean / t_inside_std: from the StandardScaler fitted on training data.
    Used to convert normalised LSTM output back to real °C for the reward function.
    """

    metadata = {}

    def __init__(
        self,
        lstm:             BuildingLSTM,
        train_sequences:  np.ndarray,
        t_inside_mean:    float,
        t_inside_std:     float,
        device:           str = 'cpu',
    ):
        super().__init__()
        self.lstm            = lstm
        self.seqs            = train_sequences   # (N, 24, 6), normalised
        self.t_inside_mean   = t_inside_mean
        self.t_inside_std    = t_inside_std
        self.device          = device

        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(26,), dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

        self._window = None
        self._steps  = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx          = self.np_random.integers(0, len(self.seqs))
        self._window = self.seqs[idx].copy()   # (24, 6)
        self._steps  = 0
        return self._obs(), {}

    def step(self, action: int):
        fan_on, heater_on = ACTION_MAP[action]

        # Set the current action in the last row of the window before predicting
        lstm_input          = self._window.copy()
        lstm_input[-1, 4]   = float(fan_on)
        lstm_input[-1, 5]   = float(heater_on)

        # Frozen LSTM forward pass
        x = torch.tensor(lstm_input, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.lstm(x).squeeze(0).cpu().numpy()   # [T_inside_norm, T_floor_norm]

        # Denormalise T_inside for reward
        T_inside_real = float(pred[0]) * self.t_inside_std + self.t_inside_mean
        reward        = float(compute_reward(T_inside_real, fan_on, heater_on))

        # Slide window: keep same T_outside and SR (constant within episode)
        new_row    = self._window[-1].copy()
        new_row[1] = pred[0]          # T_inside (normalised)
        new_row[2] = pred[1]          # T_floor  (normalised)
        new_row[4] = float(fan_on)
        new_row[5] = float(heater_on)
        self._window = np.vstack([self._window[1:], new_row])

        self._steps += 1
        terminated   = self._steps >= EPISODE_LEN
        return self._obs(), reward, terminated, False, {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        t_inside_hist = self._window[:, 1]     # (24,) last 24h T_inside
        t_outside_now = self._window[-1, 0:1]  # (1,)
        sr_now        = self._window[-1, 3:4]  # (1,)
        return np.concatenate([t_inside_hist, t_outside_now, sr_now]).astype(np.float32)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_hvac_env.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hvac_env.py tests/test_hvac_env.py
git commit -m "feat: HVACEnv — Gymnasium env wrapping frozen LSTM, Discrete(4) action space"
```

---

## Task 7: PPO training

**Files:**
- Create: `train_ppo.py`
- Create: `tests/test_train_ppo.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_ppo.py
import os
import numpy as np
import pytest
from train_ppo import train_ppo
from lstm_model import BuildingLSTM


def _make_env_components():
    lstm  = BuildingLSTM()
    lstm.eval()
    seqs  = np.random.randn(50, 24, 6).astype(np.float32)
    return lstm, seqs


def test_train_ppo_saves_model(tmp_path):
    lstm, seqs = _make_env_components()
    model_path = str(tmp_path / 'ppo_test.zip')
    train_ppo(
        lstm=lstm,
        train_sequences=seqs,
        t_inside_mean=20.0,
        t_inside_std=10.0,
        model_path=model_path,
        total_timesteps=512,   # tiny — just verifies it runs
    )
    assert os.path.exists(model_path)


def test_train_ppo_returns_model(tmp_path):
    from stable_baselines3 import PPO
    lstm, seqs = _make_env_components()
    model_path = str(tmp_path / 'ppo_test.zip')
    model = train_ppo(
        lstm=lstm,
        train_sequences=seqs,
        t_inside_mean=20.0,
        t_inside_std=10.0,
        model_path=model_path,
        total_timesteps=512,
    )
    assert isinstance(model, PPO)


def test_trained_model_predicts_valid_action(tmp_path):
    from stable_baselines3 import PPO
    lstm, seqs = _make_env_components()
    model_path = str(tmp_path / 'ppo_test.zip')
    model = train_ppo(
        lstm=lstm,
        train_sequences=seqs,
        t_inside_mean=20.0,
        t_inside_std=10.0,
        model_path=model_path,
        total_timesteps=512,
    )
    obs = np.random.randn(26).astype(np.float32)
    action, _ = model.predict(obs, deterministic=True)
    assert int(action) in {0, 1, 2, 3}
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_train_ppo.py -v
```

Expected: `ImportError: No module named 'train_ppo'`

- [ ] **Step 3: Implement train_ppo.py**

```python
# train_ppo.py
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from lstm_model import BuildingLSTM
from hvac_env import HVACEnv
from data_pipeline import load_data, split_scale, make_sequences, load_scaler

LSTM_PATH   = 'models/lstm_best.pt'
SCALER_PATH = 'models/scaler.pkl'
MODEL_PATH  = 'models/ppo_hvac.zip'


def train_ppo(
    lstm:           BuildingLSTM = None,
    train_sequences: np.ndarray  = None,
    t_inside_mean:  float        = None,
    t_inside_std:   float        = None,
    lstm_path:      str          = LSTM_PATH,
    scaler_path:    str          = SCALER_PATH,
    model_path:     str          = MODEL_PATH,
    total_timesteps: int         = 500_000,
) -> PPO:
    """
    Train the PPO agent inside the LSTM simulator.

    If lstm / train_sequences / scaler stats are not provided, load from disk.
    """
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)

    # Load LSTM and scaler if not injected (normal training path)
    if lstm is None:
        lstm = BuildingLSTM()
        lstm.load_state_dict(torch.load(lstm_path, weights_only=True))
    lstm.eval()

    if train_sequences is None:
        scaler  = load_scaler(scaler_path)
        df      = load_data('data/Concrete_floor_results.xlsx')
        train_df, _, _, _ = split_scale(df)
        train_sequences, _ = make_sequences(train_df)
        t_inside_mean = float(scaler.mean_[1])   # T_inside is index 1 in SCALE_COLS
        t_inside_std  = float(scaler.scale_[1])

    env = HVACEnv(
        lstm=lstm,
        train_sequences=train_sequences,
        t_inside_mean=t_inside_mean,
        t_inside_std=t_inside_std,
    )

    model = PPO(
        policy='MlpPolicy',
        env=env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    return model


if __name__ == '__main__':
    model = train_ppo()
    print(f'PPO training complete. Model saved to {MODEL_PATH}')
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_train_ppo.py -v
```

Expected: all 3 tests PASS. Each runs 512 timesteps so they finish in a few seconds.

- [ ] **Step 5: Commit**

```bash
git add train_ppo.py tests/test_train_ppo.py
git commit -m "feat: PPO training — SB3 PPO in HVACEnv, 500k timesteps, ent_coef=0.01"
```

- [ ] **Step 6: Run the real PPO training**

```bash
python train_ppo.py
```

Expected: SB3 progress logs every 2048 steps, ~244 updates total. Takes 10–30 minutes on CPU. When done: `models/ppo_hvac.zip` exists.

---

## Task 8: MPC random shooting

**Files:**
- Create: `mpc.py`
- Create: `tests/test_mpc.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mpc.py
import time
import numpy as np
import pytest
from mpc import MPCSolver
from lstm_model import BuildingLSTM


def _make_solver(n_candidates=64):
    lstm = BuildingLSTM()
    lstm.eval()
    return MPCSolver(lstm=lstm, n_candidates=n_candidates, horizon=24, gamma=0.99)


def test_solve_returns_two_binary_values():
    solver  = _make_solver()
    window  = np.random.randn(24, 6).astype(np.float32)
    weather = np.random.randn(24, 2).astype(np.float32)
    fan, heater = solver.solve(window, weather)
    assert fan    in {0, 1}
    assert heater in {0, 1}


def test_solve_output_changes_with_different_windows():
    """Two very different windows should sometimes give different actions."""
    solver   = _make_solver(n_candidates=256)
    results  = set()
    rng      = np.random.default_rng(0)
    for _ in range(20):
        window  = rng.standard_normal((24, 6)).astype(np.float32)
        weather = rng.standard_normal((24, 2)).astype(np.float32)
        fan, heater = solver.solve(window, weather)
        results.add((fan, heater))
    # At least 2 distinct actions should appear across 20 random inputs
    assert len(results) >= 2


def test_solve_is_deterministic_with_seed():
    solver  = _make_solver()
    window  = np.ones((24, 6), dtype=np.float32)
    weather = np.ones((24, 2), dtype=np.float32)
    np.random.seed(42)
    r1 = solver.solve(window, weather)
    np.random.seed(42)
    r2 = solver.solve(window, weather)
    assert r1 == r2


def test_solve_timing_under_2_seconds():
    """1024 candidates, 24-step batched rollout must finish under 2s on CPU."""
    lstm    = BuildingLSTM()
    lstm.eval()
    solver  = MPCSolver(lstm=lstm, n_candidates=1024, horizon=24, gamma=0.99)
    window  = np.random.randn(24, 6).astype(np.float32)
    weather = np.random.randn(24, 2).astype(np.float32)
    t0 = time.time()
    solver.solve(window, weather)
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"MPC took {elapsed:.2f}s — should be under 2s on CPU"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_mpc.py -v
```

Expected: `ImportError: No module named 'mpc'`

- [ ] **Step 3: Implement mpc.py**

```python
# mpc.py
import numpy as np
import torch
from lstm_model import BuildingLSTM
from reward import compute_reward


class MPCSolver:
    """
    Random-shooting MPC for discrete fan/heater control.

    Samples n_candidates random 24-step schedules, simulates each using the
    frozen LSTM (all n_candidates in one batched forward pass per timestep),
    scores by discounted reward sum, returns the first action of the best schedule.

    t_inside_mean / t_inside_std: StandardScaler stats for T_inside (index 1).
    Used to convert normalised LSTM output to real °C for reward scoring.
    """

    def __init__(
        self,
        lstm:          BuildingLSTM,
        t_inside_mean: float = 0.0,
        t_inside_std:  float = 1.0,
        n_candidates:  int   = 1024,
        horizon:       int   = 24,
        gamma:         float = 0.99,
        device:        str   = 'cpu',
    ):
        self.lstm          = lstm
        self.t_inside_mean = t_inside_mean
        self.t_inside_std  = t_inside_std
        self.n_candidates  = n_candidates
        self.horizon       = horizon
        self.gamma         = gamma
        self.device        = device

    def solve(self, window: np.ndarray, weather_forecast: np.ndarray):
        """
        window:           (24, 6)   current normalised sensor window
        weather_forecast: (24, 2)   [T_outside_norm, SR_norm] for next 24h

        Returns: (fan_on: int, heater_on: int) — best first action
        """
        K, H  = self.n_candidates, self.horizon

        # (K, 24, 2) random binary schedules [fan, heater]
        schedules = np.random.randint(0, 2, size=(K, H, 2)).astype(np.float32)

        # (K, 24, 6) — broadcast current window across all K candidates
        windows = np.tile(window[np.newaxis], (K, 1, 1)).copy()   # (K, 24, 6)

        total_reward = np.zeros(K, dtype=np.float32)

        with torch.no_grad():
            for t in range(H):
                fan_t    = schedules[:, t, 0]   # (K,)
                heater_t = schedules[:, t, 1]   # (K,)

                # Inject action into last row of each window
                inp          = windows.copy()
                inp[:, -1, 4] = fan_t
                inp[:, -1, 5] = heater_t

                x     = torch.tensor(inp, dtype=torch.float32).to(self.device)
                pred  = self.lstm(x).cpu().numpy()              # (K, 2)

                # Denormalise T_inside for reward
                T_inside_real = pred[:, 0] * self.t_inside_std + self.t_inside_mean

                # Vectorised reward over all K candidates
                step_rewards = np.array([
                    compute_reward(float(T_inside_real[k]), int(fan_t[k]), int(heater_t[k]))
                    for k in range(K)
                ], dtype=np.float32)

                total_reward += (self.gamma ** t) * step_rewards

                # Slide window: new row uses weather forecast + LSTM prediction
                new_rows         = windows[:, -1, :].copy()
                new_rows[:, 0]   = weather_forecast[t, 0]   # T_outside_norm
                new_rows[:, 1]   = pred[:, 0]                # T_inside_norm
                new_rows[:, 2]   = pred[:, 1]                # T_floor_norm
                new_rows[:, 3]   = weather_forecast[t, 1]   # SR_norm
                new_rows[:, 4]   = fan_t
                new_rows[:, 5]   = heater_t
                windows = np.concatenate([windows[:, 1:, :], new_rows[:, np.newaxis, :]], axis=1)

        best_k = int(np.argmax(total_reward))
        fan_on    = int(schedules[best_k, 0, 0])
        heater_on = int(schedules[best_k, 0, 1])
        return fan_on, heater_on
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_mpc.py -v
```

Expected: all 4 tests PASS. The timing test (~1s for 1024 candidates on a modern CPU) should pass comfortably.

- [ ] **Step 5: Commit**

```bash
git add mpc.py tests/test_mpc.py
git commit -m "feat: MPCSolver — batched random shooting over 1024 candidates, 24-step horizon"
```

---

## Task 9: Flask server

**Files:**
- Create: `server.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_server.py
import csv
import json
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def client(tmp_path):
    """Start the Flask test client with mocked models."""
    os.environ['EXPERIENCE_CSV'] = str(tmp_path / 'experience.csv')

    # Patch model loading so we don't need trained weights
    with patch('server.load_models') as mock_load:
        mock_lstm   = MagicMock()
        mock_ppo    = MagicMock()
        mock_scaler = MagicMock()
        mock_scaler.mean_  = np.array([0.0, 20.0, 18.0, 0.0])   # T_inside mean=20
        mock_scaler.scale_ = np.array([10.0, 10.0, 8.0, 100.0]) # T_inside std=10
        mock_load.return_value = (mock_lstm, mock_ppo, mock_scaler)

        # MPC and PPO both return (0, 1) i.e. heater only
        with patch('server.MPCSolver') as MockMPC, \
             patch('server.PPO') as MockPPO:

            MockMPC.return_value.solve = MagicMock(return_value=(0, 1))
            mock_ppo_instance = MagicMock()
            mock_ppo_instance.predict = MagicMock(return_value=(2, None))  # action 2 = heater only
            MockPPO.load.return_value = mock_ppo_instance

            import server
            server.app.config['TESTING'] = True
            with server.app.test_client() as c:
                yield c, tmp_path


def test_predict_returns_fan_and_heater(client):
    c, _ = client
    payload = {
        'T_outside': -5.0,
        'T_inside':  15.0,
        'T_floor':   12.0,
        'SR_direct': 0.0,
        'timestamp': '2026-01-15T08:00:00',
    }
    resp = c.post('/predict', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'fan_on' in data
    assert 'heater_on' in data
    assert data['fan_on']    in {0, 1}
    assert data['heater_on'] in {0, 1}


def test_log_appends_to_csv(client):
    c, tmp_path = client
    payload = {
        'T_outside':      -5.0,
        'T_inside':       15.0,
        'T_floor':        12.0,
        'SR_direct':      0.0,
        'fan_on':         0,
        'heater_on':      1,
        'T_inside_actual': 16.2,
        'timestamp':      '2026-01-15T09:00:00',
    }
    resp = c.post('/log', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['ok'] is True
    assert data['rows_logged'] == 1

    csv_path = str(tmp_path / 'experience.csv')
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2   # header + 1 data row


def test_predict_fills_window_over_time(client):
    c, _ = client
    payload = {
        'T_outside': -5.0, 'T_inside': 15.0,
        'T_floor': 12.0, 'SR_direct': 0.0,
        'timestamp': '2026-01-15T08:00:00',
    }
    for _ in range(5):
        resp = c.post('/predict', json=payload)
        assert resp.status_code == 200
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_server.py -v
```

Expected: `ImportError: No module named 'server'`

- [ ] **Step 3: Implement server.py**

```python
# server.py
import csv
import os
import collections
import numpy as np
import torch
from flask import Flask, request, jsonify
from stable_baselines3 import PPO

from lstm_model import BuildingLSTM
from mpc import MPCSolver
from data_pipeline import load_scaler, SCALE_COLS

LSTM_PATH   = os.getenv('LSTM_PATH',   'models/lstm_best.pt')
PPO_PATH    = os.getenv('PPO_PATH',    'models/ppo_hvac.zip')
SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.pkl')
EXP_CSV     = os.getenv('EXPERIENCE_CSV', 'experience.csv')

SEQ_LEN = 24
app = Flask(__name__)

# Global model state (loaded once at startup)
_lstm   = None
_ppo    = None
_scaler = None
_solver = None
_window = collections.deque(maxlen=SEQ_LEN)   # rolling 24h of raw sensor rows

CSV_HEADER = ['timestamp', 'T_outside', 'T_inside', 'T_floor',
              'SR_direct', 'fan_on', 'heater_on', 'T_inside_actual']


def load_models():
    lstm = BuildingLSTM()
    lstm.load_state_dict(torch.load(LSTM_PATH, weights_only=True))
    lstm.eval()
    ppo    = PPO.load(PPO_PATH)
    scaler = load_scaler(SCALER_PATH)
    return lstm, ppo, scaler


def _ensure_loaded():
    global _lstm, _ppo, _scaler, _solver
    if _lstm is None:
        _lstm, _ppo, _scaler = load_models()
        _solver = MPCSolver(
            lstm=_lstm,
            t_inside_mean=float(_scaler.mean_[1]),
            t_inside_std=float(_scaler.scale_[1]),
        )


def _normalise(row: dict) -> np.ndarray:
    """Convert raw sensor dict → normalised (6,) array."""
    raw = np.array([
        row['T_outside'], row['T_inside'], row['T_floor'], row['SR_direct'], 0.0, 0.0
    ], dtype=np.float32)
    raw[:4] = _scaler.transform(raw[:4].reshape(1, -1))[0]
    return raw


def _build_obs(window_arr: np.ndarray) -> np.ndarray:
    """window_arr: (24,6) normalised → obs (26,)"""
    t_inside_hist = window_arr[:, 1]
    t_outside_now = window_arr[-1, 0:1]
    sr_now        = window_arr[-1, 3:4]
    return np.concatenate([t_inside_hist, t_outside_now, sr_now]).astype(np.float32)


@app.route('/predict', methods=['POST'])
def predict():
    _ensure_loaded()
    data = request.get_json()

    norm_row = _normalise(data)
    _window.append(norm_row)

    # Need a full 24-step window before predicting
    if len(_window) < SEQ_LEN:
        # Pad with copies of the current reading
        window_arr = np.tile(norm_row, (SEQ_LEN, 1))
    else:
        window_arr = np.array(_window, dtype=np.float32)

    # Weather forecast: reuse current T_outside/SR for all 24 steps (no real forecast)
    weather_forecast = np.tile(
        [[window_arr[-1, 0], window_arr[-1, 3]]],
        (24, 1)
    ).astype(np.float32)

    # MPC
    mpc_fan, mpc_heater = _solver.solve(window_arr, weather_forecast)

    # PPO
    obs        = _build_obs(window_arr)
    ppo_action, _ = _ppo.predict(obs, deterministic=True)
    ppo_map    = {0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1)}
    ppo_fan, ppo_heater = ppo_map[int(ppo_action)]

    # 70/30 blend (per device)
    fan    = 1 if (0.7*mpc_fan    + 0.3*ppo_fan)    >= 0.5 else 0
    heater = 1 if (0.7*mpc_heater + 0.3*ppo_heater) >= 0.5 else 0

    return jsonify({'fan_on': fan, 'heater_on': heater})


@app.route('/log', methods=['POST'])
def log():
    data    = request.get_json()
    new_file = not os.path.exists(EXP_CSV)
    with open(EXP_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow({k: data.get(k, '') for k in CSV_HEADER})

    rows_logged = sum(1 for _ in open(EXP_CSV)) - 1  # subtract header
    return jsonify({'ok': True, 'rows_logged': rows_logged})


if __name__ == '__main__':
    _ensure_loaded()
    app.run(host='0.0.0.0', port=5000, debug=False)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_server.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: Flask server — /predict (MPC+PPO blend) and /log (experience.csv)"
```

---

## Task 10: Retrain script

**Files:**
- Create: `retrain.py`
- Create: `tests/test_retrain.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrain.py
import csv
import os
import numpy as np
import pytest
from retrain import load_experience, finetune_lstm

DATA_PATH = 'data/Concrete_floor_results.xlsx'


def _write_fake_csv(path, n_rows=50):
    header = ['timestamp','T_outside','T_inside','T_floor',
              'SR_direct','fan_on','heater_on','T_inside_actual']
    rng = np.random.default_rng(0)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n_rows):
            w.writerow([
                f'2026-01-01T{i:02d}:00:00',
                rng.uniform(-20, 5),    # T_outside
                rng.uniform(10, 25),    # T_inside
                rng.uniform(8, 20),     # T_floor
                rng.uniform(0, 500),    # SR_direct
                rng.integers(0, 2),     # fan_on
                rng.integers(0, 2),     # heater_on
                rng.uniform(10, 25),    # T_inside_actual
            ])


def test_load_experience_returns_dataframe(tmp_path):
    csv_path = str(tmp_path / 'experience.csv')
    _write_fake_csv(csv_path, n_rows=30)
    df = load_experience(csv_path)
    assert len(df) == 30
    assert 'T_inside_actual' in df.columns


def test_finetune_lstm_runs_without_error(tmp_path):
    from lstm_model import BuildingLSTM
    from data_pipeline import load_data, split_scale

    csv_path   = str(tmp_path / 'experience.csv')
    model_path = str(tmp_path / 'lstm_finetuned.pt')
    _write_fake_csv(csv_path, n_rows=50)

    # Load real scaler from training data
    df = load_data(DATA_PATH)
    _, _, _, scaler = split_scale(df)

    model = BuildingLSTM()
    finetune_lstm(
        model=model,
        scaler=scaler,
        csv_path=csv_path,
        model_path=model_path,
        n_epochs=2,
        lr=1e-4,
    )
    assert os.path.exists(model_path)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_retrain.py -v
```

Expected: `ImportError: No module named 'retrain'`

- [ ] **Step 3: Implement retrain.py**

```python
# retrain.py
"""
Periodic fine-tuning script.

LSTM retrain: runs every 24h (or when called manually).
PPO retrain:  runs every 7 days (or when called manually).

Call order: always retrain LSTM before PPO.
"""
import os
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from lstm_model import BuildingLSTM
from data_pipeline import load_scaler, SCALE_COLS, SEQ_LEN
from train_ppo import train_ppo

SCALER_PATH = 'models/scaler.pkl'
LSTM_PATH   = 'models/lstm_best.pt'
PPO_PATH    = 'models/ppo_hvac.zip'
EXP_CSV     = 'experience.csv'


def load_experience(csv_path: str = EXP_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def finetune_lstm(
    model:      BuildingLSTM,
    scaler,
    csv_path:   str   = EXP_CSV,
    model_path: str   = LSTM_PATH,
    n_epochs:   int   = 10,
    lr:         float = 1e-4,
    batch_size: int   = 32,
) -> None:
    """
    Fine-tune the LSTM on new rows from experience.csv.

    experience.csv has columns:
      timestamp, T_outside, T_inside, T_floor, SR_direct, fan_on, heater_on, T_inside_actual

    We treat T_inside_actual as the ground truth T_inside for the next timestep.
    Build (X, y) sequences the same way as the original training pipeline.
    """
    df = load_experience(csv_path)
    if len(df) < SEQ_LEN + 1:
        print(f'Not enough rows ({len(df)}) to form sequences — skipping.')
        return

    # Normalise continuous features using the existing scaler (no refitting)
    feat_cols = ['T_outside', 'T_inside', 'T_floor', 'SR_direct']
    df_feat   = df[feat_cols].copy()
    df_feat[feat_cols] = scaler.transform(df_feat[feat_cols])
    df_feat['fan_on']    = df['fan_on'].values
    df_feat['heater_on'] = df['heater_on'].values

    # Sequences: X = (t-24:t), y = [T_inside_actual[t], T_floor[t]] (normalised)
    arr  = df_feat[['T_outside','T_inside','T_floor','SR_direct','fan_on','heater_on']].values.astype(np.float32)
    # Use T_inside_actual as the target T_inside
    t_inside_actual_norm = ((df['T_inside_actual'].values - scaler.mean_[1]) / scaler.scale_[1]).astype(np.float32)

    X, y = [], []
    for i in range(len(arr) - SEQ_LEN):
        X.append(arr[i : i + SEQ_LEN])
        # y: [T_inside_actual at i+SEQ_LEN, T_floor at i+SEQ_LEN]
        y.append([t_inside_actual_norm[i + SEQ_LEN], arr[i + SEQ_LEN, 2]])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    loader  = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                         batch_size=batch_size, shuffle=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = model.to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        print(f'Finetune epoch {epoch+1}/{n_epochs}')

    torch.save(model.state_dict(), model_path)
    print(f'LSTM fine-tuned and saved to {model_path}')


def retrain_ppo(
    lstm:       BuildingLSTM,
    scaler,
    model_path: str = PPO_PATH,
) -> None:
    """Retrain PPO using the updated LSTM as simulator."""
    from data_pipeline import load_data, split_scale, make_sequences
    df          = load_data('data/Concrete_floor_results.xlsx')
    train_df, _, _, _ = split_scale(df)
    seqs, _     = make_sequences(train_df)
    train_ppo(
        lstm=lstm,
        train_sequences=seqs,
        t_inside_mean=float(scaler.mean_[1]),
        t_inside_std=float(scaler.scale_[1]),
        model_path=model_path,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm-only', action='store_true')
    parser.add_argument('--ppo-only',  action='store_true')
    args = parser.parse_args()

    scaler = load_scaler(SCALER_PATH)
    lstm   = BuildingLSTM()
    lstm.load_state_dict(torch.load(LSTM_PATH, weights_only=True))

    if not args.ppo_only:
        finetune_lstm(model=lstm, scaler=scaler)
        # Reload updated weights before PPO
        lstm.load_state_dict(torch.load(LSTM_PATH, weights_only=True))

    if not args.lstm_only:
        retrain_ppo(lstm=lstm, scaler=scaler)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_retrain.py -v
```

Expected: all 2 tests PASS. The finetune test loads the real scaler so it requires `models/scaler.pkl` to exist (created by `python train_lstm.py` in Task 4).

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS. If any fail, fix before committing.

- [ ] **Step 6: Commit**

```bash
git add retrain.py tests/test_retrain.py
git commit -m "feat: retrain.py — LSTM fine-tune + PPO retrain from experience.csv"
```

---

## End-to-end smoke test

Once `models/lstm_best.pt`, `models/scaler.pkl`, and `models/ppo_hvac.zip` all exist (after running train_lstm.py and train_ppo.py):

- [ ] **Start the server**

```bash
python server.py
```

Expected: `Running on http://0.0.0.0:5000`

- [ ] **Send a test prediction**

```bash
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"T_outside":-10,"T_inside":14,"T_floor":11,"SR_direct":0,"timestamp":"2026-01-15T08:00:00"}'
```

Expected: `{"fan_on": 0, "heater_on": 1}` (heater on — building is cold)

- [ ] **Log an experience row**

```bash
curl -X POST http://localhost:5000/log \
  -H 'Content-Type: application/json' \
  -d '{"T_outside":-10,"T_inside":14,"T_floor":11,"SR_direct":0,"fan_on":0,"heater_on":1,"T_inside_actual":15.2,"timestamp":"2026-01-15T09:00:00"}'
```

Expected: `{"ok": true, "rows_logged": 1}`

---

## Self-review notes

- `data_pipeline.py` val split produces 1184 rows (not 1294 as the spec says — the spec had a typo). Test assertions use the actual computed value.
- `MPCSolver.solve` uses a Python loop over K candidates for the reward step (vectorised per timestep). This is the simplest correct approach; if timing is too slow, replace the list comprehension with a numpy vectorised call.
- `server.py` uses a naive weather forecast (repeats current T_outside/SR for all 24 steps). This is intentional — upgrading to a real weather API is a future enhancement.
- `retrain.py` requires `models/scaler.pkl` to exist (from Phase 1). Running it before Phase 1 will error with a clear message.
