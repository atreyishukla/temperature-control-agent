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
    y[i]: ΔT = T(i+seq_len) - T(i+seq_len-1) for [T_inside, T_floor], shape (2,)

    Predicting delta instead of absolute temperature breaks the spurious
    correlation between heater_on=1 and cold conditions in the training data.
    """
    arr = df.values.astype(np.float32)
    n   = len(arr)
    X = np.stack([arr[i : i + seq_len]           for i in range(n - seq_len)])
    y = np.stack([arr[i + seq_len, 1:3] - arr[i + seq_len - 1, 1:3]
                  for i in range(n - seq_len)])
    return X, y


def save_scaler(scaler: StandardScaler, path: str = 'models/scaler.pkl') -> None:
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(path: str = 'models/scaler.pkl') -> StandardScaler:
    with open(path, 'rb') as f:
        return pickle.load(f)
