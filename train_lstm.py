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
        df = load_data(data_path)
        df_train, df_val, _, scaler = split_scale(df)
        save_scaler(scaler, scaler_path)

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
    N_TRIALS = 5
    best_val  = float('inf')
    best_seed = None

    df = load_data(DATA_PATH)
    df_train, df_val, _, scaler = split_scale(df)
    save_scaler(scaler, SCALER_PATH)

    import numpy as np
    for seed in range(N_TRIALS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f'\n--- Trial {seed + 1}/{N_TRIALS} (seed={seed}) ---')
        log = []
        train(df_train=df_train, df_val=df_val,
              model_path=f'models/lstm_trial_{seed}.pt', loss_log=log)
        trial_best = min(log)
        print(f'  best val_loss={trial_best:.6f}')
        if trial_best < best_val:
            best_val  = trial_best
            best_seed = seed

    import shutil
    shutil.copy(f'models/lstm_trial_{best_seed}.pt', MODEL_PATH)
    print(f'\nBest trial: seed={best_seed}, val_loss={best_val:.6f}')
    print(f'Training complete. Model saved to {MODEL_PATH}')
