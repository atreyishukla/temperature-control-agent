import pickle, torch, numpy as np
from data_pipeline import load_data, split_scale, make_sequences
from lstm_model import BuildingLSTM, load_ensemble

df = load_data('data/Concrete_floor_results.xlsx')
_, _, test, scaler = split_scale(df)
X, y = make_sequences(test)
X_t  = torch.tensor(X)

with open('models/scaler.pkl', 'rb') as f: sc = pickle.load(f)
t_std = sc.scale_[1]

def report(name, model):
    model.eval()
    with torch.no_grad():
        pred = model(X_t).numpy()
    err      = abs(pred[:, 0] - y[:, 0]) * t_std
    print(f"\n{name}")
    print(f"  MAE:          {err.mean():.3f}°C")
    print(f"  Within ±0.5°C: {(err < 0.5).mean() * 100:.1f}%")
    print(f"  Within ±1.0°C: {(err < 1.0).mean() * 100:.1f}%")

single = BuildingLSTM()
single.load_state_dict(torch.load('models/lstm_best.pt', weights_only=True))
report("Single best model", single)
report("Ensemble (5 trials)", load_ensemble())

