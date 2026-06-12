
import pickle, torch, numpy as np
from data_pipeline import load_data, split_scale, make_sequences
from lstm_model import BuildingLSTM

df = load_data('data/Concrete_floor_results.xlsx')
_, _, test, scaler = split_scale(df)
X, y = make_sequences(test)

with open('models/scaler.pkl','rb') as f: sc = pickle.load(f)
t_std = sc.scale_[1]

lstm = BuildingLSTM()
lstm.load_state_dict(torch.load('models/lstm_best.pt', weights_only=True))
lstm.eval()

with torch.no_grad():
    pred = lstm(torch.tensor(X)).numpy()  # predicted ΔT_inside_norm

mae_norm = abs(pred[:,0] - y[:,0]).mean()
mae_c    = mae_norm * t_std
within1  = (abs(pred[:,0] - y[:,0]) * t_std < 1.0).mean() * 100
within05 = (abs(pred[:,0] - y[:,0]) * t_std < 0.5).mean() * 100

print(f"LSTM MAE on test set:        {mae_c:.3f}°C")
print(f"Within ±0.5°C:               {within05:.1f}%")
print(f"Within ±1.0°C:               {within1:.1f}%")

