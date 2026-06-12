import pickle, torch, numpy as np
from data_pipeline import load_data, split_scale, make_sequences
from lstm_model import BuildingLSTM

df = load_data('data/Concrete_floor_results.xlsx')
_, _, df_test, scaler = split_scale(df)
X_test, y_test = make_sequences(df_test)

model = BuildingLSTM()
model.load_state_dict(torch.load('models/lstm_best.pt', weights_only=True))
model.eval()

with torch.no_grad():
    preds = model(torch.tensor(X_test)).numpy()

# RMSE in real °C (denormalise T_inside only — column 1)
t_std  = scaler.scale_[1]   # std of T_inside
t_mean = scaler.mean_[1]

rmse_norm = np.sqrt(np.mean((preds[:, 0] - y_test[:, 0])**2))
rmse_real = rmse_norm * t_std
print(f"T_inside RMSE (normalised): {rmse_norm:.4f}")
print(f"T_inside RMSE (°C):         {rmse_real:.3f} °C")
print(f"T_floor  RMSE (normalised): {np.sqrt(np.mean((preds[:,1]-y_test[:,1])**2)):.4f}")
