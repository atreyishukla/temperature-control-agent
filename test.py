import torch, numpy as np, pickle
from data_pipeline import load_data, split_scale, make_sequences
from lstm_model import BuildingLSTM
from hvac_env import HVACEnv

lstm = BuildingLSTM()
lstm.load_state_dict(torch.load('models/lstm_best.pt', weights_only=True))
lstm.eval()

df = load_data('data/Concrete_floor_results.xlsx')
df_train, _, _, scaler = split_scale(df)
X_train = make_sequences(df_train)[0]

t_mean, t_std   = float(scaler.mean_[1]), float(scaler.scale_[1])
to_mean, to_std = float(scaler.mean_[0]), float(scaler.scale_[0])

env = HVACEnv(lstm=lstm, train_sequences=X_train,
              t_inside_mean=t_mean, t_inside_std=t_std)

env.reset(seed=100)
t_out_start = env._window[-1, 0] * to_std + to_mean
t_in_start  = env._window[-1, 1] * t_std  + t_mean
print(f"Episode start — T_outside={t_out_start:.1f}°C  T_inside={t_in_start:.1f}°C")
print()

for label, action in [("heater always on", 2), ("all off", 0)]:
    env.reset(seed=100)
    temps = []
    for _ in range(48):
        env.step(action)
        temps.append(env._window[-1, 1] * t_std + t_mean)
    print(f"{label:20s}: start={temps[0]:.1f}  end={temps[-1]:.1f}  min={min(temps):.1f}  max={max(temps):.1f}°C")
