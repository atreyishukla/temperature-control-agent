import torch, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from data_pipeline import load_data, split_scale, make_sequences
from lstm_model import BuildingLSTM
from hvac_env import HVACEnv

# Load everything
lstm = BuildingLSTM()
lstm.load_state_dict(torch.load('models/lstm_best.pt', weights_only=True))
lstm.eval()

df = load_data('data/Concrete_floor_results.xlsx')
df_train, _, _, scaler = split_scale(df)
X_train = make_sequences(df_train)[0]

env = HVACEnv(lstm=lstm, train_sequences=X_train,
              t_inside_mean=float(scaler.mean_[1]),
              t_inside_std=float(scaler.scale_[1]))

model = PPO.load('models/ppo_hvac')

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
