import torch
import torch.nn as nn
import torch.nn.functional as F


class BuildingLSTM(nn.Module):
    """
    2-layer LSTM predicting ΔT = [ΔT_inside, ΔT_floor] for the next hour.

    Causal action separation:
      - The LSTM sees the full 24-step window but with the CURRENT timestep's
        fan_on/heater_on zeroed out. This prevents it learning spurious
        correlations between actions and temperature levels.
      - Action effects are added as explicit learnable terms AFTER the LSTM:
          delta = lstm_dynamics(window) + heater_effect * heater_on
                                        + fan_effect    * fan_on
      - heater_effect is passed through softplus so it is always >= 0
        (heater always warms — physically enforced).
      - fan_effect is passed through -softplus so it is always <= 0
        (fan is Cooling_power — physically enforced to only cool).
    """

    def __init__(
        self,
        input_size:  int   = 8,
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

        # Learnable action contributions (one value per output: T_inside, T_floor)
        self._heater_raw = nn.Parameter(torch.full((2,),  0.5))  # softplus  → always > 0 (heater warms)
        self._fan_raw    = nn.Parameter(torch.full((2,), -0.5))  # -softplus → always < 0 (fan cools)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len=24, 6)
        fan_now    = x[:, -1, 4]   # (batch,)
        heater_now = x[:, -1, 5]   # (batch,)

        # Zero out current actions so LSTM learns uncontrolled thermal dynamics
        x_masked = x.clone()
        x_masked[:, -1, 4] = 0.0
        x_masked[:, -1, 5] = 0.0

        out, _ = self.lstm(x_masked)
        last   = out[:, -1, :]
        delta  = self.fc2(self.relu(self.fc1(last)))          # (batch, 2)

        heater_contrib = F.softplus(self._heater_raw)  * heater_now.unsqueeze(1)
        fan_contrib    = -F.softplus(self._fan_raw)    * fan_now.unsqueeze(1)

        return delta + heater_contrib + fan_contrib
