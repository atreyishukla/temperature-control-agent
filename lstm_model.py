import torch
import torch.nn as nn


class BuildingLSTM(nn.Module):
    """
    2-layer LSTM predicting next-hour [T_inside, T_floor] from a 24-hour window of
    [T_outside, T_inside, T_floor, SR_direct, fan_on, heater_on].

    nn.LSTM with num_layers=2 and dropout=0.2 applies dropout between layer 1 and
    layer 2 — the output of layer 2's last timestep feeds the FC head.
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
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        last   = out[:, -1, :]         # (batch, hidden_size) — final timestep
        return self.fc2(self.relu(self.fc1(last)))  # (batch, 2)
