import torch
import torch.nn as nn
import torch.nn.functional as F
import glob


class BuildingLSTM(nn.Module):
    """
    2-layer LSTM predicting ΔT = [ΔT_inside, ΔT_floor] for the next hour.

    Causal action separation:
      - Actions at the current timestep are zeroed out before the LSTM sees
        the window. This prevents spurious correlations between actions and
        temperature level (e.g. heater was historically on when it was already
        warm, so without separation the LSTM learns heater → high T).
      - Action effects are added as explicit learnable scalars AFTER the LSTM:
          delta = lstm_dynamics(window) + heater_effect * heater_on
                                        + fan_effect    * fan_on
      - heater_effect is constrained via softplus to be always >= 0
        (heater can only warm — physically enforced).
      - fan_effect is left unconstrained. Causal separation means it is
        estimated from the actual cooling signal in the data, not from the
        correlation with temperature level. It will converge to a small
        negative value; hard-constraining it to negative hurts fitting because
        solar gains can dominate on fan-on timesteps.
    """

    def __init__(
        self,
        input_size:  int   = 8,
        hidden_size: int   = 512,
        num_layers:  int   = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc1  = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(128, 2)

        self._heater_raw = nn.Parameter(torch.full((2,),  0.5))  # softplus → always > 0 (heater warms)
        self.fan_effect  = nn.Parameter(torch.zeros(2))           # unconstrained; causal separation removes spurious level correlation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fan_now    = x[:, -1, 4]   # (batch,)
        heater_now = x[:, -1, 5]   # (batch,)

        x_masked = x.clone()
        x_masked[:, -1, 4] = 0.0
        x_masked[:, -1, 5] = 0.0

        out, _ = self.lstm(x_masked)
        last   = out[:, -1, :]
        delta  = self.fc2(self.relu(self.fc1(last)))          # (batch, 2)

        heater_contrib = F.softplus(self._heater_raw) * heater_now.unsqueeze(1)
        fan_contrib    = self.fan_effect               * fan_now.unsqueeze(1)

        return delta + heater_contrib + fan_contrib


class EnsembleLSTM(nn.Module):
    """Average predictions from multiple independently-trained BuildingLSTM checkpoints."""

    def __init__(self, paths: list[str]):
        super().__init__()
        members = []
        for p in paths:
            m = BuildingLSTM()
            m.load_state_dict(torch.load(p, weights_only=True))
            m.eval()
            members.append(m)
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.members]).mean(dim=0)


def load_ensemble(pattern: str = 'models/lstm_trial_*.pt') -> 'EnsembleLSTM':
    """Load all trial checkpoints matching *pattern* as an ensemble."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No checkpoints found matching {pattern!r}')
    return EnsembleLSTM(paths)
