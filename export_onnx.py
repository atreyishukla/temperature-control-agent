"""
Export trained LSTM to ONNX and scaler parameters to JSON.
Run once after training, before installing the Node-RED node.

Output:
  node-red-contrib-hvac-agent/models/lstm_model.onnx
  node-red-contrib-hvac-agent/models/scaler.json
"""

import json
import os
import pickle
import torch
from lstm_model import BuildingLSTM

LSTM_PATH   = 'models/lstm_best.pt'
SCALER_PATH = 'models/scaler.pkl'
OUT_DIR     = 'node-red-contrib-hvac-agent/models'

os.makedirs(OUT_DIR, exist_ok=True)

# --- LSTM → ONNX ---
lstm = BuildingLSTM()
lstm.load_state_dict(torch.load(LSTM_PATH, weights_only=True))
lstm.eval()

dummy = torch.randn(1, 24, 8)
onnx_path = os.path.join(OUT_DIR, 'lstm_model.onnx')
torch.onnx.export(
    lstm, dummy, onnx_path,
    input_names=['input'],
    output_names=['delta'],
    dynamic_axes={'input': {0: 'batch_size'}, 'delta': {0: 'batch_size'}},
    opset_version=17,
)
print(f'Exported LSTM  → {onnx_path}')

# --- Scaler → JSON ---
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

scaler_data = {
    'mean':  scaler.mean_.tolist(),   # [T_outside, T_inside, T_floor, SR_direct]
    'scale': scaler.scale_.tolist(),
}
scaler_path = os.path.join(OUT_DIR, 'scaler.json')
with open(scaler_path, 'w') as f:
    json.dump(scaler_data, f, indent=2)
print(f'Exported scaler → {scaler_path}')
print('Done. Copy node-red-contrib-hvac-agent/ to the OCN+ and run: npm install')
