const path = require('path');
const fs   = require('fs');
const ort  = require('onnxruntime-node');

const MODEL_PATH  = path.join(__dirname, 'models', 'lstm_model.onnx');
const SCALER_PATH = path.join(__dirname, 'models', 'scaler.json');

const SEQ_LEN    = 24;
const N_FEATURES = 8;
const ACTION_MAP = [[0,0],[1,0],[0,1],[1,1]];  // [fan, heat] for actions 0-3

// ---------- reward (mirrors reward.py) ----------
function computeReward(tInside, fanOn, heaterOn) {
    const coldDev = Math.max(0, 18.0 - tInside);
    const hotDev  = Math.max(0, tInside - 24.0);

    let rComfort;
    if (coldDev > 0)      rComfort = -(coldDev ** 2) * 3.0;
    else if (hotDev > 0)  rComfort = -(hotDev  ** 2) * 1.0;
    else                  rComfort = 2.0;

    let rInaction = 0;
    if (coldDev > 3.0 && heaterOn === 0) rInaction = -10.0 * coldDev;
    else if (hotDev > 3.0 && fanOn === 0) rInaction = -4.0 * hotDev;

    const rEnergy = -(0.05 * fanOn + 0.10 * heaterOn);
    return rComfort + rInaction + rEnergy;
}

// ---------- MPC random shooting ----------
async function mpcSolve(session, window24x8, scaler, nCandidates, horizon, gamma) {
    const N = nCandidates;
    const H = horizon;
    const tInsideMean = scaler.mean[1];
    const tInsideStd  = scaler.scale[1];

    // Random action sequences (N, H)
    const actions = new Int32Array(N * H);
    for (let i = 0; i < actions.length; i++) actions[i] = Math.floor(Math.random() * 4);

    // Tile window: (N, SEQ_LEN, N_FEATURES)
    const windows = new Float32Array(N * SEQ_LEN * N_FEATURES);
    for (let n = 0; n < N; n++) {
        windows.set(window24x8, n * SEQ_LEN * N_FEATURES);
    }

    const totalReward = new Float64Array(N);
    let discount = 1.0;

    for (let h = 0; h < H; h++) {
        // Inject current actions into last row of each candidate window
        const lstmInput = new Float32Array(windows);
        for (let n = 0; n < N; n++) {
            const [fan, heat] = ACTION_MAP[actions[n * H + h]];
            const offset = n * SEQ_LEN * N_FEATURES + (SEQ_LEN - 1) * N_FEATURES;
            lstmInput[offset + 4] = fan;
            lstmInput[offset + 5] = heat;
        }

        // Batch ONNX inference: (N, SEQ_LEN, N_FEATURES) → (N, 2)
        const tensor  = new ort.Tensor('float32', lstmInput, [N, SEQ_LEN, N_FEATURES]);
        const results = await session.run({ input: tensor });
        const delta   = results.delta.data;  // Float32Array length N*2

        // Update rewards and slide windows
        for (let n = 0; n < N; n++) {
            const [fan, heat] = ACTION_MAP[actions[n * H + h]];
            const wOffset     = n * SEQ_LEN * N_FEATURES;
            const lastOffset  = wOffset + (SEQ_LEN - 1) * N_FEATURES;

            const tInNorm  = windows[lastOffset + 1] + delta[n * 2];
            const tFlNorm  = windows[lastOffset + 2] + delta[n * 2 + 1];
            const tInReal  = tInNorm * tInsideStd + tInsideMean;

            totalReward[n] += discount * computeReward(tInReal, fan, heat);

            // Slide window left by one row, append new row
            windows.copyWithin(wOffset, wOffset + N_FEATURES, wOffset + SEQ_LEN * N_FEATURES);
            const newOffset = wOffset + (SEQ_LEN - 1) * N_FEATURES;
            windows.set(lstmInput.slice(lastOffset, lastOffset + N_FEATURES), newOffset);
            windows[newOffset + 1] = tInNorm;
            windows[newOffset + 2] = tFlNorm;
            windows[newOffset + 4] = fan;
            windows[newOffset + 5] = heat;
        }

        discount *= gamma;
    }

    // Best first action
    let bestN = 0;
    for (let n = 1; n < N; n++) {
        if (totalReward[n] > totalReward[bestN]) bestN = n;
    }
    return actions[bestN * H];
}

// ---------- Node-RED node definition ----------
module.exports = function(RED) {
    const scaler  = JSON.parse(fs.readFileSync(SCALER_PATH, 'utf8'));
    let   session = null;

    ort.InferenceSession.create(MODEL_PATH).then(s => {
        session = s;
        RED.log.info('[hvac-agent] ONNX model loaded');
    }).catch(err => {
        RED.log.error('[hvac-agent] Failed to load ONNX model: ' + err.message);
    });

    function HVACAgentNode(config) {
        RED.nodes.createNode(this, config);
        const node        = this;
        const nCandidates = parseInt(config.nCandidates) || 256;
        const horizon     = parseInt(config.horizon)     || 4;
        const gamma       = parseFloat(config.gamma)     || 0.95;

        // Rolling 24-step window — persisted in node context
        const ctx    = node.context();
        let   window = ctx.get('window') || null;

        node.on('input', async function(msg) {
            if (!session) {
                node.warn('ONNX model not loaded yet');
                return;
            }

            const { T_outside, T_inside, T_floor, SR_direct } = msg.payload;
            if ([T_outside, T_inside, T_floor, SR_direct].some(v => v == null)) {
                node.error('msg.payload must contain T_outside, T_inside, T_floor, SR_direct');
                return;
            }

            // Normalise sensor readings
            const hour   = new Date().getHours();
            const hSin   = Math.sin(2 * Math.PI * hour / 24);
            const hCos   = Math.cos(2 * Math.PI * hour / 24);
            const normed = new Float32Array([
                (T_outside - scaler.mean[0]) / scaler.scale[0],
                (T_inside  - scaler.mean[1]) / scaler.scale[1],
                (T_floor   - scaler.mean[2]) / scaler.scale[2],
                (SR_direct - scaler.mean[3]) / scaler.scale[3],
                0,     // fan_on    — filled by MPC
                0,     // heater_on — filled by MPC
                hSin,
                hCos,
            ]);

            // Initialise or update rolling window
            if (!window) {
                window = new Float32Array(SEQ_LEN * N_FEATURES);
                for (let i = 0; i < SEQ_LEN; i++) window.set(normed, i * N_FEATURES);
            } else {
                window.copyWithin(0, N_FEATURES);
                window.set(normed, (SEQ_LEN - 1) * N_FEATURES);
            }
            ctx.set('window', window);

            try {
                const action = await mpcSolve(session, window, scaler, nCandidates, horizon, gamma);
                const [fanOn, heaterOn] = ACTION_MAP[action];

                // Write chosen action into last row so the window reflects reality
                window[(SEQ_LEN - 1) * N_FEATURES + 4] = fanOn;
                window[(SEQ_LEN - 1) * N_FEATURES + 5] = heaterOn;
                ctx.set('window', window);

                node.status({ fill: heaterOn ? 'red' : fanOn ? 'blue' : 'green',
                              shape: 'dot',
                              text: `heat=${heaterOn} fan=${fanOn}` });

                msg.payload = { fan_on: fanOn, heater_on: heaterOn, action, source: 'mpc' };
                node.send(msg);
            } catch (err) {
                node.error('MPC error: ' + err.message, msg);
            }
        });
    }

    RED.nodes.registerType('hvac-agent', HVACAgentNode);
};
