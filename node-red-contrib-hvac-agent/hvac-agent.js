const path = require('path');
const fs   = require('fs');
const ort  = require('onnxruntime-node');

const MODEL_PATH  = path.join(__dirname, 'models', 'lstm_model.onnx');
const SCALER_PATH = path.join(__dirname, 'models', 'scaler.json');

const SEQ_LEN    = 24;
const N_FEATURES = 8;
const ACTION_MAP = [[0,0],[1,0],[0,1],[1,1]];  // [fan, heat] for actions 0-3

// ---------- reward (mirrors reward.py exactly) ----------
function computeReward(tInside, fanOn, heaterOn, tLow = 18.0, tHigh = 24.0) {
    const coldDev = Math.max(0, tLow  - tInside);
    const hotDev  = Math.max(0, tInside - tHigh);

    let rComfort;
    if (coldDev > 0)      rComfort = -(coldDev ** 2) * 2.0;
    else if (hotDev > 0)  rComfort = -(hotDev  ** 2) * 2.0;
    else                  rComfort = 2.0;

    let rInaction = 0;
    if (coldDev > 1.0 && heaterOn === 0) rInaction = -5.0 * coldDev;
    else if (hotDev > 3.0 && fanOn === 0) rInaction = -5.0 * hotDev;

    const rWarming = (heaterOn === 1 && coldDev > 0) ? 1.0 : 0.0;
    const rEnergy  = -(0.05 * fanOn + 0.10 * heaterOn);

    return rComfort + rInaction + rWarming + rEnergy;
}

// ---------- MPC helpers ----------

function sampleActions(probs, N, H) {
    const actions = new Int32Array(N * H);
    for (let h = 0; h < H; h++) {
        const p = probs[h];
        const cdf = [p[0], p[0]+p[1], p[0]+p[1]+p[2], 1.0];
        for (let n = 0; n < N; n++) {
            const r = Math.random();
            actions[n * H + h] = cdf.findIndex(c => r < c);
        }
    }
    return actions;
}

async function rolloutBatch(session, window24x8, actions, N, H, gamma, tInsideMean, tInsideStd) {
    const windows = new Float32Array(N * SEQ_LEN * N_FEATURES);
    for (let n = 0; n < N; n++) windows.set(window24x8, n * SEQ_LEN * N_FEATURES);

    const totalReward = new Float64Array(N);
    let discount = 1.0;

    for (let h = 0; h < H; h++) {
        const lstmInput = new Float32Array(windows);
        for (let n = 0; n < N; n++) {
            const [fan, heat] = ACTION_MAP[actions[n * H + h]];
            const offset = n * SEQ_LEN * N_FEATURES + (SEQ_LEN - 1) * N_FEATURES;
            lstmInput[offset + 4] = fan;
            lstmInput[offset + 5] = heat;
        }

        const tensor  = new ort.Tensor('float32', lstmInput, [N, SEQ_LEN, N_FEATURES]);
        const results = await session.run({ input: tensor });
        const delta   = results.delta.data;

        for (let n = 0; n < N; n++) {
            const [fan, heat] = ACTION_MAP[actions[n * H + h]];
            const wOffset    = n * SEQ_LEN * N_FEATURES;
            const lastOffset = wOffset + (SEQ_LEN - 1) * N_FEATURES;

            const tInNorm = windows[lastOffset + 1] + delta[n * 2];
            const tFlNorm = windows[lastOffset + 2] + delta[n * 2 + 1];
            const tInReal = tInNorm * tInsideStd + tInsideMean;

            totalReward[n] += discount * computeReward(tInReal, fan, heat);

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

    return totalReward;
}

// CEM MPC — mirrors mpc.py
async function mpcSolve(session, window24x8, scaler, nCandidates, horizon, gamma, cemIterations, cemEliteFrac) {
    const N       = nCandidates;
    const H       = horizon;
    const nElite  = Math.max(1, Math.floor(N * cemEliteFrac));
    const tInsideMean = scaler.mean[1];
    const tInsideStd  = scaler.scale[1];

    // Uniform prior
    let probs   = Array.from({ length: H }, () => [0.25, 0.25, 0.25, 0.25]);
    let actions = sampleActions(probs, N, H);

    for (let iter = 0; iter < cemIterations; iter++) {
        const scores = await rolloutBatch(session, window24x8, actions, N, H, gamma, tInsideMean, tInsideStd);

        // Select elite indices
        const sorted = Array.from({ length: N }, (_, i) => i)
            .sort((a, b) => scores[b] - scores[a]);
        const elite = sorted.slice(0, nElite);

        // Re-estimate action probabilities from elite
        probs = Array.from({ length: H }, () => [0, 0, 0, 0]);
        for (const n of elite) {
            for (let h = 0; h < H; h++) probs[h][actions[n * H + h]]++;
        }
        for (let h = 0; h < H; h++) {
            for (let a = 0; a < 4; a++) probs[h][a] /= nElite;
        }

        actions = sampleActions(probs, N, H);
    }

    const finalScores = await rolloutBatch(session, window24x8, actions, N, H, gamma, tInsideMean, tInsideStd);
    let bestN = 0;
    for (let n = 1; n < N; n++) {
        if (finalScores[n] > finalScores[bestN]) bestN = n;
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
        const node          = this;
        const nCandidates   = parseInt(config.nCandidates)   || 1024;
        const horizon       = parseInt(config.horizon)       || 24;
        const gamma         = parseFloat(config.gamma)       || 0.95;
        const cemIterations = parseInt(config.cemIterations) || 3;
        const cemEliteFrac  = parseFloat(config.cemEliteFrac) || 0.1;

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

            const hour   = new Date().getHours();
            const hSin   = Math.sin(2 * Math.PI * hour / 24);
            const hCos   = Math.cos(2 * Math.PI * hour / 24);
            const normed = new Float32Array([
                (T_outside - scaler.mean[0]) / scaler.scale[0],
                (T_inside  - scaler.mean[1]) / scaler.scale[1],
                (T_floor   - scaler.mean[2]) / scaler.scale[2],
                (SR_direct - scaler.mean[3]) / scaler.scale[3],
                0, 0,  // fan_on, heater_on — filled by MPC
                hSin, hCos,
            ]);

            if (!window) {
                window = new Float32Array(SEQ_LEN * N_FEATURES);
                for (let i = 0; i < SEQ_LEN; i++) window.set(normed, i * N_FEATURES);
            } else {
                window.copyWithin(0, N_FEATURES);
                window.set(normed, (SEQ_LEN - 1) * N_FEATURES);
            }
            ctx.set('window', window);

            try {
                const action = await mpcSolve(
                    session, window, scaler,
                    nCandidates, horizon, gamma, cemIterations, cemEliteFrac
                );
                const [fanOn, heaterOn] = ACTION_MAP[action];

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
