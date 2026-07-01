/**
 * ONNX inference wrapper for Ultimate Tic-Tac-Toe.
 *
 * Depends on: uttt_engine.js (MINI_INDICES, X, O, DRAW must be in scope).
 * Depends on: onnxruntime-web loaded as `ort` before this file.
 *
 * To swap the agent: run scripts/export_onnx.py and replace docs/models/model_config.json
 * + the ONNX file. Nothing in this file changes.
 *
 * Public API:
 *   loadAgent(configUrl)           → Promise<AgentRunner>
 *   runner.selectMove(state, temp) → Promise<cellIndex>
 *   runner.name                    → string (from config)
 */

const _WASM_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

// ---------------------------------------------------------------------------
// Tensor builder — mirrors agents/agent_base.py board_to_tensor_from_gamestate
// Input shape: (1, 7, 9, 9) float32, channel-first NCHW.
// Offset formula: ch * 81 + cell_idx, where cell_idx = row*9 + col.
// ---------------------------------------------------------------------------

function _buildInputTensor(state) {
  const data = new Float32Array(7 * 81); // zero-filled

  const board = state.board;

  // Ch 0: X pieces; Ch 1: O pieces.
  for (let i = 0; i < 81; i++) {
    if      (board[i] === X) data[     i] = 1.0;
    else if (board[i] === O) data[81 + i] = 1.0;
  }

  // Ch 2: current-player uniform (+1 if X to move, -1 if O).
  const pv = (state.player === X) ? 1.0 : -1.0;
  data.fill(pv, 162, 243);

  // Ch 3: legal move mask.
  for (const cell of state.validMoves()) data[243 + cell] = 1.0;

  // Ch 4: mini-board winners (+1 X, -1 O, 0 draw/empty).
  for (let m = 0; m < 9; m++) {
    const mw = state.miniWinner[m];
    if (mw === X || mw === O) {
      const v = (mw === X) ? 1.0 : -1.0;
      for (const cell of MINI_INDICES[m]) data[324 + cell] = v;
    }
  }

  // Ch 5: last move.
  if (state.lastMove >= 0) data[405 + state.lastMove] = 1.0;

  // Ch 6: bias (all 1.0).
  data.fill(1.0, 486, 567);

  return new ort.Tensor('float32', data, [1, 7, 9, 9]);
}

// ---------------------------------------------------------------------------
// AgentRunner
// ---------------------------------------------------------------------------

class AgentRunner {
  /**
   * @param {ort.InferenceSession} session
   * @param {object} config  parsed model_config.json
   */
  constructor(session, config) {
    this.session      = session;
    this.name         = config.name;
    this._policyName  = config.outputs.policy;
  }

  /**
   * Run one inference step and return the chosen cell index.
   * @param {GameState} state
   * @param {number}    temperature  0 = argmax (deterministic), >0 = softmax sample
   */
  async selectMove(state, temperature = 0.0) {
    const valid = state.validMoves();
    if (valid.length === 0) return -1;
    if (valid.length === 1) return valid[0];

    const input   = _buildInputTensor(state);
    const results = await this.session.run({ input });
    const logits  = results[this._policyName].data; // Float32Array[81]

    if (temperature === 0.0) {
      // Deterministic argmax over legal moves.
      let best = valid[0], bestVal = logits[valid[0]];
      for (const cell of valid) {
        if (logits[cell] > bestVal) { bestVal = logits[cell]; best = cell; }
      }
      return best;
    }

    // Temperature-scaled sampling.
    let maxLog = -Infinity;
    for (const cell of valid) if (logits[cell] > maxLog) maxLog = logits[cell];

    const probs = valid.map(c => Math.exp((logits[c] - maxLog) / temperature));
    const total = probs.reduce((a, b) => a + b, 0);
    let r = Math.random() * total;
    for (let i = 0; i < valid.length; i++) {
      r -= probs[i];
      if (r <= 0) return valid[i];
    }
    return valid[valid.length - 1]; // floating-point fallback
  }
}

// ---------------------------------------------------------------------------
// loadAgent — public entry point
// ---------------------------------------------------------------------------

/**
 * Fetch model_config.json, configure ORT WASM paths, create an inference
 * session, and return an AgentRunner.
 *
 * @param {string} configUrl  URL to model_config.json
 * @returns {Promise<AgentRunner>}
 */
async function loadAgent(configUrl) {
  const cfgResp = await fetch(configUrl);
  if (!cfgResp.ok) {
    throw new Error(
      `Could not load model config (HTTP ${cfgResp.status}). ` +
      `Run scripts/export_onnx.py to generate docs/models/model_config.json.`
    );
  }
  const config = await cfgResp.json();

  if (config.name === 'pending') {
    throw new Error(
      'Model not exported yet. Run: python -m scripts.export_onnx ' +
      '--candidate arena:21@hof --quantize'
    );
  }

  // WASM paths must be set before any InferenceSession.create() call.
  ort.env.wasm.wasmPaths = _WASM_CDN;

  const modelUrl = new URL(config.file, configUrl).href;
  const session  = await ort.InferenceSession.create(modelUrl, {
    executionProviders:    ['wasm'],
    graphOptimizationLevel: 'all',
  });

  return new AgentRunner(session, config);
}
