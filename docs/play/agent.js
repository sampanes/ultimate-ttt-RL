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
// 1-ply tactical pool -- mirrors engine/tactics.py tactical_filter().
// Returns the move pool the policy argmax must choose within:
//   - immediate game-winning moves if any exist,
//   - else moves that do not hand the opponent an immediate game win
//     (falling back to all legal moves if every move loses).
// Pure engine logic; the net breaks ties within the pool.
// ---------------------------------------------------------------------------

function _tacticalPool(state) {
  const valid = state.validMoves();
  const wins = [];
  for (const mv of valid) {
    const s = state.clone();
    s.makeMove(mv);
    if (s.winner === state.player) wins.push(mv);
  }
  if (wins.length > 0) return wins;

  const safe = [];
  for (const mv of valid) {
    const s = state.clone();
    s.makeMove(mv);
    if (s.winner !== null) { safe.push(mv); continue; } // game over on our move (draw) -- no reply
    let losing = false;
    for (const omv of s.validMoves()) {
      const s2 = s.clone();
      s2.makeMove(omv);
      if (s2.winner === s.player) { losing = true; break; }
    }
    if (!losing) safe.push(mv);
  }
  return safe.length > 0 ? safe : valid;
}

// ---------------------------------------------------------------------------
// PUCT MCTS — mirrors agents/mcts.py (serial, wave_size=1)
// ---------------------------------------------------------------------------

class _MCTSNode {
  constructor(parent, prior, move) {
    this.parent   = parent;
    this.children = {};    // cell index (int) -> _MCTSNode
    this.prior    = prior;
    this.N        = 0;
    this.W        = 0.0;
    this.move     = move;
  }
  Q() { return this.N > 0 ? this.W / this.N : 0.0; }
  U(cPuct, parentN) { return cPuct * this.prior * Math.sqrt(parentN) / (1 + this.N); }
}

// Expand one leaf: run network, create child nodes, return net value.
async function _mctsExpand(node, state, session, policyName, valueName) {
  const valid = state.validMoves();
  if (valid.length === 0) return 0.0;

  const input   = _buildInputTensor(state);
  const results = await session.run({ input });
  const logits  = results[policyName].data;
  const value   = results[valueName].data[0];

  // Masked softmax over legal moves only.
  let maxLog = -Infinity;
  for (const c of valid) if (logits[c] > maxLog) maxLog = logits[c];
  const exps = valid.map(c => Math.exp(logits[c] - maxLog));
  const sum  = exps.reduce((a, b) => a + b, 0.0);

  for (let i = 0; i < valid.length; i++) {
    node.children[valid[i]] = new _MCTSNode(node, exps[i] / sum, valid[i]);
  }
  return value;
}

/**
 * Run PUCT MCTS for `nSims` simulations.
 * @returns {Float32Array} visit-count distribution over 81 cells (unnormalised)
 */
async function _mctsSearch(rootState, session, policyName, valueName, nSims, cPuct) {
  const root = new _MCTSNode(null, 0.0, -1);
  await _mctsExpand(root, rootState, session, policyName, valueName);

  for (let sim = 0; sim < nSims; sim++) {
    let   node  = root;
    const state = rootState.clone();

    // Selection: traverse to a leaf via PUCT.
    while (Object.keys(node.children).length > 0 && state.winner === null) {
      let best = null, bestScore = -Infinity;
      for (const child of Object.values(node.children)) {
        const score = -child.Q() + child.U(cPuct, node.N);
        if (score > bestScore) { bestScore = score; best = child; }
      }
      node = best;
      state.makeMove(node.move);
    }

    // Evaluate leaf.
    let value;
    if (state.winner !== null) {
      // Terminal: winner is always the player who just moved (not the next mover).
      // From the next mover's perspective: win = -1, draw = 0.
      value = (state.winner === DRAW) ? 0.0 : -1.0;
    } else {
      value = await _mctsExpand(node, state, session, policyName, valueName);
    }

    // Backup: alternate sign each ply (zero-sum).
    let v = value, n = node;
    while (n !== null) { n.N += 1; n.W += v; v = -v; n = n.parent; }
  }

  // Return visit-count distribution + MCTS value at root (from root player's perspective).
  const pi = new Float32Array(81);
  for (const [mv, child] of Object.entries(root.children)) {
    pi[parseInt(mv, 10)] = child.N;
  }
  const rootValue = root.N > 0 ? root.W / root.N : 0.0;
  return { pi, rootValue };
}

// ---------------------------------------------------------------------------
// AgentRunner
// ---------------------------------------------------------------------------

class AgentRunner {
  /**
   * @param {ort.InferenceSession} session
   * @param {object} config  parsed model_config.json
   *
   * nSims controls MCTS depth (0 = raw policy, no search).
   * Only used when temperature === 0.0 (Hard / argmax mode).
   * Easy / Medium keep temperature > 0 and bypass MCTS entirely.
   */
  constructor(session, config) {
    this.session      = session;
    this.name         = config.name;
    this._policyName  = config.outputs.policy;
    this._valueName   = config.outputs.value;
    this.nSims        = 50;    // MCTS budget for Hard mode; set to 0 to use raw policy
    this.cPuct        = 1.5;
    this.tactical     = false; // 1-ply win/block pool on the argmax path (certified mode)
    this.lastEval     = null;  // { xAdv, topMoves, mode } — updated after each selectMove
  }

  /**
   * Run one inference step and return the chosen cell index.
   * @param {GameState} state
   * @param {number}    temperature  0 = MCTS/argmax, >0 = raw policy temperature sample
   */
  async selectMove(state, temperature = 0.0) {
    const valid = state.validMoves();
    if (valid.length === 0) return -1;
    if (valid.length === 1) return valid[0];

    // Hard mode: MCTS when nSims > 0.
    if (temperature === 0.0 && this.nSims > 0) {
      const { pi, rootValue } = await _mctsSearch(
        state, this.session, this._policyName, this._valueName,
        this.nSims, this.cPuct,
      );
      // Argmax over legal moves by visit count.
      let best = valid[0], bestN = pi[valid[0]], totalN = 0;
      for (const cell of valid) {
        totalN += pi[cell];
        if (pi[cell] > bestN) { bestN = pi[cell]; best = cell; }
      }
      const topCells = valid.slice().sort((a, b) => pi[b] - pi[a]).slice(0, 3);
      this.lastEval = {
        xAdv:     (state.player === X) ? rootValue : -rootValue,
        topMoves: topCells.map(c => ({ cell: c, pct: totalN > 0 ? Math.round(pi[c] / totalN * 100) : 0 })),
        mode:     'mcts',
      };
      return best;
    }

    // Easy / Medium: raw policy + temperature sampling (no MCTS).
    const input    = _buildInputTensor(state);
    const results  = await this.session.run({ input });
    const logits   = results[this._policyName].data;
    const netValue = results[this._valueName].data[0];

    // Temp=1 softmax over valid moves for display; temperature-scaled for sampling.
    let maxLog = -Infinity;
    for (const cell of valid) if (logits[cell] > maxLog) maxLog = logits[cell];
    const exps1  = valid.map(c => Math.exp(logits[c] - maxLog));
    const sum1   = exps1.reduce((a, b) => a + b, 0);
    const probs1 = exps1.map(v => v / sum1);
    const probMap  = new Map(valid.map((c, i) => [c, probs1[i]]));
    const topCells = valid.slice().sort((a, b) => probMap.get(b) - probMap.get(a)).slice(0, 3);

    let chosen;
    if (temperature === 0.0) {
      // Argmax, optionally constrained to the provably-correct tactical pool.
      const pool = this.tactical ? _tacticalPool(state) : valid;
      chosen = pool[0];
      for (const cell of pool) if (logits[cell] > logits[chosen]) chosen = cell;
    } else {
      const expsT = valid.map(c => Math.exp((logits[c] - maxLog) / temperature));
      const sumT  = expsT.reduce((a, b) => a + b, 0);
      let r = Math.random() * sumT;
      chosen = valid[valid.length - 1];
      for (let i = 0; i < valid.length; i++) { r -= expsT[i]; if (r <= 0) { chosen = valid[i]; break; } }
    }

    this.lastEval = {
      xAdv:     (state.player === X) ? netValue : -netValue,
      topMoves: topCells.map(c => ({ cell: c, pct: Math.round(probMap.get(c) * 100) })),
      mode:     'policy',
    };
    return chosen;
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

  // cfgResp.url is always absolute (fetch resolves relative paths);
  // configUrl may be a bare relative path which new URL() rejects as a base.
  const modelUrl = new URL(config.file, cfgResp.url).href;
  const session  = await ort.InferenceSession.create(modelUrl, {
    executionProviders:    ['wasm'],
    graphOptimizationLevel: 'all',
  });

  return new AgentRunner(session, config);
}
