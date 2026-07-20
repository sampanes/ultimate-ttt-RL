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
 *   loadAgent(configUrl)           -> Promise<AgentRunner>
 *   runner.selectMove(state, temp) -> Promise<cellIndex>
 *   runner.name                    -> string (from config)
 */

const _WASM_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

// ---------------------------------------------------------------------------
// Tensor builder -- mirrors agents/agent_base.py board_to_tensor_from_gamestate
// Input shape: (1, 7, 9, 9) float32, channel-first NCHW.
// Offset formula: ch * 81 + cell_idx, where cell_idx = row*9 + col.
// ---------------------------------------------------------------------------

// Write one board's 7x9x9 planes into `data` starting at flat offset `off`
// (off = sampleIndex * 567). Factored out so batched search can pack many
// boards into one contiguous buffer with the exact same per-board layout as a
// single input -- verified batch-invariant against the ONNX model.
function _fillInput(data, off, state) {
  const board = state.board;

  // Ch 0: X pieces; Ch 1: O pieces.
  for (let i = 0; i < 81; i++) {
    if      (board[i] === X) data[off +      i] = 1.0;
    else if (board[i] === O) data[off + 81 + i] = 1.0;
  }

  // Ch 2: current-player uniform (+1 if X to move, -1 if O).
  const pv = (state.player === X) ? 1.0 : -1.0;
  data.fill(pv, off + 162, off + 243);

  // Ch 3: legal move mask.
  for (const cell of state.validMoves()) data[off + 243 + cell] = 1.0;

  // Ch 4: mini-board winners (+1 X, -1 O, 0 draw/empty).
  for (let m = 0; m < 9; m++) {
    const mw = state.miniWinner[m];
    if (mw === X || mw === O) {
      const v = (mw === X) ? 1.0 : -1.0;
      for (const cell of MINI_INDICES[m]) data[off + 324 + cell] = v;
    }
  }

  // Ch 5: last move.
  if (state.lastMove >= 0) data[off + 405 + state.lastMove] = 1.0;

  // Ch 6: bias (all 1.0).
  data.fill(1.0, off + 486, off + 567);
}

function _buildInputTensor(state) {
  const data = new Float32Array(567); // zero-filled
  _fillInput(data, 0, state);
  return new ort.Tensor('float32', data, [1, 7, 9, 9]);
}

// Pack K boards into one [K,7,9,9] tensor for a single batched forward.
function _buildBatchTensor(states) {
  const K = states.length;
  const data = new Float32Array(K * 567); // zero-filled
  for (let k = 0; k < K; k++) _fillInput(data, k * 567, states[k]);
  return new ort.Tensor('float32', data, [K, 7, 9, 9]);
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
// PUCT MCTS -- mirrors agents/mcts.py (serial, wave_size=1)
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

// Create child nodes from a policy-logit row (masked softmax over legal moves).
// `logits` may be a full-batch buffer; `polOff` is this leaf's row offset
// (sampleIndex * 81). Shared by the serial and batched search paths so both
// expand identically -- this is what makes wave-size-1 batched search bit-for-bit
// equal to the serial search.
function _expandNode(node, valid, logits, polOff, value) {
  let maxLog = -Infinity;
  for (const c of valid) if (logits[polOff + c] > maxLog) maxLog = logits[polOff + c];
  const exps = valid.map(c => Math.exp(logits[polOff + c] - maxLog));
  const sum  = exps.reduce((a, b) => a + b, 0.0);
  for (let i = 0; i < valid.length; i++) {
    node.children[valid[i]] = new _MCTSNode(node, exps[i] / sum, valid[i]);
  }
  return value;
}

// Expand one leaf: run network (batch 1), create child nodes, return net value.
async function _mctsExpand(node, state, session, policyName, valueName) {
  const valid = state.validMoves();
  if (valid.length === 0) return 0.0;

  const input   = _buildInputTensor(state);
  const results = await session.run({ input });
  return _expandNode(node, valid, results[policyName].data, 0, results[valueName].data[0]);
}

// Virtual loss added to each node on a leaf's path while it is in flight in a
// wave, so the other descents in the same wave avoid it; removed at backup.
// Value units are [-1, 1]; 1.0 is the conventional magnitude.
const _VLOSS = 1.0;

/**
 * Run PUCT MCTS. Dispatches to the batched (leaf-parallel) search when
 * waveSize > 1 -- that path evaluates a wave of leaves in ONE forward, which on
 * WebGPU costs ~the same as a single forward, so far more sims fit the per-move
 * time budget (deeper search, same wall clock). waveSize <= 1 runs the serial
 * search unchanged (and is bit-for-bit identical to the batched path at wave 1).
 * @returns {{pi: Float32Array, rootValue: number, sims: number}}
 */
async function _mctsSearch(rootState, session, policyName, valueName, nSims, cPuct, deadlineMs, waveSize) {
  if (waveSize && waveSize > 1) {
    return _mctsSearchBatched(rootState, session, policyName, valueName,
                              nSims, cPuct, deadlineMs, waveSize);
  }
  const root = new _MCTSNode(null, 0.0, -1);
  await _mctsExpand(root, rootState, session, policyName, valueName);

  const _deadline = deadlineMs ? performance.now() + deadlineMs : Infinity;
  let _completed = 0;
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

    // Time budget: stop early once the per-move deadline passes, so every move
    // takes ~the same wall clock on every device and the depth (quality) is
    // whatever fits. Checked every sim for tight adherence -- each sim already
    // runs a millisecond-scale WASM forward, so performance.now() is negligible.
    // deadlineMs falsy -> run the full fixed nSims (unchanged path).
    _completed++;
    if (deadlineMs && performance.now() >= _deadline) break;
  }

  // Return visit-count distribution + MCTS value at root (from root player's perspective).
  const pi = new Float32Array(81);
  for (const [mv, child] of Object.entries(root.children)) {
    pi[parseInt(mv, 10)] = child.N;
  }
  const rootValue = root.N > 0 ? root.W / root.N : 0.0;
  return { pi, rootValue, sims: _completed };
}

/**
 * Batched (leaf-parallel) PUCT MCTS. Each wave descends to up to `waveSize`
 * distinct leaves using virtual loss to fan out, evaluates them all in ONE
 * network forward, then expands and backs them up. Semantically a standard
 * parallel MCTS; at waveSize 1 it reduces exactly to the serial search above
 * (verified by scripts/test_batched_mcts.js). The win is that batch-K forwards
 * on WebGPU cost ~the same wall clock as batch-1, so the deadline-bounded search
 * reaches many more sims per move on capable hardware.
 */
async function _mctsSearchBatched(rootState, session, policyName, valueName,
                                  nSims, cPuct, deadlineMs, waveSize) {
  const root = new _MCTSNode(null, 0.0, -1);
  await _mctsExpand(root, rootState, session, policyName, valueName);

  const _deadline = deadlineMs ? performance.now() + deadlineMs : Infinity;
  let completed = 0;

  while (completed < nSims) {
    if (deadlineMs && performance.now() >= _deadline) break;
    const budget = Math.min(waveSize, nSims - completed);

    // --- Collect a wave of leaves, fanning out via virtual loss. ---
    const leaves  = [];          // { node, state, terminalValue }
    const pending = new Set();   // leaf nodes already claimed this wave
    for (let w = 0; w < budget; w++) {
      let   node  = root;
      const state = rootState.clone();
      while (Object.keys(node.children).length > 0 && state.winner === null) {
        let best = null, bestScore = -Infinity;
        for (const child of Object.values(node.children)) {
          const score = -child.Q() + child.U(cPuct, node.N);
          if (score > bestScore) { bestScore = score; best = child; }
        }
        node = best;
        state.makeMove(node.move);
      }
      if (pending.has(node)) break;   // collision: evaluate what we have
      pending.add(node);

      // Virtual loss along the path: N is the real visit (kept); +_VLOSS on W is
      // provisional and removed at backup. Makes this path look worse so the
      // rest of the wave explores elsewhere.
      for (let n = node; n !== null; n = n.parent) { n.N += 1; n.W += _VLOSS; }

      let terminalValue = null;
      if (state.winner !== null) {
        terminalValue = (state.winner === DRAW) ? 0.0 : -1.0;
      } else if (state.validMoves().length === 0) {
        terminalValue = 0.0;   // mirror serial _mctsExpand's no-moves guard
      }
      leaves.push({ node, state, terminalValue });
    }
    if (leaves.length === 0) break;

    // --- One forward for every non-terminal leaf in the wave. ---
    const evalLeaves = leaves.filter(l => l.terminalValue === null);
    if (evalLeaves.length > 0) {
      const input   = _buildBatchTensor(evalLeaves.map(l => l.state));
      const results = await session.run({ input });
      const logits  = results[policyName].data;   // length K*81
      const values  = results[valueName].data;    // length K
      for (let k = 0; k < evalLeaves.length; k++) {
        const l = evalLeaves[k];
        l.value = _expandNode(l.node, l.state.validMoves(), logits, k * 81, values[k]);
      }
    }

    // --- Backup: remove virtual loss, add the real value (alternating sign). ---
    for (const l of leaves) {
      let v = (l.terminalValue !== null) ? l.terminalValue : l.value;
      for (let n = l.node; n !== null; n = n.parent) { n.W += (v - _VLOSS); v = -v; }
      completed++;
    }
  }

  const pi = new Float32Array(81);
  for (const [mv, child] of Object.entries(root.children)) {
    pi[parseInt(mv, 10)] = child.N;
  }
  const rootValue = root.N > 0 ? root.W / root.N : 0.0;
  return { pi, rootValue, sims: completed };
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
    this.moveBudgetMs = 0;     // >0 = time-bounded search: constant move time, depth scales with device
    this.waveSize     = 1;     // >1 = batched leaf-parallel search (set by loadAgent per EP)
    this.ep           = 'wasm';// execution provider actually chosen for this session
    this.tactical     = false; // 1-ply win/block pool on the argmax path (certified mode)
    this.lastEval     = null;  // { xAdv, topMoves, mode } -- updated after each selectMove
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
        this.nSims, this.cPuct, this.moveBudgetMs, this.waveSize,
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
// loadAgent -- public entry point
// ---------------------------------------------------------------------------

// Probe a session on a fixed representative board: return { ms, pol, val } where
// ms is the per-forward time (warm-up discarded) and pol/val are the raw outputs
// used to check that an alternative provider AGREES with WASM before we trust it.
async function _probeSession(session, policyName, valueName, reps = 8) {
  const g = new GameState();
  for (let i = 0; i < 3 && g.winner === null; i++) {  // a few legal plies: nontrivial input
    const v = g.validMoves();
    g.makeMove(v[Math.floor(v.length / 2)]);
  }
  const input = _buildInputTensor(g);
  for (let i = 0; i < 3; i++) await session.run({ input });   // warm / compile
  const t0 = performance.now();
  let last;
  for (let i = 0; i < reps; i++) last = await session.run({ input });
  return {
    ms:  (performance.now() - t0) / reps,
    pol: Float32Array.from(last[policyName].data),
    val: last[valueName].data[0],
  };
}

// Max abs difference of the softmax-over-81 policies plus the value gap. Used to
// reject a provider whose numerics disagree with the WASM reference, so a buggy
// GPU backend can never win the race and play wrong moves -- it just falls back.
function _outputDisagreement(a, b) {
  const sm = (logits) => {
    let m = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > m) m = logits[i];
    let s = 0; const e = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) { e[i] = Math.exp(logits[i] - m); s += e[i]; }
    for (let i = 0; i < logits.length; i++) e[i] /= s;
    return e;
  };
  const pa = sm(a.pol), pb = sm(b.pol);
  let maxP = 0;
  for (let i = 0; i < pa.length; i++) maxP = Math.max(maxP, Math.abs(pa[i] - pb[i]));
  return Math.max(maxP, Math.abs(a.val - b.val));
}

// Wave size for the batched search when WebGPU wins the race. Batch-K forwards
// are ~free relative to batch-1 on a GPU, so a wide wave multiplies search
// depth in the fixed per-move budget. WASM stays serial (wave 1): batched gives
// it little (it is compute-bound) and this keeps the new code path off the
// majority of devices.
const _WAVE_WEBGPU = 32;

/**
 * Fetch model_config.json, configure ORT, create an inference session on the
 * fastest available execution provider, and return an AgentRunner.
 *
 * @param {string} configUrl  URL to model_config.json
 * @param {object} [opts]      { race: boolean } -- when true (the searching net)
 *   also try WebGPU and keep whichever provider measures faster.
 * @returns {Promise<AgentRunner>}
 */
async function loadAgent(configUrl, opts = {}) {
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

  // Multi-threaded WASM needs SharedArrayBuffer, which needs the page to be
  // cross-origin isolated (COOP+COEP). GitHub Pages is not, so this stays a
  // no-op there and safely activates only where isolation is present.
  try {
    if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated
        && navigator.hardwareConcurrency) {
      ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency, 8);
    }
  } catch (_) { /* env may be read-only on some builds */ }

  // cfgResp.url is always absolute (fetch resolves relative paths);
  // configUrl may be a bare relative path which new URL() rejects as a base.
  const modelUrl = new URL(config.file, cfgResp.url).href;

  // Candidate providers. WASM is the always-available floor. Add WebGPU only
  // when the browser exposes it AND the caller opted in (the searching net).
  const wantGpu = !!opts.race && typeof navigator !== 'undefined' && !!navigator.gpu;
  const eps = wantGpu ? ['webgpu', 'wasm'] : ['wasm'];

  // Fetch the model bytes once so racing two providers does not double-download.
  let source = modelUrl;
  if (eps.length > 1) {
    const buf = await (await fetch(modelUrl)).arrayBuffer();
    source = new Uint8Array(buf);
  }

  const polName = config.outputs.policy, valName = config.outputs.value;
  const candidates = [];
  for (const ep of eps) {
    try {
      const s     = await ort.InferenceSession.create(source, {
        executionProviders:     [ep],
        graphOptimizationLevel:  'all',
      });
      const probe = await _probeSession(s, polName, valName);
      candidates.push({ session: s, ep, ms: probe.ms, probe });
    } catch (_) { /* provider unsupported for this model/device -> skip */ }
  }
  if (candidates.length === 0) {
    throw new Error('No execution provider could load the model.');
  }

  // WASM is the trusted reference. Disqualify any other provider whose numerics
  // disagree with it (a broken GPU backend must never win on speed alone).
  const ref = candidates.find(c => c.ep === 'wasm');
  const _AGREE_TOL = 0.05;
  const qualified = candidates.filter(c =>
    c.ep === 'wasm' || !ref || _outputDisagreement(ref.probe, c.probe) <= _AGREE_TOL);

  qualified.sort((a, b) => a.ms - b.ms);
  const winner = qualified[0];
  for (const c of candidates) {
    if (c !== winner) { try { c.session.release(); } catch (_) { /* best-effort */ } }
  }

  const runner = new AgentRunner(winner.session, config);
  runner.ep       = winner.ep;
  runner.waveSize = (winner.ep === 'webgpu') ? _WAVE_WEBGPU : 1;
  return runner;
}
