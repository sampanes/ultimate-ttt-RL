/*
 * Parity + sanity guard for the batched leaf-parallel MCTS in docs/play/agent.js.
 *
 * Loads the real uttt_engine.js + agent.js in one function scope with a mocked,
 * deterministic, BATCH-INVARIANT ort session (per-row output depends only on
 * that row's bytes, exactly like the real ONNX model). Then asserts:
 *   (1) batched search at waveSize 1 == the serial search bit-for-bit, and
 *   (2) batched search at waveSize > 1 is a valid search (right sim count,
 *       visits only on legal root moves, finite root value).
 *
 * This is what lets the batched search ship without a browser: the only thing
 * that differs from the proven serial path at wave 1 is the batching machinery,
 * and this pins it. Run: node scripts/test_batched_mcts.js
 */
const fs   = require('fs');
const path = require('path');
const PLAY = path.join(__dirname, '..', 'docs', 'play');
const engineSrc = fs.readFileSync(path.join(PLAY, 'uttt_engine.js'), 'utf8');
const agentSrc  = fs.readFileSync(path.join(PLAY, 'agent.js'), 'utf8');

// Mock ort.Tensor (records data + dims; that is all agent.js needs).
const mockOrt = { Tensor: function (t, d, s) { this.type = t; this.data = d; this.dims = s; } };

// Deterministic, batch-invariant mock session: policy/value for row k are a pure
// hash of that row's 567 floats -> identical whether run as batch 1 or batch K.
function makeMockSession() {
  return {
    run: async ({ input }) => {
      const K = input.dims[0];
      const data = input.data;
      const pol = new Float32Array(K * 81);
      const val = new Float32Array(K);
      for (let k = 0; k < K; k++) {
        const base = k * 567;
        let seed = 2166136261 >>> 0;
        for (let i = 0; i < 567; i++) {
          seed ^= ((data[base + i] * 1000) | 0) >>> 0;
          seed = Math.imul(seed, 16777619) >>> 0;
        }
        for (let c = 0; c < 81; c++) {
          seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
          pol[k * 81 + c] = ((seed >>> 8) / 16777216 - 0.5) * 6.0;
        }
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        val[k] = ((seed >>> 8) / 16777216 - 0.5) * 2.0;
      }
      return { policy: { data: pol }, value: { data: val } };
    },
  };
}

const test = `
return (async () => {
  const sess = makeMockSession();
  let s = 123456789 >>> 0;
  const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };

  function reached(nMoves) {
    const g = new GameState();
    for (let i = 0; i < nMoves && g.winner === null; i++) {
      const v = g.validMoves();
      g.makeMove(v[Math.floor(rnd() * v.length)]);
    }
    return g;
  }

  const N = 96;
  let failures = [], checked = 0;

  for (const nMoves of [0, 1, 3, 6, 10, 15]) {
    const g = reached(nMoves);
    if (g.winner !== null || g.validMoves().length < 2) continue;
    checked++;

    // (1) serial vs batched-wave-1 : must be identical
    const serial = await _mctsSearch(g.clone(), sess, 'policy', 'value', N, 1.5);
    const wave1  = await _mctsSearchBatched(g.clone(), sess, 'policy', 'value', N, 1.5, 0, 1);
    let maxDiff = 0;
    for (let c = 0; c < 81; c++) maxDiff = Math.max(maxDiff, Math.abs(serial.pi[c] - wave1.pi[c]));
    const vDiff = Math.abs(serial.rootValue - wave1.rootValue);
    if (maxDiff !== 0 || vDiff > 1e-12 || serial.sims !== wave1.sims) {
      failures.push('pos ' + nMoves + ': wave1!=serial maxPiDiff=' + maxDiff +
                    ' vDiff=' + vDiff + ' sims ' + serial.sims + '/' + wave1.sims);
    }

    // (2) batched wave>1 : valid search
    for (const W of [8, 32]) {
      const b = await _mctsSearchBatched(g.clone(), sess, 'policy', 'value', N, 1.5, 0, W);
      const legal = new Set(g.validMoves());
      let total = 0, illegal = 0;
      for (let c = 0; c < 81; c++) {
        if (b.pi[c] > 0) { total += b.pi[c]; if (!legal.has(c)) illegal++; }
      }
      if (b.sims !== N) failures.push('pos ' + nMoves + ' W' + W + ': sims=' + b.sims + ' != ' + N);
      if (total !== N)  failures.push('pos ' + nMoves + ' W' + W + ': visit sum=' + total + ' != ' + N);
      if (illegal > 0)  failures.push('pos ' + nMoves + ' W' + W + ': ' + illegal + ' illegal-cell visits');
      if (!Number.isFinite(b.rootValue)) failures.push('pos ' + nMoves + ' W' + W + ': non-finite rootValue');
    }
  }

  console.log('positions checked:', checked);
  if (failures.length === 0) {
    console.log('[OK] batched wave-1 == serial (bit-for-bit); wave 8/32 are valid searches');
    return 0;
  }
  console.log('[X] FAILURES:');
  for (const f of failures) console.log('   ' + f);
  return 1;
})();
`;

const fn = new Function('ort', 'performance', 'console', 'makeMockSession',
                        engineSrc + '\n' + agentSrc + '\n' + test);
fn(mockOrt, performance, console, makeMockSession).then(code => process.exit(code));
