/**
 * Device-speed benchmark for the Ultimate Tic-Tac-Toe play page.
 *
 * Measures REAL inference on the currently loaded ONNX net (not a synthetic
 * loop -- a synthetic loop does not predict WASM/WebGPU inference speed across
 * devices). From the measurement it derives a device tier, a persisted result,
 * and a device-tuned search depth (autoSims): a slow phone gets a shallow
 * search, a fast desktop gets a deep one.
 *
 * Depends on globals from agent.js / uttt_engine.js / ort:
 *   _buildInputTensor, _mctsSearch, GameState, ort
 *
 * Public API:
 *   runDeviceBenchmark(runner, onProgress) -> Promise<result>
 *   formatBenchmark(result) -> html string
 *   saveBenchResult(result) / loadBenchResult() -> localStorage persistence
 *   benchRelTime(ts) -> "3 days ago"
 *   shareBenchmark(result) -> Promise<bool>   (opt-in; no-op unless endpoint set)
 *
 * Privacy: nothing is sent anywhere unless BENCH_TELEMETRY_ENDPOINT is set to a
 * collector URL AND the visitor clicks Share. The result is persisted only in
 * this browser's localStorage. Payload is device/perf only -- no identity, no
 * game data, no cookies.
 */

// Set to your collector URL (e.g. a Cloudflare Worker) to enable opt-in Share.
// Empty = local-only. See docs/DEVICE_BENCH.md.
const BENCH_TELEMETRY_ENDPOINT = '';

const _BENCH_SCHEMA = 'uttt-devbench-2';
const BENCH_STORE_KEY = 'uttt_devbench_v1';

// A deliberately thorough measurement: several timed rounds, median-of-rounds,
// so thermal spikes and scheduler jitter do not skew the read. Takes a few
// seconds on a fast device, longer on a slow one -- that is the point.
const _FWD_ROUNDS   = 3;
const _FWD_WINDOW_MS = 1500;   // per round; fixed wall-clock -> self-scales
const _WARMUP_FWDS  = 6;       // discard: JIT / first-run allocation
const _MCTS_REPS    = 5;       // real search reps; report the median

// Hard mode plays with a fixed per-move time budget (see _MOVE_BUDGET_MS in
// index.html): every move takes ~the same wall-clock time on every device, and
// the search depth (quality) is whatever fits. This constant here mirrors that
// budget so the benchmark can report the depth your device reaches in it.
const _TARGET_MOVE_MS = 1000;
const _SIMS_FLOOR = 4;
const _SIMS_CEIL  = 4096;   // safety cap only; the time budget is the real limit

// --------------------------------------------------------------------------
// Capability detection (best-effort; every field may be null on some browsers)
// --------------------------------------------------------------------------

function _wasmSimdSupported() {
  try {
    return WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3,
      2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
    ]));
  } catch (_) {
    return false;
  }
}

function _benchCapabilities() {
  const nav = navigator || {};
  return {
    ua:          nav.userAgent || null,
    platform:    nav.platform || null,
    cores:       nav.hardwareConcurrency || null,
    deviceMemGB: nav.deviceMemory || null,   // Chromium only; coarse buckets
    webgpu:      !!nav.gpu,
    wasmSimd:    _wasmSimdSupported(),
    screen:      (typeof screen !== 'undefined')
                   ? `${screen.width}x${screen.height}` : null,
    dpr:         (typeof devicePixelRatio !== 'undefined')
                   ? Math.round(devicePixelRatio * 100) / 100 : null,
  };
}

// --------------------------------------------------------------------------
// Measurement primitives
// --------------------------------------------------------------------------

async function _forwardRound(session, input, windowMs) {
  let count = 0;
  const t0 = performance.now();
  let now = t0;
  while (now - t0 < windowMs) {
    await session.run({ input });
    count++;
    now = performance.now();
  }
  return (now - t0) / count;   // ms per forward this round
}

async function _measureForwards(session, onProgress) {
  const state = new GameState();       // empty board: valid, representative input
  const input = _buildInputTensor(state);

  for (let i = 0; i < _WARMUP_FWDS; i++) await session.run({ input });

  const perRound = [];
  for (let r = 0; r < _FWD_ROUNDS; r++) {
    perRound.push(await _forwardRound(session, input, _FWD_WINDOW_MS));
    if (onProgress) onProgress(0.1 + 0.55 * ((r + 1) / _FWD_ROUNDS));
    await new Promise(res => setTimeout(res, 0));   // let the UI breathe
  }
  perRound.sort((a, b) => a - b);
  const median = perRound[Math.floor(perRound.length / 2)];
  return {
    msPerForward:   median,
    forwardsPerSec: Math.round(1000 / median),
    roundsMs:       perRound.map(v => Math.round(v * 100) / 100),
    spreadPct:      Math.round((perRound[perRound.length - 1] - perRound[0]) / median * 100),
  };
}

async function _measureMcts(runner, nSims, reps, onProgress, base, span) {
  const times = [];
  for (let r = 0; r < reps; r++) {
    const state = new GameState();
    const t0 = performance.now();
    // Measure the SAME search path the device will actually play (batched on
    // WebGPU, serial on WASM), so the per-sim cost -- and the autoSims we derive
    // from it -- reflect real play, not a slower path.
    await _mctsSearch(state, runner.session, runner._policyName,
                      runner._valueName, nSims, runner.cPuct || 1.5,
                      0, runner.waveSize || 1);
    times.push(performance.now() - t0);
    if (onProgress) onProgress(base + span * ((r + 1) / reps));
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];   // median
}

// --------------------------------------------------------------------------
// Interpretation
// --------------------------------------------------------------------------

// Tier reflects the SEARCH DEPTH the device reaches in the ~1s move budget --
// the thing that actually varies and decides play quality -- NOT a guess at the
// hardware class. The pocket net is tiny, so raw forward throughput saturates on
// any modern device (a fast phone posts nearly the same forwards/sec as a
// desktop); per-sim search cost, which includes tree overhead, is what separates
// them. Bucket on autoSims so a genuine potato reads "Shallow" and a fast phone
// reads by what it can actually do, not a bogus "Desktop" label.
function _tier(autoSims) {
  if (autoSims >= 300) return { key: 'ultra', label: 'Deep search' };
  if (autoSims >= 120) return { key: 'high',  label: 'Strong search' };
  if (autoSims >= 40)  return { key: 'mid',   label: 'Moderate search' };
  return { key: 'low', label: 'Shallow search' };
}

// Hard mode is time-bounded: every move takes ~_TARGET_MOVE_MS regardless of
// device, and the search depth (quality) is whatever the device fits in it.
// autoSims is the estimated depth reached in that budget, from the MEASURED
// per-sim cost (mcts50 includes tree overhead, not just the raw forward).
function _recommend(mcts50ms) {
  const perSimMs = mcts50ms / 50;
  const autoSims = Math.max(_SIMS_FLOOR,
                            Math.min(_SIMS_CEIL, Math.round(_TARGET_MOVE_MS / perSimMs)));
  const budgetS = (_TARGET_MOVE_MS / 1000).toFixed(0);
  let hard, note;
  if (autoSims >= 200) {
    hard = 'strong';
    note = `Strong: in ~${budgetS}s per move your device searches ~${autoSims} nodes -- deep, sharp play.`;
  } else if (autoSims >= 40) {
    hard = 'ok';
    note = `Moderate: ~${autoSims} nodes per ~${budgetS}s move -- solid search depth.`;
  } else {
    hard = 'shallow';
    note = `Shallow: only ~${autoSims} nodes fit in ~${budgetS}s here, so Hard moves are quick but weaker. `
         + 'Medium (the instant raw net) may play similarly and with no wait.';
  }
  return { autoSims, perSimMs: Math.round(perSimMs * 100) / 100, hard, note };
}

// --------------------------------------------------------------------------
// Orchestration
// --------------------------------------------------------------------------

async function runDeviceBenchmark(runner, onProgress) {
  if (!runner || !runner.session) throw new Error('No model loaded to benchmark.');
  const report = onProgress || (() => {});
  const started = performance.now();

  report(0.03);
  const caps = _benchCapabilities();
  await new Promise(r => setTimeout(r, 0));   // let the UI paint the panel first

  const fwd = await _measureForwards(runner.session, report);   // -> ~0.65

  const mcts16 = await _measureMcts(runner, 16, _MCTS_REPS, report, 0.65, 0.12);
  const mcts50 = await _measureMcts(runner, 50, _MCTS_REPS, report, 0.77, 0.2);
  report(0.99);

  const rec  = _recommend(mcts50);
  const tier = _tier(rec.autoSims);

  report(1.0);
  return {
    schema:  _BENCH_SCHEMA,
    ts:      Date.now(),
    model:   { name: runner.name, kind: (typeof _modelKind !== 'undefined' ? _modelKind : null) },
    ep:      runner.ep || 'wasm',
    waveSize: runner.waveSize || 1,
    caps,
    perf: {
      msPerForward:    Math.round(fwd.msPerForward * 100) / 100,
      forwardsPerSec:  fwd.forwardsPerSec,
      roundsMs:        fwd.roundsMs,
      spreadPct:       fwd.spreadPct,
      mcts16ms:        Math.round(mcts16),
      mcts50ms:        Math.round(mcts50),
    },
    tier:      tier.key,
    tierLabel: tier.label,
    recommend: rec,
    benchMs:   Math.round(performance.now() - started),
  };
}

// --------------------------------------------------------------------------
// Persistence
// --------------------------------------------------------------------------

function saveBenchResult(res) {
  try { localStorage.setItem(BENCH_STORE_KEY, JSON.stringify(res)); return true; }
  catch (_) { return false; }
}

function loadBenchResult() {
  try {
    const raw = localStorage.getItem(BENCH_STORE_KEY);
    if (!raw) return null;
    const r = JSON.parse(raw);
    // Accept only recognised schema shapes with the fields the UI needs.
    if (!r || !r.recommend || typeof r.recommend.autoSims !== 'number') return null;
    return r;
  } catch (_) { return null; }
}

function benchRelTime(ts) {
  const s = Math.max(0, (Date.now() - ts) / 1000);
  if (s < 90)      return 'just now';
  if (s < 5400)    return `${Math.round(s / 60)} min ago`;
  if (s < 129600)  return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} days ago`;
}

// --------------------------------------------------------------------------
// Rendering + optional sharing
// --------------------------------------------------------------------------

function _row(k, v) {
  return `<div class="bench-row"><span class="bench-k">${k}</span>`
       + `<span class="bench-v">${v}</span></div>`;
}

function formatBenchmark(res) {
  const p = res.perf, c = res.caps;
  const cores = c.cores ? `${c.cores} cores` : 'cores n/a';
  const mem   = c.deviceMemGB ? `${c.deviceMemGB} GB` : 'RAM n/a';
  const accel = [c.wasmSimd ? 'WASM SIMD' : 'WASM (no SIMD)',
                 c.webgpu ? 'WebGPU avail' : null].filter(Boolean).join(', ');
  const stab  = (p.spreadPct != null) ? ` (+/-${p.spreadPct}% across rounds)` : '';
  const engine = (res.ep === 'webgpu')
    ? `WebGPU${(res.waveSize > 1) ? ` (batched x${res.waveSize})` : ''}`
    : 'WASM (CPU)';
  return ''
    + `<div class="bench-tier bench-tier-${res.tier}">${res.tierLabel}</div>`
    + _row('Model tested', `${res.model.name}`)
    + _row('Engine', engine)
    + _row('Speed', `${p.forwardsPerSec} forwards/sec (${p.msPerForward} ms each)${stab}`)
    + _row('Hard move (MCTS-50)', `${(p.mcts50ms / 1000).toFixed(2)} s`)
    + _row('Tuned search depth', `${res.recommend.autoSims} sims/move`)
    + _row('Acceleration', accel)
    + _row('Hardware', `${cores}, ${mem}`)
    + `<div class="bench-note">${res.recommend.note}</div>`;
}

async function shareBenchmark(res) {
  if (!BENCH_TELEMETRY_ENDPOINT) return false;
  try {
    await fetch(BENCH_TELEMETRY_ENDPOINT, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(res),
      keepalive: true,
    });
    return true;
  } catch (_) {
    return false;
  }
}
