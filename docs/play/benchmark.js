/**
 * Device-speed benchmark for the Ultimate Tic-Tac-Toe play page.
 *
 * Measures REAL inference on the currently loaded ONNX net (not a synthetic
 * loop -- a synthetic loop does not predict WASM/WebGPU inference speed across
 * devices). From the measured cost it derives a device tier and a recommended
 * difficulty, so a phone and a gaming PC each get an honest, self-scaled read.
 *
 * Depends on globals from agent.js / uttt_engine.js / ort:
 *   _buildInputTensor, _mctsSearch, GameState, ort
 *
 * Public API:
 *   runDeviceBenchmark(runner, onProgress) -> Promise<result>
 *   formatBenchmark(result) -> html string
 *   shareBenchmark(result) -> Promise<bool>   (opt-in; no-op unless endpoint set)
 *
 * Privacy: nothing is sent anywhere unless BENCH_TELEMETRY_ENDPOINT is set to a
 * collector URL AND the visitor clicks Share. The payload is device/perf only
 * (user-agent, core count, timings) -- no identity, no game data, no cookies.
 */

// Set to your collector URL (e.g. a Cloudflare Worker) to enable opt-in Share.
// Empty = local-only: the benchmark shows each visitor their own result and a
// Copy button; nothing leaves the browser. See docs/DEVICE_BENCH.md.
const BENCH_TELEMETRY_ENDPOINT = '';

const _BENCH_SCHEMA = 'uttt-devbench-1';
const _FWD_WINDOW_MS = 1200;   // fixed wall-clock probe: self-scales fast<->slow
const _WARMUP_FWDS = 4;        // discard: JIT / first-run allocation
const _MCTS_REPS = 3;          // real search reps; report the median

// --------------------------------------------------------------------------
// Capability detection (best-effort; every field may be null on some browsers)
// --------------------------------------------------------------------------

function _wasmSimdSupported() {
  // Minimal valid module whose body uses a v128 (SIMD) opcode. validate() is
  // true only if the engine understands the SIMD proposal.
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

async function _measureForwards(session, onProgress) {
  const state = new GameState();       // empty board is a valid, representative input
  const input = _buildInputTensor(state);

  for (let i = 0; i < _WARMUP_FWDS; i++) await session.run({ input });

  let count = 0;
  const t0 = performance.now();
  let now = t0;
  while (now - t0 < _FWD_WINDOW_MS) {
    await session.run({ input });
    count++;
    now = performance.now();
    if (onProgress && (count & 7) === 0) {
      onProgress(Math.min(0.7, 0.15 + 0.55 * ((now - t0) / _FWD_WINDOW_MS)));
    }
  }
  const elapsed = now - t0;
  return {
    forwardsSampled: count,
    msPerForward:    elapsed / count,
    forwardsPerSec:  Math.round((count / elapsed) * 1000),
  };
}

async function _measureMcts(runner, nSims, reps) {
  const times = [];
  for (let r = 0; r < reps; r++) {
    const state = new GameState();
    const t0 = performance.now();
    await _mctsSearch(state, runner.session, runner._policyName,
                      runner._valueName, nSims, runner.cPuct || 1.5);
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];   // median
}

// --------------------------------------------------------------------------
// Interpretation
// --------------------------------------------------------------------------

function _tier(forwardsPerSec) {
  if (forwardsPerSec >= 400) return { key: 'ultra', label: 'Desktop / high-end' };
  if (forwardsPerSec >= 130) return { key: 'high',  label: 'Laptop / fast tablet' };
  if (forwardsPerSec >= 35)  return { key: 'mid',   label: 'Phone / tablet' };
  return { key: 'low', label: 'Low-end / older phone' };
}

function _recommend(msPerForward, mcts50ms) {
  // Aim for a Hard move that feels responsive (~600 ms budget).
  const TARGET_MS = 600;
  const autoSims = Math.max(8, Math.min(100, Math.round(TARGET_MS / msPerForward)));
  let hard, note;
  if (mcts50ms <= 700) {
    hard = 'smooth';
    note = 'Hard (MCTS-50) runs smoothly on this device.';
  } else if (mcts50ms <= 2500) {
    hard = 'slow';
    note = `Hard is playable but slow here (~${(mcts50ms / 1000).toFixed(1)}s/move). `
         + `Medium is snappier; ~${autoSims} sims would hit a ~0.6s move.`;
  } else {
    hard = 'too-slow';
    note = `Hard is heavy on this device (~${(mcts50ms / 1000).toFixed(1)}s/move). `
         + 'Easy or Medium will feel much better.';
  }
  return { autoSims, hard, note };
}

// --------------------------------------------------------------------------
// Orchestration
// --------------------------------------------------------------------------

async function runDeviceBenchmark(runner, onProgress) {
  if (!runner || !runner.session) throw new Error('No model loaded to benchmark.');
  const report = onProgress || (() => {});
  const started = performance.now();

  report(0.05);
  const caps = _benchCapabilities();
  await new Promise(r => setTimeout(r, 0));   // let the UI paint the panel first

  const fwd = await _measureForwards(runner.session, report);

  report(0.8);
  const mcts16 = await _measureMcts(runner, 16, _MCTS_REPS);
  report(0.9);
  const mcts50 = await _measureMcts(runner, 50, _MCTS_REPS);
  report(0.98);

  const tier = _tier(fwd.forwardsPerSec);
  const rec  = _recommend(fwd.msPerForward, mcts50);

  report(1.0);
  return {
    schema:  _BENCH_SCHEMA,
    ts:      Date.now(),
    model:   { name: runner.name, kind: (typeof _modelKind !== 'undefined' ? _modelKind : null) },
    ep:      'wasm',
    caps,
    perf: {
      msPerForward:    Math.round(fwd.msPerForward * 100) / 100,
      forwardsPerSec:  fwd.forwardsPerSec,
      forwardsSampled: fwd.forwardsSampled,
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
  return ''
    + `<div class="bench-tier bench-tier-${res.tier}">${res.tierLabel}</div>`
    + _row('Model tested', `${res.model.name}`)
    + _row('Speed', `${p.forwardsPerSec} forwards/sec (${p.msPerForward} ms each)`)
    + _row('Hard move (MCTS-50)', `${(p.mcts50ms / 1000).toFixed(2)} s`)
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
