# Device-speed benchmark

The play page (`docs/play/`) has a "Test device speed" button. It measures the
device by running the REAL loaded ONNX net -- timed forward passes and timed
MCTS searches -- not a synthetic loop (a synthetic loop does not predict
WASM/WebGPU inference speed across devices). From the measurement it derives a
device tier and a recommended difficulty, so a low-end phone and a gaming PC
each get an honest, self-scaled read.

Implementation: `docs/play/benchmark.js` (self-contained; reuses
`_buildInputTensor` + `_mctsSearch` from `agent.js`). Wired into
`docs/play/index.html`.

## What it measures

- **Forward throughput**: a fixed ~1.2 s wall-clock probe counts completed
  `session.run` calls (warm-up runs discarded). Reports forwards/sec and
  ms/forward. The fixed window self-scales: fast devices do thousands, slow
  devices do dozens, both finish in ~1.2 s.
- **Real move latency**: an actual MCTS search at 16 and 50 sims (median of 3),
  giving the true "Hard move" cost the player would feel.
- **Capabilities** (best-effort, may be null): user-agent, core count,
  `navigator.deviceMemory`, WebGPU availability, WASM SIMD support, screen,
  device-pixel-ratio.
- **Tier + recommendation**: buckets by forwards/sec into
  low / mid / high / ultra, and recommends keeping Hard or dropping to Medium
  based on the measured MCTS-50 latency. "Apply recommended" sets the
  difficulty for the visitor.

## Result payload (`uttt-devbench-1`)

```json
{
  "schema": "uttt-devbench-1",
  "ts": 1721460000000,
  "model": { "name": "expert-iter-v2 gen-19 (champion)", "kind": "champion" },
  "ep": "wasm",
  "caps": { "ua": "...", "platform": "...", "cores": 8, "deviceMemGB": 8,
            "webgpu": true, "wasmSimd": true, "screen": "1920x1080", "dpr": 1 },
  "perf": { "msPerForward": 2.4, "forwardsPerSec": 417, "forwardsSampled": 500,
            "mcts16ms": 42, "mcts50ms": 130 },
  "tier": "ultra", "tierLabel": "Desktop / high-end",
  "recommend": { "autoSims": 100, "hard": "smooth", "note": "..." },
  "benchMs": 3600
}
```

No identity, no game data, no cookies -- device and timing fields only.

## Collecting real-world data (opt-in)

The page is a static GitHub Pages site with no backend, so by default the
benchmark is **local-only**: each visitor sees their own result and a "Copy
result" button; nothing leaves the browser. `BENCH_TELEMETRY_ENDPOINT` at the
top of `benchmark.js` is empty.

To aggregate results across visitors, stand up a tiny collector and set that
constant to its URL. Then a "Share anonymously" button appears; a visitor who
clicks it POSTs the payload above. Nothing is sent without that click.

### Minimal Cloudflare Worker collector

```js
// wrangler.toml: bind a KV namespace as BENCH_KV
export default {
  async fetch(req, env) {
    const cors = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };
    if (req.method === "OPTIONS") return new Response(null, { headers: cors });
    if (req.method !== "POST") return new Response("post only", { status: 405, headers: cors });
    let body;
    try { body = await req.json(); } catch { return new Response("bad json", { status: 400, headers: cors }); }
    if (body.schema !== "uttt-devbench-1") return new Response("bad schema", { status: 400, headers: cors });
    const key = `${Date.now()}-${crypto.randomUUID()}`;
    await env.BENCH_KV.put(key, JSON.stringify(body), { expirationTtl: 60 * 60 * 24 * 90 });
    return new Response("ok", { headers: cors });
  },
};
```

Deploy with `wrangler deploy`, then set in `benchmark.js`:

```js
const BENCH_TELEMETRY_ENDPOINT = 'https://<your-worker>.workers.dev';
```

Read results back with `wrangler kv key list` / `kv key get`, or point the
Worker at D1 / an analytics sink instead of KV. Any endpoint that accepts a
JSON POST works (Vercel/Netlify function, Apps Script web app, etc.).

Privacy: keep it opt-in (the Share button), keep the payload device/perf only,
and set a retention TTL as above.
