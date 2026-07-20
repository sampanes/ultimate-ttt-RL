# Device-speed benchmark

The play page (`docs/play/`) has a "Test device speed" button. It measures the
device by running the REAL loaded ONNX net -- timed forward passes and timed
MCTS searches -- not a synthetic loop (a synthetic loop does not predict
WASM/WebGPU inference speed across devices). The result is persisted to the
browser and feeds the constant-time play model below.

Implementation: `docs/play/benchmark.js` (self-contained; reuses
`_buildInputTensor` + `_mctsSearch` from `agent.js`). Wired into
`docs/play/index.html`.

## Constant-time play, device-scaled quality

**Impossible** (the searching tier -- champion net + tactical-guarded MCTS) is
**time-bounded**: the MCTS runs until a fixed ~1 s per-move budget
(`_MOVE_BUDGET_MS` in `index.html`), then plays its best move. So every device
answers in about the same wall-clock time; the number of simulations it fits in
that second -- and therefore the move quality -- scales with the hardware. A
gaming PC (esp. on WebGPU) searches deep; an old phone searches shallow; both
reply in ~1 s. (`agent.js` `_mctsSearch` takes an optional `deadlineMs`; when set
it caps the loop by wall clock instead of a fixed sim count.) The other tiers are
a single forward: Easy/Medium the pocket net, Hard the champion net with a 1-ply
tactical overlay and no search.

The benchmark's job is to tell you HOW MUCH search you get in that second
(the device tier and the estimated nodes/move), and to persist it.

## Execution provider: WebGPU with a safe fallback

At load, `loadAgent` (in `agent.js`) creates the searching net on WASM and, when
the browser exposes `navigator.gpu`, also on WebGPU, then keeps the provider with
the lowest **per-leaf forward cost for the workload it will actually run**: WASM
serial (one batch-1 forward per leaf) vs WebGPU **batched** (one batch-wave
forward per wave, cost divided by the wave size). This matters because a GPU
loses a batch-1 latency race -- dispatch overhead swamps a tiny forward -- but
wins decisively at batch-K, which is what the search actually does; racing
batch-1 would wrongly bench the GPU and it would never engage. WebGPU is trusted
only if it also **agrees numerically** with WASM at both batch-1 and every row of
a batch-wave forward (a buggy GPU backend is disqualified; WASM is the floor, so
WASM devices are unchanged). When WebGPU wins, the search runs a **batched
(leaf-parallel) MCTS**: descend to a wave of leaves via virtual loss and evaluate
them all in ONE forward. A batch-of-K forward costs ~the same wall clock as
batch-1 on a GPU, so the same ~1 s budget buys many more simulations.
The batched search is bit-for-bit identical to the serial one at wave size 1
(guarded by `scripts/test_batched_mcts.js`), so it is a pure speed lever, not a
behavior change. The champion net is raw 1-ply tactical (one forward per move),
so it stays on WASM; the provider choice there is irrelevant.

## What it measures

- **Forward throughput**: several fixed ~1.5 s wall-clock rounds count completed
  `session.run` calls (warm-up discarded), reported as the median across rounds
  plus a spread %, so a thermal spike in one round does not skew the read. The
  fixed window self-scales: fast devices do thousands, slow devices do dozens.
- **Real move cost**: actual MCTS searches at 16 and 50 sims (median of 5) on
  the SAME engine the device will play (batched on WebGPU, serial on WASM),
  giving the true per-simulation cost -- including tree overhead and any batching
  benefit -- so the derived `autoSims` reflects real play.
- **Capabilities** (best-effort, may be null): user-agent, core count,
  `navigator.deviceMemory`, WebGPU availability, WASM SIMD support, screen, DPR.
- **Tier + tuned depth**: estimates `autoSims` = how many simulations fit in the
  ~1 s move budget (from the measured per-sim cost), then buckets on THAT into
  shallow/moderate/strong/deep. Tier describes reachable search depth, not a
  guess at the hardware class -- the pocket net is tiny enough that raw
  forwards/sec saturates on any modern device (a fast phone matches a desktop),
  so throughput cannot tell device classes apart; per-sim search cost can.

## Persistence

The result is saved to `localStorage` under `uttt_devbench_v1`. On a return
visit from the same browser/device, the page restores it and shows a one-line
chip ("Your device: <tier> -- about N nodes per 1 s move, tested <when>").
Running the test again overwrites it. Cleared if the user clears site data.

## Result payload (`uttt-devbench-2`)

```json
{
  "schema": "uttt-devbench-2",
  "ts": 1721460000000,
  "model": { "name": "arena:21@hof", "kind": "pocket" },
  "ep": "webgpu",
  "waveSize": 32,
  "caps": { "ua": "...", "platform": "...", "cores": 8, "deviceMemGB": 8,
            "webgpu": true, "wasmSimd": true, "screen": "1920x1080", "dpr": 1 },
  "perf": { "msPerForward": 2.4, "forwardsPerSec": 417, "roundsMs": [2.3,2.4,2.5],
            "spreadPct": 8, "mcts16ms": 42, "mcts50ms": 130 },
  "tier": "ultra", "tierLabel": "Deep search",
  "recommend": { "autoSims": 384, "perSimMs": 2.6, "hard": "strong", "note": "..." },
  "benchMs": 9500
}
```

No identity, no game data, no cookies -- device and timing fields only.

## Collecting real-world data (opt-in)

The page is a static GitHub Pages site with no backend, so by default the
benchmark is **local-only**: each visitor sees and stores their own result;
nothing leaves the browser. `BENCH_TELEMETRY_ENDPOINT` at the top of
`benchmark.js` is empty.

To aggregate results across visitors, stand up a tiny collector and set that
constant to its URL. A "Share anonymously" button then appears; a visitor who
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
    if (body.schema !== "uttt-devbench-2") return new Response("bad schema", { status: 400, headers: cors });
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
