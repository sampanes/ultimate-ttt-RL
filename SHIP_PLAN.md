# Ship Plan — Current Milestones

This is the authoritative execution order. Open work lives in `PENDING.md`.

## Product goals

1. **Pocket champion:** smallest strong player that runs as a static client-side app on GitHub Pages. No Python server.
2. **Oracle champion:** practically unbeatable player using deep search, hosted on Hugging Face. Also the teacher for the pocket model.
3. **Proof track:** exact endgame solving and adversarial validation.

The long-term pipeline: `large search teacher → distilled tiny policy/value net → small browser MCTS`

## Milestone board

| Milestone | Status | Exit gate |
|---|---|---|
| M0. Produce viable candidates | **COMPLETE** | Two-point frontier locked with peak checkpoints and SHA-256 hashes |
| M1. One-command benchmark suite | **COMPLETE** | `scripts/benchmark_suite.py` resolves arch from state, fail-closed on any mismatch, writes JSON + Markdown provenance |
| M2. Select pocket and oracle bases | **COMPLETE** | Both finalists cleared independent panel; results in `RESULT_M2.md` |
| M3. Ship Pocket v1 in the browser | **NEXT** | Static client runs quantized net + search; measured size and latency |
| M4. Train and host the oracle champion | READY (after M3) | Search-trained champion clears independent panel at deployment budget |
| M5. Distill and squeeze Pocket v2 | BLOCKED BY M4 | Student preserves agreed fraction of oracle strength at smaller footprint |
| M6. Harden claims and exactness | LATER | Tactical/opening/endgame audits and external-bot results published |

Do not restart long Arena training before M3. More closed-loop ELO is not the missing information.

## M0 — Candidate generation — COMPLETE

Arena stopped at **1,630 completed chunks**. Two-point size/strength frontier:

| Role | Arena record | Checkpoint bytes | SHA-256 |
|---|---|---:|---|
| Pocket finalist | id 21 `06-26-26` | 5,156,981 | `7498a31f…` |
| Strength finalist | id 22 `06-27-26` | 27,074,677 | `400374b1…` |

Files: `models/arena/hall_of_fame/06-26-26_elo1819.pt` and `06-27-26_elo1864.pt`.
Note: ELO in filename is one chunk stale — use the state record, not the filename.

## M1 — One-command benchmark suite — COMPLETE

```powershell
python -m scripts.benchmark_suite `
  --candidate arena:21@hof `
  --anchors lottery,nn_big8,winblock,center,first `
  --candidate-sims 0,25,100 `
  --oracle-sims 400 `
  --openings standard `
  --out results/arena-21
```

Resolves architecture from `arena_state.json`, validates checkpoint by SHA-256, alternates sides across frozen legal openings, writes machine-readable JSON + Markdown provenance with git commit and ruleset hashes.

## M2 — Independent finalist playoff — COMPLETE

Result in `RESULT_M2.md`: **pocket base = `arena:21@hof`** (clears int8 ≤ 5 MB gate), **oracle base = `arena:22@hof`** (strongest by every aggregate). Key finding: 1-ply tactical filtering matches or beats MCTS-100, so the value head — not search depth — is the current limiter. `winblock` is the shared tactical blind spot.

Endgame grader (`RESULT_GRADING.md`): tactical overlay provably quarters the late-game blunder rate on both finalists.

## M3 — Pocket v1: static browser player

Prep numbers in `RESULT_M3_PREP.md`: pocket base estimates ~1.29 MB int8 (4× under the 5 MB gate), ~0.8 ms/move on CPU vs ~1.1 ms on CUDA. Build WASM CPU path first; WebGPU is optional.

Deliverables:
- Export pocket base to ONNX with fixed input/output contract
- Apply static int8 quantization using a representative UTTT position set
- Verify PyTorch vs ONNX vs int8 policy/value parity; rerun M2 benchmark on quantized model
- Port rules and MCTS to TypeScript/WASM; lock golden vectors against Python
- Measure cold download, bundle bytes, peak browser memory, move latency on desktop and phone
- Publish static GitHub Pages build

Size gates: quantized model ≤ 5 MB; entire static bundle ≤ 10 MB compressed.

M3 exit: static page plays a complete legal game offline, identifies its model hash, passes post-quantized M2 benchmark and golden-vector suite.

## M4 — Oracle champion and Hugging Face deployment

Optimizes strength, not footprint. See `M4_DESIGN.md` for the build spec.

Deliverables:
- Replace unbounded shaped-return value head with bounded outcome value target (tanh + game outcome)
- AlphaZero-style self-play using MCTS visit distributions as policy targets
- Gate promotions against frozen prior champion and M1 anchor panel
- Tune deep-search simulation budget against actual latency and strength curves
- Deploy champion + Python MCTS to `ultimate-tic-tac-toe-hf` on Hugging Face

M4 exit: hosted app reports model/search provenance and clears M2 panel at deployment budget.

## M5 — Distill and squeeze Pocket v2

Use M4 oracle as teacher:
- Generate diverse position corpus with MCTS visit distributions and outcome values
- Train smaller students against teacher targets; use QAT if post-training int8 loses strength
- Search architecture as a Pareto problem: benchmark score and blunder rate vs bytes and latency
- Keep browser search budget in the objective

M5 exit: Pocket v2 is strictly smaller or stronger than Pocket v1 without regressing the fixed tactical/opening suite.

## M6 — Hardening, exactness, and public claims

- Build committed adversarial opening suite from observed losses
- Add live exact endgame alpha-beta where the remaining subtree is tractable
- Generate a tiny opening book only if it changes measured failures
- Benchmark against compatible external public UTTT bots (see `GRADING_AND_ORACLE.md` Part 8)
- Publish reproducible W/D/L, score, latency, footprint, and model hashes
- Use bounty workflow in `BOUNTY.md` to turn real losses into regression tests

M6 exit: every public strength claim maps to a reproducible report. "Perfect play" requires exact-solver evidence, not MCTS confidence.

## Stop/go rules

- No multi-hour training run without a named milestone gate it can satisfy
- No candidate selection from closed-loop Arena ELO alone
- No benchmark result without checkpoint hash and architecture provenance
- No repeated empty-board deterministic games presented as a large sample
- No quantization or browser port without parity and strength regression tests
- Prefer the smallest experiment that can reject the current hypothesis
