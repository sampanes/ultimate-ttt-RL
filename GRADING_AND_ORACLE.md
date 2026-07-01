# Grading and oracle design

> The arena is a **finder and a ranker**, never a **prover**. Trust the blunder-rate; distrust the closed-loop ELO.

## Design principles (Parts 1-5 summary)

- **Anchors are ruler tick-marks, not foils.** ELO is purely relative; frozen non-learning anchors keep the scale from drifting. Random (600) pins the floor; deterministics (700-950) give low-variance readings; lottery/nn_big8 (1300/1400) are upper rungs.
- **Beating siblings proves almost nothing.** A lineage that only competes against its own descendants co-adapts to shared blind spots. Closed-loop arena ELO inflates without real skill improvement.
- **Orthogonality beats count.** Diversity of the panel buys confidence, not size. Axes: siblings / random / first-move handicap / gregory (second-best by shallow search).
- **Gregory is a depth-limited oracle.** Shallow α-β playing second-best (training foil) and full-depth α-β proving truth (certifier) are the same instrument. Build once, dial depth.
- **Confidence ladder:** beats siblings (weak) → beats diverse panel (strong) → zero blunder-rate vs complete solver (proof). Only the solver crosses the last gap.
- **Open check before Phase 6:** whether the league selects on anchor-performance or only on pool ELO lives in `league_manager.py:217+`. Verify this is torch-free to fix.

---

## Part 6 -- The endgame oracle: grading on KNOWN-CORRECT answers

This is **Phase 5c made concrete** (`SHIP_PLAN.md`), not new scope and *not* a
reversal of the crossed-off tablebase. The distinction is the unlock:

> The tablebase **crossed off in `CEILING_AND_CAPACITY.md`** was crossed off for
> **footprint** -- it would have shipped *inside the deployed artifact*, and the whole
> goal is minimum total footprint. This table is an **offline grader. It never ships.**
> It lives on the dev box. So the footprint objection **simply does not apply.**

### "Only in the endgame" is the feature, not the limitation

UTTT only ever **adds** pieces (no captures), so a near-terminal board has few empty
cells -> a **shallow** remaining tree -> **trivially alpha-beta-solvable**. You *can* prove
endgame moves cheaply; you *can't* cheaply prove opening moves (deep tree). Grading
the provable subset gives **rigorous partial signal**: "of the moves made in
positions I can actually solve, what fraction were optimal?" That is the honest-signal
discipline in its purest form -- a trustworthy number on *part* of the game beats a
hand-wavy number on *all* of it.

### Two ways to build it

1. **Memoize-on-demand (best first cut).** Don't enumerate anything. Solve each
   endgame position with live alpha-beta the first time it's hit, cache the result, cap the
   cache. It fills *only* with positions the grading suite actually touches -- this is
   the existing **5a live solve + ephemeral TT**, just given a disk budget and
   persisted between runs. For a *grader*, the working set is far smaller than the
   full frontier, so this may make the whole GB-vs-TB question moot.
2. **Precomputed retrograde table.** Build the whole "<= K empty cells" frontier once
   via backward induction, store it, reuse across **thousands** of gradings. The fixed
   GB/TB earns its keep **only if you grade often** -- and Phase 6 (architecture search
   over a zoo of tiny nets) is exactly "evaluate thousands of candidates," so the
   amortization argument holds *there* and nowhere else.

### When the cache fills -- not automatically, and not only at a full board

The cache only grows when the **solver actually runs**, which is *not* during normal
self-play training (there the agent just does NN forward passes -- the alpha-beta solver is
nowhere in that loop, so nothing is solved or cached). Two things invoke it: the
**grading audit** (you choose the positions) and **5a live-solve** (the playing engine
switches to exact alpha-beta near the end of a game -- the only "fills during play" path, and
only if enabled).

And "near the end" is **not** "board nearly full." What makes a position cheap to solve
is **effective subtree size**, not empty-cell count. Empty-count is the naive proxy
(and with it you *do* wait for a full board); the better trigger is **budget-capped
alpha-beta** -- try to solve within a node budget, take the proven value if it returns, else
fall back to net+MCTS. That opportunistically solves any position whose true subtree is
small -- including **forced wins with the board half-empty** (forcing lines, and UTTT's
constraint rule pinning you to one mini, both collapse the branching). So you catch the
*meaningful* endgames early, not just the full-board grind.

### Dedup as you cache -- one canonical key (the "alphabetical order" instinct)

Right instinct: canonicalize so equivalents collapse to a single entry. Three layers:

1. **Transpositions** (same position, different move order) -- key the cache by the
   **position** `(board, constraint, side-to-move)`, never by the move path, and
   transpositions merge for free. This is exactly what a *transposition table* is.
2. **D4 board symmetry** (~8x) -- the 3x3-of-3x3 board is symmetric under the 8
   rotations/reflections of the square (applied to macro and micro together; the
   constraint rule is preserved). Store the **lexicographically smallest** of a
   position's 8 images -- literally your "alphabetical order so duplicates are obvious."
   Remember which transform you applied and **un-apply it to the returned move** (a
   color swap doesn't move cells, so only the D4 part needs undoing).
3. **Side-to-move / color swap** (~2x) -- canonicalize so the mover is always "X" (swap
   colors + turn). This is the *same* change as the pending perspective fix (item 1):
   one canonicalization helps the net **and** halves the cache.

Stacked, that's a ~16-fold equivalence class stored once. Tradeoff by mode: a small
**in-memory** memoize cache may bother only with transpositions (cheap, and the working
set is small anyway); a **precomputed on-disk** table should fold the full symmetry --
it's one of the biggest storage levers (Part 7's compression table).

### How to grade -- and the bias to dodge

- **agent vs oracle** (oracle plays perfectly) -> outcome signal: can it hold a drawn
  endgame, can it convert a won one;
- **grade each agent endgame move against the oracle's proven-best** -> per-move
  **blunder-rate** (much finer);
- **GOLD -- seed from known-solved positions.** Free-play has selection bias: a
  *strong* agent rarely *enters* a lost endgame, so organic games under-sample exactly
  the positions where blunders are most diagnostic -- and the *dual* is just as biased:
  decisive games usually end **before** the board fills (a macro line completes with
  cells to spare), so the games that *do* reach a near-full endgame skew toward
  high-level draws or low-level flailing -- degenerate boards a good game never enters.
  So don't just watch games -- take
  solved positions (won, lost, **and** drawn) and force the agent to play *from* them,
  checking whether it finds the proven best move. That's a real test suite, not a
  passive observation. This *is* the "test suite + a number" Phase 5c describes.

---

## Part 7 -- Compression & the GB-vs-TB question *(open -- measure before committing disk)*

The "endgame is cheap to *solve*" claim is solid (shallow tree). Whether a *complete*
frontier is cheap to *store* is the open question, and **UTTT is not chess**: chess
endgame tables are small because *material is removed* (few pieces, few combinations);
UTTT endgames are near-**full** boards, and the count of reachable near-terminal
positions is **large** because the board is 81 cells. So size depends entirely on how
deep "endgame" goes.

### The decision reduces to two curves to measure

- **N(K)** = number of *reachable* positions with <= K empty cells. Grows **fast** per
  K -- each extra ply roughly multiplies, it does not add.
- **b** = compressed **bytes per position** after the tricks below.

Then `storage(K) ~= N(K) x b`; pick the largest K with `N(K).b <= budget`. The blunt
consequence: **GB -> TB buys ~1000x positions, which is only a *few more plies* of K**,
because N(K) explodes -- not proportionally "deeper." Cheap experiment to run first: a
counting pass (retrograde BFS from terminals, or forward BFS capped at distance-from-
full) that just *plots N(K)*. That curve answers "how much bang for a TB" before a
single byte of table is committed.

### The compression toolkit (chess Syzygy ideas, mapped to UTTT)

Chess's modern **Syzygy** tablebases (Ronald de Man) are the clever-compression
reference. The transferable ideas, and what each does here:

| Chess trick | What it does | UTTT mapping |
|---|---|---|
| **WDL bitbase** (store win/draw/loss, *not* distance-to-mate) | 3 values ~= ~1.6 bits vs many bits for DTM; the classic 8-16x win | UTTT has **no 50-move rule / no DTZ** -- additive game, so you *only ever need WDL*. This small representation is free here. (Keep DTM only if you also want "win *fastest*.") |
| **Symmetry reduction** (store one representative per symmetry class) | ~8x in chess (fewer without pawns) | **D4 board symmetry** of the 9x9 macro/micro structure -- already used for the 5b opening book -- folds ~8x (must respect the constraint mechanic). |
| **Side-to-move canonicalization** | negamax symmetry -> store one side | ~2x, and it's **literally the pending perspective-canonicalization (item 1)** -- the same change that helps the net also halves the table. |
| **Don't-care exploitation** | illegal/unreachable/dominated slots may take *any* value -> choose values that maximize compressibility | UTTT legality is **extremely sparse** (turn parity + constraint + mini-winner consistency), so a raw index is almost all don't-cares -- huge freedom to spend on run-length. |
| **Dense combinatorial index** (material-class ranking) | map only legal positions to a contiguous integer range | **The single biggest lever.** Raw 3^81 ~= 4x10^38 configs but legal reachable positions are a vanishing fraction; a dense rank over *legal* positions vs a sparse raw index is the difference between feasible and absurd. |
| **Block compression** (RE-PAIR / LZ on top) | squeeze the residual after the above | general-purpose, applied last. |

> **Net read:** the representation you actually need (WDL, additive, no DTZ) is
> already the small one, and D4 + side-to-move + dense legal indexing stack
> multiplicatively before any general-purpose compressor runs. The real lever isn't
> the compressor -- it's **never storing illegal positions and folding symmetry.** But
> the *deciding* number is N(K), and that's an experiment, not a guess.

---

## Part 8 -- External bots: the pre-deploy reality check

Before shipping, **play strong public UTTT bots** -- manually for a gut check,
**programmatically** for a real record. They're the one validator you can get *before*
deploy that is genuinely **independent of your gene pool**: different training,
different heuristics, different blind spots -- the "outsider with an unexplored
heuristic" from the confidence ladder (Part 5), available early. Beating your own
siblings proves nothing (Part 2); holding your own against *someone else's* bot is the
first real evidence you won't embarrass yourself in public.

**Treat "undefeated" claims with salt.** Most mean "undefeated vs casual humans" or
"in my own testing," not *proven optimal*. So an external bot is a **strong anchor,
not an oracle** -- it calibrates, it doesn't certify. The signal is asymmetric:

- **Losing to one is a strong negative** -- an independent agent found a real,
  exploitable hole. High value: harvest that losing line as a test position / patch
  target (same loop as the bounty and the oracle).
- **Beating one is reassuring, not proof** -- same honest-signal caveat as everything
  else. The oracle still certifies.

**Manual vs programmatic:**

- *Manual* -- play a few games against a web-playable bot yourself. Cheap, fast,
  catches gross embarrassment; noisy and low-N.
- *Programmatic* -- if it's open-source or speaks a protocol, write a thin **adapter**
  (translate board encoding + move format + turn convention) and play N games
  automatically for a statistically meaningful W/D/L record. This is the real
  benchmark. **Prefer a local clone / open-source bot** over hammering someone's web
  service (rate limits, ToS -- just be a good citizen).

**The one gotcha that voids the whole comparison: rule variants.** UTTT differs on
what happens when you're sent to an *already-decided* mini-board (free move anywhere
vs. other conventions). If the other bot uses a different variant than yours, the
games are **meaningless** -- confirm the ruleset matches before reading anything into
the score.

This is a **Phase-4 pre-deploy gate**: hold-or-win vs the best public bot you can find,
programmatically, over enough games -- *then* ship.

---

## Part 9 -- Deep-MCTS compute oracle: slow, any-position, empirical strength

*Added 2026-06-26, after the `--parallel 64` strength run confirmed that closed-loop
ELO is meaningless in stage 6 (1300 -> 4437, pure self-play inflation). We need an
honest external probe that does not depend on the gene pool.*

The alpha-beta endgame oracle (Part 6) gives **proof** but only in positions it can solve --
near-terminal boards with small effective subtrees. The external-bot benchmark (Part 8)
gives **independence** but its strength ceiling is bounded by whoever built that bot.

There is a third instrument: **an MCTSAgent at very high simulation budget.** The same
`MCTSAgent` that lives in `agents/mcts.py` (wired for training/inference) can be cranked
to arbitrarily high `n_sims` and used as a standalone strength oracle:

- **Any position** -- not restricted to endgame. You can evaluate opening moves, mid-game
  transitions, and anything else the alpha-beta oracle can't cheaply solve.
- **Continuously tunable accuracy** -- more simulations ~= stronger play. At 10k-100k
  sims per move the MCTS policy approaches minimax quality (with a good value head);
  you choose the quality/cost tradeoff.
- **No proof** -- MCTS is empirical, not exact. It can still be fooled by a very deep
  tactical line that exceeds its tree depth. But in practice, high-sim MCTS on a game
  as bounded as UTTT (81 cells, binary outcome) gets *very* close.
- **Honest signal** -- it's **independent of the training loop**. The same closed-loop
  inflation that let the arena ELO ratchet to 4437 can't happen here: the deep-MCTS
  oracle plays with the same fixed search budget every time, against whatever bot you
  hand it, and reports W/D/L with no ELO drift.

### How to use it

Benchmark `best.pt` by playing it *as the agent* against a **frozen deep-MCTS opponent**
(the oracle), at a sim count high enough that the oracle is clearly stronger than the
trained net. Record W/D/L over N games. Vary sim count to get a strength curve:

```
n_sims = 100        # ~same as training MCTS; upper bound on what best.pt could match
n_sims = 1 000      # ~10x deeper; starts to expose real holes
n_sims = 10 000     # a full game takes minutes; very strong reference
n_sims = 100 000    # multi-hour per game; near-proof territory for UTTT's bounded tree
```

You are not trying to *beat* it -- you are measuring **at what sim count the trained net
stops winning**. That crossover point is an honest, comparable, reproducible strength
reading.

### What this adds (relative to the other validators)

| Validator | Positions covered | Strength ceiling | Speed | Certainty |
|---|---|---|---|---|
| Arena ELO | training games only | self-inflating | fast | [X] inflates |
| Frozen-ELO anchors | training games | bounded by anchor quality | fast | [OK] calibrated |
| External bot (Part 8) | any | someone else's ceiling | medium | [OK] independent |
| alpha-beta endgame oracle (Part 6) | near-terminal only | **exact** | fast (shallow) | [OK] proven |
| **Deep-MCTS oracle (this part)** | **any position** | **scales with compute** | **slow** | [OK] empirical |

The deep-MCTS oracle fills the gap: **full-game coverage at any desired accuracy**, at
the cost of compute time. For a periodic benchmark (not every chunk -- maybe before each
SHIP_PLAN phase gate) the wall-clock is acceptable.

### When to run it

- **Right now:** `best.pt` (medium, trained to stage 6) has an inflated ELO and no
  honest absolute rating. A deep-MCTS benchmark is the fastest way to get one.
- **After MCTS is wired for inference (Phase 2):** the trained net *with* MCTS at
  eval-time should beat the same oracle at lower sim counts than the raw net -- that
  delta is a clean measure of how much MCTS inference buys.
- **Phase 6 (architecture search):** when evaluating dozens of tiny nets, a few
  deep-MCTS spot-checks anchor the blunder-rate and oracle results to an absolute scale,
  so "net A scores better than net B on the oracle" is meaningful, not just relative.

### Implementation note

`MCTSAgent` exists and is playable (`agents/mcts.py`). To use it as a compute oracle:

1. Load `best.pt` into a `NeuralNetAgentPG` (the candidate).
2. Load a **separate** checkpoint (or even a random-weight net -- the oracle's policy just
   seeds the prior; at high sim counts the tree dominates) into a second `MCTSAgent` with
   `n_sims` cranked up.
3. Play N games, alternate first-mover, record W/D/L.
4. No training -- both agents are in eval mode, `torch.no_grad()`. Pure inference.

`scripts/benchmark_vs_mcts.py` does exactly this (authored 2026-06-26). One command:

```bash
python -m scripts.benchmark_vs_mcts --checkpoint models/league_pg/best.pt \
    --network medium --games 40 --oracle_sims 800
# or sweep the budget and find the crossover:
python -m scripts.benchmark_vs_mcts --checkpoint models/league_pg/best.pt \
    --network medium --games 20 --sim_ladder 100,400,1600
```

The output is a single honest number: **win-rate of `best.pt` vs an X-sim MCTS
oracle.** No ELO, no curriculum, no closed loop. The `--sim_ladder` form reports the
**crossover** -- the sim count at which the candidate stops winning -- which is the
strength reading you actually want.

> The MCTS value-**sign** convention is verified (`agents/test_mcts.py` -- a
> deterministic immediate-win test that fails on a sign flip). The value-**scale**
> caveat (unbounded shaped-return value head mixed with clean +/-1 terminals) is
> documented in `agents/mcts.py`; if the oracle looks weak at high sims, that's the
> first suspect, not the sign.

---

## TL;DR sequencing

1. **Anchors** are the ruler; keep the duds, keep them frozen, keep them **diverse**.
2. **Make the diverse panel the *fitness* signal, not just the displayed ELO** -- else
   you inbreed regardless. (Verify the selection path before Phase 6.)
3. **Arena explores; the oracle judges.** Win condition is **5c blunder-rate**, never
   arena ELO.
4. **Grade the endgame first** (provable, cheap), seed from solved positions to dodge
   selection bias. Memoize-on-demand before precomputing a table.
5. **Before committing GB vs TB**, measure **N(K)**. Compression (WDL + D4 +
   side-to-move + dense legal index) shifts the curve, but N(K) decides it.
6. **gregory the foil and the grading oracle are the same alpha-beta at different depths** --
   build it once.
7. **Before shipping, benchmark vs strong *external* public bots** (Part 8) -- the only
   gene-pool-independent validator available pre-deploy; confirm the ruleset matches,
   and harvest any loss as a patch target.
8. **Deep-MCTS oracle (Part 9) = any-position, tunable-accuracy empirical strength.**
   Not a replacement for the alpha-beta proof or the external bot -- a third instrument.
   Run it now against `best.pt` to get an honest absolute reading. Build
   `scripts/benchmark_vs_mcts.py` when MCTS inference is wired.
