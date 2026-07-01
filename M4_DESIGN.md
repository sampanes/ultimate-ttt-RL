# M4 design -- the oracle: bounded value, tactical pressure, self-play

*Handoff spec written 2026-06-29. This is a design note for the authoring box, not
training code. It turns the M2 findings (`RESULT_M2.md`) into a concrete recipe for
the heavy "oracle" track (SHIP_PLAN M4). Inputs settled in M2: oracle base =
`arena:22@hof`, pocket base = `arena:21@hof`.*

## The one thing M2 established

On the **current, fixed-reward** nets, a 1-ply "take the win / block the loss" filter
matched or beat 100-sim MCTS against every anchor, at ~50-100x less compute. That is
direct evidence that the binding constraint is the net's **judgment** (the value head),
not search depth and not parameter count. So the M4 lever is the **value target and the
training signal**, not raw scale.

Corollary on size: a 2-3x architecture bump is reasonable headroom for self-play to
fill, but going bigger than that is not supported by any current evidence and is gated
by free-tier *inference* cost (a live searching oracle on CPU is the real limit, not
file storage). Grow only after measuring that capacity actually binds. Do not treat
`lottery` as evidence either way -- it was trained under the old broken reward/lookback
and tells us nothing about capacity.

## 1. Bounded value head (the root fix)

Replace the unbounded shaped-return value target with a **bounded game-outcome** target:

- Target is the final result of the game from the current player's perspective:
  win = +1, draw = 0, loss = -1 (or [0,1] with 0.5 draw -- pick one and document it).
- Squash the value head with `tanh` (for [-1,1]) so the estimate is calibrated and
  comparable across positions. The current unbounded shaped return makes the value
  uncomparable between states, which is what caps search sharpness (see `GOAT_NEXT.md`,
  `RESULT_MCTS_ORACLE.md`).
- Keep the policy head as-is; this is a value-target change, not an architecture
  rewrite.

Why this is the headline: MCTS quality is bounded by how trustworthy the leaf value is.
A bounded, calibrated value is the precondition for deep search to *pay off* -- which is
the entire reason to host a heavy oracle. Until this lands, the M2 MCTS numbers are a
**floor**, not a ceiling.

## 2. Tactical pressure in training (the win/block question)

There are four distinct mechanisms hiding in "make win/block an opponent / overlay."
They are NOT equivalent -- they differ in how they affect credit assignment:

1. **winblock as an opponent in the pool -- DO.** A cheap deterministic adversary that
   punishes the exact blind spot M2 measured. To not lose to it, the net must learn to
   set up its own immediate wins and to block. Keep it a *minority* of the pool (a
   sparring partner, not the diet): winblock is a weak overall player and is
   deterministic, so over-weighting it overfits to its quirks.

2. **tactical OVERLAY on the opponent nets -- DO (underrated).** Wrap pool opponents
   (archive nets, etc.) with the same 1-ply win/block filter so they never blunder a
   free win or miss a block. This sharpens the whole pool and removes free games,
   giving the learner real tactical pressure. Likely a bigger quality lift than adding
   winblock alone, and nearly free.

3. **tactical overlay on the LEARNER during self-play -- AVOID.** Overriding the
   policy's own move with the tactical move gives the net credit/blame for an action it
   did not choose, muddying the gradient; the net leans on the crutch instead of
   learning the tactic. (The overlay is fine at *inference* time -- that is the M3
   pocket layer -- just not as a training mechanism.)

4. **auxiliary tactical loss -- DO (the principled fix).** When the position has an
   immediate win or a forced block, add a supervised term that pushes policy mass onto
   that move. This teaches the blind spot directly into the weights rather than hoping
   self-play stumbles onto it. Highest-signal version of "would win/block improve
   training": yes, as a loss term.

Recommended combination: (1) + (2) as opponent-pool changes, (4) as an auxiliary loss,
and explicitly NOT (3).

## 3. Self-play loop (AlphaZero-style)

Once the value is bounded (1) and tactical pressure is in (2):

- Generate games by search-in-the-loop (MCTS over the net's own head) rather than raw
  policy sampling, so the policy trains toward the searched visit distribution.
- Train policy toward MCTS visit counts; train value toward the bounded game outcome.
- Mix the opponent pool: self-play + archive (tactical-overlaid) + a winblock minority.
- Re-run the M2 panel (`scripts/benchmark_suite.py`) each iteration as the regression
  gate. The pass condition is the `winblock` score climbing and `oracle_mcts_400`
  finally beating the cheaper modes (which would prove the value head is no longer the
  limiter).

## 4. Adversarial suite (shared with the pocket net)

Build a `winblock`-derived adversarial test set per `BOUNTY.md`: positions with an
immediate win, a forced block, and forced-board legality traps. This is the clearest
measured failure from M2 and serves three roles -- training data for the auxiliary loss
(2.4), a regression gate, and the seed of the post-ship bounty's blind-spot audit.

## Pocket net (`arena:21`) -- scope note

Settled in conversation, recorded here so M3 is not derailed: the pocket net is at
diminishing returns under *general* training (it is near the ceiling of what the current
signal can teach 1.3M params). It still has ONE cheap gain -- the same winblock blind
spot -- addressable by a short, targeted fine-tune on the adversarial suite (4), not by
more general self-play. Otherwise: ship it for M3 with the inference-time tactical layer.

## Design principles carried into this work

Inferred from how this project is being steered; recorded so the authoring box keeps the
same priorities:

- **Cheap, high-ROI moves before brute force.** Prefer a targeted fix (bounded value,
  auxiliary loss, a deterministic sparring opponent) over scaling up. Scale is a last
  resort justified by measurement, not a default.
- **Compose simple deterministic heuristics as tools.** win/block and friends are not
  just baselines -- they are usable as opponents, as overlays, and as a source of
  training signal.
- **The heavy model must be genuinely strong, not merely large.** "Biggest that fits"
  is the wrong target; "strongest within the inference budget" is the right one.
- **Decisions come from the independent panel, not arena ELO.** Keep certifying with
  `scripts/benchmark_suite.py`; treat arena ELO as engagement, not proof.
- **Reason first, then commit; update on evidence.** Surface the why and the trade-off
  before changing the recipe.
