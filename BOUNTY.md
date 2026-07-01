# The $5 bounty -- beat the bot, prove it, get paid

*Written 2026-06-24. Planning note for a **post-ship** feature (after Phase 4). Cheap
to define now, free to run later. Not started.*

> **The headline:** this bounty is a **crowdsourced, incentivized version of the
> external validator** from `GRADING_AND_ORACLE.md`. Humans are the "outsider with an
> unexplored heuristic" from the confidence ladder -- the one thing a closed arena can
> never produce. Every *legitimate* win is a **proven blind spot**, handed to you for
> $5. The win condition isn't "nobody ever beats it" -- it's "every beat teaches the
> bot something." The bounty **funds your adversarial audit.**

---

## The offer

Beat the deployed bot **legitimately** and get **$5**. The catch is in
"legitimately," and that's the whole engineering point: a claim isn't "trust me, I
won" -- it's a **move log** that your deterministic engine **re-executes** and
verifies. No replay, no payout.

The bounty is pinned to a **specific published model** (by hash / commit), so beating
last month's weaker net doesn't claim a bounty on today's.

## "Legitimately" = move-replay verification (the core)

A claimed game is just a list of moves. Verification re-runs it through the **same
deterministic rules engine** the bot plays on, checking, in order -- defense in depth,
any layer fails => rejected:

1. **Legality replay.** Replay every claimed move (human *and* bot) from the empty
   board; each must be legal under the UTTT rules at that state. Kills fabricated or
   impossible games outright.
2. **Bot-move determinism.** At every bot turn, re-run *your actual bot* on the
   reconstructed state -- its move **must match** the claimed bot move. This is the
   load-bearing check: it proves the record is consistent with **your bot**, not a
   strawman that "conveniently blundered." Requires the bounty-mode bot to be
   **deterministic** (argmax, fixed-seed MCTS); if it samples, the submission must
   include the seed/RNG state and replay reproduces it bit-for-bit.
3. **Result check.** The replayed terminal state is a genuine **bot loss** (not a
   draw, not a bot win, not an unfinished game).
4. **Provenance.** The claim names **which model version** (hash) it beat, and that
   matches a published release.

All four pass => it's real. This is exactly how chess/speedrun bounties verify a PGN:
re-simulate, don't trust.

## Redundant checks & safety

- **Untrusted input.** A submitted move log is hostile until proven otherwise: bound
  its length, range-check every move index, **never `eval`** it -- it may only be fed
  to the rules engine and replayed. The engine is the sandbox.
- **Determinism is the linchpin.** If the bot isn't reproducible, check #2 is
  meaningless. Bounty-mode bot = deterministic by construction, or seed-logged.
- **Dedup claimed lines.** A *deterministic* bot has **fixed** losing lines -- once one
  is published, anyone can replay it. So **canonicalize each claimed game** (the same
  D4 + transposition machinery as the cache in `GRADING_AND_ORACLE.md`) and reject
  duplicates, or one discovered line gets farmed for infinite $5.
- **Payout economics.** Cap it: one bounty per distinct line, and/or **reset on
  patch** (below). A fixed total budget so the wallet can't be drained.

## The economics of a deterministic bot (the deep bit)

A deterministic bot is a **puzzle, not an opponent** -- its losing lines are fixed, and
the model is open / free-hosted, so a challenger can hunt for one **offline with their
own solver.** That sounds like a hole; it's the *feature*: you're crowdsourcing the
exact blunder-search the oracle does, and paying $5 a hit instead of running the
compute yourself.

The elegant resolution to "one line = infinite claims": **each legitimate win is a
patch target.** Fix the hole -- retrain on it, add the refuted line to the opening
book (Phase 5b), or deepen the live solve (5a) so that line is now played perfectly --
and the bounty **resets, harder.** The bounty becomes a ratchet: pay to discover a
flaw, close it, raise the bar. That's the arena's external-validator role, gamified
and distributed.

## Free to host (the cost constraint)

- **Gameplay = zero server cost.** The shipped bot already runs **client-side**
  (WASM/ONNX, Phase 3/4) -- the browser does the inference, you host a static page
  (GitHub Pages / HF Space, free tier). Nothing to pay per game, nothing to attack.
- **Verification = no backend for v1.** A challenger submits their move log as a
  **GitHub issue / gist / email**; you run the **local verifier script** (the
  deterministic replay above) and pay out **manually** ($5 by whatever rail you like).
  No payment API, no database, no server process = no recurring cost and almost no
  attack surface. Add automation only if volume ever demands it -- it won't, early.

## What the bounty *is* and *isn't* (honest signal)

- **It is:** engagement, plus a crowdsourced finder of blind spots, plus a *weak*
  signal -- a bounty unclaimed over time is mild evidence of strength.
- **It is not a proof.** Absence of claims could just be low traffic. The **proof**
  still comes from the endgame oracle's blunder-rate (`GRADING_AND_ORACLE.md`). Arena
  to explore, humans to probe, **oracle to certify.**
- **End state:** if the bot ever becomes provably optimal where the game theory allows
  it, the bounty becomes unwinnable *except* on the side the theory already concedes
  (e.g. if UTTT is a first-player win, the second player simply cannot win against
  perfect play -- so the live bounty would be "beat it when **it** moves first"). That
  asymmetry in which bounties stay claimable is itself informative.

---

## TL;DR / sequencing

1. **Post-Phase-4** (needs the shipped, client-side bot first). Cheap to run.
2. **v1 = manual:** submit a move log -> local deterministic verifier -> manual $5
   payout. No backend, no DB, no payment integration.
3. **Verify by replay, four layers** -- legality, bot-determinism, result, provenance.
   Treat the submission as untrusted input.
4. **Pin to a model hash; dedup claimed lines (canonicalize); cap the budget.**
5. **Each legitimate win -> patch the hole -> reset the bounty harder.** The $5 buys
   adversarial search; the patch is the actual prize.
