# MCTS_STATUS -- current search implementation baseline

Status: factual implementation snapshot as of 2026-07-18 (gen 15 live).
This is the "what exists today" record. It is deliberately separate from
STRENGTH_ROADMAP.md: the roadmap states intended direction; this file states
the present baseline and will be updated as features land.

Source of record: `agents/mcts.py`. Audited 2026-07-18.

## The one line that matters

> Current MCTS is a fresh-root, strict-tree PUCT implementation with no symmetry
> handling, no state-based transposition or inference cache, no cross-move tree
> reuse, no solved-node propagation, and an unswept fixed `c_puct=1.5`.
> Consequently, the Generation 13 search-scaling results (0/50/100/200/400 sims
> vs gregory d3: 0.39 / 0.630 / 0.672 / 0.757 / 0.778) characterize THIS specific
> baseline implementation and should NOT be interpreted as establishing a
> fundamental 200-simulation limit. The apparent 200-sim knee is a property of
> the current vanilla, untuned search stack -- not evidence that the network or
> the game has inherently saturated at 200 sims.

That reinterpretation is the most important consequence of the audit. See the
search curve in memory `uttt-gen13-search-curve` and `loss_logs/sims_sweep_gen13.log`.

## What exists today (the audit)

1. **No 8-fold canonicalization.** Rotations/reflections are not folded. Each of
   the 8 symmetric orientations of a position is searched as a distinct state.

2. **No transposition merging.** Strict tree, one node per path. The only dedup
   is by `id(node)` WITHIN a single wave (`agents/mcts.py` ~206-213): two sims in
   the same wave landing on the same unexpanded leaf expand it once. Positions
   reached by different move orders never share statistics.

3. **No cross-move tree reuse.** `search()` builds a fresh root every call
   (`agents/mcts.py:123`) and the whole tree is discarded when the function
   returns. The subtree the opponent walks into is rebuilt from scratch next move.

4. **Proven-win/loss propagation: IMPLEMENTED 2026-07-27, opt-in
   (`MCTS(..., solve=True)`), still OFF by default.** `node.solved` holds
   None / +1 / 0 / -1 from that node's to-play perspective, stored separately
   from N/W so a proof can never be averaged into the neural estimate.
   Terminal children are marked at EXPANSION via one engine clone+make_move per
   legal move, so every expansion is an exact 1-ply search; `_solve_from_children`
   does the AND/OR backward induction; `_best_child` takes a proven win outright
   and refuses proven losses; descent stops at any solved node and backs up the
   exact value, so a proven subtree costs no network evaluation. `search()` also
   reconciles the returned visit policy with the proof (one-hot on a proven win,
   zero on refuted moves), because raw counts can otherwise hand back a target
   whose argmax is known to lose.

   Soundness is tested against exhaustive minimax on random late-game positions
   (`agents/test_mcts.py::test_solved_claims_agree_with_exhaustive_minimax`), and
   `tools/measure_solved_targets.py --parity-check` asserts that with solve=False
   the search still reproduces the frozen distillation-pilot corpus bit-for-bit.

   Was previously described here as absent; the old text read: "There is no
   solved-status flag, no exact proof, and no pruning of solved branches --
   search keeps sampling a branch whose outcome is already forced." That is
   exactly the defect RESULT_DISTILL_PILOT.md later measured, and is why this
   moved to the front of the queue. See RESULT_SOLVED_NODES.md.

5. **Parameters are not budget-tuned.** `c_puct=1.5` is fixed and flagged in the
   module docstring as never tuned (compounded by the value-SCALE mismatch:
   the value head is unbounded shaped-return, compared against clean +/-1.0
   terminal values -- see docstring lines 19-30). The only budget-adaptive knob
   is `_MIN_WAVES=16` (line 102), a floor that clamps `eff_wave` so a large
   `wave_size` still yields at least 16 waves of depth.

## Priority order for improvement (owner call, 2026-07-18)

Not yet actioned -- this is the intended sequence, cheapest/cleanest first.

1. **Tree reuse across moves.** Probably the cleanest immediate gain. After a
   root is searched, both our chosen move and the opponent's reply normally lead
   into already-built subtrees. Detach the selected node, make it the new root,
   retain its N/W/children/priors and cached net evals. Training root Dirichlet
   noise must be re-applied to the new root, NOT left permanently baked into
   stored priors.

2. **Tune `c_puct`.** Cheaper than anything structural and could materially move
   the curve. Sweep several values independently at 50, 100, and 200 sims -- the
   best exploration constant may differ by budget and by network generation.
   RESULT (2026-07-18, gen-15 teacher, 50 sims vs gregory d3, 300 games seed
   8801): 1.0 -> 0.697, 1.5 -> 0.765, 2.5 -> 0.738, 4.0 -> 0.750. The current
   default 1.5 is ALREADY optimal at the page's budget; below it hurts (~2 SE),
   above it is flat within the ~3-pt noise floor. No free win here at 50 sims.
   100/200-sim sweeps remain unswept but are now low priority (the deployed and
   teacher-generation budgets are what matter, and 50 is closed). Same sweep
   incidentally showed gen-15 at 50 sims (0.765) ~= gen-13 at 200 sims (0.757):
   +13.5 pts from two generations at FIXED search -- the network is the lever.

3. **Exact-state neural-evaluation cache.** Before a true DAG, cache
   policy/value outputs by state hash so identical positions reached by different
   paths are not re-evaluated by the net. Keep ordinary tree-local visit stats
   untouched. Much simpler and safer than sharing N/W/Q across parents.

4. **Proven-result propagation.** Expected to be high-value in tactical UTTT
   positions. Store solved status SEPARATELY from the averaged neural value,
   propagate forced wins/losses exactly, and stop spending sims on proven
   branches. Solved value must be defined consistently from the side-to-move
   perspective.

5. **Symmetry exploitation.** Several distinct features hide under this label:
   8-way training augmentation; canonical state hashing; transforming policy
   moves into/out of canonical orientation; possibly averaging predictions over
   all symmetries. Training augmentation is relatively straightforward. Full
   canonical search sharing is invasive -- it must correctly transform the forced
   destination board, local-board indices, cell indices, AND policy outputs, and
   existing networks may need retraining if all inputs become canonicalized.

6. **Full transposition merging.** Instrument FIRST: add hashing and measure how
   often exact transpositions actually occur. UTTT's forced-board rule may make
   them rarer than in games with freer move ordering. A full DAG also raises hard
   accounting questions -- sharing visit counts across nodes with multiple parents
   can distort standard PUCT assumptions. Start with the cached neural evals
   (item 3), then implement shared search statistics only if the measured
   transposition rate justifies it.

## Experimental sequence to attribute the plateau

The audit implies a clean way to find out whether the gen-13 plateau belongs to
the network or merely to inefficient search:

1. Land tree reuse (item 1) and c_puct tuning (item 2).
2. Rerun the full 0/50/100/200/400-sim curve vs gregory(d3).
3. Add proven-result propagation (item 4).
4. Rerun the same curve again.

If the knee moves right / the ceiling rises after these changes, the plateau was
the search stack. If the curve barely moves, the network capacity is the real
limit -- which is what `uttt-gen13-search-curve` currently assumes, and which
would then be confirmed rather than inferred.

Harness for the reruns already exists: `scripts.baseline_vs_gregory --sims N`
(300 fixed color-swapped openings, seed 8801, CPU, GPU-safe while a run is live).
