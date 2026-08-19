"""PUCT-style MCTS wrapper around a (policy, value) network.

Compatible with any agent whose `.model` exposes
`forward_both(x) -> (policy_logits[81], value_scalar)` -- currently
NeuralNetAgent3 / NeuralNetAgentPG via the shared ConvNet.

Value sign convention -- VERIFIED CORRECT (2026-06-26, audit + tests):
  The leaf value is the expected outcome for the player to move at that leaf.
  The backup flips sign per ply (zero-sum), so root.Q is the expected outcome
  for the root player. This matches how the value head was trained: in
  `scripts/train_league.py:play_one_game`, rewards are recorded ONLY at the
  active agent's own moves, always from the active player's perspective
  (+1 win / -0.1 draw / -1 loss, +/-0.3 per mini-board), and the value head is
  regressed to that discounted return. So V(s) is the to-move player's expected
  outcome -- exactly what `_expand` returns and `_backup` propagates. The
  per-ply negation in `_best_child` and the `_terminal_value(state, to_play)`
  perspective are consistent with this. `agents/test_mcts.py` locks it in.

KNOWN CAVEAT -- value SCALE mismatch (not a sign error; does NOT invert search):
  The ConvNet value head is `Linear(256, 1)` with NO tanh, trained on a SHAPED,
  discounted return, so its output is unbounded and NOT calibrated to [-1, 1].
  But terminal leaves here use a clean +/-1.0 / 0.0 (`_terminal_value`). MCTS
  therefore mixes two scales: a net leaf rated, say, +1.8 (won a couple of minis,
  expects to win) will outrank a PROVEN terminal win (+1.0), which can pull search
  toward shaky shaped positions over certain wins, and means c_puct was never
  tuned against a clean value scale. This degrades quality, it does not flip the
  search. Fix: retrain with --value_tanh (scripts/train_alphazero.py or
  scripts/train_league.py --value_tanh) so the value head outputs a calibrated
  [-1, 1] result and this comment no longer applies.
  Measure impact with `scripts/benchmark_vs_mcts.py` before tuning c_puct.

Batched leaf evaluation (wave_size > 1):
  When wave_size > 1, each round of the search loop collects `wave_size` leaf
  nodes via virtual loss, then evaluates all leaves in ONE batched forward pass.
  Virtual loss temporarily adds N+1 / W+1 to visited nodes (W is stored from the
  child's to-play perspective and selection scores -Q, so RAISING W penalizes
  the path -- see the sign-bug comment in _run_wave) so each selection in the
  same wave diverges to a different path. After the batch eval the VL is undone
  and real backups are applied. wave_size=1 is the original serial path
  (byte-identical to previous behaviour). wave_size=8 is a good default for the
  oracle benchmark; 16-32 helps for AlphaZero self-play at high n_sims.
  HISTORY: from its introduction until 2026-07-04 the VL sign was inverted
  (W-1), which collapsed whole waves onto a single line and made wave>1 search
  WEAKER than the raw policy -- it poisoned the M4a/M4b AlphaZero runs' visit
  targets. agents/test_mcts.py now locks the wave path too.

Solved-node propagation (solve=True, OFF by default):
  Exact game-theoretic status, tracked SEPARATELY from N/W/Q so it can never
  contaminate the averaged neural estimate. `node.solved` is None (unresolved)
  or +1 / 0 / -1 = proven win / draw / loss FOR THAT NODE'S to_play -- the same
  perspective convention as `terminal_value`, so the two are interchangeable at
  backup time.

  Propagation (`_solve_from_children`) is standard proof-number-free AND/OR
  backward induction: a node is a proven WIN as soon as any child is a proven
  loss for its own mover; a proven LOSS only once EVERY legal child is a proven
  win for the opponent; a proven DRAW when all children are solved, none wins,
  and at least one draws.

  Two things make this pay rather than just being bookkeeping:

  1. Terminal children are marked AT EXPANSION (`_mark_terminal_children`), not
     when a simulation happens to wander into them. One clone+make_move per
     legal move, using the engine itself so no win rule is duplicated here. That
     turns every expansion into an exact 1-ply search, so a mate-in-1 at the
     root is proven BEFORE the first simulation and every visit thereafter goes
     to the win.
  2. Selection short-circuits: `_best_child` takes a proven win outright and
     refuses proven losses, and descent stops at any solved node and backs up
     its exact value. A proven subtree then costs one cheap descent per sim with
     NO network evaluation.

  Motivation is measured, not aesthetic. RESULT_DISTILL_PILOT.md found that on
  mate-in-1 positions an 800-sim teacher puts a visit on EVERY legal move and
  keeps only 0.693 mass on the win, where 50 sims keeps 0.825: PUCT's sqrt(N)
  exploration bonus grows with the budget, so more search DILUTES decided
  positions. Proving them stops the sampling instead of correcting it afterwards
  with a temperature hack.

  `search()` also reconciles the returned visit policy with the proof, because
  raw counts can disagree with it: a move that looked good early can collect
  visits and only later be refuted. See the comment at the pi-extraction site.
"""

import math
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from .agent_base import Agent, board_to_tensor_from_gamestate, wave_planes
from . import native_select as _ns
from engine.rules import rule_utl_valid_moves
from engine.constants import X, O, DRAW, EMPTY, WIN_PATTERNS


class MCTSNode:
    # The last six slots are the native selection mirror (#45a) and exist only
    # when MCTS(native_select=True) built them. `sel` is the one that is always
    # initialised, because `_best_child` reads it on every descent to decide
    # which selector to run; the rest are written when the mirror is created
    # and read only through a node that has one. Initialising all six in
    # __init__ would put five stores on the incumbent's node-creation path
    # (~5,000 nodes a move) to no purpose.
    __slots__ = ("parent", "children", "prior", "N", "W",
                 "move", "to_play", "is_terminal", "terminal_value", "solved",
                 "sel", "kids", "selN", "selW", "selS", "cidx")

    def __init__(self, parent=None, prior=0.0, move=None, to_play=None):
        self.parent = parent
        self.children = {}
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.move = move
        self.to_play = to_play
        self.is_terminal = False
        self.terminal_value = 0.0
        self.sel = None
        # Exact status from THIS node's to_play perspective, or None if
        # unresolved. Deliberately NOT folded into W: a proof is not an average,
        # and mixing them would let one backup of +1 masquerade as certainty.
        self.solved = None

    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, c_puct, parent_N):
        return c_puct * self.prior * math.sqrt(parent_N) / (1 + self.N)


_DEEPCOPY_WARNED = False


def _clone(state):
    """Fast clone for MCTS rollouts. Uses state.clone() if available."""
    if hasattr(state, "clone"):
        return state.clone()
    global _DEEPCOPY_WARNED
    if not _DEEPCOPY_WARNED:
        # ASCII-only: this can print under a redirected/cp1252 stdout, where a
        # non-ASCII glyph raises UnicodeEncodeError (same class as the engine and
        # train_league import-banner fixes).
        print("[!] MCTS falling back to deepcopy for GameState -- add a clone() "
              "method to the active engine for ~20-50x speedup.")
        _DEEPCOPY_WARNED = True
    return deepcopy(state)


# The macro board's eight lines, taken from the SAME constant
# `GameState.check_ultimate_win` iterates rather than re-listed here, so the
# predicate below cannot drift away from the rule it is derived from. Retupled
# once at import: `WIN_PATTERNS` is a list of lists and this is a hot loop.
_MACRO_TRIPLES = tuple(tuple(p) for p in WIN_PATTERNS)


def could_end(mini_winners, mover):
    """Could ANY legal move from this position end the game? (#48)

    A NECESSARY condition, derived from the rule rather than guessed.
    `make_move` calls `check_ultimate_win` only when a move newly DECIDES a
    mini-board, and that function returns a result in exactly two cases: a macro
    line of three mini-boards owned by one player, or all nine decided (a draw).
    So a child of this node can be terminal only if the move decides a
    mini-board AND:

      (a) `mover` already owns two mini-boards of some macro triple whose third
          is still undecided -- otherwise no line can complete this ply; or
      (b) only one mini-board is still undecided -- otherwise deciding one
          still leaves another undecided and the macro board cannot be full.

    A mini-board with `mini_winners[m] == EMPTY` always has an empty cell -- it
    is marked the moment its last cell is filled, DRAW if no line formed -- so
    "undecided" and "still playable" are the same set and (b) needs no separate
    fullness test. A DRAW mini-board is decided and can never complete a line,
    which is why (a) tests ownership by `mover` and not merely non-emptiness.

    WHEN THIS RETURNS FALSE, SKIPPING THE PROBE LOOP IS EXACTLY EQUIVALENT, not
    approximately: no child can be marked terminal, so every child keeps
    `solved is None`, and `_solve_from_children` over children that are all
    unsolved and none refuted returns None -- the same None it would return
    after running the loop. Nothing downstream can tell the difference.

    HOW LOOSE IT IS, measured rather than assumed. Over all 391,550 live macro
    configurations times both movers it is EXACT -- 177,566 admitted against
    177,566 that can really end, zero slack -- so the two conditions are not an
    over-approximation of the rule, they are the rule projected onto
    `mini_winners`. All the remaining looseness is at the position level, where
    "some undecided mini could become the mover's or a draw" does not imply any
    single legal move achieves it: on random reachable positions it admits
    1.47% and 0.13% really have a terminal child.

    None of that is what licenses the skip. NECESSITY is, and it is checked
    separately: against the real probe over 526,097 probe roots
    (`tools/probe_ablation --mode filter`) it admits 18.88% of roots and 17.52%
    of the per-child work with ZERO false negatives. Zero is the gate rather
    than an agreement rate, because one false negative would mean this is not a
    necessary condition and the engine would quietly stop finding a proof it
    used to find. agents/test_probe_filter.py is where both halves live.
    """
    undecided = 0
    for m in mini_winners:
        if m == EMPTY:
            undecided += 1
    if undecided <= 1:
        return True
    for a, b, c in _MACRO_TRIPLES:
        x, y, z = mini_winners[a], mini_winners[b], mini_winners[c]
        owned = (x == mover) + (y == mover) + (z == mover)
        if owned == 2 and (x == EMPTY or y == EMPTY or z == EMPTY):
            return True
    return False


class MCTS:
    _VL = 1.0        # virtual loss magnitude (standard AlphaZero)
    _MIN_WAVES = 16  # floor on waves per search: the tree only deepens between
                     # waves (leaf expansion is deferred to the batched forward
                     # pass), so wave_size ~ n_sims degenerates into a one-ply
                     # breadth probe. Empirical (2026-07-04, edge vs raw net,
                     # after the VL sign fix): 38 waves 0.95, 16 waves 0.80-0.925,
                     # 10 waves 0.70-0.80, 5 waves 0.375-0.80, 1 wave 0.00.

    def __init__(self, model, device, n_sims=100, c_puct=1.5,
                 add_dirichlet_at_root=False, dir_alpha=0.3, dir_eps=0.25,
                 wave_size=1, solve=False,
                 time_budget_ms=None, max_sims=0, min_sims=1,
                 deadline_margin=1.15, reserve_ms=None,
                 batched_expand=None, graph_wave=False,
                 native_select=False, probe_filter=False):
        self.model    = model
        self.device   = device
        self.n_sims   = n_sims
        self.c_puct   = c_puct
        self.add_dirichlet = add_dirichlet_at_root
        self.dir_alpha = dir_alpha
        self.dir_eps   = dir_eps
        self.wave_size = max(1, int(wave_size))
        # OFF by default on purpose: production expert iteration must not change
        # behaviour just because this landed. Callers opt in explicitly.
        self.solve = bool(solve)
        # Wall-clock mode. None keeps the fixed-simulation behaviour exactly.
        # See _run_budget for why the deadline is predictive rather than checked
        # after the fact, and what max_sims / min_sims / deadline_margin bound.
        self.time_budget_ms = time_budget_ms
        self.max_sims = int(max_sims)
        self.min_sims = max(1, int(min_sims))
        self.deadline_margin = float(deadline_margin)
        # Headroom held back from the stated budget. Predictive admission
        # handles the ordinary case, but it can only foresee a chunk that
        # behaves like recent chunks: a GC pause or a driver hiccup lands
        # mid-chunk and no predictor can see it coming. Measured on the first
        # 1 s baseline -- p99 1000.6 ms, worst move 1054.9 -- so the tail is
        # real, small, and exactly what a reserve is for.
        self.reserve_ms = (max(5.0, 0.02 * time_budget_ms)
                           if reserve_ms is None and time_budget_ms
                           else (reserve_ms or 0.0))
        # Batched wave expansion (see _expand_wave). DEFAULTS TO TIMED MODE
        # ONLY, and that restriction is deliberate rather than cautious: the
        # wave path is what generated the frozen distillation corpora, and
        # softmaxing a (K, 81) block reduces in a different order than 81
        # separate vectors, so the priors differ in the last bits. Under a clock
        # that is irrelevant; under a fixed simulation count it would make
        # tools/extract_child_q report drift against artifacts already hashed.
        self.batched_expand = (bool(time_budget_ms) if batched_expand is None
                               else bool(batched_expand))
        # CUDA-graph replay of the wave's device sequence (agents/graph_wave).
        # OFF by default, and it must stay off until it has been promoted on
        # equal-clock strength -- a faster wave is not a stronger engine, which
        # this project has already learned once from the model-size result.
        # Captured for ONE wave size; every other size keeps the eager path,
        # which also remains the correctness oracle.
        # The REQUEST, recorded separately from the captured object. A
        # fingerprint is a statement about configuration, and
        # `engine_registry --freeze` runs on CPU where capture cannot succeed
        # -- keying identity off the object would have frozen the graph engine
        # as graph_wave=False and made it indistinguishable from the
        # incumbent. Falling back is also not allowed to silently redefine
        # which engine you are running; TimedPlayer refuses outright.
        self.graph_wave_requested = bool(graph_wave)
        self.graph_wave = None
        if graph_wave and self.batched_expand:
            from agents.graph_wave import GraphedWave
            gw = GraphedWave(model, device, k=self.wave_size)
            # A dud is not fatal: no CUDA, a driver that refuses capture, or an
            # uncapturable op all leave the engine running eagerly.
            self.graph_wave = gw if gw.ok else None
            self.graph_wave_reason = gw.reason
        # Native PUCT selection over a mirrored child array (#45a). OFF by
        # default and, unlike the graph wave, it does NOT degrade quietly:
        # falling back would leave an engine that is named for a selector it is
        # not running, and #41 already taught this project what that costs when
        # a fingerprint keys off a realised object instead of a request.
        self.native_select = bool(native_select)
        if self.native_select:
            _ns.require("MCTS(native_select=True)")
        # Hot-path alias. `self.native_select` is the configuration; `_mirror`
        # is what the descent, backup and proof loops test, hoisted to a local
        # before each loop so the check is not repaid per iteration.
        self._mirror = self.native_select
        # Selective terminal probing (#48). OFF by default like every candidate
        # before it, but unlike them it is not a different player under a clock
        # EITHER: `could_end` is a necessary condition, so the loop it skips
        # could not have marked anything, and the search that follows is the
        # same search with the dead work removed. What it buys is time, and
        # time is what the clock spends on more of the same search.
        #
        # It only does anything when `solve` is on: probing is what it filters.
        self.probe_filter = bool(probe_filter)
        self.reset_stats()

    def reset_stats(self):
        """Search-cost counters for the search-quality report. Cumulative across
        searches; the caller resets when it wants a per-move figure."""
        self.stat_searches   = 0
        self.stat_expansions = 0   # nodes given children (== net evals of leaves)
        # Children actually constructed. Cumulative and never reset by a
        # search, because what reads it is TreeReuseSearcher's retirement
        # watermark, which needs "how much has this game built" and not "how
        # much did this move build". One add per expanded node (~4,700 a move),
        # which is the cheapest place any bound on tree size can come from --
        # counting on the way OUT would mean a per-node increment inside
        # `release`, and that walk is shared with engines this feature is off
        # for.
        self.stat_nodes_created = 0
        self.stat_nn_evals   = 0   # positions pushed through the net
        self.stat_nn_batches = 0   # forward_both calls
        self.stat_probes     = 0   # clone+make_move terminal probes
        # Probe ROOTS -- expanded nodes offered to the one-ply scan -- and how
        # many of those `could_end` sent home without a single clone. Two adds
        # a probe root, paid only by solve=True engines, and they are what the
        # filter's whole claim is read off: "roots considered" against "roots
        # actually scanned" is not derivable from `stat_probes`, which counts
        # children. About 4,100 roots a move, so this is ~0.05% of a move and
        # is stated rather than glossed.
        self.stat_probe_roots = 0
        self.stat_probe_skips = 0
        self.stat_solved_roots = 0
        self.stat_sims       = 0   # simulations actually run (varies under a clock)
        self.stat_early_stops = 0  # deadline searches that returned on a proof
        self.stopped_early   = False
        # Own wall clock. A match's total elapsed time cannot be split between
        # two arms after the fact, so each search times itself.
        self.stat_seconds    = 0.0
        # Per-search record, overwritten each call. See _finish_record.
        self.last = {}

    # ------------------------------------------------------------------
    # Per-search proof bookkeeping. All of this is skipped when solve=False.
    # ------------------------------------------------------------------

    def _note_proof(self, root, sims_done):
        """First moment the ROOT becomes proven, and the state of the target then.

        `visits_off_proven` is the metric that matters for distillation: how many
        visits had already been spent on moves other than the proven win by the
        time the proof landed. Zero means the target was never distorted; a large
        number means the proof arrived too late to help the target even though it
        still helps play.
        """
        if self._proof_sim is not None or root.solved is None:
            return
        refuted = sum(c.N for c in root.children.values() if c.solved == 1)
        wins = [c for c in root.children.values() if c.solved == -1]
        off_proven = None
        if wins:
            best = max(wins, key=lambda c: (c.N, -c.move))
            off_proven = root.N - best.N
        self._proof_sim = sims_done
        self._proof_refuted_visits = refuted
        self._proof_off_visits = off_proven
        self._proof_nn_evals = self.stat_nn_evals

    def _begin_record(self):
        """Snapshot the cumulative counters so the per-search deltas are exact."""
        self._search_nn_start = self.stat_nn_evals
        self._search_exp_start = self.stat_expansions
        self._search_probe_start = self.stat_probes
        self._search_proot_start = self.stat_probe_roots
        self._search_pskip_start = self.stat_probe_skips
        self._proof_sim = None
        self._proof_refuted_visits = None
        self._proof_off_visits = None
        self._proof_nn_evals = None

    def _finish_record(self, root, pi_raw_argmax, corrected_argmax,
                       sims_done=0, elapsed=0.0, inherited=0, reused=False):
        """One row per search: what it cost, what it decided, what it inherited.

        Always written, including when solve=False. Under a wall clock the
        per-move cost IS the measurement -- `simulations_completed` is an output
        rather than a setting -- so a caller must be able to read it off every
        search, not only the solved ones.
        """
        nn_total = self.stat_nn_evals - self._search_nn_start
        rec = {
            # --- wall-clock budget accounting -----------------------------
            "elapsed_ms": elapsed * 1000.0,
            "budget_ms": self.time_budget_ms,
            "reserve_ms": self.reserve_ms,
            "worst_chunk_ms": getattr(self, "worst_chunk_ms", 0.0),
            "simulations_completed": sims_done,
            "neural_evaluations": nn_total,
            "nodes_expanded": self.stat_expansions - self._search_exp_start,
            "tree_nodes_reused": None,   # filled by the caller that re-roots
            "inherited_simulations": inherited,
            "reused_tree": reused,
            "stopped_early": self.stopped_early,
            "transposition_hits": 0,     # no DAG yet; key kept so the schema is
                                         # stable across the search-engine work
            "chosen_move": int(corrected_argmax),
            # --- pre-existing fields --------------------------------------
            "root_solved": root.solved,
            "root_n": root.N,
            "n_sims": self.n_sims,
            "expansions": self.stat_expansions - self._search_exp_start,
            "nn_evals": nn_total,
            "probes": self.stat_probes - self._search_probe_start,
            "probe_roots": self.stat_probe_roots - self._search_proot_start,
            "probe_skips": self.stat_probe_skips - self._search_pskip_start,
            "raw_argmax": int(pi_raw_argmax),
            "corrected_argmax": int(corrected_argmax),
            "reconciled": int(pi_raw_argmax) != int(corrected_argmax),
            "raw_argmax_solved": None,
            "corrected_argmax_solved": None,
            "proof_sim": self._proof_sim,
            "visits_on_refuted_at_proof": self._proof_refuted_visits,
            "visits_off_proven_at_proof": self._proof_off_visits,
            "nn_evals_at_proof": (None if self._proof_nn_evals is None else
                                  self._proof_nn_evals - self._search_nn_start),
            "visits_on_refuted_final": sum(
                c.N for c in root.children.values() if c.solved == 1),
        }
        raw_child = root.children.get(int(pi_raw_argmax))
        cor_child = root.children.get(int(corrected_argmax))
        if raw_child is not None:
            rec["raw_argmax_solved"] = raw_child.solved
        if cor_child is not None:
            rec["corrected_argmax_solved"] = cor_child.solved
        self.last = rec

    def _run_sim(self, root, root_state):
        """One simulation of the original serial path (wave_size == 1)."""
        node  = root
        state = _clone(root_state)

        while node.children and not node.is_terminal:
            node = self._best_child(node)
            state.make_move(node.move)
            if state.winner is not None:
                node.is_terminal    = True
                node.terminal_value = self._terminal_value(state, node.to_play)
                if self.solve:
                    self._mark_solved(node, node.terminal_value)
                break
            if node.solved is not None:
                break   # proven subtree: no point searching it again

        if node.solved is not None:
            value = float(node.solved)
        elif node.is_terminal:
            value = node.terminal_value
        else:
            value = self._expand(node, state, add_noise=False)
            # Expansion may itself have proved the node (a terminal child was
            # found). An exact value beats the net's guess.
            if node.solved is not None:
                value = float(node.solved)

        self._backup(node, value)

    def _run_budget(self, root, root_state, deadline):
        """Drive simulations until the budget -- count or clock -- runs out.

        Returns the number of simulations actually completed.

        DEADLINE ADMISSION. A chunk is started only if it is PREDICTED to finish
        before the deadline. Checking the clock afterwards instead would
        overshoot by a whole chunk on essentially every move, and a chunk here
        is a batched forward pass -- exactly the p99 tail a move budget exists
        to bound. The predictor is deliberately pessimistic (fast up, slow down,
        times a margin) because overshooting the budget FAILS the requirement
        while stopping early only costs a few simulations.

        The fixed-simulation path is unchanged: with `deadline is None` the
        chunking, the _MIN_WAVES clamp and the proof-note points are the same
        ones the pre-existing loops used.
        """
        serial = self.wave_size == 1
        if deadline is None:
            # Clamp so every search gets at least _MIN_WAVES waves of depth.
            eff = 1 if serial else max(
                1, min(self.wave_size, self.n_sims // self._MIN_WAVES))
        else:
            # Under a clock the total is not known in advance, so the caller's
            # wave_size stands; a 1 s search runs hundreds of waves and the
            # depth floor _MIN_WAVES protects is never in danger. The arena
            # asserts the realised wave count anyway.
            eff = 1 if serial else self.wave_size

        sims_done = 0
        pred = None
        self.stopped_early = False
        # Worst single chunk in this search. A chunk is atomic -- once a
        # batched forward pass is launched the deadline cannot interrupt it --
        # so this is the floor on how far a search can overrun, and the number
        # to look at when the reserve turns out to be too small.
        self.worst_chunk_ms = 0.0
        while True:
            if self.max_sims and sims_done >= self.max_sims:
                break
            # A PROVEN root cannot be improved on. search() returns the proof's
            # move regardless of how the remaining visits fall, so every further
            # simulation is a descent into an already-settled subtree that
            # cannot change the answer -- and because those descents skip the
            # network they are nearly free, so the loop would happily burn the
            # whole deadline on tens of thousands of them. Returning at once is
            # both correct and the single cheapest latency win available.
            #
            # Deadline mode ONLY. Under a fixed simulation count the visit
            # counts are the product (they are distillation targets), and the
            # frozen pilot corpora were generated by that path -- changing it
            # would silently invalidate artifacts already on disk.
            #
            # The min_sims guard is load-bearing, not tidiness. A root proven at
            # EXPANSION (before any simulation) has zero visits everywhere, and
            # a proven loss or draw falls through to the visit counts -- so
            # breaking at sims_done == 0 hands back an all-zero policy whose
            # argmax is cell 0, an illegal move. One chunk gives the counts
            # something to order.
            if (deadline is not None and self.solve
                    and root.solved is not None
                    and sims_done >= self.min_sims):
                self.stopped_early = True
                self.stat_early_stops += 1
                break
            if deadline is None:
                if sims_done >= self.n_sims:
                    break
                chunk = min(eff, self.n_sims - sims_done)
            else:
                chunk = eff
                if sims_done >= self.min_sims and pred is not None:
                    if time.perf_counter() + pred * self.deadline_margin > deadline:
                        break

            t0 = time.perf_counter()
            if serial:
                self._run_sim(root, root_state)
            else:
                self._run_wave(root, root_state, chunk, sims_done)
            dt = time.perf_counter() - t0
            if dt * 1000.0 > self.worst_chunk_ms:
                self.worst_chunk_ms = dt * 1000.0
            # Rise to a slow chunk immediately, decay away from it gradually:
            # a single stall must not be averaged into invisibility, because
            # the next chunk is the one that would blow the deadline.
            pred = dt if pred is None else max(dt, 0.6 * pred + 0.4 * dt)

            sims_done += chunk
            if self.solve:
                self._note_proof(root, sims_done)
        return sims_done

    @torch.no_grad()
    def search(self, root_state, root=None):
        """Run one search; return (visit policy, root node).

        `root` continues from a subtree kept across a move (see TreeReuseSearcher).
        It must already be expanded, its `to_play` must match `root_state.player`,
        and it must have been DETACHED from its old parent -- otherwise every
        backup walks up into the discarded tree. The caller owns those checks;
        `_reroot` is the supported way to satisfy them.
        """
        t_search = time.perf_counter()
        deadline = (t_search + (self.time_budget_ms - self.reserve_ms) / 1000.0
                    if self.time_budget_ms else None)
        reused = root is not None
        inherited = root.N if reused else 0
        if not reused:
            root = MCTSNode(to_play=root_state.player)
        self.stat_searches += 1
        self._begin_record()
        if not reused:
            self._expand(root, root_state, add_noise=self.add_dirichlet)
        if self.solve:
            # A mate-in-1 root is proven by the root expansion itself, before a
            # single simulation runs. Recording that as sim 0 is exact, and it
            # is the case the whole feature exists for.
            self._note_proof(root, 0)

        sims_done = self._run_budget(root, root_state, deadline)
        self.stat_sims += sims_done

        pi = np.zeros(81, dtype=np.float32)
        for mv, child in root.children.items():
            pi[mv] = child.N
        raw_argmax = int(pi.argmax())

        if self.solve:
            # Visit counts and the proof can disagree, and when they do the
            # proof is right. A move the net loved can bank hundreds of visits
            # and only later be refuted; raw counts would then hand back a
            # target whose argmax is a move known to LOSE. Two corrections:
            #
            #   proven win  -> one-hot. Any proven win ends the game, so the
            #                  honest target mass is 1.0 on it. This is the
            #                  forced-win dilution from RESULT_DISTILL_PILOT.md
            #                  removed at the source rather than smoothed over.
            #   proven loss -> zero. Unless EVERY move loses, in which case the
            #                  counts stand: something must still be played, and
            #                  visits are as good an ordering as any.
            #
            # Ties among several proven wins go to the most-visited, lowest move
            # index -- fully deterministic, which the fixed panels require.
            wins = [c for c in root.children.values() if c.solved == -1]
            if wins:
                best = max(wins, key=lambda c: (c.N, -c.move))
                pi[:] = 0.0
                pi[best.move] = 1.0
                self.stat_solved_roots += 1
                elapsed = time.perf_counter() - t_search
                self.stat_seconds += elapsed
                self._finish_record(root, raw_argmax, best.move, sims_done,
                                    elapsed, inherited, reused)
                return pi, root
            losses = [c for c in root.children.values() if c.solved == 1]
            if losses and len(losses) < len(root.children):
                for c in losses:
                    pi[c.move] = 0.0

        s = pi.sum()
        if s > 0:
            pi /= s
        elif root.children:
            # Nothing to normalise. An all-zero pi argmaxes to cell 0, which is
            # illegal in almost every position, so a caller taking argmax would
            # crash the game rather than play badly. The priors are a proper
            # distribution over exactly the legal moves, so they are the correct
            # degenerate answer.
            for mv, child in root.children.items():
                pi[mv] = child.prior
            t = pi.sum()
            if t > 0:
                pi /= t
            else:
                for mv in root.children:
                    pi[mv] = 1.0 / len(root.children)
        elapsed = time.perf_counter() - t_search
        self.stat_seconds += elapsed
        self._finish_record(root, raw_argmax, int(pi.argmax()), sims_done,
                            elapsed, inherited, reused)
        return pi, root

    # ------------------------------------------------------------------
    # Batched wave (wave_size > 1 path)
    # ------------------------------------------------------------------

    def _run_wave(self, root, root_state, wave_size, sims_base=0):
        """Select `wave_size` leaves with virtual loss, batch-evaluate, backup.

        Each selection applies virtual loss (N+1, W+VL) to visited nodes so
        subsequent selections in the same wave score those paths lower and
        diverge. After the batch forward pass the VL is undone before real
        backup, leaving tree statistics as if each sim ran independently.
        """
        pending = []    # (leaf_node, leaf_state, path_nodes_that_got_vl)
        mirror = self._mirror
        vl = self._VL

        for _ in range(wave_size):
            node  = root
            state = _clone(root_state)
            path  = []

            while node.children and not node.is_terminal:
                node = self._best_child(node)
                state.make_move(node.move)
                # Virtual loss. SIGN MATTERS: this tree stores W from the CHILD's
                # to-play perspective and _best_child scores -c.Q(), so to make a
                # visited path look BAD to the next sim we must RAISE the child's
                # W (a virtual win for the child = a loss for the selecting
                # parent). W -= VL here inverts that: Q -> -1, -Q -> +1, and the
                # whole wave collapses onto one path (found 2026-07-04: made
                # wave=64 search score 0.000 vs the raw league net; wave=1
                # scored 0.875).
                node.N += 1
                node.W += vl
                if mirror:
                    # `node.parent` is the node we just selected from, so it is
                    # never None here and it always has a mirror -- guarding
                    # would only hide the day that stops being true.
                    p = node.parent
                    i = node.cidx
                    p.selN[i] = node.N
                    p.selW[i] = node.W
                path.append(node)
                if state.winner is not None:
                    node.is_terminal    = True
                    node.terminal_value = self._terminal_value(state, node.to_play)
                    if self.solve:
                        self._mark_solved(node, node.terminal_value)
                    break
                if node.solved is not None:
                    break   # proven subtree: costs one descent, no net eval

            pending.append((node, state, path))

        # NOTE: proofs are NOT recorded here, mid-wave. Virtual loss has inflated
        # every path child's N by +1 per selection and it is not undone until the
        # end of this function, so a visit snapshot taken now would be wrong (and
        # root.N - child.N could even go negative). search() notes proofs at wave
        # boundaries instead, where the counts are real. That costs granularity
        # in proof_sim -- one wave, i.e. n_sims//16 -- and the reported figure is
        # therefore an upper bound on when the proof actually landed.

        # Collect unique unexpanded non-terminal leaves for the batch forward pass.
        # Dedup by node identity: two sims landing on the same unexpanded leaf
        # (possible in a shallow tree early in search) expand it once.
        seen    = {}    # id(node) -> index in to_eval
        to_eval = []    # (pending_idx, node, state)
        for i, (node, state, _) in enumerate(pending):
            if not node.is_terminal and node.solved is None and not node.children:
                nid = id(node)
                if nid not in seen:
                    seen[nid] = len(to_eval)
                    to_eval.append((i, node, state))

        # Batch forward pass for all unique leaves.
        leaf_values = {}    # id(node) -> float
        if to_eval:
            # Counted once, before the branch, so the two paths cannot disagree
            # about what a wave cost. They did while these lived inside each
            # branch, and the parity test caught it.
            self.stat_nn_batches += 1
            self.stat_nn_evals += len(to_eval)
            gw = self.graph_wave
            if gw is not None and gw.accepts(len(to_eval)):
                # One dispatch for the whole device sequence, and one event
                # wait instead of four accidental stream synchronizations.
                # Bit-identical to the eager branch below; see
                # agents/test_graph_wave.py.
                leaf_values = self._expand_wave_graphed(to_eval)
            else:
                # S8: fill all K leaves into one buffer (C++ fill_planes when
                # available) instead of K per-leaf builds + torch.stack.
                xs = wave_planes([s for _, _, s in to_eval], self.device)

                logits_b, values_b = self.model.forward_both(xs)
                # forward_both squeezes batch=1 to (81,)/scalar; restore dims.
                if logits_b.dim() == 1:
                    logits_b = logits_b.unsqueeze(0)
                    values_b = values_b.unsqueeze(0)

                if self.batched_expand:
                    leaf_values = self._expand_wave(to_eval, logits_b, values_b)
                else:
                    for k, (_, node, state) in enumerate(to_eval):
                        self._expand_from_logits(node, state, logits_b[k],
                                                 add_noise=False)
                        leaf_values[id(node)] = float(values_b[k].item())

        # Undo virtual loss then apply real backup for each pending sim.
        for node, state, path in pending:
            if mirror:
                for n in path:
                    n.N -= 1
                    n.W -= vl
                    p = n.parent
                    i = n.cidx
                    p.selN[i] = n.N
                    p.selW[i] = n.W
            else:
                for n in path:
                    n.N -= 1
                    n.W -= vl

            # solved first: expansion above may have proved this very node, and
            # an exact result outranks both the terminal cache and the net.
            if node.solved is not None:
                value = float(node.solved)
            elif node.is_terminal:
                value = node.terminal_value
            elif id(node) in leaf_values:
                value = leaf_values[id(node)]
            else:
                # Node already had children (expanded by a prior wave or by a
                # sibling sim in this wave) -- use its current Q as estimate.
                value = node.Q()

            self._backup(node, value)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Solved-node propagation (all no-ops unless self.solve)
    # ------------------------------------------------------------------

    def _mark_terminal_children(self, node, state):
        """Exact 1-ply search: mark every child that ends the game, then try to
        solve `node` from what that revealed.

        Uses the engine (clone + make_move) rather than a hand-rolled win test.
        A bespoke detector would have to re-derive mini-win, macro-win and
        board-full-draw rules, and a subtle disagreement with the engine would
        silently poison every target built from this search. Measured cost is
        ~2 us per legal move on the C++ engine and ~7 us on the Python one --
        against ~600 ms for an 800-sim search, that is noise.

        THAT LAST SENTENCE IS NOW WRONG AND IS KEPT AS WRITTEN. Under a 1 s
        clock the same loop is 4,113 roots and 33,699 probed children a move --
        126.9 ms inclusive, 16.7% of a move, the largest single host line item
        once native selection landed -- and 99.26% of those children are not
        terminal. `probe_filter` is the answer to that measurement: `could_end`
        is a NECESSARY condition, so when it is False this loop provably cannot
        mark a child and the early return is exactly equivalent, not an
        approximation traded for speed. See RESULT_PROBE_ABLATION.md.
        """
        self.stat_probe_roots += 1
        if self.probe_filter and not could_end(state.mini_winners,
                                               state.player):
            # No legal move from here can end the game, so no child can be
            # marked, so every child keeps `solved is None` -- and
            # `_solve_from_children` over children that are all unsolved with
            # none refuted returns None, which is what it would have returned
            # after the loop ran. Returning here therefore skips dead work and
            # nothing else; it is not a heuristic cut.
            self.stat_probe_skips += 1
            return
        sS = node.selS if self._mirror else None
        for mv, child in node.children.items():
            probe = _clone(state)
            probe.make_move(mv)
            self.stat_probes += 1
            if probe.winner is not None:
                child.is_terminal    = True
                child.terminal_value = self._terminal_value(probe, child.to_play)
                child.solved         = int(child.terminal_value)
                if sS is not None:
                    sS[child.cidx] = child.solved
        status = self._solve_from_children(node)
        if status is not None:
            self._mark_solved(node, status)

    @staticmethod
    def _solve_from_children(node):
        """Backward induction over `node`'s children, in NODE's perspective.

        Returns +1 / 0 / -1 (proven win / draw / loss for node.to_play) or None.

        A child's `solved` is in the CHILD's perspective, and the child's mover
        is our opponent, so a child solved -1 (lost for whoever moves there) is
        a WIN for us. That sign flip is the same one `_best_child` and `_backup`
        make; getting it backwards here would make the search chase its own
        losses, so it is stated explicitly rather than inlined.
        """
        kids = node.children
        if not kids:
            return None          # unexpanded: absence of children proves nothing
        all_solved = True
        has_draw = False
        for c in kids.values():
            s = c.solved
            if s is None:
                all_solved = False
            elif s == -1:
                return 1         # one refuted reply is enough to prove the win
            elif s == 0:
                has_draw = True
        if not all_solved:
            return None          # loss/draw claims need EVERY legal reply solved
        return 0 if has_draw else -1

    def _mark_solved(self, node, value):
        # Every sim that walks into an already-terminal node re-reaches this;
        # returning early keeps that from re-walking the ancestor chain each
        # time. A proof never changes once made, so re-deriving it is pure cost.
        if node.solved is not None:
            return
        node.solved = int(value)
        if self._mirror:
            p = node.parent
            if p is not None:
                p.selS[node.cidx] = node.solved
        self._propagate_solved(node)

    def _propagate_solved(self, node):
        """Carry a fresh proof as far up the tree as it can be justified."""
        mirror = self._mirror
        cur = node.parent
        while cur is not None and cur.solved is None:
            status = self._solve_from_children(cur)
            if status is None:
                return
            cur.solved = status
            p = cur.parent
            if mirror and p is not None:
                p.selS[cur.cidx] = status
            cur = p

    # ------------------------------------------------------------------

    def _expand_wave(self, to_eval, logits_b, values_b):
        """Expand a whole wave with TWO device transfers instead of two per leaf.

        `_expand_from_logits` masks and softmaxes ON the device and then pulls
        the row back, so every leaf costs a host->device tensor build
        (`torch.tensor(list, device=cuda)`, already known to be slow for small
        lists) plus a device->host sync. For a wave of K that is 2K round trips
        AFTER the single batched forward pass, which is why batch size stopped
        paying: measured at a 1 s budget, 1,399 nn-evals/s at wave 8 against
        1,525 at wave 64 -- 9% for an 8x batch. The forward pass was never the
        bottleneck; the per-leaf traffic around it was.

        Masking the whole (K, 81) block at once leaves exactly two transfers.

        Returns {id(node): leaf_value}.
        """
        k = len(to_eval)
        valids = [rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
                  for _, _, s in to_eval]
        mask = np.zeros((k, 81), dtype=bool)
        for i, v in enumerate(valids):
            mask[i, v] = True
        mask_t = torch.from_numpy(mask).to(logits_b.device)
        probs = F.softmax(logits_b.masked_fill(~mask_t, float("-inf")), dim=1)
        probs_np = probs.cpu().numpy()                       # one transfer
        values_np = values_b.reshape(-1).cpu().numpy()       # one transfer
        return self._expand_children(to_eval, valids, probs_np, values_np)

    def _expand_wave_graphed(self, to_eval):
        """The same wave, replayed from a CUDA graph instead of issued.

        Only the DEVICE work moves: the legal moves are still computed here,
        the children are still built by `_expand_children`, and the priors and
        leaf values are required to be bit-identical to the eager path. What
        goes away is 36 kernel launches at ~26 us of CPU each and the four
        stream synchronizations that a pageable copy emits behind your back.
        """
        valids = [rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
                  for _, _, s in to_eval]
        probs_np, values_np = self.graph_wave.run(
            [s for _, _, s in to_eval], valids)
        return self._expand_children(to_eval, valids, probs_np, values_np)

    def _build_children_mirrored(self, node, valid, row, next_to_play):
        """Create `node`'s children AND the native selection mirror beside them.

        The mirror's row order is `valid` order, which is what makes the whole
        thing work: `node.children` is a dict populated in this same order, so
        `children.values()` and the native columns are the same sequence, and
        `max()`'s first-maximal tie-break and a forward scan with a strict `>`
        agree. That order is NOT ascending board index -- on a send-anywhere
        position rule_utl_valid_moves runs mini-major, and 11 of 400 random
        reachable positions come back unsorted -- so a mirror that sorted by
        move, or that walked an 81-cell mask, would break ties differently
        several times a game. See tools/select_parity.py, which asserts the
        two orders are identical rather than trusting this comment.

        Priors are collected here rather than re-read from `row` because
        `float(row[mv])` is the value Python stores, and the mirror must hold
        that exact double -- not a second float32 round trip that happens to
        agree today.
        """
        kids = []
        priors = []
        append_kid = kids.append
        append_prior = priors.append
        children = node.children
        for j, mv in enumerate(valid):
            pr = float(row[mv])
            c = MCTSNode(parent=node, prior=pr, move=mv, to_play=next_to_play)
            c.cidx = j
            children[mv] = c
            append_kid(c)
            append_prior(pr)
        sel = _ns.ChildArray(valid, priors, self.c_puct, self.solve)
        node.kids = kids
        node.selN = sel.N
        node.selW = sel.W
        node.selS = sel.S
        # Published LAST. `_best_child` dispatches on `node.sel`, so until this
        # line the node is simply a node without a mirror and the Python path
        # handles it correctly. Assigning it first would open a window where a
        # re-entrant selection could index columns that are not attached yet.
        node.sel = sel

    def _expand_children(self, to_eval, valids, probs_np, values_np):
        """Build the children. ONE implementation, shared by both paths.

        Kept shared on purpose: if the graph path had its own copy of this
        loop, a parity test between the two would be testing two copies of the
        same code rather than the thing that actually differs.
        """
        out = {}
        mirror = self._mirror
        for i, (_pi, node, state) in enumerate(to_eval):
            row = probs_np[i]
            next_to_play = O if state.player == X else X
            if mirror:
                self._build_children_mirrored(node, valids[i], row,
                                              next_to_play)
            else:
                for mv in valids[i]:
                    node.children[mv] = MCTSNode(parent=node,
                                                 prior=float(row[mv]),
                                                 move=mv,
                                                 to_play=next_to_play)
            self.stat_expansions += 1
            self.stat_nodes_created += len(valids[i])
            if self.solve:
                self._mark_terminal_children(node, state)
            out[id(node)] = float(values_np[i])
        return out

    def _expand_from_logits(self, node, state, logits, add_noise=False):
        """Populate node.children from pre-computed 1-D (81,) policy logits."""
        valid   = rule_utl_valid_moves(state.board, state.last_move, state.mini_winners)
        mask    = torch.full((81,), float('-inf'), device=logits.device)
        valid_t = torch.tensor(valid, device=logits.device, dtype=torch.long)
        mask.scatter_(0, valid_t, logits[valid_t])
        probs   = F.softmax(mask, dim=0).cpu().numpy()

        if add_noise and len(valid) > 1:
            noise = np.random.dirichlet([self.dir_alpha] * len(valid))
            for i, mv in enumerate(valid):
                probs[mv] = (1 - self.dir_eps) * probs[mv] + self.dir_eps * noise[i]

        next_to_play = O if state.player == X else X
        if self._mirror:
            self._build_children_mirrored(node, valid, probs, next_to_play)
        else:
            for mv in valid:
                node.children[mv] = MCTSNode(parent=node,
                                             prior=float(probs[mv]),
                                             move=mv, to_play=next_to_play)
        self.stat_expansions += 1
        self.stat_nodes_created += len(valid)
        if self.solve:
            self._mark_terminal_children(node, state)

    def _expand(self, node, state, add_noise=False):
        """Run one network forward pass for a single leaf, expand it, return value."""
        x = board_to_tensor_from_gamestate(state).unsqueeze(0).to(self.device)
        logits, value = self.model.forward_both(x)
        self.stat_nn_batches += 1
        self.stat_nn_evals += 1
        # forward_both squeezes batch=1 -> (81,) and scalar; both correct here.
        self._expand_from_logits(node, state, logits, add_noise=add_noise)
        # NOTE: `value` is the net's UNBOUNDED shaped-return estimate, not a
        # calibrated [-1, 1] outcome (see module docstring "value SCALE mismatch").
        # It is compared against clean +/-1.0 terminal values in backup/Q.
        return float(value.item())

    def _best_child(self, node):
        # child.Q() is stored from the CHILD's to_play perspective (backup flips
        # sign per ply). The child's mover is the opponent of `node`'s mover, so to
        # score a move from `node`'s perspective we must NEGATE the child's Q.
        # Using +c.Q() here would pick the move that is best for the opponent.
        #
        # #45a: when the mirror exists the whole scan below happens in C++ over
        # five contiguous columns and comes back as an index. The Python
        # version stays as the definition of the answer -- tools/select_parity
        # requires the two to return the SAME CHILD, not a similar score --
        # and it is what runs when native_select is off.
        sel = node.sel
        if sel is not None:
            return node.kids[sel.best(node.N)]
        kids = node.children.values()
        if self.solve:
            # A proven win ends the discussion -- there is nothing PUCT could
            # find that beats a forced win, and continuing to explore is exactly
            # the dilution this feature exists to stop. Deterministic tie-break.
            wins = [c for c in kids if c.solved == -1]
            if wins:
                return max(wins, key=lambda c: (c.N, -c.move))
            # Never spend another visit on a refuted move. If they are ALL
            # refuted this node is lost regardless, so fall through to PUCT and
            # keep the statistics well-formed rather than special-casing.
            live = [c for c in kids if c.solved != 1]
            if live:
                kids = live
        return max(kids,
                   key=lambda c: -c.Q() + c.U(self.c_puct, node.N))

    def _backup(self, node, leaf_value):
        v = leaf_value
        if self._mirror:
            # Write THROUGH, do not accumulate independently. Python's node is
            # authoritative; copying its post-update value into the parent's
            # column means the mirror cannot drift even if some other site
            # forgets to call here -- it can only be stale, and the next backup
            # through that edge repairs it. A native `+= v` alongside the
            # Python one would be a second accumulator and a second answer.
            while node is not None:
                node.N += 1
                node.W += v
                p = node.parent
                if p is not None:
                    i = node.cidx
                    p.selN[i] = node.N
                    p.selW[i] = node.W
                v = -v
                node = p
            return
        while node is not None:
            node.N += 1
            node.W += v
            v = -v
            node = node.parent

    @staticmethod
    def _terminal_value(state, to_play):
        if state.winner in (DRAW, None):
            return 0.0
        return 1.0 if state.winner == to_play else -1.0


class TreeReuseSearcher:
    """Carry the search tree across moves instead of rebuilding it every turn.

    After we move and the opponent replies, the subtree under the position
    actually reached was already searched. Throwing it away and starting from an
    empty root spends the whole move budget re-deriving statistics we owned a
    moment ago. Re-rooting keeps them, so a fixed wall clock buys strictly more
    effective search without touching the network.

    ADOPTION IS PROVEN, NOT ASSUMED. The stored root is only adopted when the
    board differs from it by exactly one of our marks and one of theirs, both
    moves exist as nodes, the survivor is already expanded, and its `to_play`
    matches the position handed in. Anything else -- a new game, an opponent
    that took back a move, a caller reusing one searcher for two games -- is a
    MISS and silently falls back to a fresh tree. A wrongly adopted tree would
    score a different position's statistics, which is a silent strength bug of
    the worst kind, so every condition is checked rather than inferred from the
    caller's good behaviour.

    Detaching (`parent = None`) is load-bearing: `_backup` walks parents to the
    top, so an attached survivor would push every new simulation up into the
    discarded tree and inflate counts nobody reads.

    DEFERRED RETIREMENT (`defer_release=True`, #46). Destroying the discarded
    tree is O(nodes) and, measured on `pocket_sel`, it IS the deployment latency
    tail: 20.4 ms a move on average, 53.3 at p99, against a caller-side overhead
    p99 of 53.3 -- release is essentially the whole of it, and the reserve that
    covers it had grown 20 -> 35 -> 50 -> 95 ms across four engines because the
    walk gets longer every time throughput improves. It is also pure per-node
    work: 0.552 us/node with an intercept of -0.02 ms and R^2 0.986, so there is
    no fixed cost to attack and nothing per-node to shave.

    So it is not made faster, it is moved. Detaching is already O(1) -- `_adopt`
    does it -- and destruction is deferred to `reset()`, the game boundary,
    which no deadline covers. What that costs is memory: measured at 430 bytes
    a node and 1,148,422 nodes in the worst of eight games, so 471 MB held at
    the end of the worst game, 1.4% of this box. `retire_watermark` bounds it
    for the pathological case and is the ONLY thing that can put the walk back
    on the move path.
    """

    # Nodes that may be alive before the watermark forces a drain. It counts
    # the live tree as well as the queue, which is what a MEMORY bound wants.
    # 3,000,000 nodes is about 1.2 GB at the measured 430 bytes/node and about
    # 2.6x the worst game observed, so it is a backstop against a pathological
    # game and not a mechanism that runs in normal play. Set it to 0 to disable
    # the backstop entirely.
    DEFAULT_WATERMARK = 3_000_000

    def __init__(self, mcts, enabled=True, count_nodes=False,
                 defer_release=False, retire_watermark=None):
        self.mcts = mcts
        self.enabled = enabled
        # Node counting is two full subtree walks per move. Cheap next to a 1 s
        # search, but it is instrumentation, so it is opt-in and never on in a
        # timing run that is measuring the search itself.
        self.count_nodes = count_nodes
        self.defer_release = defer_release
        self.retire_watermark = (self.DEFAULT_WATERMARK
                                 if retire_watermark is None
                                 else retire_watermark)
        self._retired = []
        self._created_at_drain = 0
        self._root = None          # before reset(), which now releases it
        # Before reset(), which drains and therefore touches the drain
        # counters. The queue is empty here so it takes the early return, but
        # relying on that would make the order load-bearing for no reason.
        self.reset_stats()
        self.reset()

    @staticmethod
    def release(root, keep=None):
        """Break parent<->children cycles so refcounting can free the tree now.

        THE LATENCY TAIL IS THE CYCLIC COLLECTOR. A discarded search tree is
        cyclic garbage -- every node points at its parent and every parent at
        its children -- so refcounting cannot touch it and only gc can. At a 1 s
        budget with ~2,700 expansions per move that is a lot of garbage per
        move, and the collector's pauses land mid-chunk where no deadline
        predictor can see them. Measured, wave 8 batched:

            gc on   worst chunk mean 25.2 ms, p99 87.6, max 90.6  -> p99 1037 ms
            gc off  worst chunk mean  3.1 ms, p99  5.2, max  7.9  -> p99  980 ms

        Simply disabling gc would leak, because these cycles are exactly what
        needs collecting. Breaking them ourselves means the tree dies by
        refcount, at a moment we choose, for one O(nodes) walk instead of an
        unpredictable full-heap scan.

        `keep` is a subtree to spare -- it and everything under it are left
        untouched, which is what makes this safe to call right after a re-root.
        """
        stack = [root]
        while stack:
            n = stack.pop()
            if n is keep:
                continue          # retained: leave its subtree wholly intact
            kids = n.children
            n.children = {}
            n.parent = None
            # #45a: `n.kids` is a SECOND list of strong references to exactly
            # the children `n.children` just dropped. Leaving it would keep the
            # parent<->child cycle intact and quietly undo the whole reason
            # this function exists -- gc back on, worst chunk from 3.1 ms to
            # 25.2, p99 from 980 ms to 1037. Cleared under the `sel` test so
            # the non-mirrored engine pays nothing for a slot it never set.
            if n.sel is not None:
                n.sel = None
                n.kids = None
                n.selN = None
                n.selW = None
                n.selS = None
            stack.extend(kids.values())

    def reset(self):
        """Drop the tree. Call between games.

        This is where deferred retirement is paid for, and it is the right
        place: a game boundary belongs to no move, so no deadline covers it.
        The drain runs BEFORE the live root is released, because every queued
        root's subtree contains the live one (see `drain`) and doing it in this
        order means the live tree is walked once, by the drain, rather than
        twice.
        """
        self.drain(keep=self._root)
        if self._root is not None:
            self.release(self._root)
        self._root = None
        self._board = None
        self._to_play = None
        self._created_at_drain = self.mcts.stat_nodes_created

    def alive_bound(self):
        """Upper bound on the nodes this searcher is keeping reachable.

        Children created since the last drain, which is the queue PLUS the live
        tree plus anything already destroyed by an intervening non-deferred
        release. Deliberately an over-estimate and deliberately free: the exact
        number would need a per-node counter inside `release`, and that walk is
        shared with the engines this feature is off for.
        """
        return self.mcts.stat_nodes_created - self._created_at_drain

    def drain(self, keep=None):
        """Destroy every retired subtree. Never called from inside a search.

        ONE WALK IS ENOUGH FOR THE WHOLE QUEUE, and the reason is worth stating
        because it is what makes the queue a list rather than a graph. Retired
        root k's subtree CONTAINS retired root k+1: k+1 is the survivor that was
        adopted out of k. So releasing the oldest entry with `keep` set to the
        live root destroys everything that follows it, and the later entries
        then walk an already-emptied structure for nothing. Adoption misses
        start a new chain, which is why this is a list and not a single slot.
        """
        if not self._retired:
            self._created_at_drain = self.mcts.stat_nodes_created
            return 0
        t0 = time.perf_counter()
        n = len(self._retired)
        while self._retired:
            self.release(self._retired.pop(0), keep=keep)
        self.stat_drains += 1
        self.stat_drained_roots += n
        self.stat_drain_seconds += time.perf_counter() - t0
        self._created_at_drain = self.mcts.stat_nodes_created
        return n

    # Why a candidate subtree was not adopted. `unexpanded` is the interesting
    # one: the survivor is a perfectly real node, the opponent just replied with
    # a move this search barely looked at, so there is nothing under it to keep.
    # That is a statement about search coverage, not about the reuse mechanism,
    # and lumping it in with the genuine misses would hide it.
    MISS_REASONS = ("no_tree", "not_two_ply", "our_move_missing",
                    "reply_missing", "unexpanded", "to_play_mismatch")

    def reset_stats(self):
        self.stat_hits = 0
        self.stat_misses = 0
        self.stat_inherited_sims = 0
        self.stat_reused_nodes = 0
        self.stat_prev_nodes = 0
        self.stat_miss_reason = {k: 0 for k in self.MISS_REASONS}
        self.stat_drains = 0
        self.stat_drained_roots = 0
        self.stat_drain_seconds = 0.0
        # Drains forced by the watermark rather than by a game boundary. A
        # non-zero value here means the walk went back onto the move path, so
        # it is a number a latency report has to be able to see.
        self.stat_forced_drains = 0

    @staticmethod
    def _count(node):
        n, stack = 0, [node]
        while stack:
            n += 1
            stack.extend(stack.pop().children.values())
        return n

    def _adopt(self, state):
        """(survivor subtree, reason). The node is None whenever reason is set."""
        if self._root is None:
            return None, "no_tree"
        b0, b1 = self._board, tuple(state.board)
        changed = ([] if len(b0) != len(b1) else
                   [i for i in range(len(b0)) if b0[i] != b1[i]])
        # Exactly two plies: ours then theirs. One ply means the game ended on
        # our move (so we are never asked again) and anything else is not our
        # tree -- a new game, or a caller sharing one searcher across games.
        if len(changed) != 2:
            return None, "not_two_ply"
        ours = [i for i in changed if b1[i] == self._to_play]
        if len(ours) != 1:
            return None, "not_two_ply"
        our_mv = ours[0]
        opp_mv = changed[1] if changed[0] == our_mv else changed[0]
        node = self._root.children.get(our_mv)
        if node is None:
            return None, "our_move_missing"
        node = node.children.get(opp_mv)
        if node is None:
            return None, "reply_missing"
        # An unexpanded survivor has no priors, so adopting it would hand
        # search() a root it documents as already expanded. It also has N == 0,
        # so a fresh tree inherits exactly as much: nothing.
        if not node.children:
            return None, "unexpanded"
        if node.to_play != state.player:
            return None, "to_play_mismatch"
        node.parent = None
        return node, None

    def search(self, state):
        root, reason = (self._adopt(state) if self.enabled
                        else (None, "no_tree"))
        if self.count_nodes and self._root is not None:
            self.stat_prev_nodes += self._count(self._root)
        if root is None:
            self.stat_misses += 1
            self.stat_miss_reason[reason] += 1
            reused_nodes = 0
        else:
            self.stat_hits += 1
            self.stat_inherited_sims += root.N
            reused_nodes = self._count(root) if self.count_nodes else 0
            self.stat_reused_nodes += reused_nodes

        # Free last move's tree BEFORE searching, sparing whatever was adopted.
        # Order matters: after _adopt (which needs the old root) and after
        # _count (which needs it intact), but before the clock starts.
        #
        # "Before the clock starts" was never the same as "free". The caller
        # waits for this walk, the frozen requirement is written against what
        # the caller waits for, and #46a measured it at 100% of the caller-side
        # overhead p99. Deferring makes this an append: `_adopt` has already
        # detached the survivor, so the old root is the head of a subtree
        # nothing live points up into, and it can simply be put aside until
        # `reset()`.
        if self._root is not None:
            if self.defer_release:
                self._retired.append(self._root)
            else:
                self.release(self._root, keep=root)
        # The backstop, and the only path that can put the walk back on the
        # move path. It degrades to exactly the behaviour above rather than to
        # something new: a pathological game should get slow, not novel.
        if (self.defer_release and self.retire_watermark
                and self.alive_bound() > self.retire_watermark):
            self.stat_forced_drains += 1
            self.drain(keep=root)

        pi, new_root = self.mcts.search(state, root=root)
        if isinstance(self.mcts.last, dict):
            self.mcts.last["tree_nodes_reused"] = reused_nodes if root else 0
        self._root = new_root
        self._board = tuple(state.board)
        self._to_play = state.player
        return pi, new_root

    def stats(self):
        moves = self.stat_hits + self.stat_misses
        return {
            "moves": moves,
            "reuse_hits": self.stat_hits,
            "reuse_miss": self.stat_misses,
            "reuse_rate": self.stat_hits / moves if moves else 0.0,
            "inherited_sims_per_move": (self.stat_inherited_sims / moves
                                        if moves else 0.0),
            "reused_nodes_per_move": (self.stat_reused_nodes / moves
                                      if moves else 0.0),
            "node_retention": (self.stat_reused_nodes / self.stat_prev_nodes
                               if self.stat_prev_nodes else None),
            "miss_reason": dict(self.stat_miss_reason),
            "defer_release": self.defer_release,
            "retired_queued": len(self._retired),
            "alive_bound": self.alive_bound(),
            "drains": self.stat_drains,
            "drained_roots": self.stat_drained_roots,
            "drain_ms": self.stat_drain_seconds * 1000.0,
            # Non-zero means the watermark fired and the walk went back onto a
            # move. It is reported next to the reuse statistics because that is
            # where anyone reading a latency regression will look.
            "forced_drains": self.stat_forced_drains,
        }


class MCTSAgent(Agent):
    """Drop-in arena/eval agent: wraps a trained NN with PUCT search."""

    def __init__(self, nn_agent, n_sims=100, c_puct=1.5, temperature=0.0,
                 add_dirichlet_at_root=False, dir_alpha=0.3, dir_eps=0.25,
                 wave_size=1, solve=False):
        super().__init__(f"MCTS({nn_agent.name}, n={n_sims})")
        self.nn   = nn_agent
        self.mcts = MCTS(
            model=nn_agent.model,
            device=nn_agent.device,
            n_sims=n_sims,
            c_puct=c_puct,
            add_dirichlet_at_root=add_dirichlet_at_root,
            dir_alpha=dir_alpha,
            dir_eps=dir_eps,
            wave_size=wave_size,
            solve=solve,
        )
        self.temperature = temperature

    def select_move(self, gamestate):
        pi, _ = self.mcts.search(gamestate)
        if self.temperature == 0.0:
            return int(np.argmax(pi))
        probs  = pi ** (1.0 / self.temperature)
        probs /= probs.sum()
        return int(np.random.choice(81, p=probs))
