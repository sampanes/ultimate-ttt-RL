"""#45a -- the parity oracle for native PUCT selection.

`ChildArray.best` must return the SAME CHILD as `MCTS._best_child`. Not a
similar score, the same index: the output is discrete, so there is no epsilon
in which a disagreement can be small. One divergent selection sends the rest of
that descent into a different subtree, and a tree does thousands of descents a
move.

    python -m tools.select_parity --mode all --device cuda

FOUR MODES, AND THE ORDER IS THE ARGUMENT.

  fixtures  The named cases from the task brief, written out one at a time:
            proven win, all-but-one refuted, unvisited children, N=0 beside a
            visited Q=0, huge N, tiny prior, exact ties, one-ULP ties, every
            child refuted. Each is a claim about a BRANCH, so each is separate
            and named -- a fuzz that happens to cover them tells you nothing
            when it goes green.
  sweep     Exhaustive over small child counts: every assignment of solved in
            {None,-1,0,1} and N in {0,1,7} to 2 and 3 children, times a set of
            W and prior values chosen to tie. Combinatorial coverage of the
            branch structure, which hand-written cases cannot claim.
  fuzz      Millions of random arrays, deliberately TIE-HEAVY. Random doubles
            essentially never collide, so a uniform fuzz would test the
            comparison and never the tie-break; the generator quantises N, W
            and prior onto small grids so exact ties are common, and injects
            one-ULP neighbours so near-ties are too.
  shadow    The one that matters. Real searches by the real deployment engine,
            with BOTH selectors run at every call and the answers compared.

WHY SHADOW IS THE PRIMARY ORACLE AND REPLAY IS NOT. A replay harness
reconstructs the native input from Python's authoritative node state and then
checks the scoring. That validates the arithmetic and nothing else -- it cannot
see a mirror that has drifted, because it rebuilds the mirror from the truth
every time. Drift is the whole risk of this design: five columns updated by
hand at eight call sites, one of which is virtual loss, which applies and undoes
`W += 1.0` at magnitudes where float addition does not round-trip. So shadow
mode reads the mirror THE ENGINE MAINTAINED, compares the index, and (with
--deep) compares all five columns against the nodes cell by cell.

SHADOW RETURNS THE PYTHON ANSWER. On a disagreement the search follows the
reference trajectory, so every later comparison is still a comparison against
the canonical tree and the disagreement count is a count of independent events.
Returning the native answer would fork the tree at the first miss and make
everything after it unattributable.

TIE SEMANTICS, STATED RATHER THAN DISCOVERED. `max()` keeps the FIRST maximal
element and walks `node.children.values()`, i.e. dict insertion order, i.e.
`rule_utl_valid_moves` order -- which is mini-major and NOT ascending board
index on a send-anywhere position (11 of 400 random reachable positions come
back unsorted). The native side stores that order and scans forward with a
strict `>`. The proven-win branch is the exception: it carries an explicit
`(N, -move)` key, so there the tie-break is lowest move index regardless of
order. `test_send_anywhere_order` and `--deep` both assert the stored order
equals the dict order rather than trusting this paragraph.
"""
import argparse
import gc
import json
import math
import os
import random
import sys
import time

import numpy as np

from agents import native_select as ns
from agents.mcts import MCTS, MCTSNode, TreeReuseSearcher
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from tools import engine_registry


SOLVED_NONE = ns.SOLVED_NONE
SOLVED_CODES = (None, -1, 0, 1)


def encode_solved(v):
    return SOLVED_NONE if v is None else int(v)


# ----------------------------------------------------------------------
# One case, both implementations.
# ----------------------------------------------------------------------

class Case:
    """A parent and its children, in one order, with everything selection reads.

    Deliberately holds REAL MCTSNode objects and calls the REAL
    `MCTS._best_child`. A hand-written Python replica of the selection rule
    would be a second implementation to keep in step, and the thing being
    tested is agreement with what the engine actually runs.
    """

    __slots__ = ("name", "moves", "priors", "N", "W", "S", "parent_N",
                 "c_puct", "solve")

    def __init__(self, name, moves, priors, N, W, S, parent_N,
                 c_puct=1.5, solve=True):
        self.name = name
        self.moves = list(moves)
        self.priors = [float(p) for p in priors]
        self.N = [int(x) for x in N]
        self.W = [float(x) for x in W]
        self.S = list(S)
        self.parent_N = int(parent_N)
        self.c_puct = float(c_puct)
        self.solve = bool(solve)

    def python_index(self):
        mcts = _bare_mcts(self.c_puct, self.solve)
        parent = MCTSNode(to_play=1)
        parent.N = self.parent_N
        order = []
        for j, mv in enumerate(self.moves):
            c = MCTSNode(parent=parent, prior=self.priors[j], move=mv,
                         to_play=2)
            c.N = self.N[j]
            c.W = self.W[j]
            c.solved = self.S[j]
            parent.children[mv] = c
            order.append(c)
        chosen = mcts._best_child(parent)
        return order.index(chosen)

    def native_index(self):
        ca = ns.ChildArray(self.moves, self.priors, self.c_puct, self.solve)
        ca.load(self.N, self.W, [encode_solved(s) for s in self.S])
        return ca.best(self.parent_N)

    def describe(self):
        rows = []
        ca = ns.ChildArray(self.moves, self.priors, self.c_puct, self.solve)
        ca.load(self.N, self.W, [encode_solved(s) for s in self.S])
        sc = ca.scores(self.parent_N)
        for j in range(len(self.moves)):
            rows.append(
                f"      [{j}] move={self.moves[j]:2d} N={self.N[j]:6d} "
                f"W={self.W[j]!r:>24} prior={self.priors[j]!r:>24} "
                f"solved={self.S[j]!s:>4} score={sc[j]!r}")
        return (f"    parent_N={self.parent_N} c_puct={self.c_puct} "
                f"solve={self.solve}\n" + "\n".join(rows))


_BARE = {}


def _bare_mcts(c_puct, solve):
    """An MCTS with no model, used only for `_best_child`.

    Cached because the constructor is not free and this is called once per
    fuzz case. It never searches, so a None model is exactly right -- and it
    would fail loudly rather than quietly if some future `_best_child` reached
    for one.
    """
    key = (c_puct, solve)
    m = _BARE.get(key)
    if m is None:
        m = MCTS(None, "cpu", n_sims=1, c_puct=c_puct, solve=solve)
        _BARE[key] = m
    return m


def check(case):
    """(ok, python_index, native_index)."""
    p = case.python_index()
    n = case.native_index()
    return p == n, p, n


# ----------------------------------------------------------------------
# Mode 1: the named fixtures from the brief.
# ----------------------------------------------------------------------

def fixtures():
    """Every case the task brief named, plus the ones the code structure
    demands. One function, one list, because a reader should be able to check
    coverage against the brief without running anything."""
    out = []
    A = out.append

    # -- proven winning child ------------------------------------------------
    # solved == -1 is a LOSS for the child's mover, i.e. a win for us. It must
    # beat any PUCT score, including a child with vastly more visits.
    A(Case("proven_win_beats_everything",
           moves=[4, 40, 76], priors=[0.9, 0.05, 0.05],
           N=[9000, 3, 0], W=[-8000.0, 1.0, 0.0], S=[None, -1, None],
           parent_N=9003))
    # Two proven wins: (N, -move) picks the most visited.
    A(Case("two_proven_wins_most_visited",
           moves=[4, 40, 76], priors=[0.4, 0.3, 0.3],
           N=[5, 11, 2], W=[0.0, 0.0, 0.0], S=[-1, -1, None],
           parent_N=18))
    # Two proven wins on EQUAL visits: lowest move index, and the array order
    # is deliberately descending so first-in-order would give the other one.
    A(Case("two_proven_wins_tied_visits_lowest_move",
           moves=[76, 40, 4], priors=[0.3, 0.3, 0.4],
           N=[7, 7, 7], W=[0.0, 0.0, 0.0], S=[-1, -1, -1],
           parent_N=21))
    # A proven win with ZERO visits still ends the discussion.
    A(Case("proven_win_zero_visits",
           moves=[0, 1], priors=[0.99, 0.01],
           N=[4000, 0], W=[3900.0, 0.0], S=[None, -1],
           parent_N=4000))

    # -- refutation ----------------------------------------------------------
    # All but one refuted: the survivor is forced even though PUCT loves a
    # refuted sibling.
    A(Case("all_but_one_refuted",
           moves=[3, 4, 5, 6], priors=[0.97, 0.01, 0.01, 0.01],
           N=[500, 1, 1, 1], W=[-400.0, 0.9, 0.9, 0.9],
           S=[1, 1, 1, None], parent_N=503))
    # EVERY child refuted: `live` is empty and Python falls THROUGH to PUCT
    # over all of them rather than special-casing a lost node.
    A(Case("every_child_refuted_falls_through",
           moves=[3, 4, 5], priors=[0.2, 0.7, 0.1],
           N=[10, 10, 10], W=[1.0, -1.0, 5.0], S=[1, 1, 1],
           parent_N=30))
    # Draws are live: solved == 0 is not filtered by either branch.
    A(Case("solved_draw_is_live",
           moves=[3, 4], priors=[0.5, 0.5],
           N=[1, 1], W=[0.5, -0.5], S=[0, None], parent_N=2))
    # A refuted child that would otherwise win outright.
    A(Case("refuted_child_would_have_won_puct",
           moves=[3, 4], priors=[0.99, 0.01],
           N=[0, 300], W=[0.0, 250.0], S=[1, None], parent_N=300))

    # -- visit structure -----------------------------------------------------
    A(Case("all_unvisited_equal_priors_first_wins",
           moves=[10, 11, 12, 13], priors=[0.25, 0.25, 0.25, 0.25],
           N=[0, 0, 0, 0], W=[0.0, 0.0, 0.0, 0.0],
           S=[None] * 4, parent_N=0))
    # N == 0 gives Q() == 0.0 by the guard; a VISITED child with W == 0 gives
    # Q == 0.0 by division. Same Q, different U, and the guard must not be
    # mistaken for the division.
    A(Case("zero_visits_beside_visited_zero_q",
           moves=[10, 11], priors=[0.5, 0.5],
           N=[0, 40], W=[0.0, 0.0], S=[None, None], parent_N=40))
    A(Case("very_large_n",
           moves=[10, 11], priors=[0.5, 0.5],
           N=[2_000_000, 1_999_999], W=[1_500_000.25, 1_499_999.75],
           S=[None, None], parent_N=3_999_999))
    A(Case("very_small_prior",
           moves=[10, 11], priors=[5e-324, 1e-300],
           N=[0, 0], W=[0.0, 0.0], S=[None, None], parent_N=1))
    A(Case("zero_prior_everywhere",
           moves=[10, 11, 12], priors=[0.0, 0.0, 0.0],
           N=[0, 0, 0], W=[0.0, 0.0, 0.0], S=[None] * 3, parent_N=0))
    # parent_N == 0 kills the whole U term, so every score is -Q. This is the
    # `sims=0` trap from the registry, reached inside a live search.
    A(Case("parent_n_zero_scores_are_all_minus_q",
           moves=[10, 11, 12], priors=[0.6, 0.3, 0.1],
           N=[0, 0, 0], W=[0.0, 0.0, 0.0], S=[None] * 3, parent_N=0))
    A(Case("negative_w_large",
           moves=[10, 11], priors=[0.5, 0.5],
           N=[100, 100], W=[-99.5, -99.5], S=[None, None], parent_N=200))

    # -- ties ----------------------------------------------------------------
    A(Case("exact_tie_two_identical_children",
           moves=[10, 11], priors=[0.5, 0.5],
           N=[7, 7], W=[1.25, 1.25], S=[None, None], parent_N=14))
    A(Case("exact_tie_all_identical_five",
           moves=[10, 11, 12, 13, 14], priors=[0.2] * 5,
           N=[3] * 5, W=[-0.5] * 5, S=[None] * 5, parent_N=15))
    # The tie is on the SCORE, not on the inputs: different W and N that
    # divide to the same Q, and priors that multiply to the same U.
    A(Case("exact_tie_via_different_inputs",
           moves=[10, 11], priors=[0.5, 0.25],
           N=[1, 3], W=[0.5, 1.5], S=[None, None], parent_N=4))
    # One ULP apart, in both directions, so a comparison that collapses them
    # fails one of the two.
    _p = 0.5
    A(Case("one_ulp_higher_second",
           moves=[10, 11], priors=[_p, math.nextafter(_p, 1.0)],
           N=[0, 0], W=[0.0, 0.0], S=[None, None], parent_N=100))
    A(Case("one_ulp_higher_first",
           moves=[10, 11], priors=[math.nextafter(_p, 1.0), _p],
           N=[0, 0], W=[0.0, 0.0], S=[None, None], parent_N=100))

    # -- ordering ------------------------------------------------------------
    # A descending move order with a tie: first-in-order and lowest-move give
    # DIFFERENT answers, which is the only way this case can fail informatively.
    A(Case("tie_with_descending_move_order",
           moves=[76, 40, 4], priors=[0.3, 0.3, 0.3],
           N=[5, 5, 5], W=[1.0, 1.0, 1.0], S=[None] * 3, parent_N=15))
    # The same array with solve OFF, so the win branch cannot mask an ordering
    # bug in the PUCT branch.
    A(Case("tie_with_descending_move_order_solve_off",
           moves=[76, 40, 4], priors=[0.3, 0.3, 0.3],
           N=[5, 5, 5], W=[1.0, 1.0, 1.0], S=[None] * 3, parent_N=15,
           solve=False))
    # solve=False with solved values PRESENT. They must be ignored entirely --
    # this is the guard against a native side that reads S unconditionally.
    A(Case("solve_off_ignores_solved_column",
           moves=[10, 11], priors=[0.01, 0.99],
           N=[0, 0], W=[0.0, 0.0], S=[-1, None], parent_N=10, solve=False))

    # -- degenerate ----------------------------------------------------------
    A(Case("single_child", moves=[40], priors=[1.0], N=[0], W=[0.0],
           S=[None], parent_N=0))
    A(Case("single_child_refuted", moves=[40], priors=[1.0], N=[3],
           W=[3.0], S=[1], parent_N=3))
    A(Case("full_board_81_children",
           moves=list(range(81)), priors=[1.0 / 81] * 81,
           N=[0] * 81, W=[0.0] * 81, S=[None] * 81, parent_N=0))
    return out


# ----------------------------------------------------------------------
# Mode 2: exhaustive sweep over small child counts.
# ----------------------------------------------------------------------

def sweep_cases(max_kids=3):
    """Every combination of the branch-relevant values, for 2 and 3 children.

    The value grids are small ON PURPOSE. N in {0, 1, 7} covers the Q() guard,
    the 1+N denominator at its smallest, and a generic count; W in {-1, 0, 1}
    covers both signs and the zero that collides with the guard; priors in
    {0, 0.5} collide the U term. Ties are therefore everywhere, which is what
    an exhaustive sweep is for.
    """
    Ns = (0, 1, 7)
    Ws = (-1.0, 0.0, 1.0)
    Ps = (0.0, 0.5)
    from itertools import product
    idx = 0
    for k in range(2, max_kids + 1):
        # Descending moves so first-in-order and lowest-move never coincide.
        moves = [80 - 7 * j for j in range(k)]
        for solve in (False, True):
            for parent_N in (0, 1, 25):
                for combo in product(product(Ns, Ws, Ps, SOLVED_CODES),
                                     repeat=k):
                    idx += 1
                    yield Case(f"sweep{idx}", moves,
                               [c[2] for c in combo],
                               [c[0] for c in combo],
                               [c[1] for c in combo],
                               [c[3] for c in combo],
                               parent_N, solve=solve)


# ----------------------------------------------------------------------
# Mode 3: tie-heavy fuzz.
# ----------------------------------------------------------------------

def fuzz_cases(n, seed, tie_heavy=True):
    rng = random.Random(seed)
    for i in range(n):
        k = rng.randint(1, 12) if i % 7 else rng.randint(1, 81)
        moves = rng.sample(range(81), k)
        if tie_heavy and i % 3:
            # Small grids -> exact collisions are common.
            priors = [rng.choice((0.0, 0.125, 0.25, 0.5)) for _ in range(k)]
            Ns = [rng.choice((0, 1, 2, 5)) for _ in range(k)]
            Ws = [rng.choice((-1.0, -0.5, 0.0, 0.5, 1.0)) for _ in range(k)]
        else:
            priors = [rng.random() for _ in range(k)]
            s = sum(priors) or 1.0
            priors = [p / s for p in priors]
            Ns = [rng.choice((0, rng.randint(1, 50), rng.randint(1, 100000)))
                  for _ in range(k)]
            Ws = [rng.uniform(-1, 1) * max(1, Ns[j]) for j in range(k)]
        # One-ULP neighbours, so the near-tie path is exercised too.
        if k > 1 and i % 5 == 0:
            j = rng.randrange(k - 1)
            priors[j + 1] = math.nextafter(priors[j],
                                           1.0 if rng.random() < 0.5 else -1.0)
            Ns[j + 1] = Ns[j]
            Ws[j + 1] = Ws[j]
        Ss = [rng.choice(SOLVED_CODES) if rng.random() < 0.35 else None
              for _ in range(k)]
        parent_N = rng.choice((0, 1, sum(Ns), sum(Ns) + rng.randint(0, 999)))
        yield Case(f"fuzz{i}", moves, priors, Ns, Ws, Ss, parent_N,
                   c_puct=rng.choice((1.0, 1.5, 2.5)),
                   solve=rng.random() < 0.8)


# ----------------------------------------------------------------------
# Whole-tree verification.
# ----------------------------------------------------------------------

def node_faults(node):
    """Every way `node`'s mirror can disagree with `node`'s children.

    Bit-exact on W and prior. `!=` between two doubles is the correct
    comparison here for the same reason it is usually the wrong one: a value
    that has drifted by one ULP through a virtual-loss apply and undo will
    still pick the same child almost every time, so a tolerance would hide
    exactly the defect this is looking for until the day it changes an index.
    """
    faults = []
    sel = node.sel
    if sel is None:
        return faults
    kids = node.kids
    dict_order = list(node.children.values())
    if len(kids) != len(dict_order):
        return [f"len(kids)={len(kids)} vs len(children)={len(dict_order)}"]
    for j, (a, b) in enumerate(zip(kids, dict_order)):
        if a is not b:
            faults.append(f"[{j}] kids/children ORDER differs: "
                          f"move {a.move} vs {b.move}")
    sN, sW, sS, mv, pr = node.selN, node.selW, node.selS, sel.move, sel.prior
    for j, c in enumerate(kids):
        if c.cidx != j:
            faults.append(f"[{j}] cidx={c.cidx}")
        if int(sN[j]) != c.N:
            faults.append(f"[{j}] N mirror={int(sN[j])} node={c.N}")
        if float(sW[j]) != c.W:
            faults.append(f"[{j}] W mirror={float(sW[j])!r} node={c.W!r}")
        if int(sS[j]) != encode_solved(c.solved):
            faults.append(f"[{j}] S mirror={int(sS[j])} node={c.solved}")
        if int(mv[j]) != c.move:
            faults.append(f"[{j}] move mirror={int(mv[j])} node={c.move}")
        if float(pr[j]) != c.prior:
            faults.append(f"[{j}] prior mirror={float(pr[j])!r} "
                          f"node={c.prior!r}")
    return faults


def verify_subtree(root, limit=40):
    """(nodes_checked, faults) over every mirrored node under `root`.

    Once per search this is cheap and it covers the whole tree, where the
    per-call deep check only ever sees nodes that were selected FROM. A node
    whose mirror drifted and which then stopped being visited is invisible to
    sampling and visible here.
    """
    faults = []
    n = 0
    stack = [root]
    while stack:
        node = stack.pop()
        if node.sel is not None:
            n += 1
            f = node_faults(node)
            if f and len(faults) < limit:
                faults.append({"move": node.move, "N": node.N,
                               "faults": f[:12]})
        stack.extend(node.children.values())
    return n, faults


# ----------------------------------------------------------------------
# Mode 4: shadow -- both selectors, on real searches, over the real mirror.
# ----------------------------------------------------------------------

class Shadow:
    """Patches MCTS._best_child to run both and compare.

    Restores by DELETING the instance shadow rather than writing the bound
    method back: an instance attribute holding a bound method is a reference
    cycle, and this engine runs with the cyclic collector off.
    """

    def __init__(self, mcts, deep=False, deep_every=1):
        self.mcts = mcts
        self.deep = deep
        self.deep_every = max(1, int(deep_every))
        self.calls = 0
        self.deep_calls = 0
        self.mismatches = []
        self.column_faults = []
        self.no_mirror = 0
        self.trees = 0
        self.tree_nodes = 0
        self.tree_faults = []
        self._orig = type(mcts)._best_child
        self._orig_search = type(mcts).search
        self._had = "_best_child" in mcts.__dict__

    def __enter__(self):
        orig = self._orig
        self.mcts._best_child = lambda node: self._call(orig, node)
        # Also verify the WHOLE tree once per search. The per-call deep check
        # only ever sees nodes that were selected FROM, so a node whose mirror
        # drifted and which then stopped being visited is invisible to it --
        # and "stopped being visited" is exactly what a drifted score causes.
        osearch = self._orig_search
        mcts = self.mcts

        def wrapped_search(root_state, root=None):
            pi, r = osearch(mcts, root_state, root)
            self.trees += 1
            n, f = verify_subtree(r)
            self.tree_nodes += n
            if f:
                self.tree_faults.extend(f[:5])
            return pi, r

        mcts.search = wrapped_search
        return self

    def __exit__(self, *a):
        # Delete the instance shadows rather than writing the bound methods
        # back: an instance attribute holding a bound method is a reference
        # cycle and this engine runs with the cyclic collector off.
        if not self._had:
            self.mcts.__dict__.pop("_best_child", None)
        self.mcts.__dict__.pop("search", None)
        return False

    def _call(self, orig, node):
        sel = node.sel
        if sel is None:
            # Not a silent fallback: an engine built with native_select=True
            # should never reach here, and the count is asserted to be zero.
            self.no_mirror += 1
            return orig(self.mcts, node)

        self.calls += 1
        nat_i = sel.best(node.N)
        nat = node.kids[nat_i]

        # Force the reference down the Python path over the SAME node.
        node.sel = None
        try:
            ref = orig(self.mcts, node)
        finally:
            node.sel = sel

        if ref is not nat:
            if len(self.mismatches) < 40:
                self.mismatches.append(self._snapshot(node, nat_i, ref, nat))

        if self.deep and self.calls % self.deep_every == 0:
            self.deep_calls += 1
            self._check_columns(node)

        # The PYTHON answer, so the trajectory stays canonical -- see the
        # module docstring.
        return ref

    def _check_columns(self, node):
        """The mirror against the nodes, cell by cell, bit for bit.

        This is the check a replay harness structurally cannot do: a replay
        rebuilds the native input from the truth, so it can only ever validate
        the arithmetic. Drift is the risk here, and drift is only visible
        against the mirror the engine itself maintained.
        """
        faults = node_faults(node)
        if faults and len(self.column_faults) < 40:
            self.column_faults.append({"call": self.calls,
                                       "parent_N": node.N,
                                       "faults": faults[:12]})

    def _snapshot(self, node, nat_i, ref, nat):
        kids = node.kids
        sel = node.sel
        sc = sel.scores(node.N)
        return {
            "call": self.calls,
            "parent_N": node.N,
            "python_move": ref.move, "native_move": nat.move,
            "python_index": kids.index(ref), "native_index": nat_i,
            "children": [
                {"i": j, "move": c.move, "N": c.N, "W": repr(c.W),
                 "prior": repr(c.prior), "solved": c.solved,
                 "score": repr(float(sc[j]))}
                for j, c in enumerate(kids)],
        }


def shadow_positions(engine, positions, device, deep, deep_every):
    from tools.arena_1s import TimedPlayer
    p = TimedPlayer("engine:%s" % engine, device)
    if not p.mcts.native_select:
        raise SystemExit(f"[X] engine '{engine}' does not enable native "
                         f"selection; shadow mode would compare Python with "
                         f"Python and pass for the wrong reason.")
    sh = Shadow(p.mcts, deep=deep, deep_every=deep_every)
    with sh:
        for state, _phase in positions:
            p.mcts.reset_stats()
            import torch
            with torch.no_grad():
                _pi, root = p.mcts.search(state.clone())
            TreeReuseSearcher.release(root)
            gc.collect()
    return sh


def shadow_games(engine, opponent, games, device, seed, deep, deep_every):
    """A real match, so tree REUSE and re-rooting are in the picture.

    The opponent is a different engine on purpose. A mirror match inflates
    inheritance about 1.7x and would exercise adoption under conditions
    deployment never sees (`uttt-mirror-gate-hides-reroot-cost`).
    """
    from tools.arena_1s import TimedPlayer, play_match
    pa = TimedPlayer("engine:%s" % engine, device)
    pb = TimedPlayer("engine:%s" % opponent, device)
    if not pa.mcts.native_select:
        raise SystemExit(f"[X] engine '{engine}' does not enable native "
                         f"selection.")
    sh = Shadow(pa.mcts, deep=deep, deep_every=deep_every)
    with sh:
        play_match(pa, pb, games, seed, warmup=1, gc_mode="deferred")
    return sh


# ----------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------

def run_cases(cases, label, verbose=False, limit_report=8):
    n = 0
    bad = []
    t0 = time.perf_counter()
    for c in cases:
        n += 1
        ok, pi, ni = check(c)
        if not ok:
            if len(bad) < limit_report:
                bad.append((c, pi, ni))
    dt = time.perf_counter() - t0
    print(f"  {label:24s} {n:>9,} cases  {len(bad):>4} mismatches  "
          f"{dt:6.1f}s")
    for c, pi, ni in bad:
        print(f"    [X] {c.name}: python -> index {pi}, native -> index {ni}")
        print(c.describe())
    return n, len(bad)


def report_shadow(sh, label):
    print(f"  {label}")
    print(f"    selections compared     {sh.calls:>12,}")
    print(f"    index mismatches        {len(sh.mismatches):>12,}")
    print(f"    nodes without a mirror  {sh.no_mirror:>12,}")
    if sh.deep:
        print(f"    deep column checks      {sh.deep_calls:>12,}")
        print(f"    column faults           "
              f"{len(sh.column_faults):>12,}")
    print(f"    whole trees verified    {sh.trees:>12,}")
    print(f"    mirrored nodes checked  {sh.tree_nodes:>12,}")
    print(f"    tree faults             {len(sh.tree_faults):>12,}")
    for f in sh.tree_faults[:5]:
        print(f"    [X] node move={f['move']} N={f['N']}: "
              f"{'; '.join(f['faults'])}")
    for m in sh.mismatches[:5]:
        print(f"    [X] call {m['call']}: python move {m['python_move']} "
              f"(index {m['python_index']}) vs native move "
              f"{m['native_move']} (index {m['native_index']}), "
              f"parent_N={m['parent_N']}")
        for c in m["children"]:
            print(f"        [{c['i']}] move={c['move']:2d} N={c['N']:6d} "
                  f"W={c['W']} prior={c['prior']} solved={c['solved']} "
                  f"score={c['score']}")
    for f in sh.column_faults[:5]:
        print(f"    [X] call {f['call']} parent_N={f['parent_N']}: "
              f"{'; '.join(f['faults'])}")
    return (len(sh.mismatches) + len(sh.column_faults) + sh.no_mirror
            + len(sh.tree_faults))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="offline",
                    choices=["offline", "fixtures", "sweep", "fuzz", "shadow",
                             "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--engine", default="pocket_sel")
    ap.add_argument("--opponent", default="final",
                    help="cross-network, so reuse is deployment-like")
    ap.add_argument("--fuzz", type=int, default=300000)
    ap.add_argument("--sweep-kids", type=int, default=3)
    ap.add_argument("--positions", type=int, default=8)
    ap.add_argument("--position-games", type=int, default=2)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--deep", action="store_true", default=True)
    ap.add_argument("--no-deep", dest="deep", action="store_false")
    ap.add_argument("--deep-every", type=int, default=97,
                    help="deep-check every Nth selection; a prime, so it does "
                         "not phase-lock with the wave size")
    ap.add_argument("--seed", type=int,
                    default=engine_registry.SEEDS["select"])
    ap.add_argument("--out", default="results/select_parity")
    args = ap.parse_args()

    ns.require("tools.select_parity")

    print("=" * 72)
    print("PARITY: native ChildArray.best vs Python MCTS._best_child")
    print("=" * 72)
    total = failed = 0

    do = args.mode
    if do in ("offline", "fixtures", "all"):
        n, b = run_cases(fixtures(), "fixtures (named)")
        total += n
        failed += b
    if do in ("offline", "sweep", "all"):
        n, b = run_cases(sweep_cases(args.sweep_kids), "sweep (exhaustive)")
        total += n
        failed += b
    if do in ("offline", "fuzz", "all"):
        n, b = run_cases(fuzz_cases(args.fuzz, args.seed), "fuzz (tie-heavy)")
        total += n
        failed += b

    shadow_bad = 0
    if do in ("shadow", "all"):
        from tools.profile_selection import positions_for
        print()
        print("  shadow mode: real searches, real mirror")
        pos = positions_for(args.opponent, args.position_games, args.device,
                            args.seed, args.positions)
        sh1 = shadow_positions(args.engine, pos, args.device, args.deep,
                               args.deep_every)
        shadow_bad += report_shadow(sh1, f"fixed positions ({len(pos)})")
        sh2 = shadow_games(args.engine, args.opponent, args.games,
                           args.device, args.seed + 1, args.deep,
                           args.deep_every)
        shadow_bad += report_shadow(sh2, f"games vs {args.opponent} "
                                         f"({args.games})")
        total += sh1.calls + sh2.calls
        failed += shadow_bad

    print()
    print("=" * 72)
    if failed:
        print(f"[X] PARITY FAILS: {failed} disagreement(s) over {total:,} "
              f"selections")
    else:
        print(f"[OK] PARITY HOLDS over {total:,} selections")
    print("=" * 72)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "parity.json")
    with open(path, "w") as fh:
        json.dump({"mode": do, "total": total, "failed": failed,
                   "engine": args.engine, "seed": args.seed}, fh, indent=2)
    print(f"wrote {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
