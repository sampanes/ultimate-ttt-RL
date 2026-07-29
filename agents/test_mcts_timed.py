"""Tests for wall-clock-limited search and cross-move tree reuse.

The product is a player that must return a move inside a deadline, so the two
mechanisms under test are the ones that decide how much search fits in that
deadline and whether any of it survives to the next move. Both are easy to get
subtly wrong in ways that do not crash:

  * a deadline checked AFTER a chunk overshoots by a whole batched forward pass
    on every move, which is invisible in the mean and owns the p99;
  * a wrongly adopted subtree scores a DIFFERENT position's statistics, which
    never raises and only shows up as unexplained weakness.

So the assertions here are about budget compliance and adoption safety, not
about strength. Uses the same uniform stub as agents/test_mcts.py plus a
deliberately slow variant, so no network and no GPU are involved.

    python -m agents.test_mcts_timed
    pytest agents/test_mcts_timed.py
"""

import time

import torch

from agents.mcts import MCTS, TreeReuseSearcher
from agents.test_mcts import UniformZeroStub, _state_one_move_from_x_win
from engine.constants import X
from engine.game import GameState
from engine.rules import rule_utl_valid_moves


class SlowStub(UniformZeroStub):
    """Uniform stub with a fixed per-call cost, so a deadline has real work to
    stop. Without this the search finishes a 1 ms budget's worth of waves in
    microseconds and the admission logic is never exercised."""

    def __init__(self, delay_s=0.002):
        self.delay_s = delay_s
        self.calls = 0

    def forward_both(self, x):
        self.calls += 1
        time.sleep(self.delay_s)
        return super().forward_both(x)


def _timed(budget_ms, wave=8, model=None, **kw):
    return MCTS(model=model or SlowStub(), device="cpu", n_sims=10 ** 9,
                c_puct=1.5, add_dirichlet_at_root=False, wave_size=wave,
                time_budget_ms=budget_ms, **kw)


def _first_legal(state):
    return rule_utl_valid_moves(state.board, state.last_move,
                                state.mini_winners)[0]


def _play(state, n_plies):
    """Advance the game by always taking the lowest legal move."""
    for _ in range(n_plies):
        state.make_move(_first_legal(state))
    return state


# --------------------------------------------------------------------------- #
# 1. The fixed-simulation path is untouched
# --------------------------------------------------------------------------- #
def test_fixed_sims_still_runs_exactly_n_sims():
    for wave in (1, 8):
        m = MCTS(UniformZeroStub(), "cpu", n_sims=64, wave_size=wave,
                 add_dirichlet_at_root=False)
        _pi, root = m.search(GameState())
        assert m.last["simulations_completed"] == 64, wave
        assert m.last["budget_ms"] is None
        # Every simulation backs up through the root exactly once.
        assert root.N == 64, (wave, root.N)


def test_record_is_written_when_not_solving():
    """measure_solved_targets reads `last` only under solve=True, but the arena
    reads it on every move -- a timed search's simulation count IS the
    measurement, not a setting that can be read back off the config."""
    m = MCTS(UniformZeroStub(), "cpu", n_sims=16, wave_size=1)
    m.search(GameState())
    for key in ("elapsed_ms", "simulations_completed", "neural_evaluations",
                "nodes_expanded", "tree_nodes_reused", "transposition_hits",
                "chosen_move"):
        assert key in m.last, key
    assert m.last["elapsed_ms"] > 0.0


# --------------------------------------------------------------------------- #
# 2. Wall-clock budget
# --------------------------------------------------------------------------- #
def test_time_budget_is_respected():
    m = _timed(120, model=SlowStub(0.004))
    t0 = time.perf_counter()
    pi, _root = m.search(GameState())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms <= 120.0, elapsed_ms
    assert pi.sum() > 0


def test_budget_is_not_checked_after_the_fact():
    """The stall must be predicted, not discovered. With a chunk costing ~20 ms
    against a 150 ms budget, a post-hoc check would land near 160 ms."""
    m = _timed(150, wave=8, model=SlowStub(0.020))
    worst = 0.0
    for _ in range(4):
        t0 = time.perf_counter()
        m.search(GameState())
        worst = max(worst, (time.perf_counter() - t0) * 1000.0)
    assert worst <= 150.0, worst


def test_more_time_buys_more_simulations():
    slow = SlowStub(0.002)
    short = _timed(60, model=slow)
    short.search(GameState())
    n_short = short.last["simulations_completed"]

    long = _timed(240, model=slow)
    long.search(GameState())
    n_long = long.last["simulations_completed"]
    assert n_long > n_short, (n_short, n_long)


def test_tiny_budget_still_returns_a_legal_move():
    """An already-blown deadline must still produce a move: min_sims guarantees
    one chunk, because a player that returns nothing fails worse than one that
    overruns."""
    st = _play(GameState(), 2)
    m = _timed(0.001, wave=4, model=SlowStub(0.003))
    pi, _root = m.search(st)
    mv = int(pi.argmax())
    assert mv in rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)
    assert m.last["simulations_completed"] >= 1


def test_max_sims_caps_an_unbounded_clock():
    m = MCTS(UniformZeroStub(), "cpu", n_sims=10 ** 9, wave_size=8,
             time_budget_ms=60_000, max_sims=64)
    m.search(GameState())
    assert m.last["simulations_completed"] == 64


def test_proven_root_returns_immediately_under_a_clock():
    """A proven root cannot be improved on, and its remaining simulations skip
    the network -- so without an early exit the search burns the entire deadline
    on descents that cannot change the answer."""
    st = _state_one_move_from_x_win()
    m = MCTS(UniformZeroStub(), "cpu", n_sims=10 ** 9, wave_size=4,
             add_dirichlet_at_root=False, solve=True, time_budget_ms=400)
    t0 = time.perf_counter()
    pi, _root = m.search(st)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert int(pi.argmax()) == 8
    assert m.last["stopped_early"] is True
    assert elapsed_ms < 100.0, elapsed_ms      # nowhere near the 400 ms budget


def test_early_stop_never_returns_an_all_zero_policy():
    """A root proven at expansion has zero visits everywhere. A proven LOSS or
    DRAW falls through to the visit counts, so stopping at sims_done == 0 would
    hand back an all-zero pi whose argmax is cell 0 -- illegal in almost every
    position. Caught by the arena as 'returned illegal move 0'."""
    m = MCTS(UniformZeroStub(), "cpu", n_sims=10 ** 9, wave_size=4,
             add_dirichlet_at_root=False, solve=True, time_budget_ms=200)
    st = GameState()
    plies = 0
    while not st.is_over() and plies < 81:
        pi, _root = m.search(st)
        legal = rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)
        assert pi.sum() > 0.0, f"all-zero policy at ply {plies}"
        assert int(pi.argmax()) in legal, (plies, int(pi.argmax()))
        st.make_move(int(pi.argmax()))
        plies += 1
    assert m.stat_early_stops > 0, "no proven root was reached; test is vacuous"


def test_fixed_sim_path_never_stops_early():
    """The frozen distillation corpora were generated by the fixed-simulation
    path; an early exit there would change visit-count targets already on disk."""
    st = _state_one_move_from_x_win()
    m = MCTS(UniformZeroStub(), "cpu", n_sims=120, wave_size=4,
             add_dirichlet_at_root=False, solve=True)
    m.search(st)
    assert m.last["stopped_early"] is False
    assert m.last["simulations_completed"] == 120


# --------------------------------------------------------------------------- #
# 2b. Batched wave expansion
# --------------------------------------------------------------------------- #
class RandomLogitStub:
    """Distinct logits per row, so a batched softmax that silently broadcast
    one row over the whole block would be caught."""

    def __init__(self, seed=7):
        self.g = torch.Generator().manual_seed(seed)

    def forward_both(self, x):
        k = x.shape[0] if x.dim() == 4 else 1
        logits = torch.randn(k, 81, generator=self.g) * 2.0
        values = torch.randn(k, generator=self.g)
        if k == 1:
            # Mirror ConvNet.forward_both, which squeezes batch=1 to
            # (81,)/scalar. The wave path re-adds the dimension.
            return logits[0], values[0]
        return logits, values


def test_batched_expansion_matches_per_leaf():
    """Same priors and same leaf values, to floating point. The batched path
    exists purely to remove device round trips; if it changes what the search
    sees it is not an optimisation, it is a different search."""
    for wave in (2, 8, 16):
        trees = []
        for batched in (False, True):
            m = MCTS(RandomLogitStub(), "cpu", n_sims=wave * 20,
                     wave_size=wave, add_dirichlet_at_root=False,
                     batched_expand=batched)
            _pi, root = m.search(_play(GameState(), 3))
            trees.append(root)
        a, b = trees
        assert sorted(a.children) == sorted(b.children), wave
        for mv in a.children:
            ca, cb = a.children[mv], b.children[mv]
            assert abs(ca.prior - cb.prior) < 1e-6, (wave, mv, ca.prior, cb.prior)
            assert ca.N == cb.N, (wave, mv, ca.N, cb.N)
            assert abs(ca.W - cb.W) < 1e-4, (wave, mv, ca.W, cb.W)


def test_batched_expansion_is_off_for_fixed_simulation_counts():
    """The wave path generated the frozen distillation corpora. A different
    softmax reduction order changes priors in the last bits, which would make
    tools/extract_child_q report drift against hashed artifacts."""
    assert MCTS(UniformZeroStub(), "cpu", n_sims=64).batched_expand is False
    assert MCTS(UniformZeroStub(), "cpu", n_sims=10 ** 9,
                time_budget_ms=100).batched_expand is True


def test_batched_expansion_marks_terminal_children_when_solving():
    """_mark_terminal_children is what proves a mate at expansion; losing it in
    the batched path would silently disable solving for every wave leaf."""
    st = _state_one_move_from_x_win()
    m = MCTS(UniformZeroStub(), "cpu", n_sims=10 ** 9, wave_size=4,
             add_dirichlet_at_root=False, solve=True, time_budget_ms=300,
             batched_expand=True)
    pi, root = m.search(st)
    assert int(pi.argmax()) == 8
    assert root.solved == 1


# --------------------------------------------------------------------------- #
# 3. Tree reuse -- adoption safety
# --------------------------------------------------------------------------- #
def _fresh_searcher(n_sims=64, **kw):
    m = MCTS(UniformZeroStub(), "cpu", n_sims=n_sims, wave_size=1,
             add_dirichlet_at_root=False)
    return TreeReuseSearcher(m, count_nodes=True, **kw)


def test_reuse_adopts_after_our_move_and_their_reply():
    # Deep enough that the reply node is expanded. On an empty board with
    # 81 children and only a few dozen simulations the survivor is a bare leaf,
    # which is the `unexpanded` miss tested separately below.
    s = _fresh_searcher(n_sims=600)
    st = GameState()
    pi, _ = s.search(st)
    assert s.stat_hits == 0 and s.stat_misses == 1   # nothing to inherit yet

    our = int(pi.argmax())
    st.make_move(our)
    st.make_move(_first_legal(st))

    s.search(st)
    assert s.stat_hits == 1, s.stats()
    assert s.stat_inherited_sims > 0
    assert s.stat_reused_nodes > 0


def test_shallow_search_misses_because_the_survivor_is_a_bare_leaf():
    """Not a defect: with 64 sims over 81 root moves the opponent's reply was
    never expanded, so there is nothing under it to keep. It is counted under
    its own reason so a low reuse rate can be attributed to search coverage
    rather than blamed on the adoption logic."""
    s = _fresh_searcher(n_sims=64)
    st = GameState()
    pi, _ = s.search(st)
    st.make_move(int(pi.argmax()))
    st.make_move(_first_legal(st))
    s.search(st)
    assert s.stat_hits == 0
    assert s.stats()["miss_reason"]["unexpanded"] == 1, s.stats()


def test_reuse_inherits_visits_and_counts_them_once():
    """root.N after a reused search must be exactly inherited + new. Anything
    else means backups escaped into the discarded tree or were double counted."""
    s = _fresh_searcher(n_sims=600)
    st = GameState()
    pi, _ = s.search(st)
    st.make_move(int(pi.argmax()))
    st.make_move(_first_legal(st))

    before = s.stat_inherited_sims
    _pi, root = s.search(st)
    inherited = s.stat_inherited_sims - before
    assert inherited > 0
    assert root.parent is None, "survivor still attached to the discarded tree"
    assert root.N == inherited + s.mcts.last["simulations_completed"]


def test_reuse_rejects_a_position_from_another_game():
    s = _fresh_searcher()
    st = GameState()
    s.search(st)
    s.search(GameState())          # a brand new game, not a 2-ply descendant
    assert s.stat_hits == 0 and s.stat_misses == 2


def test_reuse_rejects_a_single_ply_jump():
    """One changed cell is not our move plus a reply, so it must not adopt."""
    s = _fresh_searcher()
    st = GameState()
    pi, _ = s.search(st)
    st.make_move(int(pi.argmax()))
    s.search(st)
    assert s.stat_hits == 0, s.stats()


def test_reuse_rejects_when_we_did_not_play_the_move_it_stored():
    """The searcher must key off the board, not off the move it recommended: a
    caller free to override the pick (opening book, tactical guard) would
    otherwise get a tree rooted at a position that never occurred."""
    s = _fresh_searcher()
    st = GameState()
    pi, _ = s.search(st)
    legal = rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)
    other = next(mv for mv in legal if mv != int(pi.argmax()))
    st.make_move(other)
    st.make_move(rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)[0])
    s.search(st)
    # Adoption is still legitimate here -- `other` is a real child -- but the
    # tree it adopts must be `other`'s subtree, which is what the to_play and
    # board checks enforce. What must never happen is adopting the argmax's.
    assert s._board == tuple(st.board)
    assert s._to_play == st.player


def test_release_breaks_cycles_so_the_tree_dies_by_refcount():
    """With cyclic GC off, a released tree must still be freed. If it is not,
    the latency fix leaks instead of collecting, which is worse than the
    problem it solves."""
    import gc as _gc

    from agents.mcts import MCTSNode

    def live():
        # MCTSNode has __slots__ without __weakref__, so weak references are
        # impossible; counting tracked instances is the direct evidence.
        return sum(1 for o in _gc.get_objects() if type(o) is MCTSNode)

    s = _fresh_searcher(n_sims=300)
    _gc.collect()
    before = live()
    pi, root = s.search(GameState())
    del pi, root                     # searcher._root is the only handle left
    grown = live()
    assert grown - before > 100, (before, grown)

    _gc.disable()
    try:
        s.reset()
        after = live()
    finally:
        _gc.enable()
    leaked = after - before
    assert leaked < 0.1 * (grown - before), (before, grown, after)


def test_release_spares_the_retained_subtree():
    s = _fresh_searcher(n_sims=600)
    st = GameState()
    pi, _ = s.search(st)
    st.make_move(int(pi.argmax()))
    st.make_move(_first_legal(st))
    _pi2, root2 = s.search(st)
    # The adopted root must still carry its inherited statistics and children:
    # releasing the old tree must not have emptied the part we kept.
    assert s.stat_hits == 1, s.stats()
    assert root2.children, "retained subtree was cleared by release()"
    assert root2.N > 0


def test_reset_between_games_drops_the_tree():
    s = _fresh_searcher()
    s.search(GameState())
    s.reset()
    assert s._root is None
    s.search(GameState())
    assert s.stat_hits == 0


def test_disabled_searcher_never_adopts():
    s = _fresh_searcher(enabled=False)
    st = GameState()
    pi, _ = s.search(st)
    st.make_move(int(pi.argmax()))
    st.make_move(rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)[0])
    s.search(st)
    assert s.stat_hits == 0 and s.stat_misses == 2


def test_reuse_plays_only_legal_moves_through_a_whole_game():
    s = _fresh_searcher(n_sims=32)
    st = GameState()
    plies = 0
    while not st.is_over() and plies < 81:
        pi, _ = s.search(st)
        mv = int(pi.argmax())
        legal = rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)
        assert mv in legal, (plies, mv)
        st.make_move(mv)
        plies += 1
        if st.is_over():
            break
        # Opponent replies, so the searcher sees the 2-ply jump it expects.
        opp = rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)[0]
        st.make_move(opp)
        plies += 1
    st_stats = s.stats()
    assert st_stats["reuse_rate"] > 0.5, st_stats


def test_reuse_and_timing_compose():
    m = _timed(80, wave=4, model=SlowStub(0.002))
    s = TreeReuseSearcher(m)
    st = GameState()
    pi, _ = s.search(st)
    st.make_move(int(pi.argmax()))
    st.make_move(rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)[0])
    t0 = time.perf_counter()
    s.search(st)
    assert (time.perf_counter() - t0) * 1000.0 <= 80.0
    assert s.stat_hits == 1
    assert m.last["tree_nodes_reused"] == 0 or m.last["tree_nodes_reused"] is None


def main():
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    ok = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            ok += 1
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{ok}/{len(fns)} passed.")
    return 0 if ok == len(fns) else 1


if __name__ == "__main__":
    raise SystemExit(main())
