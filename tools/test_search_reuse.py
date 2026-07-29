"""Verify the symmetry group used by tools/measure_search_reuse.py.

An incorrect symmetry group does not crash. It quietly merges positions that
are not equivalent, which INFLATES the measured reuse ceiling and would argue
for building a cache that cannot exist. So the group law is checked directly:
the eight transforms are distinct, they preserve the mini-board structure, they
really are automorphisms of the legal-move rule, and the canonical form is
invariant under all eight.

    python -m tools.test_search_reuse
"""

import random

import numpy as np

from engine.rules import rule_utl_valid_moves
from tools.measure_search_reuse import (CELL_MINI, MACRO, PERMS, canon,
                                        constraint_of, d4_perms)
from engine.game import GameState


def test_eight_distinct_permutations():
    perms = d4_perms()
    assert len(perms) == 8
    assert len({p.tobytes() for p in perms}) == 8
    for p in perms:
        assert sorted(p.tolist()) == list(range(81))


def test_transforms_preserve_mini_structure():
    """A UTTT symmetry must map each mini board onto a whole mini board -- a
    generic permutation of 81 cells would not, and would make nonsense of both
    the macro board and the send rule."""
    for p, m in zip(PERMS, MACRO):
        for mini in range(9):
            cells = np.flatnonzero(CELL_MINI == mini)
            landed = {int(CELL_MINI[int(np.flatnonzero(p == c)[0])])
                      for c in cells}
            assert len(landed) == 1, (mini, landed)
            assert landed.pop() == int(m[mini])
        assert sorted(m.tolist()) == list(range(9))


def _reachable(rng, max_depth=50):
    st = GameState()
    for _ in range(rng.randint(2, max_depth)):
        valid = rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)
        if not valid or st.is_over():
            break
        st.make_move(rng.choice(valid))
    return st


def test_transforms_are_automorphisms_of_the_legal_move_rule():
    """Transform a real position, ask the ENGINE for its legal moves, and
    require exactly the transformed set. This is what licenses treating two
    symmetric positions as the same state."""
    rng = random.Random(20260728)
    checked = 0
    for _ in range(200):
        st = _reachable(rng)
        if st.is_over():
            continue
        board = np.asarray(st.board, dtype=np.int8)
        winners = np.asarray(st.mini_winners, dtype=np.int8)
        valid = set(rule_utl_valid_moves(st.board, st.last_move,
                                         st.mini_winners))
        if not valid:
            continue
        for p, m in zip(PERMS, MACRO):
            inv = np.empty(81, dtype=np.int64)
            inv[p] = np.arange(81)          # old index -> new index
            board_t = board[p]
            winners_t = np.empty(9, dtype=np.int8)
            winners_t[m] = winners
            last_t = -1 if st.last_move is None or st.last_move < 0 \
                else int(inv[st.last_move])
            got = set(rule_utl_valid_moves(board_t.tolist(), last_t,
                                           winners_t.tolist()))
            want = {int(inv[v]) for v in valid}
            assert got == want, (p[:9], sorted(got)[:5], sorted(want)[:5])
        checked += 1
    assert checked > 50, f"only {checked} usable positions; test is too weak"


def test_canonical_form_is_invariant_under_all_eight():
    rng = random.Random(4242)
    for _ in range(300):
        board = np.array(rng.choices([0, 1, 2], k=81), dtype=np.int8)
        player = rng.choice([1, 2])
        con = rng.choice([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        want = canon(board, player, con)
        for p, m in zip(PERMS, MACRO):
            b2 = board[p]
            c2 = -1 if con < 0 else int(m[con])
            assert canon(b2, player, c2) == want


def test_canonical_form_separates_genuinely_different_positions():
    """Invariance is only half of it: a canonical form that collapsed
    everything would also pass the test above."""
    rng = random.Random(99)
    seen = set()
    for _ in range(400):
        board = np.array(rng.choices([0, 1, 2], k=81), dtype=np.int8)
        seen.add(canon(board, 1, -1))
    assert len(seen) > 300, len(seen)
    # Same board, different mover, must not collapse.
    board = np.array(rng.choices([0, 1, 2], k=81), dtype=np.int8)
    assert canon(board, 1, -1) != canon(board, 2, -1)
    # Same board and mover, different send constraint, must not collapse.
    assert canon(board, 1, 0) != canon(board, 1, -1)


def test_constraint_matches_the_engine_on_real_positions():
    rng = random.Random(7)
    free = forced = 0
    for _ in range(300):
        st = _reachable(rng)
        if st.is_over():
            continue
        valid = rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)
        if not valid:
            continue
        con = constraint_of(st)
        minis = {int(CELL_MINI[v]) for v in valid}
        if con < 0:
            assert len(minis) > 1
            free += 1
        else:
            assert minis == {con}
            forced += 1
    assert free > 0 and forced > 0, (free, forced)


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
