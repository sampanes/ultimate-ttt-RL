"""Alpha-beta endgame solver for Ultimate Tic-Tac-Toe.

Implements GRADING_AND_ORACLE.md Part 6 -- the exact endgame oracle.

PURPOSE: offline grading only. Never ships in the bot, never runs during
training. Its job is to certify whether an agent blunders in positions the
solver can prove. UTTT is additive (claimed minis stay claimed), so near-end
positions collapse fast: K<=15 empty cells is usually tractable in <100k nodes
with alpha-beta + move ordering + a transposition table.

INTERFACE
---------
solve(state, budget=100_000) -> SolveResult
    value    : +1 (to-move player wins with best play), 0 (draw), -1 (loses)
    best_move: optimal move cell index (None if state is terminal)
    exact    : True iff provably correct (budget not exceeded)
    nodes    : nodes visited in this call

grade_move(state, move, budget=100_000) -> MoveGrade | None
    Returns None if either solve was inexact (inconclusive).
    is_blunder: True iff the played move had a strictly worse outcome than
                the best available move.

find_blunders(history, budget=50_000) -> list[MoveGrade]
    history: list of (state, move) pairs (one game). Returns only positions
    where a blunder was confirmed (both solves exact). Skips unsolvable ones.

clear_tt()
    Flush the shared transposition table (call between grading sessions if
    memory is a concern; holding it across a full game suite is a large win).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .constants import X, O, DRAW, EMPTY
from .rules import rule_utl_valid_moves
from .tactics import winning_moves, losing_moves


# --------------------------------------------------------------------------- #
# Public result types
# --------------------------------------------------------------------------- #

@dataclass
class SolveResult:
    value: int               # +1 win, 0 draw, -1 loss (to-move perspective)
    best_move: Optional[int] # None iff state is already terminal
    exact: bool              # False = budget exhausted, value is unreliable
    nodes: int               # nodes visited this call


@dataclass
class MoveGrade:
    played_move: int
    played_value: int   # outcome of the played move (to-mover perspective)
    best_value: int     # best achievable outcome from that state
    is_blunder: bool    # played_value < best_value
    exact: bool         # both solves were within budget; always True here


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

class _BudgetExceeded(Exception):
    pass


# TT entry flags (standard alpha-beta / PVS terminology)
_EXACT = 0
_LOWER = 1   # value is a lower bound (came from a beta cutoff)
_UPPER = 2   # value is an upper bound (came from an alpha cutoff)


def _key(state):
    """Canonical position key. Tuple hash -- transpositions only; no D4 for v1."""
    return (tuple(state.board), state.player, state.last_move,
            tuple(state.mini_winners))


class AlphaBetaSolver:
    """Negamax alpha-beta with a persistent transposition table.

    The TT is kept alive across calls -- huge win when grading many positions
    from the same game (they share subtrees). Call clear_tt() between sessions
    if you want a clean slate.
    """

    def __init__(self):
        self._tt = {}  # key -> (value, flag, best_move)

    def clear_tt(self):
        self._tt.clear()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def solve(self, state, budget: int = 100_000) -> SolveResult:
        """Solve `state` from the to-move player's perspective."""
        self._budget = budget
        self._nodes = 0
        try:
            val, mv = self._negamax(state, -2, 2)
            return SolveResult(value=val, best_move=mv, exact=True,
                               nodes=self._nodes)
        except _BudgetExceeded:
            return SolveResult(value=0, best_move=None, exact=False,
                               nodes=self._nodes)

    def grade_move(self, state, move: int,
                   budget: int = 100_000) -> Optional[MoveGrade]:
        """Grade a single played move. Returns None if either solve is inexact."""
        r_state = self.solve(state, budget)
        if not r_state.exact:
            return None

        # Outcome of the played move from the mover's perspective.
        after = state.clone()
        _, w = after.make_move(move)
        if w is not None:
            # Game ended immediately on this move.
            if w == state.player:
                played_val = 1
            elif w == DRAW:
                played_val = 0
            else:
                played_val = -1   # shouldn't happen (opponent can't win on our turn)
        else:
            r_after = self.solve(after, budget)
            if not r_after.exact:
                return None
            # r_after.value is from after.player's (opponent's) perspective; negate.
            played_val = -r_after.value

        return MoveGrade(
            played_move=move,
            played_value=played_val,
            best_value=r_state.value,
            is_blunder=(played_val < r_state.value),
            exact=True,
        )

    def find_blunders(self, history, budget: int = 50_000) -> List[MoveGrade]:
        """Grade every (state, move) in history; return confirmed blunders only."""
        blunders = []
        for state, move in history:
            g = self.grade_move(state, move, budget)
            if g is not None and g.is_blunder:
                blunders.append(g)
        return blunders

    # ------------------------------------------------------------------ #
    # Core negamax
    # ------------------------------------------------------------------ #

    def _negamax(self, state, alpha: int, beta: int) -> Tuple[int, Optional[int]]:
        # Terminal: winner was set by the PREVIOUS move, so the current player lost.
        if state.winner is not None:
            return (0, None) if state.winner == DRAW else (-1, None)

        key = _key(state)
        tt = self._tt.get(key)
        if tt is not None:
            val, flag, tt_move = tt
            if flag == _EXACT:
                return val, tt_move
            elif flag == _LOWER:
                alpha = max(alpha, val)
            elif flag == _UPPER:
                beta = min(beta, val)
            if alpha >= beta:
                return val, tt_move

        self._nodes += 1
        if self._nodes > self._budget:
            raise _BudgetExceeded()

        moves = rule_utl_valid_moves(state.board, state.last_move,
                                     state.mini_winners)
        if not moves:
            # No moves on a non-terminal state should not happen, but be safe.
            return 0, None

        # Move ordering: immediate wins first, non-losing moves next, rest last.
        # This is a SINGLE node-level call to tactics; worth the clone overhead
        # because it dramatically improves cut rates near the endgame.
        wins = winning_moves(state, moves)
        if wins:
            best_move = wins[0]
            self._tt[key] = (1, _EXACT, best_move)
            return 1, best_move

        losing = set(losing_moves(state, moves))
        ordered = ([mv for mv in moves if mv not in losing] +
                   [mv for mv in moves if mv in losing])

        orig_alpha = alpha
        best_val = -2
        best_move = ordered[0]

        for mv in ordered:
            child = state.clone()
            _, w = child.make_move(mv)

            if w is not None:
                # winning moves handled above, so w here can only be DRAW.
                child_val = 0
            else:
                neg_val, _ = self._negamax(child, -beta, -alpha)
                child_val = -neg_val

            if child_val > best_val:
                best_val = child_val
                best_move = mv

            alpha = max(alpha, best_val)
            if alpha >= beta:
                break  # cutoff

        # Store TT entry with flag indicating bound type.
        if best_val <= orig_alpha:
            flag = _UPPER
        elif best_val >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        self._tt[key] = (best_val, flag, best_move)

        return best_val, best_move


# --------------------------------------------------------------------------- #
# Module-level convenience API (shared TT across calls -- reuse for grading)
# --------------------------------------------------------------------------- #

_solver = AlphaBetaSolver()


def solve(state, budget: int = 100_000) -> SolveResult:
    return _solver.solve(state, budget)


def grade_move(state, move: int, budget: int = 100_000) -> Optional[MoveGrade]:
    return _solver.grade_move(state, move, budget)


def find_blunders(history, budget: int = 50_000) -> List[MoveGrade]:
    return _solver.find_blunders(history, budget)


def clear_tt():
    _solver.clear_tt()
