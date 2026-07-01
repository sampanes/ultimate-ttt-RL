"""Gregory: depth-limited alpha-beta agent for Ultimate Tic-Tac-Toe.

Gregory is both a curriculum foil and the grading oracle (GRADING_AND_ORACLE.md
Parts 6/9): the same instrument -- dial the depth. At depth 1-2 it is a fast,
gene-pool-INDEPENDENT opponent that kills closed-loop ELO inflation (its stage-6
pool has no clones of itself). At depth 4-6 it approaches a reliable partial
oracle for near-full boards.

Depth guide (rough ms/move on the Python engine, constrained branching factor ~9):
  depth 1 : ~0.1 ms  -- useful as a warm-up foil, marginally above deterministics
  depth 2 : ~1 ms    -- solid foil, beats random/first reliably
  depth 3 : ~10 ms   -- good curriculum anchor (registered as 'gregory')
  depth 5 : ~1 s     -- partial oracle; registered as 'gregory_deep'
  depth 7+: minutes  -- near-optimal on constrained boards

The heuristic (depth 0):
  * Mini-board wins: each won mini scored by macro-position weight (center > edge > corner).
  * Macro two-in-a-row threats: +2 ours (unblocked), -2 opponent's.
  Small values so terminal scores (+/-200) dominate decisively.

Value sign: always from the perspective of the player-to-move at the current node
(negamax convention). Winning sooner preferred over winning later (scores vary by
remaining depth to break ties).
"""

from engine.rules import rule_utl_valid_moves, _WIN_TRIPLES, _MINI_INDICES
from engine.constants import X, O, DRAW, EMPTY
from .agent_base import Agent


# Positional value of each macro-board cell (center > edge > corner).
_MACRO_W = (0.8, 1.0, 0.8,
            1.0, 1.5, 1.0,
            0.8, 1.0, 0.8)

# Center cell of each mini-board (global index) -- used for move ordering.
_MINI_CENTERS = tuple(_MINI_INDICES[m][4] for m in range(9))

_WIN_TERMINAL = 200.0   # terminal win/loss base score


def _opp(player):
    return O if player == X else X


def _heuristic(mini_winners, to_play):
    """Fast positional evaluation from to_play's perspective."""
    opp = _opp(to_play)
    score = 0.0

    for i, w in enumerate(mini_winners):
        mw = _MACRO_W[i]
        if w == to_play:
            score += mw * 3.0
        elif w == opp:
            score -= mw * 3.0

    for a, b, c in _WIN_TRIPLES:
        wa, wb, wc = mini_winners[a], mini_winners[b], mini_winners[c]
        line = (wa, wb, wc)
        our = line.count(to_play)
        opp_c = line.count(opp)
        if opp_c == 0 and our == 2:
            score += 2.0
        elif our == 0 and opp_c == 2:
            score -= 2.0

    return score


def _order_moves(state, valid, to_play):
    """Simple move ordering: winning moves first, then center cells."""
    wins_first = []
    centers    = []
    rest       = []
    for mv in valid:
        s = state.clone()
        _, w = s.make_move(mv)
        if w == to_play:
            wins_first.append(mv)
        elif mv in _MINI_CENTERS:
            centers.append(mv)
        else:
            rest.append(mv)
    return wins_first + centers + rest


def _negamax(state, depth, alpha, beta, to_play):
    """Negamax with alpha-beta pruning. Returns score from to_play's perspective."""
    if state.winner is not None:
        if state.winner == DRAW:
            return 0.0
        # Game is already over; whoever won is NOT to_play (we only recurse on live states).
        # Safe to write it as the explicit check -- catches both cases cleanly.
        return _WIN_TERMINAL if state.winner == to_play else -_WIN_TERMINAL

    valid = rule_utl_valid_moves(state.board, state.last_move, state.mini_winners)
    if not valid:
        return 0.0

    if depth == 0:
        return _heuristic(state.mini_winners, to_play)

    opp = _opp(to_play)
    ordered = _order_moves(state, valid, to_play)
    best = -float('inf')

    for mv in ordered:
        s = state.clone()
        _, w = s.make_move(mv)

        if w == to_play:
            val = _WIN_TERMINAL + depth      # prefer winning sooner
        elif w == DRAW:
            val = 0.0
        elif w is not None:
            val = -(_WIN_TERMINAL + depth)   # shouldn't occur (can't gift opponent a win)
        else:
            val = -_negamax(s, depth - 1, -beta, -alpha, opp)

        if val > best:
            best = val
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break

    return best


class GregoryAgent(Agent):
    """Depth-limited alpha-beta agent. depth=3 is the curriculum foil;
    depth=5 approaches a reliable partial oracle for near-full boards."""

    def __init__(self, depth: int = 3, name: str = ""):
        super().__init__(name or f"Gregory(d={depth})")
        self.depth = depth

    def select_move(self, state):
        to_play = state.player
        opp     = _opp(to_play)
        valid   = rule_utl_valid_moves(state.board, state.last_move, state.mini_winners)
        if not valid:
            raise ValueError("GregoryAgent called on a position with no valid moves.")

        best_val = -float('inf')
        best_mv  = valid[0]
        alpha    = -float('inf')
        beta     =  float('inf')

        for mv in _order_moves(state, valid, to_play):
            s = state.clone()
            _, w = s.make_move(mv)

            if w == to_play:
                return mv       # immediate win -- play it now
            elif w == DRAW:
                val = 0.0
            elif w is not None:
                val = -_WIN_TERMINAL
            else:
                val = -_negamax(s, self.depth - 1, -beta, -alpha, opp)

            if val > best_val:
                best_val = val
                best_mv  = mv
            if val > alpha:
                alpha = val

        return best_mv
