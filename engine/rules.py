from typing import List, Optional
from .constants import *

# Tuple-of-tuples: faster to iterate than a list of lists.
_WIN_TRIPLES = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))

def rule_utl_get_indices_of_mini_once(mini_index: int) -> List[int]:
    """Returns the list of global indices (0-80) that belong to the given mini-board (0-8)."""
    row_offset = (mini_index // 3) * 3
    col_offset = (mini_index % 3) * 3
    indices = []
    for dr in range(3):
        for dc in range(3):
            r = row_offset + dr
            c = col_offset + dc
            indices.append(r * 9 + c)
    return indices

# Tuple of tuples: tuple iteration is faster than list iteration in CPython.
_MINI_INDICES = tuple(tuple(rule_utl_get_indices_of_mini_once(i)) for i in range(9))

# Public alias: cached lookup of a mini-board's 9 global indices (used by tests/agents).
def rule_utl_get_indices_of_mini(mini_index: int) -> List[int]:
    """Return the 9 global indices (0-80) of mini-board `mini_index` (0-8)."""
    return list(_MINI_INDICES[mini_index])

def rule_utl_check_mini_win(cells: List[int]) -> Optional[int]:
    """Check if a 3x3 mini board has a winner.
    Returns X, O, DRAW (full, no line), or EMPTY (still in progress)."""
    for a, b, c in _WIN_TRIPLES:
        v = cells[a]
        if v and v == cells[b] == cells[c]:  # EMPTY=0 is falsy; v!=EMPTY iff v is truthy
            return v
    if EMPTY not in cells:  # faster than all(cell != EMPTY ...)
        return DRAW
    return EMPTY

def rule_utl_get_mini_board_state_by_idx(gameboard, mini_idx):
    start_row = (mini_idx // 3) * 3
    start_col = (mini_idx % 3) * 3
    indices = []
    for dr in range(3):
        for dc in range(3):
            r = start_row + dr
            c = start_col + dc
            indices.append(r * 9 + c)
    return [gameboard[i] for i in indices]

def rule_utl_get_mini_index(idx):
    row, col = divmod(idx, 9)          # correctly get 0-8 row, col
    mini_row = row // 3                # 0,1,2
    mini_col = col // 3                # 0,1,2
    return mini_row*3 + mini_col      # maps into 0-8

def rule_utl_get_next_mini(idx):
    row, col = divmod(idx, 9)
    local_row = row % 3
    local_col = col % 3
    return local_row*3 + local_col

def rule_utl_valid_moves(board: List[int], last_move: Optional[int], mini_winners: List[int]) -> List[int]:
    # Inline macro-board win check (avoids function-call overhead; skips DRAW case).
    for a, b, c in _WIN_TRIPLES:
        v = mini_winners[a]
        if 0 < v < DRAW and v == mini_winners[b] == mini_winners[c]:
            return []

    if last_move is None or last_move == -1:
        return [i for i in range(81) if board[i] == EMPTY]

    forced_mini = rule_utl_get_next_mini(last_move)

    # If that mini is already won or full, allow any open cell in any unclaimed mini-board.
    if mini_winners[forced_mini] != EMPTY or all(
        board[i] != EMPTY for i in _MINI_INDICES[forced_mini]
    ):
        # Single-pass comprehension: avoids building an intermediate allowed_minis list.
        return [
            idx
            for m in range(9) if mini_winners[m] == EMPTY
            for idx in _MINI_INDICES[m] if board[idx] == EMPTY
        ]

    # Locked into the forced mini.
    return [idx for idx in _MINI_INDICES[forced_mini] if board[idx] == EMPTY]
