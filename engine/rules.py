from typing import List, Optional
from .constants import *


def rule_utl_check_mini_win(cells: List[int]) -> Optional[int]:
    """Check if a 3x3 mini board has a winner (X or O).
    Returns X, O, or None."""
    for a,b,c in WIN_PATTERNS:
        if cells[a] != EMPTY and cells[a] == cells[b] == cells[c]:
            return cells[a]
    if all(cell != EMPTY for cell in cells):
        return DRAW
    return EMPTY

def rule_utl_get_mini_index(idx):
    row, col = divmod(idx, 9)          # correctly get 0–8 row, col
    mini_row = row // 3                # 0,1,2
    mini_col = col // 3                # 0,1,2
    return mini_row*3 + mini_col      # maps into 0–8

def rule_utl_get_next_mini(idx):
    row, col = divmod(idx, 9)
    local_row = row % 3
    local_col = col % 3
    return local_row*3 + local_col

def rule_utl_get_indices_of_mini(mini_index: int) -> List[int]:
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

def rule_utl_valid_moves(board: List[int], last_move: Optional[int], mini_winners: List[int]) -> List[int]:
    if last_move is None:
        # First move: anywhere
        return [i for i in range(81) if board[i] == EMPTY]

    forced_mini = rule_utl_get_next_mini(last_move)

    # If that mini is already won or full, allow any open cell in any unclaimed mini-board
    if mini_winners[forced_mini] != EMPTY or all(
        board[i] != EMPTY for i in rule_utl_get_indices_of_mini(forced_mini)
    ):
        allowed_minis = [i for i in range(9) if mini_winners[i] == EMPTY]
        allowed_indices = []
        for m in allowed_minis:
            allowed_indices.extend([
                idx for idx in rule_utl_get_indices_of_mini(m) if board[idx] == EMPTY
            ])
        return allowed_indices

    # Otherwise, you're locked into the forced mini
    return [
        idx for idx in rule_utl_get_indices_of_mini(forced_mini)
        if board[idx] == EMPTY
    ]
