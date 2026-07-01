import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))  # .../engine/
_cpp_build = _os.path.join(_here, "cpp", "build", "Release")
_sys.path.insert(0, _cpp_build)

try:
    from uttt_engine import GameState as _CppGameState
    _CPP_ENGINE = True
    # ASCII-only: this prints unconditionally at import time, so emoji here crash
    # with UnicodeEncodeError under a cp1252 stdout (redirected output / cmd.exe).
    print(f"[OK] C++ engine loaded from {_cpp_build}")
except ImportError as e:
    _CPP_ENGINE = False
    _CppGameState = None
    print(f"[!] C++ engine not found, using Python: {e}")

from dataclasses import dataclass, field
from typing import List, Optional
from .constants import EMPTY, X, O, DRAW, WIN_PATTERNS
from .rules import rule_utl_check_mini_win, rule_utl_valid_moves, rule_utl_get_mini_index

NOT_NEW = False
IS_NEW = True


@dataclass
class _PyGameState:
    board: List[int] = field(default_factory=lambda: [EMPTY]*81)
    player: int = X
    last_move: Optional[int] = None
    mini_winners: List[int] = field(default_factory=lambda: [EMPTY]*9)
    winner: Optional[int] = None  # X, O, DRAW, or None

    def is_over(self) -> bool:
        return self.winner is not None

    def clone(self) -> "_PyGameState":
        # Manual field copy -- ~20-50x faster than copy.deepcopy.
        return _PyGameState(
            board=self.board[:],
            player=self.player,
            last_move=self.last_move,
            mini_winners=self.mini_winners[:],
            winner=self.winner,
        )

    def make_move(self, idx: int) -> tuple:
        if not self.is_valid_move(idx):
            return False, None

        self.board[idx] = self.player
        self.last_move = idx
        mini_idx = rule_utl_get_mini_index(idx)

        _, is_new = self.mini_winner(mini_idx)
        if is_new:
            ultimate_winner = self.check_ultimate_win()
            if ultimate_winner:
                self.winner = ultimate_winner
                return True, ultimate_winner

        self.player = O if self.player == X else X
        return True, None

    def is_valid_move(self, idx: int) -> bool:
        if not (0 <= idx < 81):
            return False
        return idx in rule_utl_valid_moves(self.board, self.last_move, self.mini_winners)

    def get_mini_board(self, mini_idx: int) -> List[int]:
        """Returns the 9-square board of the mini-board at index 0-8."""
        start_row = (mini_idx // 3) * 3
        start_col = (mini_idx % 3) * 3
        indices = []
        for dr in range(3):
            for dc in range(3):
                r = start_row + dr
                c = start_col + dc
                indices.append(r * 9 + c)
        return [self.board[i] for i in indices]

    def mini_winner(self, mini_idx: int):
        if self.mini_winners[mini_idx] != EMPTY:
            return True, NOT_NEW
        mini = self.get_mini_board(mini_idx)
        winner = rule_utl_check_mini_win(mini)
        if winner != EMPTY:
            self.mini_winners[mini_idx] = winner
            return True, IS_NEW
        return False, False

    def check_ultimate_win(self) -> Optional[int]:
        """Returns X, O, DRAW, or None (if still in progress)."""
        for a, b_, c in WIN_PATTERNS:
            winner = self.mini_winners[a]
            if winner in (X, O) and winner == self.mini_winners[b_] == self.mini_winners[c]:
                return winner

        if all(m != EMPTY for m in self.mini_winners):
            return DRAW

        return None

    def print_board(self):
        for row in range(9):
            line = ""
            for col in range(9):
                val = self.board[row * 9 + col]
                c = "." if val == EMPTY else ("X" if val == X else "O")
                line += c + " "
            print(line)
        print()


if _CPP_ENGINE:
    class GameState(_CppGameState):
        # Translate C++ sentinel -1 to Python None for winner and last_move.
        # All callers (solver, mcts, train loop) use `is None` / `is not None`
        # checks; the C++ is_over() method reads the raw int directly so it
        # doesn't need overriding.
        @property
        def winner(self):
            w = self._raw_winner()
            return None if w == -1 else w

        @property
        def last_move(self):
            lm = self._raw_last_move()
            return None if lm == -1 else lm

        def clone(self):
            # C++ copy constructor: trivial memcpy of two int arrays + 3 ints.
            # ~20-50x faster than the old _PyGameState field copy for MCTS.
            return GameState(self)

        def make_move(self, idx):
            ok = _CppGameState.make_move(self, idx)
            if not ok:
                return False, None
            return True, self.winner  # property handles None translation

        def print_board(self):
            for row in range(9):
                line = ""
                for col in range(9):
                    val = self.board[row * 9 + col]
                    c = "." if val == EMPTY else ("X" if val == X else "O")
                    line += c + " "
                print(line)
            print()
else:
    GameState = _PyGameState
