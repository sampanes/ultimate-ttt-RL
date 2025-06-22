from dataclasses import dataclass, field
from typing import List, Optional

EMPTY, X, O = 0, 1, 2

@dataclass
class GameState:
    board: List[int] = field(default_factory=lambda: [EMPTY]*81)
    player: int = X
    last_move: Optional[int] = None

    def make_move(self, idx: int) -> bool:
        if not self.is_valid_move(idx):
            return False
        self.board[idx] = self.player
        self.last_move = idx
        self.player = O if self.player == X else X
        return True

    def is_valid_move(self, idx: int) -> bool:
        return self.board[idx] == EMPTY

    def print_board(self):
        for row in range(9):
            line = ""
            for col in range(9):
                val = self.board[row * 9 + col]
                c = "." if val == EMPTY else ("X" if val == X else "O")
                line += c + " "
            print(line)
        print()