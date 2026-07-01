EMPTY, X, O, DRAW = 0, 1, 2, 3

WIN_PATTERNS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]

PLAYER_MAP = {None: "None",
              EMPTY: "None",
              X: "X",
              O: "O",
              DRAW: "Draw"}