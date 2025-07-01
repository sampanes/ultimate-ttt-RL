from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from engine.constants import PLAYER_MAP, EMPTY, X, O, DRAW

def format_moves(moves):
    """Show all 9x9 slots: number if valid, '--' if already taken."""
    rows = []
    for r in range(9):
        row_entries = []
        for c in range(9):
            idx = r * 9 + c
            if idx in moves:
                row_entries.append(str(idx).rjust(2))
            else:
                row_entries.append("--")
        rows.append(f"Row {r}: " + ", ".join(row_entries))
    return "\n".join(rows)

def main():
    move_history = []
    game = GameState()
    while True:
        game.print_board()

        # Show valid move info
        moves = rule_utl_valid_moves(game.board, game.last_move, game.mini_winners)
        print("Valid move indices:")
        print(format_moves(moves))

        move = input(f"Player {game.player} move ({'X' if game.player == 1 else 'O'}): ")
        if not move.isdigit():
            print("Invalid input.")
            continue
        idx = int(move)
        success, winner = game.make_move(idx)
        if not success:
            print("Invalid move.")
            continue
        move_history.append(idx)
        if winner is not None:
            game.print_board()
            print(f"Game Over! {"This was a Draw!" if winner == DRAW else f"Winner was {PLAYER_MAP[winner]}!" }")
            print(f"        (\n            {move_history},\n            ['''idk'''],\n            {PLAYER_MAP[winner].capitalize()}\n        )")
            break

if __name__ == "__main__":
    main()
