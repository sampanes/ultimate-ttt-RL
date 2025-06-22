from engine.game import GameState

def main():
    game = GameState()
    while True:
        game.print_board()
        move = input(f"Player {game.player} move (0-80): ")
        if not move.isdigit():
            print("Invalid input.")
            continue
        idx = int(move)
        if not game.make_move(idx):
            print("Invalid move.")
            continue

if __name__ == "__main__":
    main()
