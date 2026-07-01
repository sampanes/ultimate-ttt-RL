"""scripts/play_human.py -- play a human vs a NeuralNetAgentPG checkpoint in the terminal."""
import argparse
import sys

from agents.neural_net_agent_pg import NeuralNetAgentPG
from agents.agent_base import ModelConfigCNN
from engine.game import GameState
from engine.rules import rule_utl_valid_moves, rule_utl_get_next_mini
from engine.constants import X, O, DRAW, EMPTY, PLAYER_MAP


_SYM = {EMPTY: ".", X: "X", O: "O", DRAW: "#"}  # # = mini-board won/drawn


def print_uttt_board(game: GameState):
    """Print the 9x9 board with mini-board dividers and macro-board winner overlays."""
    b = game.board
    mw = game.mini_winners

    # Build 9x9 symbol grid, replacing won mini-boards with their winner symbol
    def cell_sym(row, col):
        flat = row * 9 + col
        mini = (row // 3) * 3 + (col // 3)
        if mw[mini] != EMPTY:
            return _SYM[mw[mini]]
        return _SYM[b[flat]]

    print()
    for row in range(9):
        if row in (3, 6):
            #      . . . | . . . | . . .
            print("------+-------+------")
        line = ""
        for col in range(9):
            if col in (3, 6):
                line += "| "
            line += cell_sym(row, col) + " "
        print(line.rstrip())
    print()

    # Macro board summary (which mini-boards are won)
    print("Macro board:")
    for mr in range(3):
        row_str = ""
        for mc in range(3):
            row_str += _SYM[mw[mr * 3 + mc]] + " "
        print(" ", row_str)
    print()


def forced_mini_label(game: GameState):
    """Return a human-readable string describing where play is forced, or 'any'."""
    if game.last_move is None or game.last_move == -1:
        return "any mini-board (first move)"
    forced = rule_utl_get_next_mini(game.last_move)
    if game.mini_winners[forced] != EMPTY:
        return "any open mini-board (forced mini already won)"
    return f"mini-board {forced} (macro row {forced // 3}, col {forced % 3})"


def get_human_move(game: GameState, human_side: int) -> int:
    valid = rule_utl_valid_moves(game.board, game.last_move, game.mini_winners)

    print(f"Forced into: {forced_mini_label(game)}")
    print(f"Valid moves ({len(valid)}):")
    for i, flat in enumerate(valid):
        row, col = divmod(flat, 9)
        mini = (row // 3) * 3 + (col // 3)
        local_row, local_col = row % 3, col % 3
        print(f"  {i+1:>3}. cell {flat:>2}  (board row {row}, col {col})"
              f"  [mini {mini}, local {local_row},{local_col}]")

    while True:
        try:
            raw = input("Your move (enter number): ").strip()
            choice = int(raw) - 1
            if 0 <= choice < len(valid):
                return valid[choice]
            print(f"  Enter a number between 1 and {len(valid)}.")
        except (ValueError, EOFError):
            print("  Invalid input, try again.")


def play(checkpoint: str, human_side: int):
    cfg = ModelConfigCNN(
        conv_channels=[32, 64, 64],
        fc_hidden_sizes=[256, 512, 256],
        learning_rate=1e-4,
        label="league_pg",
        model_dir="models/league_pg",
    )
    agent = NeuralNetAgentPG(cfg=cfg, model_path=checkpoint)
    agent.set_eval(True)

    agent_side = O if human_side == X else X
    print(f"\nYou are {PLAYER_MAP[human_side]}. Agent is {PLAYER_MAP[agent_side]}.")
    print("Mini-board key: . = empty, X/O = played, # = won/drawn")

    game = GameState()

    while not game.is_over():
        print_uttt_board(game)
        print(f"--- {PLAYER_MAP[game.player]}'s turn ---")

        if game.player == human_side:
            print(f"game board: {game.board}")
            move = get_human_move(game, human_side)
        else:
            print("Agent thinking...")
            move = agent.select_move(game)
            row, col = divmod(move, 9)
            print(f"Agent plays cell {move} (board row {row}, col {col})")

        game.make_move(move)

    print_uttt_board(game)
    w = game.winner
    if w == human_side:
        print("You win!")
    elif w == DRAW:
        print("It's a draw.")
    else:
        print("Agent wins.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to a NeuralNetAgentPG .pt checkpoint.")
    ap.add_argument("--side", type=str, default="X", choices=["X", "O"],
                    help="Which side the human plays (default: X).")
    args = ap.parse_args()

    human_side = X if args.side.upper() == "X" else O
    play(args.checkpoint, human_side)


if __name__ == "__main__":
    main()
