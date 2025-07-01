from engine.game import GameState
from engine.constants import PLAYER_MAP, X, O, EMPTY, DRAW
from tests.utils import verify_eq


def test_ultimate_win():
    ret_val = True
    test_cases = [
        ([X, X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY], X),  # Top row win
        ([O, EMPTY, X, O, O, O, EMPTY, EMPTY, EMPTY], O),          # Middle row win
        ([EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, X, X, X], X),  # Bottom row win
        ([X, EMPTY, EMPTY, X, EMPTY, EMPTY, X, EMPTY, EMPTY], X),  # Left column
        ([EMPTY]*9, None),                                         # No win
        ([X, O, EMPTY, O, X, EMPTY, EMPTY, EMPTY, X], X),          # Diagonal win
    ]

    for mini_winners, expected in test_cases:
        game = GameState()
        game.mini_winners = mini_winners
        actual = game.check_ultimate_win()
        visual = "\n" + "\n".join(" ".join(f"{PLAYER_MAP[c]:^{4}}" for c in mini_winners[i:i+3]) for i in range(0, 9, 3))
        ret_val = verify_eq(actual, expected, f"Ultimate winner: {PLAYER_MAP.get(actual, 'None')} in:{visual}") and ret_val

    return ret_val

def test_actually_play_games():
    ret_val = True
    # Define some test games as tuples: (move_sequence, invalid moves, expected_winner)
    games = [
        (
            [40, 30, 1, 3, 2, 6, 0],
            [],
            None # still in progress
        ),
        (
            [40, 30, 1, 3, 0, 10, 32, 6, 2, 16, 48, 74, 60, 47, 69, 27, 78],
            [],
            X
        ),
        (
            [0, 1, 5, 6, 2, 16, 30, 10, 31, 4, 3, 19, 58, 13, 32, 26, 61, 22],
            [],
            O
        ),
        (
            [0, 1, 4, 5, 6, 2, 7, 3, 99, 9, 29, 16, 39, 28, 12, 27, 11, -1, 43, 32, 26, 62, 10, 40, 48, 54, 19, 68, 33, 20, 79, 67, 41, 53, 70, 50, 78, 56, 30, 31, 14, 35, 49, 66, 38, 34, 21, 73, 64, 63, 47, 61, 23, 80, 71, 42, 45, 65, 52, 37, 74, 69, 36, 22, 60, 51, 55, 13, 72, 44],
            [99, -1],
            DRAW
        )
    ]

    for i, (move_sequence, fail_sequence, expected_winner) in enumerate(games):
        game = GameState()
        actual_winner = None

        for move in move_sequence:
            success, winner = game.make_move(move)
            if not success:
                if fail_sequence.pop(0) != move:
                    print(f"Failure, unexpected invalid move {move} in game {i}")
            if winner:
                actual_winner = winner
                break  # optional: break early if someone wins

        ret_val = ret_val and verify_eq(actual_winner, expected_winner, f"Test Game {i}: Final winner should be {expected_winner}: {PLAYER_MAP[expected_winner]}")
    return ret_val

if __name__ == '__main__':
    pass_so_far = True

    # Test: check_ultimate_win
    if pass_so_far:
        pass_so_far = test_ultimate_win()

    if pass_so_far:
        pass_so_far = test_actually_play_games()

    if pass_so_far:
        print("✅ All tests passed.")
    else:
        print("❌ Check failures")