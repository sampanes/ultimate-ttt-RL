from engine.rules import (
    rule_utl_get_indices_of_mini,
    rule_utl_check_mini_win,
    rule_utl_valid_moves,
    rule_utl_get_mini_index
)
from tests.utils import verify_eq
from engine.constants import EMPTY, X, O, PLAYER_MAP


# TESTS
def test_get_mini_index():
    ret_val = True
    input_and_expected = [
        (0,  0),   # row=0,col=0   → mini 0
        (4,  1),   # row=0,col=4   → mini 1
        (8,  2),   # row=0,col=8   → mini 2
        (9,  0),   # row=1,col=0   → mini 0
        (13, 1),   # row=1,col=4   → mini 1
        (40, 4),   # row=4,col=4   → mini 4
        (53, 5),   # row=5,col=8   → mini 5
        (80, 8),   # row=8,col=8   → mini 8
    ]
    for input, expected in input_and_expected:
        actual = rule_utl_get_mini_index(input)
        '''some sort of visual printout? idk'''
        ret_val = ret_val and verify_eq(
            actual, expected,
            f"rule_utl_get_mini_index({input}) returned {actual}, expected {expected}"
        )
    return ret_val

def test_indices_of_mini():
    ret_val = True
    input_and_expected = [
        (4,[30, 31, 32, 39, 40, 41, 48, 49, 50]),
        (0,[0,1,2,9,10,11,18,19,20])
        ]
    for input, expected in input_and_expected:
        actual = rule_utl_get_indices_of_mini(input)
        visual = "\n" + "\n".join(
            " ".join(f"{n:2}" for n in actual[i:i+3]) for i in range(0, 9, 3)
        )
        ret_val = ret_val and verify_eq(actual, expected, f"indices for mini {input} are:{visual}")
    return ret_val


def test_check_mini_win():
    ret_val = True
    input_and_expected = [
        ([X, X, X, EMPTY, O, EMPTY, O, EMPTY, EMPTY], X),
        ([X, O, X, X, O, EMPTY, O, X, EMPTY], None)
        ]
    for board, expected in input_and_expected:
        actual = rule_utl_check_mini_win(board)
        visual = "\n" + "\n".join(" ".join("X" if c == X else "O" if c == O else "." for c in board[i:i+3]) for i in range(0, 9, 3))
        ret_val  == ret_val and verify_eq(actual, expected, f"winner: {PLAYER_MAP[actual]} in:{visual}")
    return ret_val


def test_valid_moves():
    ret_val = True
    # Blank test
    empty_board = [EMPTY] * 81
    last_move = None
    empty_mini_winners = [EMPTY] * 9
    all_squares = list(range(81))

    # one X
    one_x_board = [EMPTY] * 81
    one_x_board[40] = X
    one_x_last_move = 40
    one_x_expected_indices = rule_utl_get_indices_of_mini(4)
    one_x_expected_indices.remove(40)

    # full mini
    full_mini_board = [EMPTY] * 81
    for i in rule_utl_get_indices_of_mini(4):
        full_mini_board[i] = X
    full_mini_last_move = 40
    full_mini_winners = empty_mini_winners
    full_mini_expected = []
    for i in range(9):
        if i != 4:
            full_mini_expected.extend(rule_utl_get_indices_of_mini(i))

    # middle mini won by X but not full
    won_mini_board = [EMPTY] * 81
    center_indices = rule_utl_get_indices_of_mini(4)
    for i in [center_indices[0], center_indices[1], center_indices[2]]:
        won_mini_board[i] = X  # top row win in center mini
    won_mini_last_move = 40
    won_mini_winners = [EMPTY] * 9
    won_mini_winners[4] = X  # mini 4 is considered won by X

    won_mini_expected = []
    for i in range(9):
        if i != 4:
            won_mini_expected.extend(rule_utl_get_indices_of_mini(i))

    input_and_expected = [
        (empty_board, last_move, empty_mini_winners, all_squares,                       "Test: valid_moves with no last move → all 81"),
        (one_x_board, one_x_last_move, empty_mini_winners, one_x_expected_indices,      "Test: valid_moves with forced mini (center) not full or won"),
        (full_mini_board, full_mini_last_move, full_mini_winners, full_mini_expected,   "Test: valid_moves when forced mini is full → should allow anywhere in open minis"),
        (won_mini_board, won_mini_last_move, won_mini_winners, won_mini_expected,       "Test: valid_moves when forced mini is won → allow any cell in open minis")
    ]
    for board, last_move, mini_winers, expected, msg in input_and_expected:
        actual = rule_utl_valid_moves(board, last_move, mini_winers)
        ret_val  == ret_val and verify_eq(actual, expected, msg)
    return ret_val


# MAIN
if __name__ == "__main__":
    pass_so_far = True
    def print_board(board):
        for i in range(0, 81, 9):
            print(" ".join(["." if c == EMPTY else "X" if c == X else "O" for c in board[i:i+9]]))
        print()

    # Test: rule_utl_get_mini_index
    if pass_so_far:
        pass_so_far = test_get_mini_index()

    # Test: rule_utl_get_indices_of_mini
    if pass_so_far:
        pass_so_far = test_indices_of_mini()

    # Test: rule_utl_check_mini_win
    if pass_so_far:
        pass_so_far = test_check_mini_win()

    # Test: rule_utl_valid_moves
    if pass_so_far:
        pass_so_far = test_valid_moves()

    if pass_so_far:
        print("✅ All tests passed.")
    else:
        print("❌ Check failures")
