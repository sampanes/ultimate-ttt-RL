from engine.constants import EMPTY, X, O, DRAW
from engine.rules import rule_utl_valid_moves, rule_utl_check_mini_win

def get_mini_board(gameboard, mini_idx):
    start_row = (mini_idx // 3) * 3
    start_col = (mini_idx % 3) * 3
    indices = []
    for dr in range(3):
        for dc in range(3):
            r = start_row + dr
            c = start_col + dc
            indices.append(r * 9 + c)
    return [gameboard[i] for i in indices]

def mbw_from_board(board):
    return [rule_utl_check_mini_win(get_mini_board(board, i)) for i in range(9)]

# Efficient iterative DFS using stack
def count_possible_games():
    total_games = 0
    stack = [([EMPTY] * 81, None, X)]  # board, last_move, current_player

    while stack:
        board, last_move, player = stack.pop()
        mini_winners = mbw_from_board(board)
        moves = rule_utl_valid_moves(board, last_move, mini_winners)

        if not moves or all(m != EMPTY for m in mini_winners):
            total_games += 1
            continue

        next_player = O if player == X else X
        for move in moves:
            new_board = board[:]
            new_board[move] = player
            stack.append((new_board, move, next_player))

    return total_games

# Running a limited depth version just to check it works
def count_games_limited_depth(max_depth=3):
    total_games = 0
    stack = [([EMPTY] * 81, None, X, 0)]  # board, last_move, current_player, depth

    while stack:
        board, last_move, player, depth = stack.pop()
        if depth == max_depth:
            total_games += 1
            continue

        mini_winners = mbw_from_board(board)
        moves = rule_utl_valid_moves(board, last_move, mini_winners)

        if not moves or all(m != EMPTY for m in mini_winners):
            total_games += 1
            continue

        next_player = O if player == X else X
        for move in moves:
            new_board = board[:]
            new_board[move] = player
            stack.append((new_board, move, next_player, depth + 1))

    return total_games

def count_games_up_to_depth(max_depth=4):
    depth_counts = {i: 0 for i in range(1, max_depth + 1)}
    stack = [([EMPTY] * 81, None, X, 0)]

    while stack:
        board, last_move, player, depth = stack.pop()

        if depth >= max_depth:
            continue

        mini_winners = mbw_from_board(board)
        moves = rule_utl_valid_moves(board, last_move, mini_winners)

        if not moves or all(m != EMPTY for m in mini_winners):
            depth_counts[depth + 1] += 1
            continue

        next_player = O if player == X else X
        for move in moves:
            new_board = board[:]
            new_board[move] = player
            stack.append((new_board, move, next_player, depth + 1))
            depth_counts[depth + 1] += 1

    return depth_counts

'''
move    1  has 81 combinations
move    2  has 720 combinations
move    3  has 6,336 combinations
move    4  has 55,080 combinations
move    5  has 473,256 combinations
move    6  has 4,020,960 combinations
move    7  has 33,782,544 combinations
move    8  has 281,067,408 combinations
'''

if __name__ == '__main__':
    results = count_games_up_to_depth(8)
    for move, count in results.items():
        print(f"move\t{move:<3}has {count:,} combinations")
