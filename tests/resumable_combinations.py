import pickle
import os
import csv
from engine.constants import EMPTY, X, O, DRAW
from engine.rules import rule_utl_valid_moves, rule_utl_check_mini_win, rule_utl_get_mini_board_state_by_idx

def export_to_csv(depth_counts, filename="utt_depth_counts.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Move", "Combinations"])
        for move, count in sorted(depth_counts.items()):
            writer.writerow([move, count])
    print(f"📄 CSV exported to {filename}")


def save_checkpoint(filename, stack, depth_counts, max_depth, processed):
    with open(filename, 'wb') as f:
        pickle.dump({
            "stack": stack,
            "depth_counts": depth_counts,
            "max_depth": max_depth,
            "processed": processed,
        }, f)

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def mbw_from_board(board):
    return [rule_utl_check_mini_win(rule_utl_get_mini_board_state_by_idx(board, i)) for i in range(9)]

def resumeable_count_games_up_to_depth(max_depth=10, checkpoint_file="depth_counts.pkl", save_every=1000000):
    if os.path.exists(checkpoint_file):
        data = load_checkpoint(checkpoint_file)
        stack = data["stack"]
        depth_counts = data["depth_counts"]
        processed = data["processed"]
    else:
        stack = [([EMPTY] * 81, None, X, 0)]
        depth_counts = {i: 0 for i in range(1, max_depth + 1)}
        processed = 0

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

        processed += 1
        if processed % save_every == 0:
            print(f"[{processed:,}] Saving checkpoint...")
            save_checkpoint(checkpoint_file, stack, depth_counts, max_depth, processed)

    print(f"Done. Processed {processed:,} positions.")
    save_checkpoint(checkpoint_file, stack, depth_counts, max_depth, processed)
    return depth_counts

def print_depth_counts(depth_counts):
    print("\nMove Depth Breakdown:")
    for move, count in sorted(depth_counts.items()):
        print(f"Move {move:<2}: {count:,} combinations")

def load_and_inspect(filename):
    if not os.path.exists(filename):
        print("File not found.")
        return

    data = load_checkpoint(filename)
    print(f"\n✅ Loaded checkpoint from {filename}")
    print(f"Processed so far: {data['processed']:,}")
    print(f"Max depth: {data['max_depth']}")
    print(f"Stack size: {len(data['stack']):,}")
    print_depth_counts(data['depth_counts'])


if __name__ == '__main__':
    results = resumeable_count_games_up_to_depth(10)
    for move, count in results.items():
        print(f"move\t{move:<3}has {count:,} combinations")
