from agents.neural_net_agent import NeuralNetAgent
from engine.game import GameState
from engine.constants import X, O, DRAW, EMPTY
from collections import Counter
import argparse, glob, re, os
import time
import math
import heapq

DEFAULT_CHECKPOINT_PREFIX = "models/neural_net/self_play_trained_"


def parse_games(s: str) -> int:
    # remove any underscores then cast to int
    return int(s.replace("_", ""))


def find_latest_checkpoint(prefix=DEFAULT_CHECKPOINT_PREFIX):
    best = None
    pattern = prefix+"*.pt"
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        m = re.match(r"self_play_trained_(\d+)\.pt$", name)
        if not m:
            continue
        ver = int(m.group(1))
        if best is None or ver > best:
            best = ver
    return best


def next_version(prefix=DEFAULT_CHECKPOINT_PREFIX):
    latest = find_latest_checkpoint()
    pattern=prefix+"{:02d}.pt"
    new = (latest + 1) if (latest is not None) else 0
    return pattern.format(new)


def run_self_play(agent, verbose=False):
    game = GameState()
    agents = {X: agent, O: agent}

    while not game.is_over():
        current_agent = agents[game.player] # BOTH THE SAME, but placed here just to remember in future how to do it
        move = current_agent.select_move(game)
        valid, _ = game.make_move(move) # Winner is unused here
        if not valid:
            raise ValueError(f"Invalid move: {move}")
        if verbose:
            print(f"{'X' if game.player == O else 'O'} played {move}")
            game.print_board()

    if verbose:
        print(f"🏁 Game Over! Winner: {['None', 'X', 'O', 'Draw'][game.winner]}")

    # DRAW_SCALE = 0.4
    max_moves    = 81
    moves_played = len(agent.last_players)
    decay_rate   = math.log(100) / (max_moves - 1)

    # if draw, pretend it took the full 81 moves
    adjusted_moves = moves_played if game.winner != DRAW else max_moves
    time_factor    = math.exp(-decay_rate * (adjusted_moves - 18))

    rewards = []
    for p in agent.last_players:
        if p == game.winner:
            rewards.append(time_factor)
        elif game.winner == DRAW:
            rewards.append( -0.02 ) # decided that everyone gets dinged for draw, at least with a win/loss someone gets rewarded #Nicole says 0 is "the most neutral" setting, was #time_factor * DRAW_SCALE)
        else:
            rewards.append(-time_factor)

    result = (
        agent.last_game_states[:],
        agent.last_moves[:],
        agent.last_players[:],
        rewards,
        game.winner
    )

    agent.clear_history()

    return result


def format_elapsed(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = int(seconds // 3600)
        minutes = (seconds % 3600) / 60
        return f"{hours}h {minutes:.2f}m"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume",    action="store_true",
                   help="load the latest checkpoint instead of starting fresh")
    p.add_argument("--games",     type=parse_games, default=500,
                   help="how many self-play games to run (underscores OK, e.g. 500_000)")
    args = p.parse_args()
    '''
        500:  12.96s     (38.6 games/sec)
      2,500:  66.27s     (37.7 games/sec)
      5,000: 2.26 mins   (36.9 games/sec)
     10,000: 257.96s     (38.8 games/sec)
    400,000: 2h 57.02m   (37.7 games/sec)
    TODO print start time, maybe make progress bar?
        (20, [27, 20, 78, 54, 9, 28, 21, 74, 60, 2, 24, 64, 30, 11, 42, 38, 51, 29, 15, 47])
        (24, [19, 75, 55, 22, 67, 48, 74, 60, 1, 21, 73, 59, 7, 23, 79, 57, 10, 32, 17, 34, 70, 40, 52, 66])
        (25, [37, 31, 23, 70, 50, 60, 2, 24, 63, 27, 11, 51, 65, 34, 3, 18, 64, 30, 20, 78, 36, 46, 59, 6, 38])
        (27, [12, 46, 77, 78, 56, 16, 40, 32, 25, 59, 24, 54, 11, 43, 50, 70, 30, 10, 14, 53, 69, 45, 74, 62, 26, 21, 65])
        (27, [48, 64, 40, 49, 66, 28, 21, 55, 12, 29, 26, 70, 32, 15, 37, 22, 77, 61, 5, 25, 75, 74, 78, 73, 76, 34, 3])  
        (27, [58, 14, 35, 8, 7, 23, 60, 20, 62, 15, 46, 59, 6, 19, 67, 31, 13, 48, 54, 18, 56, 26, 61, 22, 76, 72, 55])   
        (27, [72, 64, 50, 71, 43, 30, 1, 21, 74, 69, 28, 13, 41, 35, 6, 18, 73, 59, 26, 80, 61, 5, 25, 67, 32, 8, 24])
        (27, [60, 0, 11, 51, 72, 73, 75, 64, 39, 47, 61, 23, 62, 6, 2, 25, 57, 19, 58, 13, 41, 33, 20, 28, 14, 43, 40])
        (27, [52, 66, 45, 54, 19, 67, 31, 3, 10, 41, 42, 28, 23, 79, 77, 69, 46, 68, 51, 65, 33, 9, 47, 70, 49, 16, 40])
        (27, [10, 48, 63, 28, 13, 41, 53, 69, 46, 57, 1, 14, 34, 5, 25, 59, 24, 64, 50, 60, 19, 58, 23, 62, 26, 61, 3])
        (28, [32, 25, 75, 56, 16, 41, 34, 14, 35, 8, 15, 38, 51, 55, 5, 7, 13, 30, 2, 6, 18, 54, 10, 50, 62, 77, 60, 40])      
        (28, [39, 46, 66, 28, 21, 74, 60, 11, 51, 54, 10, 50, 70, 48, 65, 53, 69, 37, 41, 35, 25, 57, 19, 59, 16, 49, 68, 44])
        (28, [37, 49, 67, 30, 19, 66, 46, 57, 10, 40, 31, 5, 17, 51, 64, 50, 61, 13, 27, 2, 15, 45, 55, 21, 54, 1, 43, 75])
        (28, [37, 49, 67, 30, 19, 66, 46, 57, 10, 40, 31, 5, 17, 51, 64, 50, 61, 13, 27, 2, 15, 45, 55, 21, 54, 1, 43, 75])
        (28, [62, 25, 57, 18, 54, 2, 26, 70, 40, 50, 69, 29, 6, 0, 10, 49, 77, 79, 67, 48, 74, 61, 4, 21, 56, 17, 33, 9])
        (28, [37, 49, 67, 30, 19, 66, 46, 57, 10, 40, 31, 5, 17, 51, 64, 50, 61, 13, 27, 2, 15, 45, 55, 21, 54, 1, 43, 75])
        (28, [77, 80, 70, 39, 29, 17, 44, 42, 38, 33, 20, 62, 8, 6, 2, 16, 41, 51, 73, 58, 21, 55, 12, 37, 50, 71, 56, 26])
        (28, [66, 47, 62, 15, 29, 17, 34, 13, 40, 50, 70, 48, 55, 5, 6, 2, 24, 73, 76, 57, 0, 20, 60, 11, 35, 16, 30, 21])
        (28, [65, 51, 56, 24, 64, 31, 13, 30, 9, 45, 55, 23, 71, 35, 16, 32, 17, 43, 63, 47, 61, 5, 25, 76, 77, 79, 66, 46])
    '''

    RUNS = args.games

    if args.resume:
        ver = find_latest_checkpoint()
        if ver is None:
            print("⚠️\tno existing checkpoints, starting new at _00")
            model_path = next_version()
        else:
            model_path = f"models/neural_net/self_play_trained_{ver:02d}.pt"
            print(f"🔄\tresuming from {model_path}")
    else:
        model_path = next_version()
        print(f"✨\tstarting new model at {model_path}")

    agent = NeuralNetAgent(model_path=None)    # always start un-loaded
    # only load if --resume AND file exists
    if args.resume and os.path.isfile(model_path):
        agent.load(model_path)

    current_time = time.localtime()
    current_time_str = f"{current_time.tm_mon}/{current_time.tm_mday}/{current_time.tm_year} @ {current_time.tm_hour:02}:{current_time.tm_min:02}:{current_time.tm_sec:02}"
    print(f"🏋️\tTraining {agent.name} via self-play for {RUNS:,} games...\nStarted {current_time_str}")
    start = time.time()

    results = []

    TOP_N = 16
    heap = [] # For shortest games

    for _ in range(RUNS):
        states, moves, players, rewards, winner = run_self_play(agent, verbose=False)

        entry = (-len(moves), moves)
        if len(heap) < TOP_N:
            heapq.heappush(heap, entry)
        else:
            heapq.heappushpop(heap, entry)

        agent.last_game_states.extend(states)
        agent.last_moves.extend(moves)
        agent.last_players.extend(players)

        agent.last_rewards.extend(rewards)

        results.append(winner)

        if len(results) % 10 == 0:
            assert len(agent.last_game_states) == len(agent.last_moves) == len(agent.last_rewards), \
                f"💥 Batch mismatch before learn: {len(agent.last_game_states)} states, {len(agent.last_moves)} moves, {len(agent.last_rewards)} rewards"
            agent.learn()

    shortest_games = sorted([(-l,m) for l, m in heap], key=lambda x: x[0])

    elapsed = time.time() - start
    formatted_time = format_elapsed(elapsed)
    
    c = Counter(results)

    print("\n✅ Training complete!")
    print(f"⏱️ Time elapsed: {formatted_time} ({RUNS/elapsed:.1f} games/sec)")
    print("📊 Results:")
    print(f"  X wins  : {c[X]:,}")
    print(f"  O wins  : {c[O]:,}")
    print(f"  Draws   : {c[DRAW]:,}")
    print(f"  Invalid : {c[0]:,}")  # In case of unexpected zero wins

    for short_game in shortest_games:
        print(f"\t{short_game}")

    agent.save(model_path)
