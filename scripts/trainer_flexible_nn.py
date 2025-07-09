from agents.agent_base import ModelConfig
from agents.neural_net_agent_2 import NeuralNetAgent2
from engine.constants import X, O, DRAW
import argparse, re, os
import time
from typing import Tuple, List, Set, Any, Dict
import heapq
import json
import argparse, os
from scripts.trainer_base import (
    next_version, find_latest_checkpoint, train_against_random, train_against_agent,
    get_current_time_str, display_results
)

DEFAULT_CHECKPOINT_PREFIX = "models/neural_net_2/self_play_trained_"


def write_interesting_games_multiline(
    data: Dict[str, List[List[Any]]],
    file_path: str = 'interesting_games.json'
) -> None:
    with open(file_path, 'w') as f:
        f.write('{\n')
        for _, key in enumerate(('shortest', 'longest')):
            f.write(f'  "{key}": [\n')
            lst = data[key]
            for i, sub in enumerate(lst):
                # serialize with spaces after commas
                line = json.dumps(sub, separators=(',', ' '))
                # only comma-separate if not the last sublist
                comma = ',' if i < len(lst) - 1 else ''
                f.write(f'    {line}{comma}\n')
            # comma-separate the two blocks, but not after 'longest'
            block_comma = ',' if key == 'shortest' else ''
            f.write(f'  ]{block_comma}\n')
        f.write('}\n')


def write_to_json(in_list: List[List[Any]], s_small_l_large: str) -> None:
    file_path = 'interesting_games.json'
    
    # 1) load existing data (or init if file doesn't exist)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"shortest": [], "longest": []}
    
    # 2) pick the right list
    if s_small_l_large == "s":
        key = "shortest"
    elif s_small_l_large == "l":
        key = "longest"
    else:
        raise ValueError("s_small_l_large must be 's' or 'l'")
    
    # 3) build a set of existing sequences as tuples
    existing_tuples = set(tuple(game) for game in data[key])
    
    # 4) only keep truly new ones, convert back to lists for JSON
    to_add = []
    for game in in_list:
        tup = tuple(game)
        if tup not in existing_tuples:
            existing_tuples.add(tup)
            to_add.append(list(game))
    
    # 5) extend and save
    if to_add:
        data[key].extend(to_add)
        write_interesting_games_multiline(data, file_path)


def consider_top_k_shortest(seq: Tuple[int, ...],
                            heap: List[Tuple[int, Tuple[int, ...]]],
                            seen: Set[Tuple[int, ...]],
                            k: int = 5):
    """
    Maintains a max‐heap of size ≤ k containing the k shortest unique seqs seen so far.
    `heap` stores entries as (-len(seq), seq) so the root is the *longest* of the shortest cluster.
    """
    if seq in seen:
        return
    length = len(seq)
    if len(heap) < k:
        heapq.heappush(heap, (-length, seq))
        seen.add(seq)
    else:
        # heap[0] is (-max_len, seq)
        max_len = -heap[0][0]
        if length < max_len:
            _, removed = heapq.heapreplace(heap, (-length, seq))
            seen.remove(removed)
            seen.add(seq)


def consider_top_k_longest(seq: Tuple[int, ...],
                           heap: List[Tuple[int, Tuple[int, ...]]],
                           seen: Set[Tuple[int, ...]],
                           k: int = 5):
    """
    Maintains a min‐heap of size ≤ k containing the k longest unique seqs seen so far.
    `heap` stores entries as (len(seq), seq) so the root is the *shortest* of the longest cluster.
    """
    if seq in seen:
        return
    length = len(seq)
    if len(heap) < k:
        heapq.heappush(heap, (length, seq))
        seen.add(seq)
    else:
        # heap[0] is (min_len, seq)
        min_len = heap[0][0]
        if length > min_len:
            _, removed = heapq.heapreplace(heap, (length, seq))
            seen.remove(removed)
            seen.add(seq)


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


def find_latest_checkpoint(model_dir: str, file_regex: str = r"version_(\d+)\.pt$") -> int:
    """
    Scan `model_dir` for files matching `file_regex` (default "version_XX.pt"),
    return the highest XX, or None if none found.
    """
    if not os.path.isdir(model_dir):
        return None
    best = None
    for fname in os.listdir(model_dir):
        m = re.match(file_regex, fname)
        if m:
            ver = int(m.group(1))
            if best is None or ver > best:
                best = ver
    return best


def next_version(model_dir: str, file_template: str = "version_{:02d}.pt") -> str:
    """
    Ensure `model_dir` exists, then return the path to the next unused
    version file inside it (e.g. "…/version_00.pt", "…/version_01.pt", etc.).
    """
    latest = find_latest_checkpoint(model_dir)
    new = (latest + 1) if latest is not None else 0
    os.makedirs(model_dir, exist_ok=True)
    version_path = os.path.join(model_dir, file_template.format(new))
    return version_path


def display_results(opponent, agent_wins_tuple, opponent_wins, draws, shortest, longest, elapsed):
    agent_wins_x, agent_wins_o = agent_wins_tuple
    agent_wins_total = agent_wins_x + agent_wins_o
    total = agent_wins_total + opponent_wins + draws
    print(f"\nResults vs {opponent}:")
    print(f"  Agent wins   X   : {agent_wins_x:,} ({100 * agent_wins_x/total:.1f}%)")
    print(f"  Agent wins   O   : {agent_wins_o:,} ({100 * agent_wins_o/total:.1f}%)")
    print(f"  Agent wins Total : {agent_wins_total:,} ({100 * agent_wins_total/total:.1f}%)\n")
    print(f"  Opponent wins    : {opponent_wins:,} ({100 * opponent_wins/total:.1f}%)\n")
    print(f"  Draws            : {draws:,} ({100 * draws/total:.1f}%)\n")
    print(f"  Time elapsed     : {format_elapsed(elapsed)} ({total/elapsed:.1f} games/sec)\n")

    current_time = time.localtime()
    current_time_str = f"{current_time.tm_mon}/{current_time.tm_mday}/{current_time.tm_year} @ {current_time.tm_hour:02}:{current_time.tm_min:02}:{current_time.tm_sec:02}"
    print(f"Finished {current_time_str}")
    
    write_to_json(shortest, 's')
    write_to_json(longest, 'l')


def validate_int(value):
    try:
        v=int(value.replace('_',''))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer.")
    return v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="load latest checkpoint")
    parser.add_argument("--games", type=validate_int, default=500, help="number of games to train")
    parser.add_argument("--opponent", type=str, default="random", help="opponent agent id (e.g. 'random', 'neural')")
    args = parser.parse_args()

    # # # #
    '''
    WHERE WE MAKE NEW NN SHAPES
    '''
    cfg = ModelConfig(
        hidden_sizes=[256, 512, 1024, 2048, 2048, 1024, 512, 256],
        learning_rate=1e-3,
        label="big_8_layer"
    )
    '''
    Update above, copy it into cfg of __init__
    '''
    # # # #

    ver = find_latest_checkpoint(cfg.model_dir)
    if ver is None:
        model_path_load = next_version(cfg.model_dir)
    else:
        model_path_load = os.path.join(cfg.model_dir, f"version_{ver:02d}.pt")

    if args.resume:
        model_path_save = model_path_load
    else:
        model_path_save = next_version(cfg.model_dir)

    agent = NeuralNetAgent2(cfg, model_path=None)
    if args.resume and os.path.isfile(model_path_load):
        agent.load(model_path_load)

    current_time_str = get_current_time_str()
    print(f"Training for {args.games:,} games vs {args.opponent}...\n\nStarting {current_time_str}")

    if args.opponent == "random":
        agent_wins, opponent_wins, draws, shortest, longest, elapsed = train_against_random(agent, args.games)
    else:
        agent_wins, opponent_wins, draws, shortest, longest, elapsed = train_against_agent(agent, args.opponent, args.games)

    display_results(args.opponent, agent_wins, opponent_wins, draws, shortest, longest, elapsed)

    os.makedirs(os.path.dirname(model_path_save), exist_ok=True)
    agent.save(model_path_save)