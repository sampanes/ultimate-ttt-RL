from agents import get_agent
from agents.base import ModelConfig
from agents.neural_net_agent_2 import NeuralNetAgent2
from engine.game import GameState
from engine.constants import X, O, DRAW
import argparse, glob, re, os
import random, time, math
from typing import Tuple, List, Set, Any, Dict
import heapq
import json

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


def get_random_x_o():
    return X if random.random() < 0.5 else O


def play_and_train(agent, opponent, runs):
    agent_wins_x = 0
    agent_wins_o = 0
    opponent_wins = 0
    draws = 0
    decay_rate = math.log(100) / (81 - 1)
    start = time.time()
    k = 5
    shortest_heap, shortest_set = [], set()
    longest_heap,  longest_set  = [], set()

    for _ in range(runs):
        agent.clear_history()
        game = GameState()
        seq = []
        agent_side = get_random_x_o()

        while not game.is_over():
            current = agent if game.player == agent_side else opponent
            move = current.select_move(game)
            valid, _ = game.make_move(move)
            if not valid:
                raise ValueError(f"Invalid move: {move}")
            seq.append(move)

        # tally results
        if game.winner == agent_side:
            if agent_side == X:
                agent_wins_x += 1
            else:
                agent_wins_o += 1
        elif game.winner == DRAW:
            draws += 1
        else:
            opponent_wins += 1

        # record unique sequence
        tup = tuple(seq)
        consider_top_k_shortest(tup, shortest_heap, shortest_set, k)
        consider_top_k_longest(tup, longest_heap,  longest_set,  k)

        # time-based reward shaping
        num_moves = len(agent.last_players)
        adjusted = num_moves if game.winner != DRAW else 81
        time_factor = math.exp(-decay_rate * (adjusted - 18))

        # assign rewards for agent's turns
        agent.last_rewards = []
        for p in agent.last_players:
            if p == game.winner:
                agent.last_rewards.append(time_factor)
            elif game.winner == DRAW:
                agent.last_rewards.append(-0.02)
            else:
                agent.last_rewards.append(-time_factor)

        agent.learn()

    elapsed = time.time() - start
    # shortest_heap holds (-len, seq)
    shortest = sorted([seq for _, seq in shortest_heap], key=len)
    # longest_heap holds (len, seq)
    longest  = sorted([seq for _, seq in longest_heap],  key=len)

    return (agent_wins_x, agent_wins_o), opponent_wins, draws, shortest, longest, elapsed


def train_against_random(agent, runs):
    return play_and_train(agent, get_agent("random"), runs)


def train_against_agent(agent, opponent_name, runs):
    return play_and_train(agent, get_agent(opponent_name), runs)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="load latest checkpoint")
    parser.add_argument("--games", type=int, default=500, help="number of games to train")
    parser.add_argument("--opponent", type=str, default="random", help="opponent agent id (e.g. 'random', 'neural')")
    args = parser.parse_args()

    cfg = ModelConfig(
        hidden_sizes=[256, 512, 512, 512, 256],
        learning_rate=1e-3
    )
    hidden_str    = "-".join(map(str, cfg.hidden_sizes + [cfg.output_size]))
    cfg.model_dir = os.path.join("models", "neural_net_2", f"{hidden_str}")

    if args.resume:
        ver = find_latest_checkpoint(cfg.model_dir)
        if ver is None:
            model_path = next_version(cfg.model_dir)
        else:
            model_path = os.path.join(cfg.model_dir, f"version_{ver:02d}.pt")
    else:
        model_path = next_version(cfg.model_dir)

    print(f"\n\n##################\nNN2 TRAINING\n\nModel save path: {model_path}")
    agent = NeuralNetAgent2(cfg, model_path=None)
    if args.resume and os.path.isfile(model_path):
        agent.load(model_path)

    current_time = time.localtime()
    current_time_str = f"{current_time.tm_mon}/{current_time.tm_mday}/{current_time.tm_year} @ {current_time.tm_hour:02}:{current_time.tm_min:02}:{current_time.tm_sec:02}"
    print(f"Training for {args.games:,} games vs {args.opponent}...\n\nStarting {current_time_str}")

    if args.opponent == "random":
        agent_wins, opponent_wins, draws, shortest, longest, elapsed = train_against_random(agent, args.games)
    else:
        agent_wins, opponent_wins, draws, shortest, longest, elapsed = train_against_agent(agent, args.opponent, args.games)

    display_results(args.opponent, agent_wins, opponent_wins, draws, shortest, longest, elapsed)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)