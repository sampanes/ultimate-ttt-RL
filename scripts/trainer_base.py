import json
from agents import get_agent
from agents.agent_base import board_to_tensor_from_gamestate, get_random_x_o
import random, time, math
from typing import Tuple, List, Set, Any, Dict
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from engine.constants import X, O, DRAW
from tqdm import trange
import heapq, re, os

SMALL_GAME = 19
BIG_GAME = 778

LOG_FILE = "loss_logs/metrics_log.jsonl"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def append_metrics(loss: float, epsilon: float, winrate: float):
    entry = {
        "loss": loss,
        "epsilon": epsilon,
        "winrate": winrate
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def clear_metrics_log():
    """Clears the metrics log file."""
    open(LOG_FILE, "w").close()

def play_and_train(agent, opponent, runs):
    agent_wins_x = 0
    agent_wins_o = 0
    opponent_wins = 0
    draws = 0

    # for time-based shaping 
    reward_decay_rate = math.log(100) / (81 - 1)

    k = 5
    shortest_heap, shortest_set = [], set()
    longest_heap,  longest_set  = [], set()

    start_epsilon = 1.0
    min_epsilon = 0.1  # or 0.01 if you want

    # This makes epsilon decay linearly on a log scale over the full `runs`
    epsilon_decay = math.exp(math.log(min_epsilon / start_epsilon) / runs)

    epsilon = start_epsilon

    min_epsilon = 0.02

    start = time.time()
    clear_metrics_log()

    # A bit hacky so I can go to work
    sneaky_saves = True
    save_interval = 10
    # checkpoint_dir # TODO find a way to automate this
    t = trange(1, runs + 1, desc="Training", unit="game", bar_format = "{desc}: {percentage:.3f}%|{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}]")
    for i in t:
        agent.clear_history()
        game = GameState()
        seq = []
        agent_side = get_random_x_o()

        while not game.is_over():
            if game.player == agent_side:
                valid = rule_utl_valid_moves(game.board, game.last_move, game.mini_winners)

                # grab the state tensor once
                state = board_to_tensor_from_gamestate(game).to(agent.device)

                # ε-greedy pick
                if random.random() < epsilon:
                    move = random.choice(valid)
                else:
                    move = agent.select_move(game)

                # record *every* decision
                agent.last_game_states.append(state)
                agent.last_moves.append(move)
                agent.last_players.append(game.player)

            else:
                move = opponent.select_move(game)

            valid_move, _ = game.make_move(move)
            if not valid_move:
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

        calculate_reward(agent, game, reward_decay_rate)

        loss = agent.learn()

        total_agent_wins = agent_wins_x + agent_wins_o
        win_rate = total_agent_wins / i if i > 0 else 0
        t.set_description(f"🏋️‍♀️ {loss:.4f} | ε={epsilon:.3f} | WR={100*win_rate:.1f}%")

        append_metrics(loss, epsilon, win_rate)

        # decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # TODO figure out save directory for sneak saves on long runs away from home
        # if sneaky_saves and i % (runs/save_interval) == 0:
        #     t = time.localtime()
        #     t_str = f"{t.tm_mday:02}_{t.tm_hour:02}"
        #     chunk = i // save_interval
        #     weekend_name = f"weekend_{t_str}_{chunk:03d}.pt"
        #     weekend_path = os.path.join(checkpoint_dir, weekend_name)
        #     agent.save(weekend_path, verbose=False)

    elapsed = time.time() - start
    # shortest_heap holds (-len, seq)
    shortest = sorted([seq for _, seq in shortest_heap], key=len)
    # longest_heap holds (len, seq)
    longest  = sorted([seq for _, seq in longest_heap],  key=len)

    return (agent_wins_x, agent_wins_o), opponent_wins, draws, shortest, longest, elapsed


def calculate_reward(agent, game, decay_rate):
    # time-based reward shaping
    # TODO add ability to do flat or other reward systems
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


def train_against_random(agent, runs):
    return play_and_train(agent, get_agent("random"), runs)


def train_against_agent(agent, opponent_name, runs):
    return play_and_train(agent, get_agent(opponent_name), runs)

def train_against_self(agent, runs):
    # Clone the current agent to be the opponent
    import copy
    opponent = copy.deepcopy(agent)
    if hasattr(agent, "set_eval"):
        opponent.set_eval(True)  # Don't let opponent learn
        return play_and_train(agent, opponent, runs)
    else:
        print(f"Add set_eval func in {agent.name} like seen in neural net agent 3")
        train_against_random(agent, runs)


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
    Maintains a max-heap of size ≤ k containing the k shortest unique seqs seen so far.
    `heap` stores entries as (-len(seq), seq) so the root is the *longest* of the shortest cluster.
    """
    if seq in seen:
        return
    length = len(seq)

    if length > SMALL_GAME:
        return
    
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

    if length < BIG_GAME:
        return

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


def get_current_time_str():
    current_time = time.localtime()
    return f"{current_time.tm_mon}/{current_time.tm_mday}/{current_time.tm_year} @ {current_time.tm_hour:02}:{current_time.tm_min:02}:{current_time.tm_sec:02}"


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

    current_time_str  = get_current_time_str()
    print(f"Finished {current_time_str}")
    
    write_to_json(shortest, 's')
    write_to_json(longest, 'l')