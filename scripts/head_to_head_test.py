from agents import get_agent
from engine.constants import X, O
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from collections import defaultdict
import argparse
import random
import time


def run_match(agent1, agent2, verbose=True):
    gs = GameState()
    agents = {X: agent1, O: agent2}

    while not gs.is_over():
        agent = agents[gs.player]
        move = agent.select_move(gs)
        if not gs.is_valid_move(move):
            raise ValueError(f"{agent.name} tried invalid move {move}. Valid moves were: {rule_utl_valid_moves(gs.board, gs.last_move, gs.mini_winners)}")
        
        success, _ = gs.make_move(move)

        if not success:
            raise ValueError(f"{agent.name} tried invalid move {move}")
        
        if verbose:
            print(f"{agent.name} played {move}")
            gs.print_board()

    winner = gs.winner
    if verbose:
        print(f"🎉 Winner: {agents[winner].name if winner else 'Draw'}")
    return winner


def print_results(n_games, x_wins, o_wins, draws, elapsed):
    print(f"\n🏁 Results after {n_games:,} games:")
    all_names = sorted(set(x_wins) | set(o_wins))
    total_wins = {}

    for name in all_names:
        x = x_wins[name]
        o = o_wins[name]
        total = x + o
        percent = (total / n_games) * 100
        total_wins[name] = total

        print(f"  {name:<15}{' as X: ':<8}{x:>6,} wins")
        print(f"  {name:<15}{' as O: ':<8}{o:>6,} wins")
        print(f"  {' '*15}{' total: ':<8}{total:>6,} wins  ( {percent:.1f}% )\n")

    print(f"  {'Draws':<15}{' ':<8}{draws:>6,} games ( {(draws / n_games) * 100:.1f}% )")

    winner = max(total_wins, key=total_wins.get)
    wins = total_wins[winner]
    ties = [k for k, v in total_wins.items() if v == wins]

    if len(ties) == 1:
        print(f"\n🏆 Overall winner: {winner} with {wins:,} wins ({(wins/n_games)*100:.1f}%)")
    else:
        print(f"\n🤝 Tie between: {', '.join(ties)} with {wins:,} wins each")

    print(f"\n⏱️ Elapsed time: {elapsed:.2f} sec ({n_games/elapsed:.2f} games/sec)")


def agent_vs_agent(a1_string, a2_string, n_games=1000):
    start = time.time()
    winner_list = []
    x_wins = defaultdict(int)
    o_wins = defaultdict(int)
    draws = 0
    a1 = get_agent(a1_string)
    a2 = get_agent(a2_string)

    for agent in (a1, a2):
        if hasattr(agent, "set_eval"):
            print(f"Setting {agent.name} to eval mode")
            agent.set_eval(True)

    for _ in range(n_games):
        if random.random() < 0.5:
            a1, a2 = a2, a1
            x_name, o_name = a2_string, a1_string
        else:
            x_name, o_name = a1_string, a2_string

        winner = run_match(a1, a2, verbose=False)
        winner_list.append(winner)

        if winner == X:
            x_wins[x_name] += 1
        elif winner == O:
            o_wins[o_name] += 1
        else:
            draws += 1

    end = time.time()
    elapsed = end - start

    print_results(n_games, x_wins, o_wins, draws, elapsed)


def validate_int(value):
    try:
        v=int(value.replace('_',''))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer.")
    return v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=validate_int, default=500, help="number of games to play")
    parser.add_argument("--a1", type=str, default="nn", help="Agent 1, i.e. nn2 or random")
    parser.add_argument("--a2", type=str, default="random", help="Agent 2, i.e. nn2 or random")
    args = parser.parse_args()

    current_time = time.localtime()
    current_time_str = f"{current_time.tm_mon}/{current_time.tm_mday}/{current_time.tm_year} @ {current_time.tm_hour:02}:{current_time.tm_min:02}:{current_time.tm_sec:02}"

    title = f"HEAD TO HEAD, {args.a1} vs {args.a2}"
    width = len(title)+8
    border = "*"*width
    print(border)
    print("*   "+title+"   *")
    print(border)
    print(f"\n\nStarting {current_time_str}")
    agent_vs_agent(args.a1, args.a2, args.games)
    
    current_time = time.localtime()
    current_time_str = f"{current_time.tm_mon}/{current_time.tm_mday}/{current_time.tm_year} @ {current_time.tm_hour:02}:{current_time.tm_min:02}:{current_time.tm_sec:02}"
    print(f"\nFinished {current_time_str}")

