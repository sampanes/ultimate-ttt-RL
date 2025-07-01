from agents import AGENT_REGISTRY
from engine.constants import X, O
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from collections import Counter
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

if __name__ == "__main__":
    N_GAMES = 10000
    start = time.time()
    winner_list = []

    for _ in range(N_GAMES):
        a1 = AGENT_REGISTRY["random"]
        a2 = AGENT_REGISTRY["first"]
        winner = run_match(a1, a2, verbose=False)
        winner_list.append(winner)

    end = time.time()
    elapsed = end - start
    c = Counter(winner_list)

    # Neat alignment
    label_width = max(len(a1.name), len(a2.name), 5)
    print(f"\n🏁 Results after {N_GAMES} games:")
    print(f"  {a1.name:<{label_width}} wins: {c[X]}")
    print(f"  {a2.name:<{label_width}} wins: {c[O]}")
    print(f"  {'Draws':<{label_width}} wins: {c[None]}")
    print(f"\n⏱️ Elapsed time: {elapsed:.2f} sec ({N_GAMES/elapsed:.2f} games/sec)")

