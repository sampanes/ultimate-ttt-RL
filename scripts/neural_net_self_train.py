from agents.neural_net_agent import NeuralNetAgent
from engine.game import GameState
from engine.constants import X, O, DRAW
from collections import Counter
import time

def run_self_play(agent, verbose=False):
    game = GameState()
    agents = {X: agent, O: agent}

    while not game.is_over():
        current_agent = agents[game.player] # BOTH THE SAME, but placed here just to remember in future how to do it
        move = current_agent.select_move(game)
        valid, _ = game.make_move(move) # Winner is unused here
        if not valid:
            raise ValueError(f"Agent tried invalid move: {move}")
        if verbose:
            print(f"{'X' if game.player == O else 'O'} played {move}")
            game.print_board()

    if verbose:
        print(f"🏁 Game Over! Winner: {['None', 'X', 'O', 'Draw'][game.winner]}")
    
    agent.learn(game.winner)
    return game.winner

if __name__ == "__main__":
    agent = NeuralNetAgent(model_path="models/neural_net/self_play_trained_00.pt")

    N = 1000
    print(f"🏋️ Training {agent.name} via self-play for {N} games...")
    start = time.time()

    results = []
    for _ in range(N):
        winner = run_self_play(agent, verbose=False)
        results.append(winner)

    elapsed = time.time() - start
    c = Counter(results)

    print("\n✅ Training complete!")
    print(f"⏱️ Time elapsed: {elapsed:.2f}s ({N/elapsed:.1f} games/sec)")
    print("📊 Results:")
    print(f"  X wins  : {c[X]}")
    print(f"  O wins  : {c[O]}")
    print(f"  Draws   : {c[DRAW]}")
    print(f"  Invalid : {c[0]}")  # In case of unexpected zero wins
    agent.save("models/neural_net/self_play_trained_00.pt")
