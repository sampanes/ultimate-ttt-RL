from agents import get_agent
from agents.neural_net_agent import NeuralNetAgent
from engine.game import GameState
from engine.constants import X, O, DRAW
from collections import Counter
import time
import sys

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

    rewards = []
    for p in agent.last_players:
        if p == game.winner:
            rewards.append( 1.0 )
        elif game.winner == DRAW:
            rewards.append( 0.2 )
        else:
            rewards.append( -1.0 )

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
    # Default number of games
    '''
       500:  12.96s (38.6 games/sec)
     2,500:  66.27s (37.7 games/sec)
    10,000: 257.96s (38.8 games/sec)
    '''
    DEFAULT_N = 5000

    # Try to parse an integer from the first CLI argument
    if len(sys.argv) > 1 and sys.argv[1].isdigit() and int(sys.argv[1]) > 0:
        N = int(sys.argv[1])
    else:
        N = DEFAULT_N
        print(f"⚠️ Invalid or missing argument, defaulting to {DEFAULT_N} games.")

    agent = get_agent("nn")

    print(f"🏋️ Training {agent.name} via self-play for {N:,} games...")
    start = time.time()

    results = []
    for _ in range(N):
        states, moves, players, rewards, winner = run_self_play(agent, verbose=False)

        agent.last_game_states.extend(states)
        agent.last_moves.extend(moves)
        agent.last_players.extend(players)

        agent.last_rewards.extend(rewards)

        results.append(winner)

        if len(results) % 10 == 0:
            assert len(agent.last_game_states) == len(agent.last_moves) == len(agent.last_rewards), \
                f"💥 Batch mismatch before learn: {len(agent.last_game_states)} states, {len(agent.last_moves)} moves, {len(agent.last_rewards)} rewards"
            agent.learn()

    elapsed = time.time() - start
    formatted_time = format_elapsed(elapsed)
    
    c = Counter(results)

    print("\n✅ Training complete!")
    print(f"⏱️ Time elapsed: {formatted_time} ({N/elapsed:.1f} games/sec)")
    print("📊 Results:")
    print(f"  X wins  : {c[X]:,}")
    print(f"  O wins  : {c[O]:,}")
    print(f"  Draws   : {c[DRAW]:,}")
    print(f"  Invalid : {c[0]:,}")  # In case of unexpected zero wins
    agent.save("models/neural_net/self_play_trained_00.pt")
