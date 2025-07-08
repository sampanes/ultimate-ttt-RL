from agents.base import ModelConfig
from agents.neural_net_agent_2 import NeuralNetAgent2
import argparse, os
from scripts.trainer_base import (
    next_version, find_latest_checkpoint, train_against_random, train_against_agent,
    get_current_time_str, display_results
)

DEFAULT_CHECKPOINT_PREFIX = "models/neural_net_2/self_play_trained_"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="load latest checkpoint")
    parser.add_argument("--games", type=int, default=500, help="number of games to train")
    parser.add_argument("--opponent", type=str, default="random", help="opponent agent id (e.g. 'random', 'neural')")
    args = parser.parse_args()

    cfg = ModelConfig(
        hidden_sizes=[256, 512, 512, 512, 256],
        learning_rate=1e-3,
        label="neural_net_2"
    )

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

    current_time_str = get_current_time_str()
    print(f"Training for {args.games:,} games vs {args.opponent}...\n\nStarting {current_time_str}")

    if args.opponent == "random":
        agent_wins, opponent_wins, draws, shortest, longest, elapsed = train_against_random(agent, args.games)
    else:
        agent_wins, opponent_wins, draws, shortest, longest, elapsed = train_against_agent(agent, args.opponent, args.games)

    display_results(args.opponent, agent_wins, opponent_wins, draws, shortest, longest, elapsed)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)