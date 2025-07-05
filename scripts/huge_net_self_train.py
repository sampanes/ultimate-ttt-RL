# File: scripts/huge_net_self_train.py

import argparse, os, time
from agents.huge_net_agent import HugeNetAgent
from agents import get_agent
from engine.config import ModelConfig
from neural_net_self_train_2 import (
    train_against_random, train_against_agent, display_results,
    find_latest_checkpoint, next_version
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="load latest checkpoint")
    parser.add_argument("--games", type=int, default=500, help="number of games to train")
    parser.add_argument("--opponent", type=str, default="random", help="opponent agent id")
    args = parser.parse_args()

    cfg = ModelConfig(
        hidden_sizes=[2048, 4096, 8192],
        learning_rate=1e-4
    )
    hidden_str = "-".join(map(str, cfg.hidden_sizes + [cfg.output_size]))
    cfg.model_dir = os.path.join("models", "huge_net", f"{hidden_str}")

    if args.resume:
        ver = find_latest_checkpoint(cfg.model_dir)
        if ver is None:
            model_path = next_version(cfg.model_dir)
        else:
            model_path = os.path.join(cfg.model_dir, f"version_{ver:02d}.pt")
    else:
        model_path = next_version(cfg.model_dir)

    print("\n\n##################\nHUGE NET TRAINING\n")
    print(f"Model save path: {model_path}\n")
    agent = HugeNetAgent(cfg)
    if args.resume and os.path.isfile(model_path):
        agent.model.load_state_dict(torch.load(model_path, map_location=agent.device))

    start = time.strftime("%m/%d/%Y @ %H:%M:%S")
    print(f"Training for {args.games:,} games vs {args.opponent}...\nStarting {start}\n")

    if args.opponent == "random":
        agent_wins, opponent_wins, draws, shortest, longest, elapsed = train_against_random(agent, args.games)
    else:
        agent_wins, opponent_wins, draws, shortest, longest, elapsed = train_against_agent(agent, args.opponent, args.games)

    display_results(args.opponent, agent_wins, opponent_wins, draws, shortest, longest, elapsed)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(agent.model.state_dict(), model_path)
