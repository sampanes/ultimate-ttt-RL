import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from agents import get_agent
from agents.agent_base import ModelConfigCNN
from agents.neural_net_agent_3 import NeuralNetAgent3
from agents.super_agent_config import SUPER_MODEL_PATH, SUPER_CFG, build_super_agent
from engine.constants import X, O
from engine.game import GameState
from scripts.trainer_base import (
    get_current_time_str,
    train_against_agent_pool,
    display_results,
)

VERSION_RE = re.compile(r"version_(\d+)\.pt$")


@dataclass
class Candidate:
    version: int
    path: str
    score: float = 0.0


def run_match(agent_x, agent_o):
    gs = GameState()
    agents = {X: agent_x, O: agent_o}
    while not gs.is_over():
        move = agents[gs.player].select_move(gs)
        ok, _ = gs.make_move(move)
        if not ok:
            raise ValueError(f"Invalid move {move}")
    return gs.winner


def score_pair(agent_a, agent_b, games: int = 12) -> float:
    a_points = 0.0
    for _ in range(games):
        if random.random() < 0.5:
            winner = run_match(agent_a, agent_b)
            if winner == X:
                a_points += 1.0
            elif winner not in (X, O):
                a_points += 0.5
        else:
            winner = run_match(agent_b, agent_a)
            if winner == O:
                a_points += 1.0
            elif winner not in (X, O):
                a_points += 0.5
    return a_points / games


def list_challenger_versions(model_dir: str) -> List[Candidate]:
    p = Path(model_dir)
    if not p.exists():
        return []

    items = []
    for f in p.glob("version_*.pt"):
        m = VERSION_RE.match(f.name)
        if not m:
            continue
        items.append(Candidate(version=int(m.group(1)), path=str(f)))

    items.sort(key=lambda c: c.version, reverse=True)
    return items


def build_challenger_agent(path: str) -> NeuralNetAgent3:
    cfg = ModelConfigCNN(
        conv_channels=[32, 64, 64],
        fc_hidden_sizes=[256, 512, 1024, 512, 128],
        learning_rate=1e-4,
        label="new_cnn",
    )
    agent = NeuralNetAgent3(cfg=cfg, model_path=path)
    agent.set_eval(True)
    return agent


def rank_recent_challengers(model_dir: str, recent_count: int, games_per_pair: int) -> List[Candidate]:
    candidates = list_challenger_versions(model_dir)[:recent_count]
    if len(candidates) <= 1:
        return candidates

    agents = {c.path: build_challenger_agent(c.path) for c in candidates}

    for i, a in enumerate(candidates):
        total_score = 0.0
        comparisons = 0
        for j, b in enumerate(candidates):
            if i == j:
                continue
            total_score += score_pair(agents[a.path], agents[b.path], games=games_per_pair)
            comparisons += 1
        a.score = total_score / max(1, comparisons)

    return sorted(candidates, key=lambda c: c.score, reverse=True)


def compare_challenger_vs_super(challenger_path: str, gate_games: int) -> Tuple[int, int, int, float]:
    challenger = build_challenger_agent(challenger_path)
    super_agent = build_super_agent()
    super_agent.set_eval(True)

    wins = 0
    losses = 0
    draws = 0
    for _ in range(gate_games):
        if random.random() < 0.5:
            winner = run_match(challenger, super_agent)
            if winner == X:
                wins += 1
            elif winner == O:
                losses += 1
            else:
                draws += 1
        else:
            winner = run_match(super_agent, challenger)
            if winner == O:
                wins += 1
            elif winner == X:
                losses += 1
            else:
                draws += 1

    score = (wins + 0.5 * draws) / max(1, (wins + losses + draws))
    return wins, losses, draws, score


def build_wake_pool(args, ranked: List[Candidate]) -> list:
    pool = []
    tokens = [t.strip() for t in args.curriculum.split(",") if t.strip()]

    if not tokens:
        raise ValueError("Curriculum is empty. Provide at least one token.")

    challenger_agents = {}

    def challenger_for(path: str):
        if path not in challenger_agents:
            challenger_agents[path] = build_challenger_agent(path)
        return challenger_agents[path]

    latest = ranked[0] if ranked else None
    best = ranked[0] if ranked else None
    second_best = ranked[1] if len(ranked) > 1 else None

    for token in tokens:
        if token in {"random", "first", "nn", "nn2", "nn_big_8", "new_cnn", "lottery", "super_agent"}:
            pool.append(get_agent(token))
            continue

        if token == "last":
            if latest:
                pool.append(challenger_for(latest.path))
            continue

        if token == "best":
            if best:
                pool.append(challenger_for(best.path))
            continue

        if token == "second_best":
            if second_best:
                pool.append(challenger_for(second_best.path))
            continue

        if token == "best_plus_random":
            if best:
                pool.append(challenger_for(best.path))
            pool.append(get_agent("random"))
            continue

        raise ValueError(f"Unknown curriculum token: {token}")

    for opponent in pool:
        if hasattr(opponent, "set_eval"):
            opponent.set_eval(True)

    if not pool:
        raise ValueError("Curriculum produced an empty pool; check available checkpoints and tokens.")

    return pool


def maybe_wake_super_agent(args):
    ranked = rank_recent_challengers(
        model_dir=args.challenger_model_dir,
        recent_count=args.rank_recent,
        games_per_pair=args.rank_games_per_pair,
    )

    if not ranked:
        print(f"No challenger checkpoint found in {args.challenger_model_dir}. Super-agent remains dormant.")
        return

    print("\nRanked challenger checkpoints (best first):")
    for c in ranked[: min(len(ranked), 5)]:
        print(f"  version_{c.version:02d}.pt score={c.score:.3f}")

    gate_target = ranked[0]
    print(f"\nGate target: version_{gate_target.version:02d}.pt")

    w, l, d, score = compare_challenger_vs_super(gate_target.path, gate_games=args.gate_games)
    print(f"Gate score vs super-agent: {score:.3f} (W/L/D={w}/{l}/{d})")

    if score < args.wake_threshold:
        print(f"Super-agent stays dormant (needs >= {args.wake_threshold:.2f}, got {score:.3f}).")
        return

    print("\n🔥 Wake condition met. Super-agent training begins.")
    pool = build_wake_pool(args, ranked)
    pool_tokens = [t.strip() for t in args.curriculum.split(",") if t.strip()]

    super_agent = build_super_agent()
    super_agent.set_eval(False)

    current_time_str = get_current_time_str()
    print(
        f"Training super-agent for {args.train_games:,} games vs curriculum:{','.join(pool_tokens)} "
        f"(autosave={args.autosave_every})\n\nStarting {current_time_str}"
    )

    agent_wins, opponent_wins, draws, shortest, longest, elapsed = train_against_agent_pool(
        super_agent,
        pool,
        args.train_games,
        checkpoint_every=args.autosave_every,
        checkpoint_path=SUPER_MODEL_PATH,
        keep_last_checkpoints=0,
    )

    display_results("curriculum:" + ",".join(pool_tokens), agent_wins, opponent_wins, draws, shortest, longest, elapsed)
    super_agent.save(SUPER_MODEL_PATH)
    print(f"Saved immortal super-agent checkpoint to {SUPER_MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dormant immortal super-agent gate + training loop using existing arena/trainer stack."
    )
    parser.add_argument(
        "--challenger-model-dir",
        default="models/new_cnn/256-512-1024-512-128-81",
        help="Folder containing version_XX.pt challenger checkpoints.",
    )
    parser.add_argument("--gate-games", type=int, default=60, help="H2H games for wake gate")
    parser.add_argument(
        "--wake-threshold",
        type=float,
        default=0.55,
        help="Required challenger score (wins+0.5*draws)/games to wake super-agent.",
    )
    parser.add_argument("--train-games", type=int, default=50000, help="Training games once super-agent wakes")
    parser.add_argument("--autosave-every", type=int, default=2000, help="Autosave frequency while super-agent trains")
    parser.add_argument(
        "--curriculum",
        default="random,first,last,best,second_best,best_plus_random,nn,nn2,nn_big_8,new_cnn,lottery",
        help=(
            "Comma-separated opponent curriculum tokens. Supported tokens: "
            "random, first, nn, nn2, nn_big_8, new_cnn, lottery, super_agent, last, best, second_best, best_plus_random"
        ),
    )
    parser.add_argument(
        "--rank-recent",
        type=int,
        default=8,
        help="Use newest N challenger checkpoints when finding best/second_best.",
    )
    parser.add_argument(
        "--rank-games-per-pair",
        type=int,
        default=6,
        help="Mini round-robin games per pair when ranking recent challengers.",
    )
    args = parser.parse_args()

    maybe_wake_super_agent(args)
