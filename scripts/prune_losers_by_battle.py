import argparse
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from agents.agent_base import ModelConfigCNN
from agents.neural_net_agent_3 import NeuralNetAgent3
from engine.constants import X, O
from engine.game import GameState


VERSION_RE = re.compile(r"version_(\d+)\.pt$")


@dataclass
class Stat:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games


def parse_version(path: Path):
    m = VERSION_RE.match(path.name)
    return int(m.group(1)) if m else None


def list_versions(model_dir: str) -> List[Path]:
    p = Path(model_dir)
    if not p.exists():
        return []
    files = []
    for f in p.glob("version_*.pt"):
        ver = parse_version(f)
        if ver is not None:
            files.append((ver, f))
    files.sort(key=lambda t: t[0], reverse=True)
    return [f for _, f in files]


def run_match(agent_x, agent_o):
    gs = GameState()
    agents = {X: agent_x, O: agent_o}
    while not gs.is_over():
        move = agents[gs.player].select_move(gs)
        ok, _ = gs.make_move(move)
        if not ok:
            raise ValueError(f"Invalid move {move}")
    return gs.winner


def build_agent(model_path: str):
    cfg = ModelConfigCNN(
        conv_channels=[32, 64, 64],
        fc_hidden_sizes=[256, 512, 1024, 512, 128],
        learning_rate=1e-4,
        label="new_cnn",
    )
    agent = NeuralNetAgent3(cfg, model_path=model_path)
    agent.set_eval(True)
    return agent


def evaluate(models: List[Path], games_per_pair: int) -> Dict[Path, Stat]:
    stats = {m: Stat() for m in models}
    agents = {m: build_agent(str(m)) for m in models}

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = models[i]
            b = models[j]
            for _ in range(games_per_pair):
                if random.random() < 0.5:
                    x, o = a, b
                else:
                    x, o = b, a
                winner = run_match(agents[x], agents[o])
                if winner == X:
                    stats[x].wins += 1
                    stats[o].losses += 1
                elif winner == O:
                    stats[o].wins += 1
                    stats[x].losses += 1
                else:
                    stats[x].draws += 1
                    stats[o].draws += 1
    return stats


def main():
    parser = argparse.ArgumentParser(description="Battle checkpoints and optionally delete clear losers.")
    parser.add_argument("--model-dir", required=True, help="directory containing version_XX.pt checkpoints")
    parser.add_argument("--max-models", type=int, default=12, help="evaluate only the newest N checkpoints")
    parser.add_argument("--games-per-pair", type=int, default=8, help="games per model pair")
    parser.add_argument("--delete-below", type=float, default=0.40, help="delete models with score below this threshold")
    parser.add_argument("--always-keep-latest", type=int, default=2, help="never delete the newest N checkpoints")
    args = parser.parse_args()

    models = list_versions(args.model_dir)
    if args.max_models > 0:
        models = models[: args.max_models]

    if len(models) < 2:
        print("Need at least two checkpoints to compare.")
        return

    stats = evaluate(models, games_per_pair=args.games_per_pair)

    rows = []
    for m in models:
        st = stats[m]
        rows.append((st.score, st.games, m, st))
    rows.sort(key=lambda t: (t[0], t[1]))

    print("\nCheckpoint ranking (worst -> best):")
    for score, games, m, st in rows:
        print(
            f"  {m.name:<14} score={score:.3f} games={games:3d} "
            f"W/L/D={st.wins}/{st.losses}/{st.draws}"
        )

    keep_protected = set(models[: args.always_keep_latest])
    losers = [m for score, _, m, _ in rows if score < args.delete_below and m not in keep_protected]

    if not losers:
        print("\nNo losers matched delete criteria. Nothing to delete.")
        return

    print("\nCandidates for deletion:")
    for m in losers:
        print(f"  {m}")

    answer = input("\nType YES exactly to delete these losers: ")
    if answer != "YES":
        print("Aborted. No files deleted.")
        return

    deleted = 0
    for m in losers:
        try:
            os.remove(m)
            deleted += 1
        except OSError as e:
            print(f"Failed deleting {m}: {e}")

    print(f"Deleted {deleted} loser checkpoints.")


if __name__ == "__main__":
    main()
