# scripts/run_tournament.py
import argparse
import itertools
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import contextlib
import io

from agents import get_agent, AGENT_FACTORIES  # uses your factory keys
from agents.neural_net_agent_pg import NeuralNetAgentPG
from agents.agent_base import ModelConfigCNN
from arena.population_manager import PopulationManager
from engine.constants import X, O
from scripts.head_to_head_test import run_match
from scripts.trainer_base import find_latest_checkpoint

ARENA_STATE_PATH = "models/arena/arena_state.json"


def load_arena_agents(n: int, state_path: str = ARENA_STATE_PATH) -> Dict[str, object]:
    """
    Load the top-n arena agents by ELO from the population state JSON.
    Returns {display_name: agent} ordered best-first.
    Prefers active agents; falls back to all agents if fewer than n are active.
    """
    pm = PopulationManager(state_path)
    if not pm.load():
        raise FileNotFoundError(f"Arena state not found: {state_path}")

    candidates = pm.active_agents()
    if len(candidates) < n:
        # top up with retired agents ranked by best_elo
        retired = sorted(
            [a for a in pm.agents if a.status == "retired"],
            key=lambda a: a.best_elo,
            reverse=True,
        )
        candidates = candidates + retired

    top = sorted(candidates, key=lambda a: a.elo, reverse=True)[:n]

    built: Dict[str, object] = {}
    for record in top:
        cfg = ModelConfigCNN(
            conv_channels=record.conv_channels,
            fc_hidden_sizes=record.fc_hidden_sizes,
            learning_rate=1e-4,
            label="arena",
            model_dir=record.model_dir,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            agent = NeuralNetAgentPG(cfg=cfg, model_path=None)

        latest_ver = find_latest_checkpoint(record.model_dir)
        if latest_ver is not None:
            ckpt = os.path.join(record.model_dir, f"version_{latest_ver:02d}.pt")
            agent.load(ckpt)

        if hasattr(agent, "set_eval"):
            agent.set_eval(True)

        label = f"{record.name}(elo={record.elo:.0f})"
        built[label] = agent
        print(f"  arena: {label}  arch={record.arch_label}  ckpt={latest_ver}")

    return built


# ----------------------------
# helpers
# ----------------------------

def validate_int(value: str) -> int:
    """Allow 10_000 style ints."""
    try:
        return int(value.replace("_", ""))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer")


def parse_agent_spec(spec: str) -> Tuple[str, Optional[str]]:
    """
    Supported forms:
      - "nn2"                 -> use factory as-is
      - "nn2@latest"          -> load latest version_XX.pt from agent.cfg.model_dir
      - "nn2@05"              -> load version_05.pt from agent.cfg.model_dir
      - "nn2@version_05.pt"   -> same as above (ver string parsing is flexible)
      - "nn2@models/.../x.pt" -> treat as explicit path
    """
    if "@" not in spec:
        return spec, None
    name, suffix = spec.split("@", 1)
    return name.strip(), suffix.strip()


def resolve_checkpoint(agent, suffix: str) -> Optional[str]:
    """
    Returns a checkpoint path if we can resolve one, else None.
    Requires agent.cfg.model_dir to exist.
    """
    if not hasattr(agent, "cfg") or not getattr(agent.cfg, "model_dir", None):
        return None

    model_dir = agent.cfg.model_dir

    # explicit path
    if suffix.endswith(".pt") and ("/" in suffix or "\\" in suffix):
        return suffix

    # latest
    if suffix.lower() == "latest":
        latest = find_latest_checkpoint(model_dir)
        if latest is None:
            return None
        return os.path.join(model_dir, f"version_{latest:02d}.pt")

    # version number like "05"
    if suffix.isdigit():
        return os.path.join(model_dir, f"version_{int(suffix):02d}.pt")

    # something like "version_05.pt"
    if "version_" in suffix and suffix.endswith(".pt"):
        # strip any leading paths
        fname = os.path.basename(suffix)
        return os.path.join(model_dir, fname)

    # fallback: treat as filename inside model_dir
    if suffix.endswith(".pt"):
        return os.path.join(model_dir, suffix)

    return None


def make_agent(spec: str):
    """
    Builds an agent from your agents.get_agent(), with optional checkpoint override.
    """
    name, suffix = parse_agent_spec(spec)

    # first build via factory
    agent = get_agent(name)

    # optional override
    if suffix:
        ckpt = resolve_checkpoint(agent, suffix)
        if ckpt is None:
            print(f"[!]  {spec}: couldn't resolve checkpoint override; using factory default.")
        else:
            # Rebuild the agent if constructor supports (cfg=..., model_path=...)
            # This matches how your factory builds NeuralNetAgent2/3. :contentReference[oaicite:1]{index=1}
            try:
                agent = agent.__class__(cfg=agent.cfg, model_path=ckpt)
            except TypeError:
                # If ctor doesn't match, just warn and keep the original.
                print(f"[!]  {spec}: agent class doesn't support cfg/model_path rebuild; using factory default.")
            else:
                print(f"[OK] {spec}: using checkpoint {ckpt}")

    # evaluation mode if supported
    if hasattr(agent, "set_eval"):
        agent.set_eval(True)

    return agent


def maybe_clear(agent):
    if hasattr(agent, "clear_history"):
        agent.clear_history()


@dataclass
class PairResult:
    a: str
    b: str
    games: int
    a_wins: int
    b_wins: int
    draws: int

    def a_score(self) -> float:
        return (self.a_wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def b_score(self) -> float:
        return (self.b_wins + 0.5 * self.draws) / self.games if self.games else 0.0


def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def elo_update(ra: float, rb: float, sa: float, k: float) -> Tuple[float, float]:
    ea = elo_expected(ra, rb)
    eb = 1.0 - ea
    ra2 = ra + k * (sa - ea)
    rb2 = rb + k * ((1.0 - sa) - eb)
    return ra2, rb2


# ----------------------------
# tournament
# ----------------------------

def play_series(a_name: str, b_name: str, a_agent, b_agent, games: int, rng: random.Random,
                elo: Optional[Dict[str, float]] = None, k: float = 16.0) -> PairResult:
    a_wins = b_wins = draws = 0

    for _ in range(games):
        # randomize who is X vs O each game
        if rng.random() < 0.5:
            agent_x, agent_o = a_agent, b_agent
            x_name, o_name = a_name, b_name
        else:
            agent_x, agent_o = b_agent, a_agent
            x_name, o_name = b_name, a_name

        maybe_clear(agent_x)
        maybe_clear(agent_o)

        winner = run_match(agent_x, agent_o, verbose=False)

        if winner == X:
            # X player won
            if x_name == a_name:
                a_wins += 1
                sa = 1.0
            else:
                b_wins += 1
                sa = 0.0
        elif winner == O:
            # O player won
            if o_name == a_name:
                a_wins += 1
                sa = 1.0
            else:
                b_wins += 1
                sa = 0.0
        else:
            draws += 1
            sa = 0.5

        if elo is not None:
            ra, rb = elo[a_name], elo[b_name]
            ra2, rb2 = elo_update(ra, rb, sa, k)
            elo[a_name], elo[b_name] = ra2, rb2

    return PairResult(a=a_name, b=b_name, games=games, a_wins=a_wins, b_wins=b_wins, draws=draws)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["new_cnn@latest", "lottery@latest", "nn2@latest", "nn_big_8@latest", "random", "first"],
        help="Agent specs. Use '@latest' to load newest version_XX.pt from that agent's cfg.model_dir.",
    )
    parser.add_argument("--games", type=validate_int, default=2000, help="Games per head-to-head pairing")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (side assignment + series order)")
    parser.add_argument("--elo", action="store_true", help="Also compute Elo ratings")
    parser.add_argument("--k", type=float, default=16.0, help="Elo K-factor (only used with --elo)")
    parser.add_argument("--out", type=str, default="", help="Optional path to write results JSON")
    parser.add_argument("--list", action="store_true", help="List available factory keys and exit")
    parser.add_argument(
        "--arena",
        nargs="?",
        const=1,
        type=int,
        default=None,
        metavar="N",
        help="Include top-N arena agents by ELO (default N=1 if flag given without a number)",
    )
    parser.add_argument(
        "--arena-state",
        type=str,
        default=ARENA_STATE_PATH,
        metavar="PATH",
        help=f"Path to arena_state.json (default: {ARENA_STATE_PATH})",
    )
    args = parser.parse_args()

    if args.list:
        keys = sorted(AGENT_FACTORIES.keys())
        print("Available agents:", ", ".join(keys))
        return

    rng = random.Random(args.seed)

    # build agents (and skip any that fail)
    built: Dict[str, object] = {}
    specs_ok: List[str] = []

    if args.arena is not None:
        print(f"\nLoading top-{args.arena} arena agent(s)...")
        try:
            arena_built = load_arena_agents(args.arena, args.arena_state)
            for name, agent in arena_built.items():
                built[name] = agent
                specs_ok.append(name)
        except Exception as e:
            print(f"  Could not load arena agents: {e}")

    for spec in args.agents:
        try:
            agent = make_agent(spec)
        except Exception as e:
            print(f"[X] Skipping {spec}: {e}")
            continue
        built[spec] = agent
        specs_ok.append(spec)

    if len(specs_ok) < 2:
        raise SystemExit("Need at least 2 working agents to run a tournament.")

    # init Elo
    elo = {spec: 1000.0 for spec in specs_ok} if args.elo else None

    # round robin
    pair_results: List[PairResult] = []
    totals = {spec: {"wins": 0, "losses": 0, "draws": 0, "points": 0.0, "games": 0} for spec in specs_ok}

    start = time.time()
    pairs = list(itertools.combinations(specs_ok, 2))
    rng.shuffle(pairs)

    for i, (a_spec, b_spec) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {a_spec}  vs  {b_spec}  ({args.games:,} games)")
        res = play_series(a_spec, b_spec, built[a_spec], built[b_spec], args.games, rng, elo=elo, k=args.k)
        pair_results.append(res)

        # update totals
        totals[a_spec]["wins"] += res.a_wins
        totals[a_spec]["losses"] += res.b_wins
        totals[a_spec]["draws"] += res.draws
        totals[a_spec]["games"] += res.games
        totals[a_spec]["points"] += res.a_wins * 1.0 + res.draws * 0.5

        totals[b_spec]["wins"] += res.b_wins
        totals[b_spec]["losses"] += res.a_wins
        totals[b_spec]["draws"] += res.draws
        totals[b_spec]["games"] += res.games
        totals[b_spec]["points"] += res.b_wins * 1.0 + res.draws * 0.5

        print(f"  -> {a_spec}: {res.a_wins}W {res.draws}D {res.b_wins}L  (score {res.a_score():.3f})")
        print(f"  -> {b_spec}: {res.b_wins}W {res.draws}D {res.a_wins}L  (score {res.b_score():.3f})")

    elapsed = time.time() - start

    # standings
    def winrate(spec: str) -> float:
        g = totals[spec]["games"]
        if g == 0:
            return 0.0
        return (totals[spec]["wins"] + 0.5 * totals[spec]["draws"]) / g

    ranked = sorted(
        specs_ok,
        key=lambda s: (totals[s]["points"], winrate(s), -totals[s]["losses"]),
        reverse=True,
    )

    print("\n" + "=" * 70)
    print("STANDINGS")
    print("=" * 70)
    for idx, spec in enumerate(ranked, 1):
        t = totals[spec]
        wr = winrate(spec)
        line = f"{idx:>2}. {spec:<20}  {t['wins']:>6}W {t['draws']:>6}D {t['losses']:>6}L  " \
               f"points={t['points']:.1f}  winrate={wr:.3f}"
        if elo is not None:
            line += f"  elo={elo[spec]:.1f}"
        print(line)

    print(f"\n[t]  Elapsed: {elapsed:.2f}s   (~{sum(totals[s]['games'] for s in specs_ok)/elapsed:.2f} games/sec)")

    # optional JSON output
    if args.out:
        payload = {
            "agents": specs_ok,
            "games_per_pair": args.games,
            "seed": args.seed,
            "elo_enabled": bool(args.elo),
            "elo_k": args.k,
            "standings": [
                {
                    "agent": spec,
                    **totals[spec],
                    "winrate": winrate(spec),
                    "elo": (elo[spec] if elo is not None else None),
                }
                for spec in ranked
            ],
            "pair_results": [res.__dict__ for res in pair_results],
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[save] Wrote results to {args.out}")


if __name__ == "__main__":
    main()