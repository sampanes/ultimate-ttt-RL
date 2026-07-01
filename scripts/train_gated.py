# scripts/train_gated.py
import argparse
import os
import random
import shutil
import time

from agents import get_agent
from engine.constants import X, O
from scripts.head_to_head_test import run_match
from scripts.trainer_base import train_against_agent, train_against_self, find_latest_checkpoint, next_version


def validate_int(value: str) -> int:
    try:
        return int(value.replace("_", ""))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer")


def set_eval_if_supported(agent, flag: bool = True):
    if hasattr(agent, "set_eval"):
        agent.set_eval(flag)


def clear_if_supported(agent):
    if hasattr(agent, "clear_history"):
        agent.clear_history()


def load_if_exists(agent, path: str):
    if path and os.path.isfile(path):
        agent.load(path)
        return True
    return False


def ensure_best_checkpoint(model_dir: str) -> str:
    """
    Ensure model_dir/best.pt exists. If not, seed it from latest version_XX.pt if present.
    Returns path to best.pt.
    """
    best_path = os.path.join(model_dir, "best.pt")
    if os.path.isfile(best_path):
        return best_path

    ver = find_latest_checkpoint(model_dir)
    if ver is None:
        # no checkpoints yet; best will be created after first promote
        return best_path

    latest_path = os.path.join(model_dir, f"version_{ver:02d}.pt")
    if os.path.isfile(latest_path):
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy2(latest_path, best_path)
    return best_path


def eval_head_to_head(a, b, games: int, seed: int) -> float:
    """
    Returns score for A in [0,1] where win=1, draw=0.5, loss=0.
    Randomizes sides each game.
    """
    rng = random.Random(seed)
    a_w = b_w = draws = 0

    set_eval_if_supported(a, True)
    set_eval_if_supported(b, True)

    for _ in range(games):
        # randomize sides
        if rng.random() < 0.5:
            agent_x, agent_o = a, b
            a_is_x = True
        else:
            agent_x, agent_o = b, a
            a_is_x = False

        clear_if_supported(agent_x)
        clear_if_supported(agent_o)

        winner = run_match(agent_x, agent_o, verbose=False)

        if winner == X:
            if a_is_x:
                a_w += 1
            else:
                b_w += 1
        elif winner == O:
            if a_is_x:
                b_w += 1
            else:
                a_w += 1
        else:
            draws += 1
        assert id(a) != id(b)
        
    return (a_w + 0.5 * draws) / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=str, default="nn_big_8", help="agent key, e.g. nn_big_8 / new_cnn / lottery")
    ap.add_argument("--train_opponent", type=str, default="self", help="self | random | first | <agentkey>")
    ap.add_argument("--chunk_games", type=validate_int, default=50_000, help="training games per chunk")
    ap.add_argument("--chunks", type=validate_int, default=20, help="number of chunks to attempt")
    ap.add_argument("--eval_games", type=validate_int, default=5_000, help="eval games per opponent")
    ap.add_argument("--promote_threshold", type=float, default=0.52,
                    help="must score >= this vs best to promote (0.50 means 'not worse')")
    ap.add_argument("--min_vs_first", type=float, default=0.60,
                    help="Candidate must score at least this vs 'first' to promote.")
    ap.add_argument("--min_vs_random", type=float, default=0.80,
                    help="Candidate must score at least this vs 'random' to promote.")
    ap.add_argument("--max_consecutive_baseline_fails", type=int, default=3,
                    help="Abort if candidate fails baseline gates this many chunks in a row.")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    agent = get_agent(args.agent)
    cfg = agent.cfg
    model_dir = cfg.model_dir
    os.makedirs(model_dir, exist_ok=True)

    best_path = ensure_best_checkpoint(model_dir)

    # start from best if it exists, otherwise from latest if present
    if os.path.isfile(best_path):
        agent.load(best_path)
        print(f"Loaded BEST: {best_path}")
    else:
        ver = find_latest_checkpoint(model_dir)
        if ver is not None:
            latest = os.path.join(model_dir, f"version_{ver:02d}.pt")
            agent.load(latest)
            print(f"Loaded LATEST: {latest}")
        else:
            print("No existing checkpoints found; starting from scratch.")

    # build eval opponents once
    best_eval = get_agent(args.agent)  # same agent class/config
    if os.path.isfile(best_path):
        best_eval.load(best_path)
    set_eval_if_supported(best_eval, True)

    first_eval = get_agent("first")
    random_eval = get_agent("random")
    set_eval_if_supported(first_eval, True)
    set_eval_if_supported(random_eval, True)

    consecutive_baseline_fails = 0
    for chunk_idx in range(1, args.chunks + 1):
        print("\n" + "=" * 80)
        print(f"Chunk {chunk_idx}/{args.chunks} | train {args.chunk_games:,} games vs {args.train_opponent}")
        print("=" * 80)

        # Always train starting from current best (prevents "drifting worse" for long)
        if os.path.isfile(best_path):
            agent.load(best_path)

        # TRAIN
        t0 = time.time()
        if args.train_opponent == "self":
            train_against_self(agent, args.chunk_games)
        else:
            train_against_agent(agent, args.train_opponent, args.chunk_games)
        t_train = time.time() - t0

        # Save candidate temporarily
        candidate_path = os.path.join(model_dir, "candidate.pt")
        agent.save(candidate_path)
        print(f"Saved candidate: {candidate_path}  (train time {t_train:.1f}s)")

        # EVAL candidate vs best + baselines
        cand_eval = get_agent(args.agent)
        cand_eval.load(candidate_path)

        score_vs_best = eval_head_to_head(cand_eval, best_eval, args.eval_games, seed=args.seed + 1000 + chunk_idx)

        score_vs_first = eval_head_to_head(cand_eval, first_eval, args.eval_games, seed=args.seed + 2000 + chunk_idx)
        score_vs_random = eval_head_to_head(cand_eval, random_eval, args.eval_games, seed=args.seed + 3000 + chunk_idx)

        msg = (
            f"Eval score vs BEST: {score_vs_best:.3f} (threshold {args.promote_threshold:.3f})"
            f" | vs first: {score_vs_first:.3f} (min {args.min_vs_first:.3f})"
            f" | vs random: {score_vs_random:.3f} (min {args.min_vs_random:.3f})"
        )
        print(msg)

        passes_best = score_vs_best >= args.promote_threshold
        passes_first = score_vs_first >= args.min_vs_first
        passes_random = score_vs_random >= args.min_vs_random

        passes_all = passes_best and passes_first and passes_random

        # PROMOTE or REJECT
        if passes_all:
            consecutive_baseline_fails = 0  # reset on success

            version_path = next_version(model_dir)
            os.replace(candidate_path, version_path)
            shutil.copy2(version_path, best_path)

            best_eval.load(best_path)

            print(f"[OK] PROMOTED -> {version_path}")
            print(f"[OK] Updated   -> {best_path}")
        else:
            # classify why (helps debugging)
            reasons = []
            if not passes_best:
                reasons.append("didn't beat BEST")
            if not passes_first:
                reasons.append("too weak vs FIRST")
            if not passes_random:
                reasons.append("too weak vs RANDOM")

            try:
                os.remove(candidate_path)
            except OSError:
                pass

            print(f"[X] Rejected candidate (kept best). Reasons: {', '.join(reasons)}")

            # baseline-fail guard (wasting GPU detector)
            if (not passes_first) or (not passes_random):
                consecutive_baseline_fails += 1
            else:
                consecutive_baseline_fails = 0  # only count baseline failures

            # TODO: if this triggers often, something is wrong (eval determinism, encoding, LR, etc.)
            if consecutive_baseline_fails >= args.max_consecutive_baseline_fails:
                print(
                    f"\n[STOP] Aborting: failed baseline gates (first/random) "
                    f"{consecutive_baseline_fails} chunks in a row.\n"
                    f"Likely wasting GPU on bad settings/bug. Check:\n"
                    f"- eval determinism (epsilon forced to 0)\n"
                    f"- state encoding (81 vs 7x9x9)\n"
                    f"- learning rate / epsilon schedule\n"
                )
                break

    print("\nDone.")


if __name__ == "__main__":
    main()