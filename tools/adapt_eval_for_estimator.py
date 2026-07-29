"""Reshape eval_distill_pilot output into what pooled_estimator reads.

The two tools were written to the same experiment but not to the same JSON
shape: eval_distill_pilot nests everything under `primary_800_vs_50.per_seed`,
while pooled_estimator documents a flat `{seed: {wins, draws, losses,
outcomes, match_seed}}` (or `{seed: [w, d, l]}`).

This is an ADAPTER, deliberately, and it exists instead of a one-line edit to
pooled_estimator.py. That file is one of the 14 hashed into
EXPERIMENT_SOLVED_PILOT.json, frozen before any solved target existed. Once the
result is on screen, editing the estimator -- even to fix a plumbing mismatch
that has nothing to do with the arithmetic -- is precisely the move the freeze
is there to prevent, and "it was only plumbing" is what that move always looks
like from the inside. So the estimator stays byte-identical and the data is
brought to it.

Nothing here estimates anything: it selects a block, renames nothing, and
copies counts and per-game score vectors through unchanged. Verified by
recomputing each seed's score from its own outcomes vector and asserting it
matches the wins/draws/losses the evaluator reported.

    python -m tools.adapt_eval_for_estimator --eval EVAL.json --out FLAT.json
    python -m tools.adapt_eval_for_estimator --eval EVAL.json --out OC.json --outcomes-only
"""
from __future__ import annotations

import argparse
import json


def extract(doc):
    """Pull the single primary block out of an eval_distill_pilot document."""
    keys = [k for k in doc if k.startswith("primary_")]
    if len(keys) != 1:
        raise SystemExit(
            f"[X] expected exactly one primary_* block, found {keys}. Was this "
            f"produced by `eval_distill_pilot --primary`?")
    block = doc[keys[0]]
    if "per_seed" not in block:
        raise SystemExit(f"[X] {keys[0]} has no per_seed block")
    return keys[0], block


def check(seed, entry):
    """The outcomes vector and the W/D/L counts must tell the same story."""
    oc = entry.get("outcomes")
    if oc is None:
        return
    w = sum(1 for x in oc if x == 1.0)
    d = sum(1 for x in oc if x == 0.5)
    lo = sum(1 for x in oc if x == 0.0)
    if (w, d, lo) != (entry["wins"], entry["draws"], entry["losses"]):
        raise SystemExit(
            f"[X] seed {seed}: outcomes vector says {w}/{d}/{lo} but the "
            f"evaluator reported {entry['wins']}/{entry['draws']}/"
            f"{entry['losses']}. One of them is wrong; refusing to pair on it.")
    if len(oc) != w + d + lo:
        raise SystemExit(f"[X] seed {seed}: outcomes contains non-{{0,.5,1}} values")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="eval_distill_pilot output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--outcomes-only", action="store_true",
                    help="emit {seed: [per-game score, ...]} for "
                         "--baseline-outcomes instead of the full record")
    args = ap.parse_args()

    with open(args.eval, encoding="utf-8") as fh:
        name, block = extract(json.load(fh))

    flat = {}
    for seed, e in block["per_seed"].items():
        check(seed, e)
        if args.outcomes_only:
            if "outcomes" not in e:
                raise SystemExit(
                    f"[X] seed {seed} has no per-game outcomes, so game-level "
                    f"pairing cannot be licensed from this file.")
            flat[str(seed)] = e["outcomes"]
        else:
            rec = {"wins": e["wins"], "draws": e["draws"],
                   "losses": e["losses"]}
            for k in ("outcomes", "match_seed"):
                if k in e:
                    rec[k] = e[k]
            flat[str(seed)] = rec

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(flat, fh)

    kind = "outcome vectors" if args.outcomes_only else "records"
    print(f"[OK] {name}: {len(flat)} seed {kind} -> {args.out}")
    if not args.outcomes_only:
        for s, r in flat.items():
            n = r["wins"] + r["draws"] + r["losses"]
            sc = (r["wins"] + 0.5 * r["draws"]) / n
            print(f"     seed {s:>3}  {r['wins']:>3}/{r['draws']:>3}/"
                  f"{r['losses']:>3}  score {sc:.4f}  "
                  f"match_seed {r.get('match_seed', 'ABSENT')}  "
                  f"outcomes {'yes' if 'outcomes' in r else 'NO'}")


if __name__ == "__main__":
    main()
