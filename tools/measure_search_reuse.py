"""How much duplicate work does a real 1 s search actually do?

Transposition merging and symmetry canonicalization are both substantial
implementations -- a DAG needs parent-independent backup bookkeeping, and
symmetry folding needs a canonical form threaded through the evaluation cache.
Neither is worth building blind. This measures the OPPORTUNITY first, on the
searches the deployed player really runs, so the decision is a number rather
than an appeal to how much these help in other games.

WHAT IS COUNTED. A search tree is walked after the fact and every node's state
is reconstructed by replaying moves from the root. Two totals matter:

    expanded nodes   nodes that cost a network evaluation
    distinct states  how many of those were genuinely different positions

The gap between them is the ceiling on what a transposition table could save.
Folding the eight board symmetries on top gives the additional ceiling for a
symmetry-canonicalized evaluation cache.

These are CEILINGS, not projections. A transposition table that avoids 20% of
evaluations does not make search 20% better: the freed budget goes into more
simulations, whose value is what the arena measures. And every hit costs a hash
and a lookup, so a small ceiling is a reason not to build the thing at all.

STATE IDENTITY. Two positions are the same when the board, the player to move,
and the legal-move constraint all match. The constraint is read off the engine
(do the legal moves span one mini or several?) rather than re-derived from
last_move here -- a private copy of the send-rule that disagreed with the engine
would silently merge positions that are not equal, which is far worse than
missing a transposition.

SYMMETRY. Ultimate tic-tac-toe's eight symmetries are exactly the eight
dihedral transforms of the 9x9 grid: rotating or reflecting the whole grid
permutes the macro board and every mini board consistently, which is precisely
the constraint a UTTT symmetry has to satisfy. The canonical form is the
lexicographic minimum over the eight.

    python -m tools.measure_search_reuse --positions 60 --ms 1000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from agents.mcts import MCTS
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from tools import provenance
from tools.analyze_search_disagreement import PHASE_BANDS, build_model
from tools.arena_1s import DEFAULT_CKPT


def d4_perms():
    """The eight dihedral transforms of the 9x9 board, as gather indices:
    new_board[a] = old_board[perm[a]]."""
    base = np.arange(81).reshape(9, 9)
    out = []
    for k in range(4):
        r = np.rot90(base, k)
        out.append(r.reshape(-1).copy())
        out.append(np.fliplr(r).reshape(-1).copy())
    return out


PERMS = d4_perms()
# Mini index of each cell, and how each transform permutes the MINI indices.
# Derived from the cell permutation rather than written out, so the two can
# never drift apart.
CELL_MINI = np.array([(i // 9 // 3) * 3 + (i % 9) // 3 for i in range(81)])
MACRO = []
for p in PERMS:
    m = np.empty(9, dtype=np.int64)
    for mini in range(9):
        src = int(np.flatnonzero(CELL_MINI == mini)[0])
        dst = int(np.flatnonzero(p == src)[0])
        m[mini] = CELL_MINI[dst]
    MACRO.append(m)


def constraint_of(state):
    """-1 when the mover may play anywhere, else the mini they are sent to.

    Read off the engine's own legal moves. A private reimplementation of the
    send rule that disagreed by one case would merge unequal positions.
    """
    valid = rule_utl_valid_moves(state.board, state.last_move,
                                 state.mini_winners)
    if not valid:
        return -1
    minis = {CELL_MINI[v] for v in valid}
    return int(next(iter(minis))) if len(minis) == 1 else -1


def state_key(state):
    board = np.asarray(state.board, dtype=np.int8)
    return (board.tobytes(), int(state.player), constraint_of(state))


def canon(board, player, con):
    """Lexicographic minimum over the eight transforms. Split out from
    canonical_key so the group law can be tested without an engine."""
    best = None
    for p, m in zip(PERMS, MACRO):
        k = (board[p].tobytes(), player, -1 if con < 0 else int(m[con]))
        if best is None or k < best:
            best = k
    return best


def canonical_key(state):
    return canon(np.asarray(state.board, dtype=np.int8), int(state.player),
                 constraint_of(state))


def walk(root, root_state):
    """Every expanded node's state, by replaying moves from the root."""
    keys, canon, nodes = [], [], 0
    stack = [(root, root_state)]
    while stack:
        node, st = stack.pop()
        nodes += 1
        keys.append(state_key(st))
        canon.append(canonical_key(st))
        for mv, ch in node.children.items():
            if ch.children:          # expanded => it cost a network evaluation
                nxt = st.clone()
                nxt.make_move(mv)
                stack.append((ch, nxt))
    return keys, canon, nodes


def sample_positions(n, seed):
    """Positions from self-play with random legal moves, spread over phases.

    Random play is the right sampler here because the question is structural --
    how often the search graph revisits a position -- and biasing the sample
    toward one engine's preferred lines would answer a narrower question than
    the one being asked.
    """
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        st = GameState()
        depth = rng.randint(2, 55)
        for _ in range(depth):
            valid = rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)
            if not valid or st.is_over():
                break
            st.make_move(rng.choice(valid))
        if not st.is_over():
            out.append(st)
    return out


def phase_of(filled):
    for name, lo, hi in PHASE_BANDS:
        if lo <= filled <= hi:
            return name
    return "late"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--ms", type=float, default=1000.0)
    ap.add_argument("--wave", type=int, default=8)
    ap.add_argument("--solve", type=int, default=1)
    ap.add_argument("--seed", type=int, default=6190)
    ap.add_argument("--output", default="results/arena_1s/search_reuse.json")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = build_model(args.checkpoint, args.device)
    mcts = MCTS(model, args.device, n_sims=10 ** 9, c_puct=1.5,
                add_dirichlet_at_root=False, wave_size=args.wave,
                solve=bool(args.solve), time_budget_ms=args.ms,
                max_sims=200_000)

    rows = []
    t0 = time.time()
    for i, st in enumerate(sample_positions(args.positions, args.seed)):
        filled = int(np.count_nonzero(st.board))
        _pi, root = mcts.search(st.clone())
        keys, canon, nodes = walk(root, st)
        rows.append({
            "filled": filled, "phase": phase_of(filled),
            "simulations": mcts.last["simulations_completed"],
            "neural_evaluations": mcts.last["neural_evaluations"],
            "expanded_nodes": nodes,
            "distinct_states": len(set(keys)),
            "distinct_canonical": len(set(canon)),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.positions}  {time.time() - t0:.0f}s",
                  flush=True)

    def agg(sel):
        sub = [r for r in rows if sel(r)]
        if not sub:
            return None
        exp = sum(r["expanded_nodes"] for r in sub)
        dis = sum(r["distinct_states"] for r in sub)
        can = sum(r["distinct_canonical"] for r in sub)
        return {
            "positions": len(sub),
            "expanded_nodes_per_search": exp / len(sub),
            "simulations_per_search":
                sum(r["simulations"] for r in sub) / len(sub),
            # Share of expansions a transposition table could have skipped.
            "transposition_ceiling": 1.0 - dis / exp if exp else None,
            # Additional share on top of that, from folding the 8 symmetries.
            "symmetry_ceiling_extra": (dis - can) / exp if exp else None,
            "combined_ceiling": 1.0 - can / exp if exp else None,
        }

    out = {"checkpoint": args.checkpoint, "budget_ms": args.ms,
           "wave": args.wave, "solve": bool(args.solve), "seed": args.seed,
           "overall": agg(lambda r: True),
           "by_phase": {p: agg(lambda r, p=p: r["phase"] == p)
                        for p, _lo, _hi in PHASE_BANDS},
           "per_search": rows, "provenance": provenance.build()}

    o = out["overall"]
    print(f"\n  {o['positions']} searches, "
          f"{o['expanded_nodes_per_search']:.0f} expanded nodes each "
          f"({o['simulations_per_search']:.0f} sims)")
    print(f"  transposition ceiling      {o['transposition_ceiling']:.4f}")
    print(f"  symmetry adds              {o['symmetry_ceiling_extra']:.4f}")
    print(f"  combined ceiling           {o['combined_ceiling']:.4f}")
    print(f"\n  {'phase':<8}{'searches':>10}{'expanded':>11}"
          f"{'transp':>9}{'+symm':>9}{'combined':>10}")
    for p, v in out["by_phase"].items():
        if v:
            print(f"  {p:<8}{v['positions']:>10}"
                  f"{v['expanded_nodes_per_search']:>11.0f}"
                  f"{v['transposition_ceiling']:>9.4f}"
                  f"{v['symmetry_ceiling_extra']:>9.4f}"
                  f"{v['combined_ceiling']:>10.4f}")
    print("\n  These are CEILINGS on avoided network evaluations, not strength. "
          "A freed\n  budget still has to be spent on more search, and the "
          "arena decides whether\n  that is worth anything.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
