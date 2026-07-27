"""Freeze an experiment manifest BEFORE any result exists, and verify it after.

The point is to make it impossible to quietly move the goalposts. Everything
that could change an outcome -- checkpoint bytes, corpus revision, code commit,
anchor implementations, seeds, evaluation opponents and game counts -- is
hashed and written down while the answer is still unknown. `--verify` re-hashes
and reports any drift.

Anchors are hashed by SOURCE FILE, not by name, because a heuristic opponent is
only a fixed ruler for as long as its code is fixed. benchmark_suite once
attested gregory against the wrong file for exactly this reason.

    python -m tools.freeze_experiment --spec distill_pilot --out EXPERIMENT_DISTILL_PILOT.json
    python -m tools.freeze_experiment --spec distill_pilot --verify EXPERIMENT_DISTILL_PILOT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time


def sha256_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state():
    def run(*a):
        try:
            return subprocess.check_output(a, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


# The pilot, as specified. Written here rather than passed on the command line
# so the frozen spec is itself under version control.
DISTILL_PILOT = {
    "name": "distill_pilot_50_vs_800",
    "question": ("Do stronger MCTS targets transfer through distillation into a "
                 "fixed 172k student? Arms are 50-sim and 800-sim teachers over "
                 "IDENTICAL positions."),
    "why_these_arms": ("RESULT_TEACHER_SIM_LADDER.md measured +0.019 strength per "
                       "doubling, so the 800-vs-200 teacher gap (~4 pts) sits under "
                       "the ~0.05 panel floor. 50-vs-800 (~17 pts) is the only "
                       "contrast with adequate signal. A 200 arm is deliberately "
                       "NOT generated yet; it is a curve-shape arm for later."),
    "hash_files": [
        "models/expert_iter_v2/teacher.pt",
        "agents/gregory.py",
        "agents/deterministics.py",
        "agents/mcts.py",
        "agents/neural_net_agent_3.py",
        "agents/agent_base.py",
        "engine/rules.py",
        "scripts/train_student_offline.py",
        "scripts/pocket_challenge.py",
        "scripts/benchmark_suite.py",
        "tools/make_distill_corpus.py",
        "tools/teacher_sim_ladder.py",
    ],
    "corpus": {
        "source": "models/corpus_gen22",
        "sample_seed": 20260727,
        "max_shards": 500,
        "positions": 50000,
        "sampling": "uniform random, natural phase distribution (NOT stratified)",
        "sims_arms": [50, 800],
        "dirichlet": False,
        "c_puct": 1.5,
        "policy_target": "full MCTS visit distribution (not top move)",
        "value_target": "original corpus outcome z, IDENTICAL across arms",
    },
    "student": {
        "arch": "squeeze",
        "arch_detail": "conv=[56]*4, fc=[256], head_squeeze=2, ~172,389 params",
        "init_seeds": [11, 22, 33],
        "data_seeds": [11, 22, 33],
        "pairing": ("For each seed the two arms share init weights AND batch "
                    "order; only the policy targets differ. Three seeds per arm "
                    "so training variance is measured, not assumed."),
        "steps": 40000,
        "batch_size": 512,
        "lr": 2e-3,
        "lr_min": 1e-4,
        "lr_half_life_steps": 15000,
        "value_coef": 1.0,
    },
    "evaluation": {
        "primary": "student_800 vs student_50 head to head",
        "primary_games": 800,
        "primary_note": ("Paired openings, colors swapped. Extend beyond 800 only "
                         "if the CI still materially overlaps 0.5. 800 games is "
                         "the known resolution scale from the teacher ladder, "
                         "where neither 400-game block separated alone."),
        "external_ladder": {
            "random": 4401,
            "winblock": 3301,
            "gregory_d3": 8801,
            "gregory_d4": 8802,
        },
        "external_games": 300,
        "elo": "NOT a primary result; internal Elo is self-play inflated",
        "success_criterion": "student_800 > student_50, replicated and separated",
        "explicitly_not_required": ("The student gap need NOT approach the "
                                    "teacher's ~17-point gap. The question is "
                                    "whether stronger targets transfer AT ALL and "
                                    "how much survives compression."),
    },
    "secondary": [
        "accuracy on the 50-vs-800 disagreement subset",
        "agreement with the 800-sim policy",
        "breakout by phase, branching factor, tactical status, teacher value gap",
        ("whether gains concentrate where adjudication says the 800-sim move is "
         "better -- CAVEAT: RESULT_SEARCH_DISAGREEMENT.md found the adjudicator "
         "cannot resolve this (independent referee at chance, same-net signals "
         "contradictory), so treat this item as exploratory, not confirmatory"),
    ],
    "decision_rules": {
        "clear_win": "run the full equal-example experiment; add 200 as a curve arm",
        "directional_unresolved": "add evaluation games first, do NOT regenerate corpora",
        "flat_across_seeds": "investigate student capacity, target weighting, optimization",
        "reversal": ("check whether higher-sim policies are softer/noisier/harder "
                     "to fit for a fixed student"),
    },
}

SPECS = {"distill_pilot": DISTILL_PILOT}


def build_manifest(spec):
    return {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_state(),
        "spec": spec,
        "file_hashes": {p: sha256_file(p) for p in spec["hash_files"]},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="distill_pilot", choices=list(SPECS))
    ap.add_argument("--out", default="EXPERIMENT_DISTILL_PILOT.json")
    ap.add_argument("--verify", default="",
                    help="path of an existing manifest to re-check instead of writing")
    args = ap.parse_args()

    spec = SPECS[args.spec]

    if args.verify:
        with open(args.verify, encoding="utf-8") as fh:
            old = json.load(fh)
        now = build_manifest(spec)
        drift = []
        for path, want in old["file_hashes"].items():
            got = now["file_hashes"].get(path)
            if got != want:
                drift.append(f"  {path}\n    frozen {want}\n    now    {got}")
        if old["git"]["commit"] != now["git"]["commit"]:
            print(f"[!] git commit moved: {old['git']['commit'][:12]} -> "
                  f"{now['git']['commit'][:12]} (expected as work proceeds)")
        if drift:
            print("[X] FROZEN FILES CHANGED since the manifest was written:")
            print("\n".join(drift))
            raise SystemExit(1)
        print(f"[OK] all {len(old['file_hashes'])} frozen files unchanged "
              f"(frozen {old['frozen_at']})")
        return

    if os.path.exists(args.out):
        raise SystemExit(f"[X] {args.out} already exists. Refusing to overwrite a "
                         f"frozen manifest -- that is the whole point. Delete it "
                         f"deliberately if the experiment is genuinely being redefined.")

    manifest = build_manifest(args.out and spec)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    missing = [p for p, h in manifest["file_hashes"].items() if h is None]
    print(f"froze {len(manifest['file_hashes'])} files at commit "
          f"{manifest['git']['commit'][:12]}"
          f"{' (DIRTY TREE)' if manifest['git']['dirty'] else ''}")
    if missing:
        print(f"[!] {len(missing)} declared files do not exist: {missing}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
