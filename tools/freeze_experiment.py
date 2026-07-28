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

SOLVED_PILOT = {
    "name": "solved_pilot_50_vs_800_solve_on",
    "question": ("Does solved-node propagation shrink the 800-target penalty "
                 "measured in RESULT_DISTILL_PILOT.md? Same 50,000 frozen "
                 "positions, same student, same panel; the ONLY change is that "
                 "the teacher's search proves forced results and stops "
                 "sampling them."),
    "why": ("The pilot's mechanism was PUCT dilution: at 800 sims the teacher "
            "put a visit on every legal move of a mate-in-1 and kept 0.693 mass "
            "on the win, vs 0.825 at 50 sims. If that dilution is the dominant "
            "causal path, removing it should move the student result."),
    "relation_to_previous": {
        "frozen_manifest": "EXPERIMENT_DISTILL_PILOT.json",
        "drifted_files": ["agents/mcts.py", "tools/make_distill_corpus.py",
                          "tools/teacher_sim_ladder.py"],
        "drift_is_expected": True,
        "why_the_comparison_still_holds": (
            "Those three files changed, so the OLD manifest's --verify now "
            "reports drift and that is correct -- it must not be re-frozen. The "
            "comparison survives on a stronger guarantee than a file hash: with "
            "solve=False the search reproduces every one of the frozen pilot's "
            "targets BIT-FOR-BIT, verified on all measured positions at both 50 "
            "and 800 sims by tools/measure_solved_targets.py. Behaviour of the "
            "default path is proven unchanged by output, not asserted by hash."),
    },
    "hash_files": [
        "models/expert_iter_v2/teacher.pt",
        "agents/gregory.py",
        "agents/deterministics.py",
        "agents/mcts.py",
        "agents/neural_net_agent_3.py",
        "agents/agent_base.py",
        "engine/rules.py",
        "scripts/train_student_offline.py",
        "tools/make_distill_corpus.py",
        "tools/measure_solved_targets.py",
        "tools/teacher_sim_ladder.py",
        "tools/eval_distill_pilot.py",
    ],
    "corpus": {
        "source": "models/corpus_gen22",
        "sample_seed": 20260727,
        "max_shards": 500,
        "positions": 50000,
        "expect_x_sha256": ("c0d695adca04e6ca1996474070ef77b89f026475020b67d"
                            "7293471a432211d1e"),
        "expect_x_sha256_note": ("the frozen pilot's shared planes hash. "
                                 "make_distill_corpus --expect-x-sha256 fails "
                                 "the run unless it matches, so 'same "
                                 "positions' is proven, not inferred from "
                                 "matching seeds."),
        "sims_arms": [50, 800],
        "solved_node_propagation": True,
        "dirichlet": False,
        "c_puct": 1.5,
    },
    "student": {
        "arch": "squeeze",
        "init_seeds": [11, 22, 33],
        "data_seeds": [11, 22, 33],
        "steps": 40000,
        "batch_size": 512,
        "lr": 2e-3,
        "lr_min": 1e-4,
        "lr_half_life_steps": 15000,
        "value_coef": 1.0,
        "unchanged_from": "EXPERIMENT_DISTILL_PILOT.json",
        "why_three_seeds": (
            "One seed was authorised as a mechanistic read, but the pilot "
            "measured an across-seed spread of 0.132 at this student size -- "
            "larger than the effect being looked for. Training costs ~4 min per "
            "student, so three seeds is ~25 extra minutes and is what makes the "
            "result comparable to the published pooled 0.4108 rather than to a "
            "single noisy draw."),
    },
    "evaluation": {
        "primary": "student_800_solve vs student_50_solve head to head",
        "primary_games": 800,
        "pooled_games": 2400,
        "external_ladder": {"random": 4401, "winblock": 3301,
                            "gregory_d3": 8801, "gregory_d4": 8802},
        "external_games": 300,
        "panel_unchanged_from": "EXPERIMENT_DISTILL_PILOT.json",
    },
    # Stated before any solve target exists, so 'materially' cannot be chosen
    # after seeing the number.
    "prereg_threshold": {
        "baseline_pooled_score": 0.4108,
        "baseline_per_seed": {"11": 0.3331, "22": 0.4650, "33": 0.4344},
        "baseline_penalty": 0.0892,
        "materially_shrinks": (
            "pooled score >= 0.4554, i.e. at least HALF the 0.0892 penalty "
            "recovered, AND the 95% interval on the delta against 0.4108 "
            "excludes zero"),
        "resolution_check": (
            "each pooled score has SE ~0.009 at n=2400, so the delta has SE "
            "~0.013 and a 0.0446 shift separates at ~3.4 SE. The design can "
            "answer its own question."),
        "gating": ("Do NOT run this at all unless the measurement pass shows: "
                   "forced-win dilution eliminated or sharply reduced; "
                   "non-tactical targets not regressed; teacher strength "
                   "maintained or improved on the ladder; and proof correction "
                   "reaching enough production-relevant positions to plausibly "
                   "alter training."),
    },
    "decision_rules": {
        "reversal_gone_or_flipped": "solved propagation addressed a major causal mechanism",
        "forced_improves_reversal_remains": ("dominant problem is non-tactical "
                                             "churn; go to target extraction or "
                                             "weighting, NOT optimizer work"),
        "strength_up_but_slower": ("keep solving as a correctness feature and "
                                   "re-evaluate under EQUAL-TIME rather than "
                                   "equal-simulation budgets"),
        "coverage_negligible_outside_tactics": ("do not spend another training "
                                                "run on it after this pilot"),
    },
}

SPECS = {"distill_pilot": DISTILL_PILOT, "solved_pilot": SOLVED_PILOT}


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
