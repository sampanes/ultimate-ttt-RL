"""The single authority for the solved-pilot provenance block.

Every artifact this branch produces -- target measurement, teacher ladder,
distillation evaluation -- carries the same block, so a reader can establish
from any ONE file that solving did not perturb the default search path.

The central claim is machine-readable:

    default_path_behaviorally_identical = true

and it is not an assertion of intent. It is backed by an exact count of
solve=off targets that reproduce the frozen pilot corpus bit for bit.

Why this exists at all: adding solved-node propagation edited three files that
EXPERIMENT_DISTILL_PILOT.json hashes (agents/mcts.py, tools/make_distill_corpus.py,
tools/teacher_sim_ladder.py), so `--verify` on that manifest now FAILS. That
failure is expected and is deliberately NOT repaired -- re-freezing would
destroy the record of what the baseline actually ran. The manifest stays
failed, and the comparison rests on the parity count instead, which is the
stronger guarantee: a file hash proves the bytes did not move, parity proves
the OUTPUT did not move, which is the thing we actually care about.

    python -m tools.provenance                        # print the block
    python -m tools.provenance --stamp FILE [FILE..]  # merge it into artifacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import numpy as np

BASELINE_MANIFEST = "EXPERIMENT_DISTILL_PILOT.json"
PILOT_CORPUS = "models/distill_pilot"
MEASURE_OUT = "results/solved_targets"

# The three files that legitimately drifted when solving was added. Any OTHER
# file failing verification means something unrelated changed and the
# comparison is NOT safe.
EXPECTED_DRIFT = {
    "agents/mcts.py",
    "tools/make_distill_corpus.py",
    "tools/teacher_sim_ladder.py",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def git_dirty():
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        return None


def verify_manifest():
    """Re-hash the baseline manifest's files. Returns (status, detail)."""
    if not os.path.isfile(BASELINE_MANIFEST):
        return "MANIFEST_ABSENT", {}
    man = json.load(open(BASELINE_MANIFEST, encoding="utf-8"))
    drifted, absent = [], []
    for rel, want in man.get("file_hashes", {}).items():
        if not os.path.isfile(rel):
            absent.append(rel)
        elif sha256_file(rel) != want:
            drifted.append(rel)

    detail = {"drifted": sorted(drifted), "absent": sorted(absent),
              "expected_drift": sorted(EXPECTED_DRIFT)}
    if absent:
        return "FAILED_UNEXPECTED", detail
    if not drifted:
        return "PASSED", detail
    if set(drifted) == EXPECTED_DRIFT:
        # The intended state of the world for this branch.
        return "FAILED_EXPECTED", detail
    return "FAILED_UNEXPECTED", detail


def parity_from_cache(sims=50, measure_out=None):
    """Independently recount solve=off / pilot agreement from the npz cache.

    Deliberately does NOT trust what measure_solved_targets printed. It reloads
    the cached policies and the frozen pilot targets and compares them here, so
    the number in the provenance block is produced by different code than the
    number in the run log.
    """
    cache = os.path.join(measure_out or MEASURE_OUT, "policies.npz")
    arm = os.path.join(PILOT_CORPUS, f"sims{sims}", "data")
    idx = os.path.join(PILOT_CORPUS, "index.npz")
    if not (os.path.isfile(cache) and os.path.isdir(arm) and os.path.isfile(idx)):
        return {"status": "UNAVAILABLE", "count": None, "failures": None,
                "note": "measurement cache or pilot corpus not present yet"}

    z = np.load(cache)
    key = f"pi_{sims}_off"
    if key not in z:
        return {"status": "UNAVAILABLE", "count": None, "failures": None,
                "note": f"{key} not in {cache}"}
    mine = z[key]
    rows = z["rows"]

    # Reuse the measurement's own shard reader rather than re-guessing the
    # on-disk layout (the arm is data/shard_*.pt, not npz). Imported lazily:
    # measure_solved_targets imports THIS module, so a top-level import here
    # would be circular.
    from tools.measure_solved_targets import load_pilot
    try:
        _, pilot_pi = load_pilot(PILOT_CORPUS, sims)
    except SystemExit as e:
        return {"status": "UNAVAILABLE", "count": None, "failures": None,
                "note": str(e)}
    pilot = pilot_pi.numpy()
    if rows.max(initial=-1) >= len(pilot):
        return {"status": "UNAVAILABLE", "count": None, "failures": None,
                "note": "pilot corpus shorter than the sampled row indices"}
    theirs = pilot[rows]

    exact = np.all(mine == theirs, axis=1)
    return {"status": "VERIFIED", "count": int(exact.sum()),
            "failures": int((~exact).sum()), "total": int(len(exact)),
            "comparison": "bitwise float32 equality, all 81 entries per row",
            "recomputed_by": "tools.provenance (independent of the run log)"}


def build(sims=50, measure_out=None):
    man_status, man_detail = verify_manifest()
    baseline = {}
    if os.path.isfile(BASELINE_MANIFEST):
        baseline = json.load(open(BASELINE_MANIFEST, encoding="utf-8"))

    par = parity_from_cache(sims, measure_out)
    identical = par["status"] == "VERIFIED" and par["failures"] == 0

    return {
        "baseline_commit": baseline.get("git", {}).get("commit"),
        "solved_commit": git_head(),
        "working_tree_dirty": git_dirty(),
        "original_manifest_hash": (sha256_file(BASELINE_MANIFEST)
                                   if os.path.isfile(BASELINE_MANIFEST) else None),
        "manifest_verification_status": man_status,
        "manifest_verification_detail": man_detail,
        "manifest_policy": (
            "NOT re-frozen by design. The baseline manifest is left in its "
            "failing state so the record of what the 0.4108 baseline actually "
            "ran survives. Comparability rests on the parity count below."),
        "solve_off_full_parity_count": par["count"],
        "solve_off_full_parity_failures": par["failures"],
        "solve_off_full_parity_detail": par,
        "solve_off_parity_sims": sims,
        "default_path_behaviorally_identical": identical,
        "default_path_claim": (
            f"{par['count']} / {par.get('total')} solve=off targets at {sims} "
            f"simulations reproduce the frozen pilot corpus exactly"
            if identical else
            "UNPROVEN -- do not compare against the 0.4108 baseline"),
    }


def stamp(paths, sims=50):
    block = build(sims)
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        if not isinstance(doc, dict):
            doc = {"_data": doc}
        doc["provenance"] = block
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        print(f"  stamped {p}")
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", nargs="*", default=None,
                    help="JSON artifacts to merge the provenance block into")
    ap.add_argument("--sims", type=int, default=50,
                    help="which solve=off arm to verify parity on")
    args = ap.parse_args()

    block = stamp(args.stamp, args.sims) if args.stamp else build(args.sims)
    print(json.dumps(block, indent=2))

    if not block["default_path_behaviorally_identical"]:
        print("\n[!] default_path_behaviorally_identical = false -- parity is "
              "not yet established, so results carrying this block must NOT be "
              "compared against the frozen pilot baseline.")


if __name__ == "__main__":
    main()
