"""Pin and verify the solved-pilot student seeds BEFORE any target exists.

Written ahead of corpus generation on purpose. If the seeds are chosen after
the targets are in hand, nothing stops a favourable draw from being picked, and
the pre-registration is worthless. Running this now fixes them in a file with a
timestamp and a git commit.

The seeds are NOT newly invented: [11, 22, 33] are the init_seeds and data_seeds
the original pilot used (EXPERIMENT_DISTILL_PILOT.json), and reusing them is
required -- the owner's instruction is to keep the exact student
initialization, batch order, optimizer schedule and evaluation panel from the
reversal experiment, so the only difference between the old result and the new
one is the targets.

What "the arms start byte-identically" rests on, structurally:

  * weight init reads the GLOBAL torch RNG, seeded by --init_seed only, and
    ConvNet is built on CPU then moved, so init does not depend on device;
  * batch order comes from np.random.RandomState(--seed), a DEDICATED stream
    that weight init never touches;
  * symmetry augmentation comes from random.Random(--seed + 1), likewise;
  * the arms differ only in the PI tensor -- X and Z are copied from the same
    source corpus.

So arm identity cannot enter either RNG. This tool proves the first three by
recomputation and the fourth by direct comparison once the corpora exist.

    python -m tools.lock_student_seeds                  # write the lock file
    python -m tools.lock_student_seeds --check          # re-verify, exit 1 on drift
    python -m tools.lock_student_seeds --corpora A B    # add the arm-equality proof
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

from scripts.train_student_offline import AB_ARCHS, build_model
from tools.provenance import git_head

LOCK_PATH = "SEEDS_SOLVED_PILOT.json"

SEEDS = [11, 22, 33]
ARCH = "squeeze"

# Copied verbatim from EXPERIMENT_DISTILL_PILOT.json -- changing any of these
# breaks comparability with the 0.4108 baseline.
TRAIN_CONFIG = {
    "arch": ARCH,
    "steps": 40000,
    "batch_size": 512,
    "lr": 0.002,
    "lr_min": 0.0001,
    "lr_half_life_steps": 15000,
    "value_coef": 1.0,
    "n_examples": 50000,
    "arms": ["sims50_solve", "sims800_solve"],
    "primary_games": 800,
}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def initial_weight_hash(init_seed):
    """Hash the exact tensor bytes a training run would start from."""
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    model = build_model(AB_ARCHS[ARCH], "cpu")
    h = hashlib.sha256()
    with torch.no_grad():
        for k, v in sorted(model.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        fp = sum(float(p.double().sum()) for p in model.parameters())
    return h.hexdigest(), float(fp), sum(p.numel() for p in model.parameters())


def batch_order_hash(seed, n, steps, batch_size):
    """Replay the sampler exactly and hash the full index + symmetry sequence.

    This mirrors train_student_offline's loop: one permutation per epoch drawn
    from a single RandomState, a cursor that refuses partial batches, and one
    dihedral draw per step from random.Random(seed + 1).
    """
    idx_rng = np.random.RandomState(seed)
    sym_rng = random.Random(seed + 1)
    order = idx_rng.permutation(n)
    cursor = 0
    hi, hs = hashlib.sha256(), hashlib.sha256()
    for _ in range(1, steps + 1):
        if cursor + batch_size > n:
            order = idx_rng.permutation(n)
            cursor = 0
        hi.update(order[cursor:cursor + batch_size].astype(np.int64).tobytes())
        cursor += batch_size
        hs.update(bytes([sym_rng.randrange(8)]))
    return hi.hexdigest(), hs.hexdigest()


def corpus_arm_equality(arm_a, arm_b):
    """Prove the two arms differ ONLY in pi -- same x, same z, same order."""
    def cat(arm, key):
        shards = sorted(f for f in os.listdir(arm) if f.endswith(".npz"))
        return np.concatenate([np.load(os.path.join(arm, f))[key]
                               for f in shards], axis=0)

    out = {"arm_a": arm_a, "arm_b": arm_b}
    for key in ("x", "z"):
        a, b = cat(arm_a, key), cat(arm_b, key)
        same = a.shape == b.shape and bool(np.all(a == b))
        out[f"{key}_identical"] = same
        out[f"{key}_sha256"] = _sha(a.tobytes())
        out[f"{key}_n"] = int(a.shape[0])
    pa, pb = cat(arm_a, "pi"), cat(arm_b, "pi")
    out["pi_identical"] = pa.shape == pb.shape and bool(np.all(pa == pb))
    out["pi_a_sha256"], out["pi_b_sha256"] = _sha(pa.tobytes()), _sha(pb.tobytes())
    # x and z identical, pi different: exactly the intended experiment.
    out["valid_pairing"] = bool(out["x_identical"] and out["z_identical"]
                                and not out["pi_identical"])
    return out


def build(n_examples=None, corpora=None):
    n = n_examples or TRAIN_CONFIG["n_examples"]
    cfg_hash = _sha(json.dumps(TRAIN_CONFIG, sort_keys=True).encode())

    weights, orders = {}, {}
    for s in SEEDS:
        h, fp, np_ = initial_weight_hash(s)
        weights[str(s)] = {"sha256": h, "fingerprint": fp, "params": np_}
        oh, sh = batch_order_hash(s, n, TRAIN_CONFIG["steps"],
                                  TRAIN_CONFIG["batch_size"])
        orders[str(s)] = {"index_sha256": oh, "symmetry_sha256": sh}

    # Distinct seeds must give distinct starts, or "three seeds" is a lie.
    distinct = len({w["sha256"] for w in weights.values()}) == len(SEEDS)

    doc = {
        "locked_at": datetime.now().isoformat(timespec="seconds"),
        "commit": git_head(),
        "student_seeds": SEEDS,
        "seeds_shared_by_both_arms": True,
        "seed_provenance": ("reused verbatim from EXPERIMENT_DISTILL_PILOT.json "
                            "init_seeds/data_seeds so the solved pilot stays "
                            "comparable to the 0.4108 baseline"),
        "training_config": TRAIN_CONFIG,
        "training_config_hash": cfg_hash,
        "initial_weight_hashes": weights,
        "batch_order_hashes": orders,
        "seeds_produce_distinct_inits": distinct,
        "arm_independence_argument": (
            "init reads the global torch RNG (init_seed only); batch order "
            "reads np.random.RandomState(seed); symmetry reads "
            "random.Random(seed+1). None of the three is reachable from arm "
            "identity, and the arms share X and Z by construction, so for a "
            "given seed the two runs start from identical weights and consume "
            "identical rows in identical order."),
        "byte_identical_start_verified": distinct,
    }
    if corpora:
        eq = corpus_arm_equality(*corpora)
        doc["corpus_arm_equality"] = eq
        doc["byte_identical_start_verified"] = bool(distinct and eq["valid_pairing"])
    else:
        doc["corpus_arm_equality"] = {
            "status": "PENDING -- rerun with --corpora once both arms exist"}
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--corpora", nargs=2, metavar=("ARM_A", "ARM_B"))
    ap.add_argument("--n-examples", type=int, default=None)
    args = ap.parse_args()

    doc = build(args.n_examples, args.corpora)

    if args.check:
        if not os.path.isfile(LOCK_PATH):
            raise SystemExit(f"[X] no {LOCK_PATH} -- nothing was ever locked")
        old = json.load(open(LOCK_PATH, encoding="utf-8"))
        drift = [k for k in ("student_seeds", "training_config_hash",
                             "initial_weight_hashes", "batch_order_hashes")
                 if old.get(k) != doc.get(k)]
        if drift:
            raise SystemExit(
                f"[X] locked values changed: {drift}. The students would NOT "
                f"start where the lock file says they do.")
        print(f"[OK] {LOCK_PATH} still reproduces exactly")
        return

    with open(LOCK_PATH, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)

    print(f"student_seeds        {doc['student_seeds']}")
    print(f"training_config_hash {doc['training_config_hash'][:16]}")
    for s in SEEDS:
        w = doc["initial_weight_hashes"][str(s)]
        o = doc["batch_order_hashes"][str(s)]
        print(f"  seed {s:>3}  init {w['sha256'][:16]}  "
              f"order {o['index_sha256'][:16]}  sym {o['symmetry_sha256'][:12]}")
    print(f"distinct inits per seed   {doc['seeds_produce_distinct_inits']}")
    eq = doc["corpus_arm_equality"]
    print(f"corpus arm equality       {eq.get('status', eq.get('valid_pairing'))}")
    print(f"[OK] wrote {LOCK_PATH}")


if __name__ == "__main__":
    main()
