"""Frozen, named engine configurations -- the anti-drift layer for the arena.

WHY THIS EXISTS. "The original engine" and "the final engine" are claims about
specific behaviour, and both were measured once. If either is reconstructed
later from whatever the defaults happen to be at that moment, every comparison
against the 0.7229 baseline silently stops meaning anything -- and nothing
crashes to tell you. The same hazard is worse for the anchor ladder, because an
anchor that drifts moves the ruler under every candidate measured against it.

So a named engine here pins EVERY resolved parameter, and is then checked
against a stored fingerprint. That is the important part: a spec string alone
would still inherit `wave`, `c_puct`, `reserve_ms`, `min_sims`,
`deadline_margin`, virtual-loss magnitude and the `_MIN_WAVES` floor from code
defaults. The fingerprint covers those too, so changing a DEFAULT -- not just a
flag -- trips the guard.

THREE LAYERS, each catching a different kind of drift:

  1. Resolved-config fingerprint. Catches a changed default or a changed spec.
     Hard failure, always.
  2. Checkpoint sha256. Catches a retrained or overwritten .pt at the same path.
     Hard failure, always.
  3. Engine source sha256. Catches an edit to the search itself, which no
     config fingerprint can see. Hard failure when the engine is being used as
     an ANCHOR, warning otherwise -- because a candidate is *expected* to change
     the search, and an anchor is precisely the thing that must not.

Layer 3 cannot prevent drift, only make it impossible to miss. The recovery
path is a git tag: every frozen configuration here is reproducible from
`arena-1s-baseline`, so a drifted anchor can always be re-run from that tree.

    python -m tools.engine_registry                 # list, and verify
    python -m tools.engine_registry --freeze        # emit fingerprints to paste
    python -m tools.engine_registry --env           # the environment baseline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

# ---------------------------------------------------------------------------
# Frozen 2026-07-30, at the commit tagged arena-1s-baseline.
# ---------------------------------------------------------------------------

BASELINE_TAG = "arena-1s-baseline"
FROZEN_ON = "2026-07-30"

# The requirement is a property of the DEPLOYMENT target, not of a player, so
# it lives with the registry rather than with any one engine. Ladder rungs above
# 1000 ms are evaluation-only and exempt -- see LADDER_EXEMPT below.
REQUIREMENT = {
    "budget_ms": 1000,
    "p99_ms": 1000.0,
    "max_ms": 1250.0,
    "frozen": "2026-07-28, before any candidate was benchmarked",
}

# Frozen seed plan. Openings come from `scripts.expert_iter._eval_openings(n,
# seed)` and colours are swapped within each opening, so a seed IS the opening
# set. Separate namespaces because the same fixed openings get reused across
# many experiments, and a tuning sweep that eliminates on the same positions it
# is later confirmed on will overfit them -- cheap to avoid, invisible if not.
SEEDS = {
    "headline": 6100,   # the original vs final comparison; already published
    "ladder": 6200,     # ordering validation between adjacent rungs
    "anchor": 6300,     # a candidate measured against a frozen anchor
    "tune": 6400,       # elimination rounds of a parameter sweep
    "confirm": 6500,    # held out: the powered final between sweep survivors
}

# Files whose bytes define how a search plays. An anchor built while any of
# these differs from the frozen hash is not the anchor that was measured.
ENGINE_SOURCES = {
    "agents/mcts.py":
        "14c8b8a6f9166f5c1bc35f0b52b5b20529d4f8f393a77a0474a9de26bd5aafdc",
    "agents/agent_base.py":
        "31bf86ea68e25a5bf60f98c4fba42db9ba4193cb4b4143ac673d5f0a74f50eda",
    "agents/neural_net_agent_3.py":
        "64e7af58ae3e5cda071391f020ef080a9655bd7b1325bac95bb999ae2e348aed",
    "engine/rules.py":
        "88c326e96a102752dfd14a341d4041049fe2399581769fdefcccf04c55d988e2",
    "engine/game.py":
        "212c2069ebb4a985e45c1201da6a49f18990ac6048edd1b09b5de2eddc2b6231",
}

CHECKPOINTS = {
    "models/expert_iter_v2/teacher.pt":
        "cfef6febd4a430368d6b9864fd4ae9e9c52d5be7cc4f560ca3723498d668d5ba",
    "models/pocket_candidate/squeeze_pocket.pt":
        "b028d3499eca8b1049c5cdbe0a6deed2f056851afad68fe0858ca778af09123b",
    "models/ab_arch/plain.pt":
        "02a90b8364885ab595a5f11a7c14ce198543a059a7b625a2e6539593be9e5908",
}

GEN22 = "models/expert_iter_v2/teacher.pt"
POCKET = "models/pocket_candidate/squeeze_pocket.pt"
MIDSIZE = "models/ab_arch/plain.pt"

# The promoted engine, one place. Every rung of the ladder and every candidate
# baseline is this dict plus a budget, so "the final engine" cannot fork.
_FINAL = {"wave": "8", "cpuct": "1.5", "solve": "1", "reuse": "1", "bexp": "1",
          "maxsims": "200000"}

# reserve_ms defaults to max(5, 2% of budget) inside MCTS. Pinned explicitly at
# every rung so the ladder does not silently re-derive it if that rule changes.
_RESERVE = {250: "5", 500: "10", 1000: "20", 2000: "40", 4000: "80"}


def _engine(ckpt, arch, ms, name, **over):
    o = dict(_FINAL)
    o.update(over)
    o.update({"ckpt": ckpt, "arch": arch, "ms": str(ms), "name": name,
              "reserve": _RESERVE[ms]})
    return o


# sims=0 means the network alone -- masked policy argmax, no tree. Only the
# four options that can affect it are pinned; wave, reserve and bexp would be
# decoration, and pinning dead knobs invites the belief that they did something.
RAW_PINNED = {"ckpt", "arch", "sims", "name"}


def _raw(ckpt, arch, name):
    return {"ckpt": ckpt, "arch": arch, "sims": "0", "name": name}


def is_raw(name):
    return ENGINES[name].get("sims") == "0"


ENGINES = {
    # -- the two configurations the 0.7229 headline was measured between ------
    # "original" is pinned, not reconstructed. It is rebuild-every-move plus
    # per-leaf expansion, and it is the only thing the combined delta means.
    "original": _engine(GEN22, "arena22", 1000, "original",
                        reuse="0", bexp="0"),
    "final": _engine(GEN22, "arena22", 1000, "final"),

    # -- the anchor ladder: same engine, same net, budget is the only knob ----
    # Shares the training gene pool by construction. That is accepted: the
    # purpose is a stronger DETERMINISTIC reference produced by a known engine,
    # after gregory(d4) saturated at 1.0000 and stopped resolving anything.
    "anchor_A": _engine(GEN22, "arena22", 250, "anchor_A_250ms"),
    "anchor_B": _engine(GEN22, "arena22", 500, "anchor_B_500ms"),
    "anchor_C": _engine(GEN22, "arena22", 2000, "anchor_C_2000ms"),
    "anchor_D": _engine(GEN22, "arena22", 4000, "anchor_D_4000ms"),

    # -- model-size arms, all at the deployment budget -----------------------
    "pocket": _engine(POCKET, "squeeze", 1000, "pocket_172k"),
    "midsize": _engine(MIDSIZE, "plain", 1000, "midsize_921k"),

    # -- the networks alone, no search ---------------------------------------
    # Without these a model-size result is uninterpretable: if the small net
    # wins at 1,000 ms you cannot tell whether the network is better or whether
    # it merely bought more search, and those imply opposite next moves.
    "gen22_raw": _raw(GEN22, "arena22", "gen22_raw"),
    "pocket_raw": _raw(POCKET, "squeeze", "pocket_raw"),
    "midsize_raw": _raw(MIDSIZE, "plain", "midsize_raw"),
}

# Evaluation-only rungs. Exempt from the deployment latency requirement by
# design -- their job is to be a hard, fixed opponent, not to ship.
LADDER_EXEMPT = {"anchor_C", "anchor_D"}

# Engines whose implementation must not move. Building one of these against
# drifted sources is a hard failure unless explicitly overridden.
ANCHOR_ROLES = {"anchor_A", "anchor_B", "anchor_C", "anchor_D"}

# Resolved-config fingerprints, emitted by --freeze. An empty dict means the
# registry has never been frozen and verification cannot run.
FINGERPRINTS = {
    "original": "fe26cffd58d2290d",
    "final": "cfb14702454ac6df",
    "anchor_A": "cf1ff7cb2b932e75",
    "anchor_B": "f70fa9f070360fc8",
    "anchor_C": "5b93848f100a9c7f",
    "anchor_D": "dbc1514fb6b28f45",
    "pocket": "036f17c9aa644aad",
    "midsize": "5f38b819e7037640",
    "gen22_raw": "fc29c23a40fc3184",
    "pocket_raw": "ee530ad9254ab7a0",
    "midsize_raw": "1da5604204f35072",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spec_of(name):
    """The frozen spec string for a named engine."""
    if name not in ENGINES:
        raise SystemExit(f"[X] unknown engine '{name}'. "
                         f"known: {sorted(ENGINES)}")
    return ",".join(f"{k}={v}" for k, v in sorted(ENGINES[name].items()))


def resolved_config(player):
    """Everything about a built player that can change how it plays.

    Deliberately wider than the spec string: min_sims, deadline_margin, the
    virtual-loss magnitude and the _MIN_WAVES floor are code defaults that no
    spec pins, and any of them moving changes the engine.
    """
    m = player.mcts
    return {
        "ckpt": player.ckpt,
        "ckpt_sha256": sha256_file(player.ckpt),
        "arch": player.arch,
        "params": player.net_info["params"],
        "value_tanh": player.net_info["value_tanh"],
        "time_budget_ms": m.time_budget_ms,
        "n_sims": None if m.time_budget_ms else m.n_sims,
        "max_sims": m.max_sims,
        "min_sims": m.min_sims,
        "deadline_margin": m.deadline_margin,
        "reserve_ms": m.reserve_ms,
        "wave_size": m.wave_size,
        "c_puct": m.c_puct,
        "solve": m.solve,
        "batched_expand": m.batched_expand,
        "add_dirichlet": m.add_dirichlet,
        "virtual_loss": type(m)._VL,
        "min_waves": type(m)._MIN_WAVES,
        "tree_reuse": player.reuse,
    }


def fingerprint(cfg):
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("ascii")).hexdigest()[:16]


def check_sources(strict):
    """Compare the engine implementation against its frozen bytes."""
    drift = [p for p, want in ENGINE_SOURCES.items()
             if not os.path.isfile(p) or sha256_file(p) != want]
    if not drift:
        return []
    msg = (f"engine source drift in {len(drift)} file(s): {sorted(drift)}\n"
           f"    The frozen configurations were measured against different "
           f"bytes.\n"
           f"    Re-run from the tag: git worktree add ../uttt-anchor "
           f"{BASELINE_TAG}")
    if strict:
        raise SystemExit(f"[X] {msg}\n"
                         f"    Override only if you know why: "
                         f"--allow-anchor-drift")
    print(f"[!] {msg}")
    return sorted(drift)


def verify(name, player, strict_sources=None):
    """Assert a built player IS the frozen engine. Returns a provenance dict."""
    if strict_sources is None:
        strict_sources = name in ANCHOR_ROLES
    cfg = resolved_config(player)

    want_ck = CHECKPOINTS.get(cfg["ckpt"])
    if want_ck and cfg["ckpt_sha256"] != want_ck:
        raise SystemExit(
            f"[X] engine '{name}': checkpoint {cfg['ckpt']} has changed.\n"
            f"    frozen {want_ck}\n    found  {cfg['ckpt_sha256']}")

    got = fingerprint(cfg)
    want = FINGERPRINTS.get(name)
    if want is None:
        raise SystemExit(f"[X] engine '{name}' has no frozen fingerprint. "
                         f"Run: python -m tools.engine_registry --freeze")
    if got != want:
        raise SystemExit(
            f"[X] engine '{name}' has DRIFTED from its frozen configuration.\n"
            f"    frozen fingerprint {want}\n    built  fingerprint {got}\n"
            f"    resolved: {json.dumps(cfg, indent=6, sort_keys=True)}\n"
            f"    A default or a spec changed. The comparison against the "
            f"frozen baseline is not valid until this is reconciled.")

    drift = check_sources(strict_sources)
    return {"engine": name, "fingerprint": got, "verified": True,
            "source_drift": drift, "baseline_tag": BASELINE_TAG,
            "resolved": cfg}


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def environment():
    """The measurement environment. Recorded, never gated.

    Hardware and driver changes do not invalidate a frozen CONFIGURATION -- they
    invalidate a frozen LATENCY, which is exactly why the requirement is
    re-measured by the regression benchmark rather than trusted from a file.
    """
    import torch
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "git_head": git_head(),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        try:
            env["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader"], text=True).strip().splitlines()[0]
        except Exception:
            env["driver"] = None
    return env


# The environment the frozen numbers were measured on. Compared, not enforced.
FROZEN_ENV = {
    "python": "3.11.9",
    "torch": "2.7.1+cu128",
    "cuda_build": "12.8",
    "cudnn": 90701,
    "gpu": "NVIDIA GeForce RTX 3080",
    "gpu_capability": [8, 6],
    "driver": "591.74",
    "platform": "Windows-10-10.0.19045-SP0",
}


def env_drift():
    cur = environment()
    return {k: {"frozen": v, "now": cur.get(k)}
            for k, v in FROZEN_ENV.items() if cur.get(k) != v}


def _build(name, device="cpu"):
    from tools.arena_1s import TimedPlayer
    return TimedPlayer(spec_of(name), device)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true",
                    help="rebuild every engine and emit the FINGERPRINTS block")
    ap.add_argument("--env", action="store_true",
                    help="print the environment and any drift from frozen")
    ap.add_argument("--device", default="cpu",
                    help="cpu is enough: a fingerprint is configuration, and "
                         "configuration does not depend on the device")
    args = ap.parse_args()

    if args.env:
        print(json.dumps(environment(), indent=2))
        drift = env_drift()
        if drift:
            print("\n[!] environment differs from the frozen baseline:")
            for k, v in drift.items():
                print(f"    {k}: frozen {v['frozen']} -> now {v['now']}")
            print("    Configurations are still valid; LATENCY numbers are "
                  "not. Re-run tools.regress_engine.")
        else:
            print("\n[OK] environment matches the frozen baseline")
        return

    if args.freeze:
        print("FINGERPRINTS = {")
        for name in ENGINES:
            cfg = resolved_config(_build(name, args.device))
            print(f'    "{name}": "{fingerprint(cfg)}",')
        print("}")
        return

    print(f"frozen {FROZEN_ON} at tag {BASELINE_TAG}\n")
    check_sources(strict=False)
    bad = 0
    for name in ENGINES:
        player = _build(name, args.device)
        cfg = resolved_config(player)
        got, want = fingerprint(cfg), FINGERPRINTS.get(name)
        ok = got == want
        bad += not ok
        role = "anchor" if name in ANCHOR_ROLES else "candidate"
        exempt = "  (latency-exempt)" if name in LADDER_EXEMPT else ""
        budget = (f"{cfg['time_budget_ms']:.0f} ms" if cfg["time_budget_ms"]
                  else f"{cfg['n_sims']} sims")
        print(f"  [{'OK' if ok else 'X '}] {name:10s} {role:9s} "
              f"{budget:>8s}  {cfg['params']:>9,} params  "
              f"reuse={int(cfg['tree_reuse'])} bexp={int(cfg['batched_expand'])} "
              f"cpuct={cfg['c_puct']}{exempt}")
        if not ok:
            print(f"       frozen {want}  built {got}")
    print(f"\n{len(ENGINES) - bad}/{len(ENGINES)} engines match their frozen "
          f"fingerprint")
    if bad:
        raise SystemExit("[X] registry verification FAILED")


if __name__ == "__main__":
    main()
