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
    "profile": 6600,    # instrumented runs; no score is ever read off these
    "expand": 6700,     # the CUDA sync-vs-transfer study, likewise unscored
    "kernels": 6800,    # the kernel/launch trace; structure only, no score
    "select": 6900,     # the post-graph tree re-profile; instrumented, no score
    # Held out for the native-selection strength match (#45a). The parity
    # oracle and the throughput study both ran on 6900, and although neither
    # reads a score off it, an effect-size match that reuses the openings its
    # own instrument was tuned on is exactly the cheap mistake this table
    # exists to prevent.
    "select_ab": 7000,
    # The 120-game effect-size match came back at 0.5708 [0.5285, 0.6131] --
    # already excluding parity, but it is the run that SIZED the effect, and
    # reading a verdict off it would be using one sample twice. A 240-game
    # confirmation on seed 7000 would not fix that either: the opening set is
    # taken in order, so the first 120 of it are the games already played.
    "select_confirm": 7100,
    # #46, the release-path work. Instrumented and unscored like `profile` and
    # `select`, and separate from them because the memory characterisation
    # plays real games and any opening set an instrument has been read off is
    # spent for scoring purposes.
    "release": 7200,
    # Held out for the strength match that follows the release change, if the
    # remeasured throughput makes one worth funding. Kept empty until then.
    "release_ab": 7300,
    # #47, the terminal-probe question. Instrumented and unscored like
    # `profile`, `select` and `release`. The ON/OFF ablation reads MOVE
    # DISAGREEMENT off these openings, which is a behaviour count and not a
    # score -- but it is still a readout, so it gets its own namespace rather
    # than spending one a future match might want.
    "probe": 7400,
    # Held out for the strength match that follows a probe change, if #47
    # produces one worth funding. Kept empty until then.
    "probe_ab": 7500,
}

# Files whose bytes define how a search plays. An anchor built while any of
# these differs from the frozen hash is not the anchor that was measured.
# agents/mcts.py was re-hashed 2026-08-09 for the CUDA-graph wave path (#41).
# The change adds an OFF-BY-DEFAULT branch and splits `_expand_wave` into the
# device half plus a shared `_expand_children`. The eager path's behaviour is
# unchanged, and that is checked rather than claimed: the frozen replicas in
# tools/test_profile_expand.py and tools/test_profile_kernels.py were written
# statement-for-statement against the PRE-change function and still require
# bit-identical children, priors and leaf values. Anchors were not re-measured
# because they do not play differently.
#
# agents/mcts.py was re-hashed AGAIN 2026-08-12 for native PUCT selection
# (#45a). The change is off by default and the off path is byte-for-byte the
# old code -- the mirror lives in `if self._mirror:` branches that duplicate
# each loop rather than adding a test inside it, precisely so the incumbent arm
# of the A/B does not get slower. The ONE unconditional cost is a `self.sel`
# store per node and a `sel is not None` load per selection, together about
# 0.1% of a move; it is stated here because "off by default" is a claim about
# behaviour, not about cost. Anchors were not re-measured: with select=0 the
# selection they run is the same `max()` over the same dict view.
#
# agents/mcts.py was re-hashed A THIRD TIME 2026-08-15 for deferred tree
# retirement (#46). Off by default, and the off path is the old code: the
# choice is one `if self.defer_release:` in `TreeReuseSearcher.search`, and
# `release()` itself -- the function every engine shares -- was NOT touched.
# There is one unconditional cost, and it is stated rather than glossed: a
# `stat_nodes_created += len(valid)` per expanded node, about 4,700 adds a
# move, which is what the retirement watermark is bounded with. Counting on the
# way OUT would have meant a per-node increment inside `release`, and that
# would have slowed every engine including the incumbent arm of the A/B.
# Anchors were not re-measured: with defer=0 they destroy the tree at exactly
# the same moment, by exactly the same walk.
ENGINE_SOURCES = {
    "agents/mcts.py":
        "9c34e122820c58c3164c49574ad51c200455687002caba265ba6ae7fd0523550",
    "agents/graph_wave.py":
        "913227d4a87925078d25dc188fe6a9058c8c37f81d7edd2db6e4712724ae6f77",
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
# graph="0" pins the CUDA-graph wave path OFF. It is stated rather than left
# to a code default because the registry's whole job is that a changed DEFAULT
# trips the guard, not only a changed flag. Adding it moved every fingerprint
# below on 2026-08-09; no engine's behaviour moved, because "0" is what they
# already did.
#
# THESE STAY "0" EVEN THOUGH THE PROMOTED ENGINE SETS ALL THREE TO "1". `_FINAL`
# is not "what we ship", it is the common base that `original`, `final`, the
# four anchor rungs and the superseded baselines are all built from, and those
# must keep playing exactly as they were measured. Promotion is recorded by
# `DEPLOYED`, not by moving the defaults underneath the ladder.
#
# select="0" pins native PUCT selection OFF, for the same reason and by the
# same rule: a changed DEFAULT must trip the guard, not only a changed flag.
# Adding it moved every fingerprint below again on 2026-08-12; no engine's
# behaviour moved, because "0" is what they already did.
#
# defer="0" pins deferred tree retirement OFF, third application of the same
# rule. Adding it moved every fingerprint below on 2026-08-15; no engine's
# behaviour moved, because "0" is what they already did.
_FINAL = {"wave": "8", "cpuct": "1.5", "solve": "1", "reuse": "1", "bexp": "1",
          "maxsims": "200000", "graph": "0", "select": "0", "defer": "0"}

# reserve_ms defaults to max(5, 2% of budget) inside MCTS. Pinned explicitly at
# every rung so the ladder does not silently re-derive it if that rule changes.
#
# THIS RESERVE ONLY COVERS CHUNK OVERRUN. It is held back from the SEARCH's own
# deadline, and the search honours it: over 5,517 measured moves the search
# never exceeded 981.1 ms against its 980 ms deadline. What it does not cover is
# the work the caller waits for OUTSIDE `MCTS.search` -- re-rooting, releasing
# the discarded tree, the argmax -- which reached 23.8 ms at p99 and pushed 106
# moves past the 1,000 ms requirement in the model-size decision match. See
# `_RESERVE_DEPLOY`.
_RESERVE = {250: "5", 500: "10", 1000: "20", 2000: "40", 4000: "80"}


def _engine(ckpt, arch, ms, name, **over):
    o = dict(_FINAL)
    o.update({"ckpt": ckpt, "arch": arch, "ms": str(ms), "name": name,
              "reserve": _RESERVE[ms]})
    o.update(over)          # a DECLARED override wins, reserve included
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
    # `pocket` is FROZEN AS MEASURED, latency failure included. It is the arm
    # that scored 0.5854 against `final`, and rewriting it to fix that failure
    # would silently detach the published number from the configuration that
    # produced it. The correction lives in a separate entry below.
    "pocket": _engine(POCKET, "squeeze", 1000, "pocket_172k"),
    "midsize": _engine(MIDSIZE, "plain", 1000, "midsize_921k"),

    # -- the latency-corrected deployment candidate ---------------------------
    # `pocket` wins on strength but misses the frozen p99 by 2.3 ms, and the
    # cause is not the search: over 5,517 moves the search never exceeded
    # 981.1 ms against its own 980 ms deadline, and the worst chunk on the 106
    # failing moves was 3.3 ms. All of the overshoot is caller-side tree work
    # -- `_adopt`, `release()` of the discarded tree -- which is inside the
    # move the requirement is written against but outside the interval the
    # search times itself over. It costs MORE here than for `final` for the
    # reason `pocket` is faster: 4,193 sims/move builds a 24% bigger tree to
    # walk. Raising the reserve is the configuration fix; moving that walk off
    # the critical path is tree-core work, not a knob.
    #
    # 35 ms covers the measured p99 overhead (23.8) plus p99 chunk overrun
    # (5.4) with room, and costs 1.5% of thinking time -- worth ~0.002 by the
    # anchor ladder against an 0.0854 edge.
    #
    # SUPERSEDED 2026-08-15 by `pocket_defer`, and deliberately left exactly as
    # it was measured. It stays in the registry because it is now the
    # HISTORICAL COMPARATOR: it was the deployment engine from 2026-07-30, it is
    # the B side of the graph, native-selection and deferred-retirement matches,
    # and it is what `c_puct = 1.5` was tuned on. A superseded baseline that
    # stops building is a published result that stops being checkable.
    "pocket_r35": _engine(POCKET, "squeeze", 1000, "pocket_172k_r35",
                          reserve="35"),

    # -- the CUDA-graph candidate (#41) --------------------------------------
    # `pocket_r35` with the wave's device sequence replayed from a captured
    # CUDA graph instead of issued kernel by kernel. The wave computes exactly
    # the same numbers -- bit-identical priors and leaf values over 320
    # distinct production waves, agents/test_graph_wave.py -- so this is not a
    # different player at a fixed simulation count. Under a clock it is: it
    # completes about 48% more network evaluations per move.
    #
    # THE RESERVE IS LARGER FOR THE SAME REASON `pocket_r35` NEEDED ONE. All of
    # the p99 risk is caller-side tree work -- `_adopt` and `release()` of the
    # discarded tree -- which is inside the move the requirement is written
    # against but outside the interval the search times. More search means a
    # bigger tree to walk, so the overhead grows with the speedup that caused
    # it. 50 ms is provisional and is checked by tools/regress_engine; it costs
    # 1.5% of thinking time against a 48% throughput gain.
    #
    # NOT PROMOTED. It is a candidate until it wins at equal wall clock.
    "pocket_graph": _engine(POCKET, "squeeze", 1000, "pocket_172k_graph",
                            reserve="50", graph="1"),

    # -- the native-selection candidate (#45a) --------------------------------
    # `pocket_graph` with PUCT selection done in C++ over a mirrored child
    # array instead of a `max()` over a dict view with a lambda key. #44
    # measured that scan at 193.85 ms/move -- 22.2% of the move and the largest
    # host term by 64% -- after the CUDA graph turned the engine host-bound
    # (59.2% host / 39.3% device).
    #
    # THE RESERVE IS 95 ms, AND THE FIRST DRAFT OF THIS COMMENT WAS WRONG. It
    # said 50 was enough because selection sits inside the search's own
    # deadline, which the search has never missed -- true, and beside the
    # point. `regress_engine` measured caller-side overhead at p99 78.99 ms
    # against the 50 ms reserve and failed the gate at p99 1027.9. The
    # mechanism is the same one that took `pocket` to 20, `pocket_r35` to 35
    # and `pocket_graph` to 50, only sharper: `release()` walks the discarded
    # tree OUTSIDE the search's deadline, and every throughput win makes that
    # tree bigger. 42.3% more simulations a move bought 84.7% more overhead.
    #
    # About 18 of those 79 ms are not the bigger tree. They are the mirror
    # itself: four extra Python objects per expanded node (a ChildArray and
    # three numpy columns) that release has to drop. That is a real cost of
    # this design and it is priced here rather than hidden in the reserve.
    #
    # 95 ms covers the measured need (overhead p99 78.99 + worst-chunk p99
    # 9.3 = 88.3) and costs 4.7% of thinking time, which the +17.6% in network
    # evaluations pays for several times over. It is provisional in exactly
    # the way 50 was: tools/regress_engine is what decides it.
    #
    # THIS ONE IS NOT A DIFFERENT PLAYER AT FIXED SIMULATIONS. The graph wave
    # was bit-identical in its OUTPUTS; this is stricter -- it must return the
    # same child index on every selection, which tools/select_parity checks
    # against the Python implementation on real searches and on fixtures. Under
    # a clock it is a different player for the same reason the graph was: more
    # search fits.
    #
    # NOT PROMOTED. It is a candidate until it wins at equal wall clock.
    "pocket_sel": _engine(POCKET, "squeeze", 1000, "pocket_172k_sel",
                          reserve="95", graph="1", select="1"),

    # -- deferred tree retirement (#46) ---------------------------------------
    # `pocket_sel` with the discarded tree DETACHED on the move path and
    # DESTROYED at the game boundary, instead of walked inside the move.
    #
    # This is the fourth engine in a row whose reserve was set by one mechanism,
    # and the first that removes the mechanism instead of paying for it. #46a
    # measured `release()` on `pocket_sel` at 20.4 ms a move, p99 53.3, against
    # a caller-side overhead p99 of 53.3 -- the walk is essentially the whole of
    # it. It is also pure per-node work (0.552 us/node, intercept -0.02 ms,
    # R^2 0.986), so there was nothing in it to optimise; the only lever was
    # not doing it here.
    #
    # THE RESERVE IS 20 ms, BACK TO WHERE `pocket` STARTED, and it is the
    # measurement that says so rather than optimism: with release removed the
    # caller-side overhead p99 is 0.06 ms and the worst-chunk p99 is 3.24, so
    # the tool's own prescription indicates 3.3. Twenty is six times that, it
    # is the family default at this budget, and tools/regress_engine is what
    # decides it. It hands 75 ms of the 95 back to the search, which is 8.3% of
    # the thinking time the engine has today.
    #
    # WHAT IT COSTS IS MEMORY, and that was measured before it was built rather
    # than discovered afterwards: 430 bytes a node, 1,148,422 nodes retired in
    # the worst of eight games, so 471 MB held at the end of it -- 1.4% of this
    # box and 2.6% of what was free. `TreeReuseSearcher.DEFAULT_WATERMARK`
    # bounds the pathological case at about 2.6x that and is the only thing
    # that can put the walk back on a move; `stats()["forced_drains"]` is
    # non-zero if it ever does.
    #
    # NOT A DIFFERENT PLAYER AT FIXED SIMULATIONS, in the same strong sense as
    # `pocket_sel`: the same nodes are built and the same statistics kept, and
    # only the moment of destruction moves. agents/test_mcts_timed.py requires
    # bit-identical visit policies across several plies of re-rooting.
    #
    # IT HAS WON AT EQUAL WALL CLOCK. 0.5625 [0.5273, 0.5977], W54/D162/L24
    # over 240 paired games against the DEPLOYED `pocket_r35` on the held-out
    # `release_ab` namespace -- inside the 0.5352-0.5744 band pre-registered
    # from throughput before the match ran. It also meets the latency
    # requirement more comfortably than the engine it beat (p99 980.2 against
    # 986.1, max 981.9 against 1003.4, 0 moves over budget against 1).
    #
    # What that match separates is the STACK against the deployed engine, not
    # the three changes from each other: `pocket_graph` alone already scored
    # 0.5458 against `pocket_r35`, and the increment to 0.5625 is inside the
    # noise on the difference.
    #
    # PROMOTED 2026-08-15. This is `DEPLOYED` -- see the block above ANCHOR_ROLES
    # for what the promotion does and does not claim, and
    # RESULT_DEFERRED_RETIREMENT.md for the evidence. Being the baseline buys it
    # one extra guard and no exemptions: source drift under it is now a hard
    # failure (STRICT_SOURCE_ROLES), and it is still latency-gated by
    # tools/regress_engine like every other deployment engine.
    "pocket_defer": _engine(POCKET, "squeeze", 1000, "pocket_172k_defer",
                            reserve="20", graph="1", select="1", defer="1"),

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

# ---------------------------------------------------------------------------
# THE DEPLOYMENT BASELINE
# ---------------------------------------------------------------------------
# One name, so "the deployed engine" is something the code can resolve instead
# of a claim in a comment that goes stale the day it changes. Anything that
# means "against what we actually ship" -- a profile, an ablation, the B side
# of an A/B -- should read this rather than hard-code an engine name. What must
# NOT read it is a published study: a reproduction command that says
# `--engine pocket_r35` has to keep building `pocket_r35` forever, so those
# stay literal on purpose.
#
# PROMOTED 2026-08-15, recorded at commit 0224669. The supported claim is one
# sentence and it is narrower than the set of changes it contains:
#
#     `pocket_defer` as a complete 1-second agent is stronger than
#     `pocket_r35` at equal wall clock.
#
# It does NOT say that native selection improves strength, and it does not say
# that deferred retirement improves strength. Those two arrived on top of a
# stack whose first member already beat the incumbent by itself (`pocket_graph`
# scored 0.5458 against `pocket_r35`), and the increment to 0.5625 is inside
# the noise on the difference. The stack is what was measured and the stack is
# what ships; nothing here licenses attributing the win to a part of it.
DEPLOYED = "pocket_defer"

# Superseded baselines, newest first. KEPT BUILDABLE rather than archived: a
# promotion whose predecessor stops being runnable can never be re-checked, and
# every deployment number published before 2026-08-15 was measured either
# against `pocket_r35` or by it.
SUPERSEDED = ("pocket_r35",)

# One row per promotion. `predicted` is the band registered BEFORE the match
# where there was one -- an after-the-fact interval is not a prediction, and
# leaving the field absent says so more honestly than filling it in.
PROMOTIONS = (
    {"engine": "pocket_defer", "replaced": "pocket_r35",
     "date": "2026-08-15", "commit": "0224669",
     "score": 0.5625, "ci": (0.5273, 0.5977), "games": 240, "seed": 7300,
     "predicted": (0.5352, 0.5744),
     "evidence": "RESULT_DEFERRED_RETIREMENT.md"},
    {"engine": "pocket_r35", "replaced": "final",
     "date": "2026-07-30", "commit": "01eada1",
     "score": 0.5854, "ci": (0.5360, 0.6349), "games": 240, "seed": 6300,
     "evidence": "RESULT_MODEL_SIZE.md"},
)

# Engines whose IMPLEMENTATION must not move underneath a measurement. Anchors
# for the obvious reason -- a candidate may not shift its own ruler -- and the
# deployment baseline for a reason that only became visible over the last four
# engines: it is the B side of every A/B, so if a source edit changes what it
# does, the candidate's edge is measured against something nobody agreed to.
# Source drift here is a hard failure; `--allow-anchor-drift` is the override,
# and re-freezing ENGINE_SOURCES with a written justification is the fix.
#
# This is deliberately NOT the same set as ANCHOR_ROLES. An anchor may never be
# overridden at all, because it is a ruler. The deployment baseline is the
# opposite -- ablating one declared knob off it (`engine:pocket_defer+solve=0`)
# is how the next question gets asked -- so it keeps the strict source check
# and does not get the override ban.
STRICT_SOURCE_ROLES = ANCHOR_ROLES | {DEPLOYED}

# Resolved-config fingerprints, emitted by --freeze. An empty dict means the
# registry has never been frozen and verification cannot run.
# RE-FROZEN 2026-08-15. Every value below moved a THIRD time, for the same
# reason as 2026-08-09 and 2026-08-12 and by the same rule: `resolved_config`
# gained `defer_release` (pinned False) and `retire_watermark` -- which is what
# these engines already did. No engine's behaviour changed and none was
# re-measured. All three previous generations are kept below so "only the new
# keys moved" stays checkable at each step rather than asserted once and then
# trusted; tools/test_engine_registry checks all three.
FINGERPRINTS = {
    "original": "a6a4289adcdf8f6a",
    "final": "1b36b08d5b812d20",
    "anchor_A": "5532136cbaca0ad7",
    "anchor_B": "3ecebd49460c5e14",
    "anchor_C": "fd573abf81d0edfe",
    "anchor_D": "154ddf537448205b",
    "pocket": "a8efc7b3ff812218",
    "midsize": "658e5d794de92d56",
    "pocket_r35": "26531023f03fd818",
    # Freeze these three with `--freeze --device cuda`: TimedPlayer refuses to
    # build a graph engine whose capture failed, and capture cannot succeed on
    # the CPU default.
    "pocket_graph": "6fcf8ce4de3f08ef",
    "pocket_sel": "4e0e8e0d422459b8",
    "pocket_defer": "d602920d80a73777",
    "gen22_raw": "370f7ef6bf3e6908",
    "pocket_raw": "c320b956e52b4e19",
    "midsize_raw": "e3dc389d122b17e1",
}

# The values every published result between 2026-08-12 and 2026-08-15 was
# measured against -- #45a's whole throughput study and both of its strength
# matches. `resolved_config(player)` minus `defer_release` and
# `retire_watermark` must still hash to these.
PRE_DEFER_FINGERPRINTS = {
    "original": "247a5d5f5a693792",
    "final": "ae965b10747d1ca2",
    "anchor_A": "97da7da71258ec65",
    "anchor_B": "0a86d2ad24188aba",
    "anchor_C": "7fb6b7a2d4be9bb0",
    "anchor_D": "3ab93ffd546a63cb",
    "pocket": "bc542e5f043f6ee0",
    "midsize": "3c51ee0174d14b56",
    "pocket_r35": "be9f7b150592c32a",
    "pocket_graph": "e82f7ddf88acec2f",
    "pocket_sel": "e33659b9f270d0c2",
    "gen22_raw": "b2605d73351a74f8",
    "pocket_raw": "04d79bccae15abf2",
    "midsize_raw": "71535604401525af",
}

# The values every published result between 2026-08-09 and 2026-08-12 was
# measured against. `resolved_config(player)` minus `native_select` must still
# hash to these.
PRE_SELECT_FINGERPRINTS = {
    "original": "6305c68d4f7e2c7c",
    "final": "180cb11bc206b84a",
    "anchor_A": "0215e5f848db39f4",
    "anchor_B": "87d79141aac50f42",
    "anchor_C": "3c97e4e10f6daaec",
    "anchor_D": "cde9982e27e6ce50",
    "pocket": "f274f655957c4852",
    "midsize": "503126dcfb4f1eb0",
    "pocket_r35": "e7bce88383363d0e",
    "pocket_graph": "df93af350ef9f906",
    "gen22_raw": "4c5659d088cca421",
    "pocket_raw": "fa04295ca54fcf24",
    "midsize_raw": "a6af84bf609e5250",
}

# The values every published result before 2026-08-09 was measured against.
# `resolved_config(player)` minus `graph_wave` must still hash to these.
PRE_GRAPH_FINGERPRINTS = {
    "original": "fe26cffd58d2290d",
    "final": "cfb14702454ac6df",
    "anchor_A": "cf1ff7cb2b932e75",
    "anchor_B": "f70fa9f070360fc8",
    "anchor_C": "5b93848f100a9c7f",
    "anchor_D": "dbc1514fb6b28f45",
    "pocket": "036f17c9aa644aad",
    "midsize": "5f38b819e7037640",
    "pocket_r35": "d9769168cae6af7c",
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


def derived_spec(name, overrides):
    """A frozen engine plus DECLARED overrides -- for sweeping one parameter.

    A sweep candidate is deliberately not the frozen engine, so it cannot match
    a frozen fingerprint. What must still be guaranteed is that it differs in
    exactly what was asked for and in nothing else, which is what the returned
    `diff` is checked against by the caller. Overrides may only replace keys the
    base already pins: adding a new one would mean the base was not fully
    specified, and silently accepting it would defeat the registry.

    Returns (spec_string, diff_keys).
    """
    if name not in ENGINES:
        raise SystemExit(f"[X] unknown engine '{name}'. "
                         f"known: {sorted(ENGINES)}")
    base = ENGINES[name]
    new = set(overrides) - set(base)
    if new:
        raise SystemExit(
            f"[X] engine '{name}' does not pin {sorted(new)}, so overriding "
            f"it would change something the frozen spec never declared. "
            f"Add it to the registry entry first.")
    merged = dict(base)
    merged.update({k: str(v) for k, v in overrides.items()})
    diff = {k for k in merged if base[k] != merged[k]} - {"name"}
    if "name" not in overrides and diff:
        merged["name"] = (base["name"] + "-"
                          + "-".join(f"{k}{merged[k]}" for k in sorted(diff)))
    return ",".join(f"{k}={v}" for k, v in sorted(merged.items())), diff


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
        # Under a clock this changes how the engine plays -- not what a wave
        # computes (that is bit-identical, gated by agents/test_graph_wave.py)
        # but how many waves fit in the budget. The REQUESTED flag, not the
        # captured object: --freeze runs on CPU, where capture cannot succeed.
        "graph_wave": m.graph_wave_requested,
        # Cannot change what is selected -- parity is the promotion gate -- but
        # it changes how much search fits in the clock, so it is configuration.
        "native_select": m.native_select,
        "add_dirichlet": m.add_dirichlet,
        "virtual_loss": type(m)._VL,
        "min_waves": type(m)._MIN_WAVES,
        "tree_reuse": player.reuse,
        # #46. Same category as `native_select`: it cannot change what is
        # computed -- the same nodes are built, the same statistics kept, only
        # the moment of destruction moves -- but it decides whether an O(nodes)
        # walk sits inside the move, which is what the reserve is sized for.
        # Read off the SEARCHER, because that is the object that acts on it; a
        # raw player has no searcher and cannot defer anything.
        "defer_release": bool(player.searcher is not None
                              and player.searcher.defer_release),
        "retire_watermark": (player.searcher.retire_watermark
                             if player.searcher is not None else None),
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


def verify(name, player, strict_sources=None, overrides=None):
    """Assert a built player IS the frozen engine. Returns a provenance dict."""
    if strict_sources is None:
        strict_sources = name in STRICT_SOURCE_ROLES
    cfg = resolved_config(player)

    want_ck = CHECKPOINTS.get(cfg["ckpt"])
    if want_ck and cfg["ckpt_sha256"] != want_ck:
        raise SystemExit(
            f"[X] engine '{name}': checkpoint {cfg['ckpt']} has changed.\n"
            f"    frozen {want_ck}\n    found  {cfg['ckpt_sha256']}")

    if overrides:
        # A derived candidate cannot match the frozen fingerprint -- that is
        # the point of it. The checkpoint and source checks still apply, and
        # the base fingerprint is recorded so the result stays traceable to a
        # frozen configuration rather than floating free.
        if name in ANCHOR_ROLES:
            raise SystemExit(
                f"[X] '{name}' is an ANCHOR and must not be overridden. A "
                f"candidate may never alter the ruler it is measured against.")
        return {"engine": name, "fingerprint": fingerprint(cfg),
                "derived_from": name, "base_fingerprint": FINGERPRINTS.get(name),
                "overrides": dict(overrides), "verified": True,
                "source_drift": check_sources(strict_sources),
                "baseline_tag": BASELINE_TAG, "resolved": cfg}

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
        role = ("anchor" if name in ANCHOR_ROLES
                else "DEPLOYED" if name == DEPLOYED
                else "past" if name in SUPERSEDED
                else "candidate")
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
