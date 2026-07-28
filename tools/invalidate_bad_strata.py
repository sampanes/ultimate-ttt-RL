"""Mark the strata poisoned by the mini-board indexing bug as INVALID.

The bug (fixed 2026-07-27 in tools/summarize_search_disagreement.py) used
`move // 9` -- the board ROW -- as the mini-board index, and `last_move % 9`
-- the COLUMN -- as the mini the mover is forced into. Measured blast radius
over 2,904 random reachable positions:

    mini_win_available   true rate 0.3705, reported 0.1977, mislabelled 17.29%
    forced_target        mislabelled 11.88%, and 'drawn' was never once emitted

That is large enough that any conclusion drawn from those two strata has to be
thrown away rather than caveated.

What is NOT affected, and is deliberately left untouched (the owner's
instruction was not to regenerate unrelated global metrics):

    overall, by_phase, by_legal_bucket   -- no mini indexing anywhere
    by_tactical.immediate_win_available  -- derived from `s.winner == mover`
    by_tactical.no_immediate_win         -- same

This script does not delete or recompute anything. It stamps the affected
objects in place so a later reader cannot pick the numbers up by accident, and
records why. Values are preserved under `_invalid_original` so the historical
record survives -- the point is to disarm them, not to erase them.

    python -m tools.invalidate_bad_strata            # stamp
    python -m tools.invalidate_bad_strata --check    # verify, exit 1 if unstamped
"""
from __future__ import annotations

import argparse
import json
import os

MARKER = "_INVALID"

REASON = (
    "INVALID -- computed with the mini-board indexing bug (move // 9 is the "
    "board ROW, not the mini index; last_move % 9 is the COLUMN, not the "
    "forced mini). Fixed 2026-07-27. mini_win_available was mislabelled on "
    "17.29% of positions (true rate 0.3705 vs reported 0.1977); forced_target "
    "on 11.88%, never emitting 'drawn'. Do not cite these numbers. Regenerate "
    "with the fixed tools/summarize_search_disagreement.py if they are needed."
)

# path-in-json -> which files carry it
TARGETS = {
    "results/disagreement/summary_analysis.json": [
        ("by_forced_target",),
        ("by_tactical", "*", "mini_win_available"),
    ],
    "results/disagreement/adjudication_200v800.json": [
        ("by_forced_target",),
        ("by_tactical", "*", "mini_win_available"),
    ],
    "results/disagreement_bak/summary_analysis.json": [
        ("by_forced_target",),
        ("by_tactical", "*", "mini_win_available"),
    ],
    # Per-position dumps: the two per-row FIELDS are wrong, not the rows.
    "results/disagreement/diagnostic_top_js.json": [
        ("[]", "forced_target"),
        ("[]", "mini_win_available"),
    ],
    "results/disagreement_bak/diagnostic_top_js.json": [
        ("[]", "forced_target"),
        ("[]", "mini_win_available"),
    ],
}

# Gzipped CSVs cannot carry an in-band marker without rewriting the data, and
# rewriting would destroy the historical record. They get a sidecar note
# instead, named so it sorts directly beside the file it condemns. Both listed
# columns came from the buggy summarizer; every other column in these files is
# clean (analyze_search_disagreement.py computes no strata at all -- verified).
CSV_TARGETS = {
    "results/disagreement/position_strata.csv.gz":
        ["forced_target", "mini_win_available"],
    "results/disagreement_bak/position_strata.csv.gz":
        ["forced_target", "mini_win_available"],
    # tools/make_distill_corpus.py IMPORTS the two buggy functions rather than
    # reimplementing them, so every corpus index it wrote before 2026-07-27
    # carries the same two poisoned columns. Nothing reads them -- every
    # consumer takes its mate-in-1 flag from `immediate_win`, which is clean --
    # but they are wrong and are marked so. Rewriting an .npz to embed a marker
    # would rewrite the whole archive, so these get a sidecar too.
    "models/distill_pilot/index.npz": ["forced_target", "mini_win"],
}


def stamp_csv(rel, cols, check):
    """Drop (or verify) a sidecar note naming the poisoned columns."""
    note = rel + ".INVALID_COLUMNS.txt"
    if check:
        return 0 if os.path.isfile(note) else 1
    if os.path.isfile(note):
        return 0
    with open(note, "w", encoding="utf-8") as fh:
        fh.write(f"INVALID COLUMNS in {os.path.basename(rel)}\n\n")
        for c in cols:
            fh.write(f"  - {c}\n")
        fh.write("\nEvery other column in this file is unaffected.\n\n")
        fh.write(REASON + "\n")
    return 1


def stamp(obj, path, check):
    """Walk `path` and replace each leaf with an invalidation record."""
    head, rest = path[0], path[1:]

    if head == "[]":
        hits = 0
        for row in obj:
            hits += stamp(row, rest, check)
        return hits

    if head == "*":
        hits = 0
        for v in obj.values():
            hits += stamp(v, rest, check)
        return hits

    if head not in obj:
        return 0

    if rest:
        return stamp(obj[head], rest, check)

    cur = obj[head]
    if isinstance(cur, dict) and cur.get(MARKER):
        return 0                      # already stamped, idempotent
    if check:
        return 1                      # found an UNstamped leaf
    obj[head] = {MARKER: True, "reason": REASON, "_invalid_original": cur}
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report unstamped strata and exit 1, changing nothing")
    args = ap.parse_args()

    total, missing = 0, []
    for rel, paths in TARGETS.items():
        if not os.path.isfile(rel):
            missing.append(rel)
            continue
        with open(rel, encoding="utf-8") as fh:
            doc = json.load(fh)
        hits = sum(stamp(doc, p, args.check) for p in paths)
        total += hits
        if hits and not args.check:
            with open(rel, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=2)
        print(f"  {'FOUND' if args.check else 'stamped'} {hits:>4}  {rel}")

    for rel, cols in CSV_TARGETS.items():
        if not os.path.isfile(rel):
            missing.append(rel)
            continue
        hits = stamp_csv(rel, cols, args.check)
        total += hits
        print(f"  {'FOUND' if args.check else 'noted'} {hits:>4}  {rel} "
              f"({', '.join(cols)})")

    for rel in missing:
        print(f"  [!] absent (nothing to stamp): {rel}")

    if args.check:
        if total:
            raise SystemExit(
                f"[X] {total} stratum leaves are still unstamped. Run "
                f"`python -m tools.invalidate_bad_strata` to disarm them.")
        print("[OK] every bug-affected stratum is marked invalid")
    else:
        print(f"[OK] {total} stratum leaves marked invalid")


if __name__ == "__main__":
    main()
