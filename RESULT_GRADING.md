# RESULT -- oracle blunder-rate grading (first provable correctness numbers)

*Run on the RTX 3080 home box, 2026-06-29, commit `41dfc5a` (working tree dirty).
Produced by the freshly pulled M4 grading oracle (`engine/solver.py` +
`scripts/grade_agent.py`, GRADING_AND_ORACLE.md Part 6). These are the project's
first PROVABLE correctness metrics: the alpha-beta endgame oracle certifies whether a
played move was strictly worse than optimal, rather than measuring relative win rate.*

## Headline

The **1-ply tactical overlay roughly quarters the provable blunder rate** on all three
nets tested, and lets each net survive into far more deep positions. Against center
(large, solid samples): `best.pt` 28.3% raw, the pocket finalist `arena:21` 28.3% -> 7.8%
with tactical, the strength finalist `arena:22` 23.2% -> 6.3%. This is a provable
confirmation of the M2 / M4_DESIGN thesis that the cheap 1-ply win/block layer patches a
large fraction of the net's tactical errors -- now replicated on the actual M2 finalists,
not just the league net.

## Numbers

Candidate: `models/league_pg/best.pt` (`medium`). 5000 games per row, grading every
candidate-to-move position with <= 26 empty cells, node budget 300k, seed 0. Blunder
rate = provable blunders / positions proven by the oracle. All rows below proved 100%
of attempted positions within budget (UTTT's additive collapse keeps alpha-beta
tractable at this depth).

| Candidate | Opponent | Proven positions | Blunders | Blunder rate | ~95% CI |
|---|---|---:|---:|---:|---:|
| raw | winblock | 206 | 36 | 17.48% | 12.9 - 23.2% |
| raw | center | 6036 | 1707 | **28.28%** | 27.1 - 29.4% |
| raw | first | 0 | -- | n/a | -- |
| raw | nn_big_8 | 0 | -- | n/a | -- |
| **tactical** | winblock | 1744 | 85 | **4.87%** | 3.9 - 5.9% |

## Arena finalists (the M2 bases)

`grade_agent.py` cannot load the finalists' custom architectures (it only knows the
`small/medium/large` presets). To grade them anyway, a scratchpad harness reused the
benchmark suite's architecture resolver (`resolve_candidate` / `_build_base_candidate`,
which read the arch from `models/arena/arena_state.json`) and fed the played positions to
the same `engine.solver` oracle. No repo code was modified. Same settings (5000 games,
<= 26 empty, budget 300k, seed 0); 100% of attempted positions proved within budget.

| Candidate | Mode | Opponent | Proven | Blunders | Blunder rate | ~95% CI |
|---|---|---|---:|---:|---:|---:|
| arena:21 (pocket) | raw | winblock | 58 | 13 | 22.41% | small n |
| arena:21 (pocket) | raw | center | 4129 | 1170 | **28.34%** | 27.0 - 29.7% |
| arena:21 (pocket) | tactical | winblock | 417 | 22 | 5.28% | 3.5 - 7.9% |
| arena:21 (pocket) | tactical | center | 5057 | 392 | **7.75%** | 7.0 - 8.5% |
| arena:22 (strength) | raw | winblock | 9 | 1 | 11.11% | n too small |
| arena:22 (strength) | raw | center | 1067 | 248 | **23.24%** | 20.7 - 25.8% |
| arena:22 (strength) | tactical | winblock | 402 | 16 | 3.98% | 2.5 - 6.4% |
| arena:22 (strength) | tactical | center | 862 | 54 | **6.26%** | 4.8 - 8.0% |

Read the center rows (large n) as the trustworthy signal: tactical cuts the provable
blunder rate by ~73% on the pocket net (28.3% -> 7.8%) and ~73% on the strength net
(23.2% -> 6.3%); the CIs are far apart. The strength net also blunders less than the
pocket net both raw and tactical, corroborating the M2 strength ordering on a provable
metric. winblock raw rows have too few proven positions to trust (the raw nets lose to
winblock too fast to reach solvable endgames), but the tactical winblock rows agree with
the center story.

## What this establishes

1. **The tactical overlay is a large, provable improvement.** Same opponent
   (winblock), tactical on vs off: blunder rate drops 17.48% -> 4.87% (the confidence
   intervals do not overlap), and the net reaches 1744 deep positions instead of 206 --
   i.e. it stops losing quickly and survives into real endgames. This is the cleanest
   controlled pair here and it confirms the standing thesis on independently graded,
   provable data.
2. **The raw league net blunders heavily in solvable endgames.** Against center the
   oracle proved 6036 positions and flagged 28.28% as strictly suboptimal -- a
   rock-solid (n=6036) signal that `best.pt`'s late-game play is weak, consistent with
   its known winblock blind spot.
3. **Organically reached positions are scarce or absent for some matchups.** vs `first`
   and vs `nn_big_8` the run collected ZERO gradeable positions in 5000 games: those
   games simply do not produce candidate-to-move positions at <= 26 empty cells (games
   end earlier, with more of the board empty). This is the GRADING_AND_ORACLE Part 6
   limitation in the raw -- a trustworthy, opponent-independent blunder rate needs the
   GOLD-seed approach (inject curated near-end positions), which is not yet implemented.

## Caveats

- **The M2 finalists were graded via a scratchpad harness, not `grade_agent.py` itself**
  (which still can't load their architectures -- see note 1). The harness only reuses
  existing, tested code paths (the benchmark suite's arch resolver + the solver), but the
  clean fix is to teach `grade_agent.py` the same resolution so this is a first-class,
  reproducible command rather than glue.
- The raw-vs-tactical comparison shares the opponent but not the exact position set
  (different play produces different games); the effect size and direction are
  unambiguous regardless.
- The proven sets are organically reached, so they are skewed toward whatever endgames
  each matchup produces (see finding 3). Rates are provable but not opponent-neutral.

## Test battery (all green on this box)

- C++ engine cross-validation (`engine/cpp/test_engine.py`): rebuilt `.pyd` matches the
  Python engine exactly over 500 random games (1000 more ran assertion-clean).
- M1 benchmark suite unit tests (`scripts/test_benchmark_suite.py`): 10/10.
- Endgame solver tests (`engine/test_solver.py`): 8/8.

## Notes for the authoring box

1. **`grade_agent.py` cannot grade the arena finalists.** It hardcodes the
   `small/medium/large` presets; it needs the same architecture-resolution path that
   `benchmark_suite.py` already uses (`models/arena/arena_state.json`) to grade
   `arena:21/22`.
2. **`--max_empty` default of 15 yields 0 gradeable positions for UTTT.** Games end with
   far more empty cells; ~26 is the usable floor. Bump the default or note it in the
   docstring.
3. **`engine/cpp/test_engine.py` fails when run as `python engine/cpp/test_engine.py`**
   (its documented invocation): `ModuleNotFoundError: No module named 'engine'`, because
   the project root is not on `sys.path` under path-style invocation. Run it with
   `PYTHONPATH=.` (or `python -m engine.cpp.test_engine` if `cpp` is made a package).
4. **After pulling `engine/cpp` changes, the home box must rebuild the extension**
   (`cmake --build engine/cpp/build --config Release`); the committed source had new
   pybind11 bindings (`_raw_last_move`, `_raw_winner`) that the stale `.pyd` lacked,
   which crashed `grade_agent` until rebuilt.

## Reproduce

```
cmake --build engine/cpp/build --config Release
python -m scripts.grade_agent --checkpoint models/league_pg/best.pt --network medium ^
    --games 5000 --opponent winblock --max_empty 26 --budget 300000 --seed 0
```

Add `--tactical` for the overlay row; swap `--opponent` for center/first/nn_big_8.
