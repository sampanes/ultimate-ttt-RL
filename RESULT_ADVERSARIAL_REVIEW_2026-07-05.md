# Expert-iteration adversarial review - 2026-07-05

## Verdict

The 48-hour run was not demonstrating reliable learning progress. It was
optimizing and promoting a search-assisted relative proxy while the requested
artifact is a strong raw-inference policy. The old run is preserved under
`models/expert_iter`, but it must not be resumed or used as a champion.

Observed old-run state:

- 582 blocks, 9,312 games, 57,700 optimizer steps, 26 teacher generations.
- Last ten gate mean: 0.574 vs random, **0.116 vs WinBlock**, 0.496 vs the
  moving raw teacher, and 0.913 MCTS edge over the raw student.
- On a fixed 40-game panel, the old seed scored 0.0625 vs WinBlock and 0.6875
  vs random; the trained student scored 0.0875 and 0.6000. Promotion count hid
  an absolute regression.

## Confirmed root causes

1. **Wrong teacher seed.** The script used `models/league_pg/best.pt`, while
   `RESULT_M2.md` had already selected Arena 22 as the strongest independent
   oracle base. The corrected seed scores 0.1125 vs WinBlock and 0.8250 vs
   random on the same fixed panel before any new training.

2. **Promotion measured the wrong artifact.** A student became teacher after
   MCTS(student) scored at least 0.55 against MCTS(teacher) in only 40 noisy
   games. Scores of 0.55-0.60 do not establish a meaningful advantage at that
   sample size, and search can conceal a weak raw policy. The run promoted 26
   times while raw WinBlock strength remained near zero.

3. **The replay window mixed incompatible teachers.** Promotions occurred
   roughly every 200-400 games while the 200k-position window retained many
   generations. On the final shard, the student had policy CE 2.060 and value
   MSE 0.887 even though the logged replay minibatch value MSE was about 0.14.
   The dashboard reported fit to stale replay, not fit to the current teacher.

4. **Training and inference policy objectives differed.** Inference masks
   illegal moves before softmax. Training normalized over all 81 logits and
   spent gradient suppressing moves that the runtime can never select. The
   fixed loss masks exactly as inference does and rejects malformed targets.

5. **Known failures were evaluated but not trained.** WinBlock was the
   documented shared blind spot, but expert generation remained pure teacher
   self-play. The corrected run sends 35% of games through WinBlock positions.

6. **Verified symmetry was discarded.** UTTT has eight exact square-board
   rotations/reflections. The corrected trainer transforms every spatial input
   plane and the policy target together, increasing effective coverage without
   more search games.

7. **The student discarded the seed policy.** The old student started from
   random weights. V2 copies the verified teacher weights into the student; the
   tanh change affects the value output, not policy logits, and outcome training
   then recalibrates that head.

## Corrected promotion contract

A teacher change now requires all of the following:

- at least 1,000 expert games from the current teacher;
- raw student score at least 0.55 against raw teacher on reproducible,
  color-swapped openings with no policy sampling or MCTS;
- an absolute WinBlock score improvement of at least 0.02;
- no more than 0.03 regression against the fixed random panel.

After promotion, old-generation replay is cleared. Resume loads only shards
created by the current teacher generation.

## Run and acceptance gates

Start or resume v2:

```bat
start_goat.bat
```

Stop gracefully:

```bat
stop_goat.bat
```

Do not judge the run by loss or teacher generation. Require:

1. raw fixed-panel WinBlock score to rise and remain above the prior best;
2. random score not to regress;
3. MCTS edge to remain at least 0.5;
4. the promoted raw checkpoint to clear the full frozen M1 anchor/opening panel;
5. checkpoint hash, architecture, and exact inference mode in the result.

## Verification completed

- `python -m scripts.test_expert_iter`: 5/5 passed.
- `python -m agents.test_mcts`: 10/10 passed.
- `python -m scripts.test_benchmark_suite`: 10/10 passed.
- `python -m engine.test_tactics`: passed.
- `python -m engine.test_solver`: 8/8 passed.
- One-block CUDA smoke run strictly loaded Arena 22, generated a shard, ran the
  fixed gate, and saved resumable v2 state.
