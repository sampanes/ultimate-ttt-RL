# RESULT: new pocket champion -- 7.5x smaller and far stronger (2026-07-26)

## What shipped

`arena:21@hof` (1,287,314 params, 5.16 MB) is replaced as pocket champion by
`pocket:squeeze-gen22` (**172,389 params, 0.70 MB**), a distilled
expert-iteration student using the `head_squeeze` 1x1-conv head from
`RESULT_ARCH_AB.md`.

The browser download drops from **5,035 KB to 680 KB (7.4x)**, with no
quantization -- the fp32 model is now far under the 5 MB gate that int8 existed
to clear.

## Why the head change is the whole size win

The legacy `ConvNet` flattens the entire conv output straight into a `Linear`,
so on the shipped shapes the heads hold the overwhelming majority of the
parameters and the conv tower that does the spatial reasoning gets the rest.
Inserting an AlphaZero-style 1x1 conv before the flatten collapses that. The
four-arm A/B measured this as 5.3x compression at parity; here it lands 7.5x
against the incumbent while also being much stronger, because the replacement is
distilled from a far better teacher.

Note this is *not* an architecture-strength claim. `RESULT_ARCH_AB.md` showed
architecture does not move deployed strength at all. The strength here comes
from the training signal -- MCTS-200 targets off the gen-22 teacher -- and the
architecture only buys the size.

## Certification

Promotion rule (CHAMPIONS.md): beat the incumbent on the M2 panel, with a
300-game fixed-opening colour-swapped raw head-to-head as the tie-breaker.

### 300-game panel, raw argmax, colours swapped

| net | params | random | winblock | gregory d3 | gregory d4 |
|---|---|---|---|---|---|
| `arena:21@hof` (incumbent) | 1,287,314 | 0.752 | 0.158 | 0.067 | 0.017 |
| **`pocket:squeeze-gen22`** | **172,389** | **0.983** | **0.905** | **0.728** | **0.653** |

**Head-to-head: 0.9833** to the challenger (300 games, seed 9901).

### M2 panel, both nets, identical anchors and seed

Challenger wins **15 of 15 cells**. Selected:

| mode | opponent | incumbent | challenger |
|---|---|---|---|
| raw | gregory d3 | 0.000 (0-0-18) | 0.833 |
| raw | winblock | 0.056 | 0.917 |
| tactical | gregory d3 | 0.028 | 0.861 |
| tactical | winblock | 0.417 | 0.861 |
| mcts_25 | gregory d3 | 0.000 (0-0-18) | 0.806 |
| mcts_25 | winblock | 0.083 | 0.972 |

Reports: `results/pocket-squeeze/`, `results/pocket-incumbent/`.

The incumbent losing 18-0 to a depth-3 alpha-beta is the historic winblock blind
spot documented in `RESULT_GEN19_CERT.md` (raw winblock 0.361 -> 0.861 on the
oracle track). The oracle track fixed it in July; the pocket slot never got the
fix and has been shipping it to players since.

## Defects found and fixed to make this shippable

None of these were the task; all of them blocked or silently corrupted it.

1. **`benchmark_suite` could not evaluate the new architecture.** Its
   `Architecture` had no `head_squeeze`/`residual`/`norm`, so the certification
   path was blind to anything from `RESULT_ARCH_AB.md`. Added as opt-in fields;
   all-falsy still resolves the legacy graph, so every pre-existing manifest is
   unaffected.
2. **The M2 anchor list had no gene-pool-independent ruler.** winblock is 30% of
   the expert-iteration training mix; gregory never is, which is why
   `expert_iter` refuses to train against it. Added `gregory`/`gregory_deep` as
   anchors -- and gregory-d3 is precisely the cell where the incumbent scores
   zero, so the panel had been unable to see its worst weakness.
3. **Anchor provenance attested the wrong file.** Every code anchor was hashed
   as `agents/deterministics.py`, so gregory's certification record pointed at a
   file that does not contain gregory; changing the ruler would not have changed
   the record. Now resolved from the agent class's own module.
4. **`export_onnx` never verified its output.** Nothing compared the exported
   graph against the net it came from, so a tracing error would ship silently to
   every browser -- a real risk for `head_squeeze`, which adds a reshape
   boundary. Added a parity gate that fails the export. This model passes at
   max|dpolicy| 1.19e-06, max|dvalue| 4.77e-07, argmax agreement 64/64.
5. **`model_config.json` misdescribed whatever it exported.** `description` was
   hardcoded to the policy-gradient blurb, so this swap would have shipped a
   distilled expert-iteration student described as arena self-play, and `name`
   was set to the manifest's file path. Both are now real fields
   (`--label`, `--description`, `--config-version`).

## Browser verification (done 2026-07-26, Chrome via the extension)

Served `docs/` locally and clicked through the real play page.

- The page loads `pocket:squeeze-gen22`, `696,567` bytes -- byte-exact match to
  the exported artifact.
- **Easy, Medium and Hard all play legal moves.** Playing X at cell 40 (centre
  of the centre mini-board) drew O at cell 37 on Easy/Medium and cell 44 on
  Hard, both inside mini-board 4 as the send-rule requires. Hard runs the
  champion and is unaffected by this swap, as expected.
- **The `policy_logits` shape warning is confirmed COSMETIC.** Running the
  deployed ONNX in-browser on identical input at batch 1 and batch 32:

  | | batch 1 | batch 32 |
  |---|---|---|
  | runtime dims | `[81]` | `[32, 81]` |
  | max abs diff vs batch 1 | -- | **0** (policy and value) |

  onnxruntime-web resolves the true shape at runtime despite the graph metadata
  declaring `{-1}`, and all 32 rows are bit-identical to the single-input
  result. Nothing is mis-sliced.

  It is *not* harmless noise, though: the warning fires at `{32,81}` on **every
  move**, roughly nine times per move, and Chrome surfaces it at ERROR level.
  So the pocket net IS driven by the batched wave-32 MCTS -- batching is not
  champion-only, which is worth knowing -- and the console is unusable for real
  debugging while it spams.

## Known issues, deliberately not fixed here

- **The exported graph mis-declares `policy_logits`.** `forward_both` squeezes
  the batch dimension at trace time, so the traced output is 1-D and the axis
  labelled `batch` is really the 81 moves. Verified above to be cosmetic, and
  **all three deployed models share this signature**, including the live
  champion -- so this ship introduces no regression. Worth fixing on its own,
  for the console noise, with its own browser test.
- `docs/models/model_int8.onnx` is now stale: it is the quantized *previous*
  model. It is gitignored, is not served (ConvInteger is unsupported by
  onnxruntime-web WASM), and the new fp32 is smaller than that int8 was, so
  quantization is moot at this budget. Left in place per the .gitignore note
  that keeps it for re-quantization experiments.
- **Not pushed.** Verified locally end to end; nothing has gone to Pages.

## Reproduce

    REM certify
    .venv\Scripts\python -m scripts.pocket_challenge --games 300
    .venv\Scripts\python -m scripts.benchmark_suite ^
      --candidate models/pocket_candidate/manifest.json ^
      --anchors gregory,winblock,nn_big8,center,first ^
      --candidate-sims 0,25 --oracle-sims 0 --out results/pocket-squeeze

    REM export (parity-gated)
    .venv\Scripts\python -m scripts.export_onnx ^
      --candidate models/pocket_candidate/manifest.json --out-dir docs/models ^
      --label "pocket:squeeze-gen22" --config-version m3-r2 ^
      --description "Distilled expert-iteration student: ..."
