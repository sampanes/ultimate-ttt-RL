# RESULT: lottery giant vs gen-22 tiny champ (skill per millisecond)

Date: 2026-07-24. Question asked: the lottery net has 77x more parameters --
if one forward costs the same as gen-22's, "we should have always had a giant
model". Measure latency, then run a fair head-to-head where each side gets the
same wall-clock budget.

Answer: the premise cannot be tested with this checkpoint. The lottery net is
not a weak player, it is **not a player at all** -- it is a constant function
that returns the same move for every board. Everything below is the evidence.

## 1. Latency (batch=1, one forward, `forward_both`)

| | gen-22 (`arena22`) | lottery giant | ratio |
|---|---|---|---|
| parameters | 6,766,386 | 518,392,402 | 76.6x |
| CUDA (RTX 3080) | 0.899 ms | 12.505 ms | 13.9x |
| CPU, 24 threads | 2.769 ms | 210.824 ms | 76.1x |
| CPU, 1 thread (phone/WASM proxy) | 4.681 ms | 691.733 ms | 147.8x |

So the giant is nowhere near the same latency. On a single-thread client-side
browser budget it needs ~692 ms for ONE raw move with no search, versus 4.7 ms
for gen-22.

**Skill-per-millisecond budget**: in the time the lottery produces one raw
forward, gen-22 can afford ~14 net evaluations on GPU and ~148 single-thread on
CPU. That is a net-compute ceiling: a real MCTS sim also costs Python tree
bookkeeping, so end-to-end a phone fits fewer than 148 sims in 692 ms.

## 2. Head-to-head, raw vs raw

1000 games, diverse reproducible openings, colors swapped, pure argmax both
sides (`_play_fixed_match`, seed 20260724):

    gen-22 raw vs lottery raw = 0.989   (989.0 / 1000 points)

## 3. Why that number is not a compliment to gen-22

A 0.989 blowout looks the same whether the opponent is weak or broken, so the
lottery was pinned against fixed anchors (300 games each):

| lottery raw vs | score |
|---|---|
| random | 0.748 |
| winblock (1-ply heuristic) | 0.157 |

Barely above random, and crushed by a few lines of if-statements. That is the
profile of a net that is not reading the board.

## 4. Root cause: the net is a constant function

Probed over 256 distinct legal positions:

- std of the policy logits ACROSS positions: **0.0000**
- most common argmax move share: **100.0%** (always move 24)

Precision was ruled out as the explanation (TF32 disabled, CPU):

| precision | max abs logit delta between any two positions |
|---|---|
| CPU float32 (WASM-equivalent) | 3.58e-07 |
| CPU float64 | 6.66e-16 |

float64 machine epsilon is 2.2e-16, so the residual is rounding noise. The
output is mathematically independent of the input.

### The mechanism

`ConvNet` (agents/neural_net_agent_3.py:19-25) stacks `Conv2d -> ReLU` 135 deep
in a plain `nn.Sequential`: **no residual connections, no normalization**.
PyTorch's default init is `kaiming_uniform_(a=sqrt(5))`, whose gain is
`sqrt(1/3)` rather than He's `sqrt(2)`, so each layer contracts activation
variance by roughly 1/3 with nothing to restore scale.

Walking the real weights layer by layer, the input-dependent signal decays and
then flatlines on a constant bias-driven attractor within six layers:

    input                 absmax 1.000
    after conv block  1   absmax 1.268
    after conv block  2   absmax 0.490
    after conv block  3   absmax 0.180
    after conv block  4   absmax 0.069
    after conv block  6   absmax 0.027
    after conv block 20   absmax 0.023   <- flat from here to block 135
    after conv block 100  absmax 0.023

The same contraction runs backward, so the gradient never reached the early
layers. This is provable exactly rather than statistically, because the model's
own LTH rewind point survives: `models/lottery/4096-.../initial.pt`
(2025-07-25), saved by `brand_new_weights()` before the run that produced
`several_weeks_no_touchy.pt` (2025-08-17). Comparing the two directly, any
tensor that is bit-identical never moved:

    conv layers total     : 135
    BIT-IDENTICAL to init :  82   <- never changed by a single bit
    moved >= 1% relative  :  23

Relative movement against depth is a clean exponential decay -- roughly five
orders of magnitude per 20 layers:

    layer #  1 .. # 60   exactly 0.0  (bit-identical)
    layer # 70           2.9e-22
    layer # 90           9.6e-12
    layer #100           3.0e-07
    layer #110           3.2e-03
    layer #120           7.8e-02
    layer #130           5.8e-01
    layer #135           8.5e-01

**"Several weeks" of training only meaningfully moved the last ~20 conv layers
and the FC head, on top of 82 layers of literally untouched random projection
that had already destroyed the board signal.**

(An earlier pass in this document estimated 115 layers "at init" using the
weaker test of whether a tensor had escaped its init bound `1/sqrt(fan_in)`.
The bit-exact comparison above supersedes it: 82 layers never moved at all, and
another 30 moved by amounts between 1e-22 and 1e-3 that are far too small to
matter. Both tests agree on the conclusion; the exact numbers here are the ones
to quote.)

The single largest weight change anywhere in the 518M-parameter network is
0.032, and the biggest movers are the **policy-head biases**
(`policy_head.13.bias` 0.0269, `.11` 0.0270, `.9` 0.0271, ...). That is the
mechanism closing: gradient could not reach the convolutions, so training
deposited what it learned in the only place it could reach -- the bias chain,
which is exactly the constant square-preference table of section 4b.

The checkpoint is genuine and healthy otherwise: `pruned_list` is empty,
`conv_channels` is exactly `[512]*135`, no dead or all-zero tensors, and 284 of
288 keys load with zero unexpected keys and zero shape mismatches.

## 4b. "But it used to beat random and other bots" -- yes, and that was real

It still does: 0.748 vs random, measured here. That is not in conflict with
being a constant function, and the win was earned rather than accidental.

Because `select_move` masks the constant logit vector to the LEGAL moves before
taking the argmax, a constant net is exactly a **fixed priority ranking over the
81 cells** -- "of the squares I am allowed this turn, take my highest-ranked
one". In Ultimate TTT, where each turn is confined to one mini-board, that is a
real if crude positional policy.

Is the ranking learned, or would any fixed order do? 300 games vs random each:

| fixed-ranking agent vs random | score |
|---|---|
| **the lottery's own learned ranking** | **0.748** |
| arbitrary ranking #0 | 0.467 |
| arbitrary ranking #1 | 0.468 |
| arbitrary ranking #2 | 0.575 |
| arbitrary ranking #3 | 0.507 |
| arbitrary ranking #4 | 0.545 |
| arbitrary mean | 0.512 |

An arbitrary fixed ordering is a coin flip against random; the lottery's is
**+0.236 better**. The FC head really did learn useful square preferences. So
the historical wins against random and the weak bots were genuine, and they came
from a static opening-preference table that never needed to read the board.

That also explains the shape of every panel it ever appeared in: it beats
anything that likewise does not read the board (random, `first`, `center`,
`nn_big_8`) and is crushed by the one anchor that does (`winblock`, 0.157).

### Where the "strong anchor" reputation came from

Not from measurement. `GRADING_AND_ORACLE.md:7` assigns the anchor ladder by
hand -- "Random (600) pins the floor; deterministics (700-950) ...
lottery/nn_big8 (1300/1400) are upper rungs". Those ELOs are design priors. The
panel data always disagreed: in `RESULT_M2.md` the candidates score 0.667-0.944
against lottery while losing to winblock 17-1. `M4_DESIGN.md:20` had already
flagged it as untrustworthy evidence, without identifying the cause.

## 4c. Control: is it just being fed the wrong input format?

Short answer: no. The training-era source was recovered and the input format is
identical to today's.

### Recovering the pre-flatten history

`e6041ce` ("Flatten history to a single root commit", 2026-07-01) is a true
parentless root and `origin/main` points at it, so the mainline history is gone
locally. **But two GitHub pull-request refs survived the flatten**, and GitHub
retains PR heads independently of branch force-pushes:

    refs/pull/1/head  2093d3f   PR#1, 2026-03-14, base 3d6712a
    refs/pull/2/head  c151c06   PR#2, 2026-03-16, base a20e96c

Those base commits sit on the OLD main, so the entire pre-flatten ancestry is
still reachable through the GitHub API (do NOT `git fetch` them casually -- that
pulls back the multi-GB binary bloat the flatten removed; query the API or
raw.githubusercontent.com instead).

Walking back, the last commit before the checkpoint is `3d6712a` (2025-07-13,
"improve trainer, add deeper cnn, save init weights"), and there are no commits
at all between then and 2026-03-14. The checkpoint (2025-08-17) was produced in
that dormant month, so **the pruning driver that wrote its `label` /
`conv_channels` / `pruned_list` / `hidden_sizes` keys was never committed** --
the era's `save()` writes a bare `torch.save(self.model.state_dict(), path)`.

The "lottery" name is the **Lottery Ticket Hypothesis**, confirmed in the era
source at `agents/neural_net_agent_3.py:184`:

    # no checkpoint -> this is a fresh network:
    # let's save its initial weights for LTH / rewinding later

### The input format never changed

`board_to_tensor_from_gamestate` at `3d6712a` versus today:

| channel | 2025-07-13 (lottery era) | today |
|---|---|---|
| 0 | X positions | X positions |
| 1 | O positions | O positions |
| 2 | current player (X=1, O=-1) | current player |
| 3 | valid moves | valid moves |
| 4 | mini-board winners (X=1, O=-1) | mini-board winners |
| 5 | last move | last move |
| 6 | constant bias plane | bias |

Both use `idx = row * 9 + col` and `divmod(idx, 9)`. The era `ConvNet` is also
the same plain `Conv2d -> ReLU` `nn.Sequential` with no normalization and no
residuals, asserting `x.shape[1] == 7` -- and its ONLY difference from today's
is the absent `value_head`, which is exactly the 4 keys missing from the
checkpoint. Everything cross-checks. The net was fed what it was trained on.

### Behavioural control (independent of the above)

A net fed the wrong channels plays BADLY but its output still VARIES per board;
only a net that cannot propagate its input is constant. Max abs logit delta over
32 positions, float64, TF32 off:

| input arrangement | max delta |
|---|---|
| canonical 7-channel | 6.66e-16 |
| 20 random channel permutations | <= 9.99e-16 |
| uniform noise x1 / x10 / x1000 | ~6.7e-16 |
| board scaled x100 | 6.66e-16 |
| gen-22 on the same boards (reference) | **8.29e+01**, 12 distinct argmax |

No input arrangement whatsoever moves the output past float64 rounding error --
not any channel permutation, not noise a thousand times larger than a board. The
input convention is not the problem.

Nor was the architecture different at training time. The checkpoint holds
exactly 284 tensors: 135 conv layers x (weight+bias) = 270, plus 7 policy
Linears x 2 = 14, with **zero unexpected keys**, and `initial.pt` has the same
284. BatchNorm or residual blocks would have left
`running_mean`/`running_var`/downsample tensors behind. It was trained as this
same plain unnormalized stack, and was already a constant function the day it
was saved.

### Recovery note for future archaeology

The pre-flatten history is NOT lost. Anything reachable from the PR base
commits can be read without cloning the bloat:

    curl -s "https://api.github.com/repos/sampanes/ultimate-ttt-RL/pulls?state=all"
    curl -s "https://api.github.com/repos/sampanes/ultimate-ttt-RL/commits?sha=<base_sha>&until=<date>"
    curl -sL "https://raw.githubusercontent.com/sampanes/ultimate-ttt-RL/<sha>/<path>"

This only holds while GitHub keeps those PR refs, and only for history
reachable from them.

## 5. Second finding: the lottery has no value head, so it can never be searched

The 4 keys absent from the checkpoint are the entire value head
(`value_head.1/.3` weight+bias), which postdates it. Consequences:

- Raw play is unaffected -- `NeuralNetAgent3.select_move`
  (agents/neural_net_agent_3.py:145) calls `self.model(x)`, the policy head
  only, and never evaluates `value_head`. The head-to-head above is valid.
- MCTS is impossible -- `agents/mcts.py` takes every leaf value from
  `forward_both` (lines 222, 274) with no rollout fallback. Searching the
  lottery searches on random noise.

So "give both sides the same time budget" could never have meant "give both
sides MCTS". Only one of these two nets can spend a larger compute budget at
all.

**Latent bug**: the registered `mcts_lottery` factory (agents/__init__.py:110)
wraps this checkpoint in `MCTSAgent(n_sims=100)` and has therefore been
searching on a random value head. Left unfixed here, flagged only.

## 6. Blast radius -- where a constant-move bot has been used as a real opponent

Not audited or changed in this pass, but any number produced against the
lottery anchor is a number against a constant bot:

- `scripts/league_manager.py:195` loads it as a league opponent; per CLAUDE.md
  the stage-6 mix is 10% lottery, so ~10% of stage-6 league games were played
  against a fixed move.
- `scripts/benchmark_suite.py:81-98,621,691` uses it as a benchmark anchor.
- `RESULT_M2.md` reports panel cells against it (0.667-0.889), which in
  hindsight say more about the anchor than the agents.
- `README.md:186,219,225,231,267,395` documents it as a legitimate opponent.

## 7. Verdict

The giant is 76.6x larger, 13.9x slower on GPU, 147.8x slower on a
single-thread CPU budget, loses 0.989 raw, cannot be given search, and does not
read the board. The "should we have gone giant?" experiment was never actually
run -- depth killed it before size could matter. 135 plain layers without
residuals or normalization is not a bigger model, it is an untrainable one.

To be fair to it: the 518M parameters bought a good static square-preference
table, which is a real learned artifact and beat every board-blind opponent it
ever faced. It just is not a board-reading policy, and no amount of further
training through 115 frozen layers was ever going to make it one.

The planned multi-hour equal-time MCTS sweep was cancelled: pouring GPU hours
into beating a constant-move bot by a wider margin measures nothing.

## Reproduce

Harness lives in the session scratchpad (throwaway, not committed):
`lottery_vs_gen22.py` (modes `time` / `raw` / `control` / `sweep`),
`init_scan.py`, `signal_collapse.py`, `depth_underflow.py`,
`collapse_cpu_control.py`. `lottery_vs_gen22.verify_lottery_load` hard-asserts
that the checkpoint gap is exactly the 4 value-head keys, so any future drift
fails loudly instead of silently poisoning a result.
