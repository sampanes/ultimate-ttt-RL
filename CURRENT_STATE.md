# CURRENT_STATE -- what is true right now

Written 2026-09-04, at the close of the engine-optimization branch.

**This file is the single canonical record.** Where any other document in this
repository disagrees with it, this one is right and that one is stale. The
`RESULT_*.md` files remain the primary evidence -- each one is the full method
and the full numbers for one experiment -- but several of them state
conclusions that later experiments overturned, and a reader who arrives at them
cold has no way to know which. Section 11 lists those explicitly.

Two things are authoritative in code rather than prose, and outrank this file
the same way it outranks the others:

    python -m tools.engine_registry        # what the engine IS
    python -m tools.regress_engine         # whether it still behaves

---

## 0. The one-paragraph summary

The product is a ~1,000 ms-per-move Ultimate Tic-Tac-Toe player: a
172,389-parameter network inside a PUCT search with cross-move tree reuse,
CUDA-graph batched expansion, native selection, deferred tree retirement and
selective terminal probes. It is frozen as `engine:pocket_filter`. Search is
about as fast as it is worth making it -- the remaining measurable
opportunities are individually smaller than the noise floor of the instrument
that would have to confirm them. The network inside it is gen-22 of a teacher
lineage that **has not promoted a new generation in 564,480 games**, because a
promotion gate demanded something the lineage does not produce. That gate was
diagnosed and repaired on 2026-07-25 and **has never been run in production**.
That is the whole of the open work.

---

## 1. The production engine

`engine:pocket_filter`, DEPLOYED since 2026-08-18. Frozen for the training
phase (section 13).

| | |
|---|---|
| network | `models/pocket_candidate/squeeze_pocket.pt`, sha256 `b028d3499eca8b10...` |
| architecture | `squeeze`: conv `[56,56,56,56]`, fc `[256]`, `head_squeeze=2`, tanh value head |
| parameters | 172,389 (0.70 MB fp32) |
| move budget | 1,000 ms wall clock |
| reserve | 20 ms |
| wave | 8 |
| c_puct | 1.5 |
| `solve` | on -- solved-node propagation, including the proven-root early exit |
| `reuse` | on -- cross-move tree reuse |
| `bexp` | on -- batched wave expansion |
| `graph` | on -- the wave's device sequence replayed from a captured CUDA graph |
| `select` | on -- native PUCT selection over a mirrored child array |
| `defer` | on -- discarded tree detached on the move path, destroyed at the game boundary |
| `pfilter` | on -- selective terminal probes, gated on `could_end` |
| `maxsims` | 200,000 (a ceiling; never binds at 1 s) |

Implicit parameters that no flag pins -- `min_sims`, `deadline_margin`, the
virtual-loss magnitude, the `_MIN_WAVES` floor, `add_dirichlet` -- are covered
by the resolved-config fingerprint, because a change to any of them changes the
engine without changing a character of the command line.

The same 172k network ships as `docs/models/model.onnx` for the browser player.
The browser runs an independent JavaScript/WASM search, so it is a different
agent with the same weights -- numbers measured here do not transfer to it.

### Why the search looks like this

Each feature below is a measured win, not a design preference. The evidence
column is where the measurement lives.

| feature | what it bought | evidence |
|---|---|---|
| wall-clock budget + tree reuse + batched expansion | 0.7229 vs the pre-search engine at equal clock | `RESULT_ARENA_1S.md` |
| 172k network in place of 6.77M | 0.5854 at equal clock | `RESULT_MODEL_SIZE.md` |
| CUDA-graph wave | +29.6% network evaluations, 0.5458 at equal clock | `RESULT_GRAPH_WAVE.md` |
| native PUCT selection | 5.5x per call, +12.2% evaluations; strength effect unresolved | `RESULT_NATIVE_SELECT.md` |
| deferred tree retirement | reserve 95 -> 20 ms; the stack won 0.5625 | `RESULT_DEFERRED_RETIREMENT.md` |
| selective terminal probes | +20.6% evaluations at proven identical search | `RESULT_PROBE_FILTER.md` |

---

## 2. Promotion history, and what each promotion actually claims

Three promotions, and **they are not the same kind of evidence.** The registry
records this in a `basis` field for exactly that reason.

| date | engine | replaced | basis | result |
|---|---|---|---|---|
| 2026-07-30 | `pocket_r35` | `final` | equal-clock strength match | 0.5854 [0.5360, 0.6349], n=240 |
| 2026-08-15 | `pocket_defer` | `pocket_r35` | equal-clock strength match | 0.5625 [0.5273, 0.5977], n=240 |
| 2026-08-18 | `pocket_filter` | `pocket_defer` | **identity + throughput** | +20.6% nn/second x deadline |

**The third row is not a win rate and must never be quoted as one.**
`pocket_filter` and `pocket_defer` are bit-identical at a fixed simulation
count -- same nodes, same statistics, same `solved` flag on every one of them,
proofs included, enforced by `agents/test_probe_filter.py`. There is no
hypothesis about play for games to resolve; more of an identical search inside
the same clock cannot be worse. The registry stores the ladder's estimate of
what that throughput is worth (0.5145 to 0.5306 in a mirror) under `expected`,
with **no `score`, `ci` or `games` field and a test asserting their absence** --
an expected band living in a field named like an observation is precisely how a
prediction gets quoted later as a result.

**The second row is a stack, not a feature.** `pocket_defer` differs from
`pocket_r35` in three things at once -- the graph wave, native selection,
deferred retirement -- and the 0.5625 does not separate them. Two of the three
were measured individually: the graph wave won its own match at 0.5458, and
native selection did **not** resolve (0.5146 [0.4807, 0.5485], needing ~1,206
games). Native selection shipped inside a stack that won; it has never
independently demonstrated strength.

Superseded engines stay buildable rather than being deleted. A promotion whose
predecessor stops running is a published result that stops being checkable.

---

## 3. Latency: the policy, the diagnosis, and why a reserve exists

    p99 move latency  <=  1000 ms     the requirement
    max move latency  <=  1250 ms     reported, and a hard reject

Frozen 2026-07-28, before any candidate was benchmarked, and unchanged since. A
mean is the wrong statistic for a deadline: a player averaging 900 ms that
occasionally takes 1.8 s has not met the objective and the mean cannot see it.

Timing is an **outer** measurement around the whole move -- re-rooting, tree
release, instrumentation and the argmax included -- not the interval the search
times itself over. The inner number would quietly exclude work the deadline
covers, and that distinction turned out to be the entire story.

**Every latency failure in this project was caller-side tree work, never search
overrun.** Over 5,517 measured moves the search never exceeded 981.1 ms against
its own 980 ms deadline. What blew the budget was `release()` walking the
discarded tree *after* the search returned. The reserve exists to cover that
gap, and it kept growing for one mechanism: every throughput win builds a bigger
tree, and a bigger tree costs more to walk.

| engine | reserve | why |
|---|---|---|
| `pocket` | 20 ms | inherited default; **failed** p99 by 2.3 ms |
| `pocket_r35` | 35 ms | measured overhead p99 23.8 + chunk overrun 5.4 |
| `pocket_graph` | 50 ms | more search, bigger tree |
| `pocket_sel` | 95 ms | 42% more sims bought 85% more overhead; 18 ms of it was the child-array mirror itself |
| **`pocket_filter`** | **20 ms** | the walk was removed from the move path entirely |

`release()` was 20.4 ms/move (p99 53.3) against a caller-side overhead p99 of
53.3 -- the walk was essentially all of it. It was also pure per-node work
(0.552 us/node, intercept -0.02 ms, R^2 0.986), so there was nothing inside it
to optimize. The only lever was not doing it there. Deferring it to the game
boundary took the reserve from 95 ms back to 20 and left caller-side overhead at
p99 0.05 ms.

A related earlier diagnosis, still true: the **latency tail was the garbage
collector**, and it needed cycle-breaking on discard plus `--gc deferred` before
the 1-second engine passed at all.

Current measured state: 5/5 on `tools.regress_engine`, p99 980.1 ms, max
981.4 ms, 0 moves over budget.

**A latency gate must never play a mirror match.** Self-play inflates tree-reuse
inheritance about 1.7x and hid 4.3 ms of p99 overhead, passing an engine that
then failed against a real opponent. The opponent defaults from the checkpoint,
not from the engine name.

---

## 4. Search performance: the current bottleneck ranking

Commits `95ae9d8` and `1e58565`, measured on `pocket_filter` with
`tools/profile_selection --mode all`. Fixed-position arm, 906.67 ms/move of
clean wall.

**This table replaces every earlier ranking in this repository**, including
"the tree dominates", the 87% figures, `_best_child` at 193.9 ms, and every
probe-path number published before 2026-08-27.

| operation | ms/move | share |
|---|---:|---:|
| wave loop (its own Python) | 90.36 | 10.0% |
| node creation | 76.69 | 8.5% |
| `state.make_move` | 72.56 | 8.0% |
| `_best_child` | 32.19 | 3.5% |
| terminal probes (own loop) | 23.46 | 2.6% |
| backup | 14.01 | 1.5% |
| legal moves | 13.37 | 1.5% |
| state clone | 11.50 | 1.3% |
| proofs | 0.62 | 0.1% |

`state.make_move` splits by caller: the wave loop 65.05 ms over 48,559 calls,
terminal probes 7.51 ms over 5,367. `_adopt` + `release()` now cost **0.00 ms**
inside the move -- they are outside the deadline by construction.

**Two things that were the top of the list are finished.** The terminal-probe
path was the largest host item before the filter landed and is now sixth. And a
dedicated run prices a **free** `_best_child` primitive -- infinitely fast, cost
zero -- at **+4.0% network evaluations**, which closes the native-selection
headroom question that had been open since #45.

**Do not read the `device:` rows as a GPU budget.** They total 570 ms/move
(62.9%) and are host-observed intervals *inside* device-facing calls: launch and
synchronization, not compute. See section 5.

### Object churn, measured as a first-class cost

Per move: 12,605 `GameState` clones, 40,610 `MCTSNode` objects, 5,378
expansions, 5,045 children probed, 164,984 peak live pymalloc blocks, 4.06
blocks/node. Retirement is clean -- 290 blocks held after release.

Two mechanisms proposed for the residual cost of that churn were tested
directly and **failed to reproduce**: pinning a real 69,639-node live tree moved
the loop +1.9%, and a 118x working-set expansion (261 kB -> 31.8 MB) moved it
+4.0%. The residual was mostly the instrument itself (section 11).

### The candidates that exist, and why none of them is being built

| candidate | ms/move | throughput if free |
|---|---:|---:|
| descent `make_move` redundancy | 26.9 | +3.3% |
| wave-loop Python | 90.4 | not characterized |
| node creation (~1.9 us/node) | 76.7 | not characterized |
| native `could_end` | ~9 | +1.1% |

The descent redundancy is fully priced and shaped: `make_move` builds a
`(True, self.winner)` tuple that the descent discards and then re-reads
`state.winner`, 53,048 times a move, structurally confirmed at
winner-reads-per-make_move = 2.00. A hot-path-only `probe_make_move(mv)` would
remove it. It is **archived, not rejected** -- see section 12 for the rule that
archived it.

---

## 5. CUDA: the correct interpretation, and the one it supersedes

> **A CUDA event pair cannot measure GPU busy time.** It spans from when the
> opening event is processed to when the closing one is, so it charges the
> device for the time it spends *waiting to be given work*. Every GPU column
> derived that way in `RESULT_EXPAND_CUDA.md` is wrong.

**CUPTI is the authority.** Measured at wave k=8: **243.7 us/wave of real
device busy inside 1,181.3 us of stream-elapsed time.** Seventy-nine percent of
the apparent "GPU time" is an idle device. Real device work is about 71.6
ms/move -- **7.8% of a move**, not the 482.5 ms and 53% the event-based reading
claimed.

Consequences, all of which stand:

- **The engine is host-bound.** Independently confirmed by behaviour rather than
  by instrument: the selective probe removed host work and returned +20.6% more
  search, which a GPU-bound engine could not have done.
- **The wave is dispatch-bound.** 36 kernels per wave, ~26 us to launch each,
  and the count does not move with wave size -- k=1 and k=8 issue the same work.
  33 of the 36 are the network forward, 3 are masking and softmax.
- **Transfers were overstated about 40x.** Actual DMA is 5.11 us/wave up and
  3.78 us/wave down, about 3.4 ms/move against the 149.2 ms/move claimed. "Do
  not optimize transfers" was the right call for the wrong reason.
- The four per-wave synchronizations were accidental pageable-memcpy syncs, not
  a bool dtype as first guessed; the CUDA graph removed them.

Authority: `RESULT_KERNEL_TRACE.md`. `RESULT_EXPAND_CUDA.md` carries the
correction inline and must not be read without it.

---

## 6. Model size: 172k and 6.77M are raw-strength peers

**The 172,389-parameter network beats the 6,766,386-parameter gen-22 network at
equal wall clock, 0.5854 [0.5360, 0.6349] over 240 games.** A 39x parameter cut.
Pre-registered in `EXPERIMENT_MODEL_SIZE.md` before the first game, with three
named ways the prediction could be wrong; one of them fired, so the prediction
was right for the wrong reason.

**The reason matters more than the result. Shrinking the network does not buy
much search.** The 39x cut bought only **1.24x** more simulations, which means
roughly three quarters of a simulation is tree bookkeeping rather than network
evaluation. Anyone reaching for a smaller network to go faster should read that
number first.

At raw strength the two are approximately peers: the 172k pocket is at parity
with the 6.77M teacher on the gregory d4 and d5 rulers, so network capacity is
not the binding constraint at this budget.

**Architecture is not the strength lever.** A four-arm A/B on a frozen gen-22
corpus found that at equal wall clock all architectures tie within a +/-0.032
band; the residual-tower hypothesis is refuted, and residual + normalization is
a net negative at 4-6 layers. The **only** change that paid was the 1x1-squeeze
head, which is where the entire 7.5x size win came from -- the legacy heads
flattened the conv stack into a Linear and held 83.6% of the parameters.
Parameters are not latency: an 8.8x CPU gap at matched parameter counts, and the
ranking flips on CUDA.

---

## 7. Distillation: what is known, and what is not

Four studies, in the order they were run. They resolve less than they look like
they resolve, and it is worth being blunt about that.

**More teacher search makes a stronger player and a worse teacher.** The
teacher genuinely plays better with more simulations -- that was measured
directly, teacher against itself at two budgets. But the student distilled from
an 800-sim teacher **lost** head-to-head to the student distilled from a 50-sim
teacher, 0.411 pooled over three seeds, and lost on all four anchors. The
diagnosed mechanism is PUCT dilution: at 800 sims the teacher puts a visit on
every legal move of a mate-in-1 and keeps only 0.693 of its mass on the winning
move, against 0.825 at 50 sims. Capacity is not the cause -- the gap is
identical at 5.3x the student size. **Raising teacher simulations is not a
lever.**

**Fixing the targets did not fix the students.** Solved-node propagation was
the direct intervention on that diagnosis, and it worked on the targets exactly
as designed: win-move mass went 0.7356 -> 1.0000 and the teacher got stronger at
every budget. The student recovered **0.97% of the distillation penalty**
(+0.0009 against a +0.0446 threshold), with 2.5x the resolution needed to see
the effect. Reconciliation changed **zero** mate-in-1 argmaxes. This is a
powered null, not an underpowered one. Correctness was never the lever -- stop
proposing tactical fixes for distillation.

**Search churn overstates search improvement about 7x.** Every doubling of
simulations changes ~15% of the teacher's moves but is worth only +0.019 win
rate (teacher against itself, 800 v 200, 800 games, p=0.0098). The visit
distribution converges while argmax churn stays flat. Refereeing individual
moves is a structural dead end -- do not rebuild it; play the match.

**Target extraction is closed with no transformed arm.** Visit-argmax and
value-argmax disagree on 42% of positions at both budgets, but only by 0.03 of
value. When 800 simulations changes the top move it is a coin flip whether the
new move is better (median +0.0022 against a 0.013 ruler), bimodal 48/46 and
cancelling. No value-gap threshold can sort the good changes from the bad. Q
stability across simulation counts was never measured and is the first thing to
run if this is reopened.

---

## 8. The gate failure: an eight-month stall caused upstream

**Expert iteration went 564,144 games -- 39% of the project's lifetime total of
1,439,792 -- without promoting a teacher.** The cause was not the lineage
running out of room. It was the promotion gate.

(Those are the counts on the day of the diagnosis. The run continued a few more
hours and stopped at 564,480 since promotion, 1,440,128 lifetime, which is where
`models/expert_iter_v2/state.json` sits today. Nothing has moved since.)

`mcts_edge`, the teaching signal, was still 0.811 at gen 22, down only 0.066
across 22 generations. Search still beat the raw network 81% of the time. There
was plenty left to distill.

The gate asked the student to beat the teacher by 0.02 on a fixed WinBlock
heuristic, where `best_heur` was the teacher's own score of 0.9267. Across 127
logged attempts the students averaged 0.9002 -- **0.0265 below the teacher** --
so they needed +0.047 and the best student the run ever produced reached 0.9317.
Replaying the decision logic: **zero of 127 should have passed.** Meanwhile
head-to-head averaged 0.545 and was above 0.500 on *every single attempt*. The
students really were better.

**These panels are deterministic.** `_play_fixed_match` reseeds Python and numpy
per game and promotion runs raw argmax, so a panel score is a reproducible
function of the weights -- measured three times against gen 22 it returned
0.926667 every time. There is no sampling noise here and nothing to get lucky
on. **Never reason about these scores as noisy samples.**

The real finding is a **non-transitivity the gate could not express**: the
students are better than the teacher head-to-head and slightly worse against a
fixed heuristic. Both facts are deterministic. A gate requiring improvement on
both requires something the lineage does not produce.

The fix (2026-07-25, `1ec0b86` + `cc346a7`): head-to-head is the *only*
improvement criterion, because it is the only panel that cannot saturate --
centred on 0.500 by construction at every generation. Everything else became a
no-regression guard with a high-water floor. `--promote_margin` is gone,
replaced by `--winblock_tolerance`. Guard floors widen with expected wander
(`max(tolerance, --noise_sigmas * _panel_sigma)`, default 2.5 sigmas) -- not as
a sampling-noise correction, but as a scale-free proxy for the student's own
weights moving as training continues. Bars stay high-water marks via `max`,
because with winblock demoted to a guard a direct assignment would slide the
floor down every generation. Failures are now recorded in the metrics row:
attributing this required replaying 90k rows because only the scores had ever
been written down.

Replayed through the new gate, the 127 real attempts promote at **39%**, and a
null student with no real edge promotes at **5%** -- an 8.3x separation. Winblock
still blocks 20% of attempts, which is the guard doing real work rather than a
malfunction.

> **The repaired gate has never been run in production.** The run stopped on
> 2026-07-25, the same day the fix landed, and the engine work started three
> days later. The 39% figure is a replay, not an outcome.

---

## 9. Evaluation: what is trusted, and what is not

**Closed-loop ELO is not trusted and never should have been.** The league
`best.pt` reached ELO 4437 and then lost **0 for 40** to a shallow MCTS running
over its own weights. Self-play ELO measures position within a gene pool, not
strength. Certify by fixed external panel, never by ladder rating.

**The anchor ladder is the current ruler.** `gregory(d4)` saturated at 1.0000
(60/0/0) and was retired as a primary discriminator; it stays as a low anchor.
The replacement is the frozen engine at other budgets -- same network, same
search, budget the only variable, asserted by a test rather than by intention.

| rung | budget | role |
|---|---|---|
| `anchor_A` | 250 ms | low |
| `anchor_B` | 500 ms | low |
| `anchor_C` | **2000 ms** | **primary anchor** |
| `anchor_D` | 4000 ms | high |

Rungs above 1,000 ms are latency-exempt by design; their job is to be a hard
fixed opponent, not to ship. The ladder is ordered, and one doubling of clock is
worth 0.59-0.69, decaying about 0.03 per doubling. The 1,000 ms agent scores
0.3750 against `anchor_C`, which is the headroom an anchor is chosen for --
pick an anchor for headroom, not for proximity to 0.5.

**The ladder's known weakness:** every rung is the same gen-22 network. It is
fixed and deterministic relative to future candidates, which is what an anchor
must be, but it is **not an independent opinion about strength**. A genuinely
external opponent remains wanted. Also, more than half of simulations never
touch the network at 4,000 ms, and throughput degrades 25% as trees grow.

**Two resolution floors bound everything above.**

- **Throughput:** `nn/second x deadline` replicates to 1.7-6.1% run to run.
  Never quote nn/move -- it swings 22% run to run on the same engine because
  the composition of positions changes.
- **Panels:** two panels correlate at only +0.266 with each other, and one
  training run is one sample of a trajectory. A real 0.02 cannot be separated
  from luck. Checkpoint wobble is real (0.080 span) but SWA and EMA both failed
  to capture it. More games buy nothing against a deterministic panel -- only
  replicate training runs would.

---

## 10. Seeds and openings

Openings come from `scripts.expert_iter._eval_openings(n, seed)`; colours are
swapped within each opening; the per-game RNG is reseeded to
`seed + opening_idx*2 + side`. A seed therefore **is** an opening set.

| namespace | seed | used for |
|---|---|---|
| headline | 6100 | the published 0.7229 |
| ladder | 6200 | ordering between adjacent rungs |
| anchor | 6300 | a candidate against a frozen anchor |
| tune | 6400 | elimination rounds of a sweep |
| confirm | 6500 | held out: the powered final between survivors |
| gregory_d4 | 6144 | the retired low anchor |
| expand | 6700 | `_expand_wave` instrumentation |
| kernels | 6800 | kernel tracing |
| select | 6900 | native-selection development |
| select_ab | 7000 | held out: native-selection confirmation |
| defer | 7300 | deferred retirement |
| probe | 7400 | probe ablation and cost |
| probe_ab | 7500 | probe A/B |

Separate namespaces because the same fixed openings are reused across many
experiments. A sweep that eliminates on the positions it is later confirmed on
overfits them -- cheap to avoid, invisible if not.

---

## 11. Instrumentation errata

Every one of these produced a number that looked fine. They are collected here
because the pattern is the finding: **this project's instruments have been wrong
more often than its hypotheses.**

**1. In-process Python samplers are 6.4x biased toward C.** A GIL-bound sampler
reported the tree at 0.0% of a move when it was 26.9% -- wrong by 353x. The real
failure was upstream: the ground-truth test that validated it had **both arms in
pure Python**. A calibration workload must span the axis the instrument is used
across.

**2. A CUDA event pair cannot measure GPU busy time.** Section 5. It charges the
device for waiting. Overstated device work by ~7x and transfers by ~40x.

**3. Wrapper timings under-price object-churning paths by 20-25%.**
`AttributedTimer.calibrate()` times a wrapper around a trivial Python no-op, but
the wrappers that matter sit on pybind bound methods reached through a Python
subclass. Measured `price()` recovers **1.250x** the bare cost (0.982 us/call
calibrated against the 1.338 actually needed), replicated independently on a
different code path (`_best_child` in situ at 1.579 us/call, 1.6x its tight-loop
calibration). The two biases point opposite ways: a wrapper-derived ms/move is
an **upper bound on time** and a **lower bound on value**. Consequence: every
`probe ms/move` published before 2026-08-27 is ~25% too high -- read
`154.6 -> 46.6` as `~123.7 -> ~37.2`. **No decision changed**, because every
gate was decided by uninstrumented arms. `tools/profile_selection` prices in
situ and deflates; `tools/probe_ablation.price()` does not.

**4. A warmup can land in the numerator and not the denominator.**
`play_match(warmup=N)` clears the *players* -- records, policies, counters,
reuse tallies -- but an instrument accumulating from outside the players keeps
counting warmup games while the denominator excludes them. Two games in
fourteen is 16.7%, and it moved a published share by four points.

**5. Simulations per move cannot price an instrument.** A free descent is not a
unit of work; identical openings swing 35-57%. Measure per network evaluation.

**6. A mini-board indexing bug** invalidated two strata of
`RESULT_SEARCH_DISAGREEMENT.md` -- the forced-target mini-board table and every
`mini_win_available` figure -- and reached two further tools. Global metrics and
conclusions were unaffected. Full account: `ERRATA_MINI_INDEX_BUG.md`.

**7. A latency gate playing a mirror match** inflates tree-reuse inheritance
~1.7x and hid 4.3 ms of p99 overhead, passing an engine that failed against a
real opponent.

**8. Two overlapping GPU runs corrupt a timing study silently.** Both wrote the
same JSON, neither crashed, and k=8 kernel time differed 1.9x from GPU sharing.
`$!` from a bash launcher is the subshell's pid, and `tasklist | head` truncated
the proof. Timing tools need an exclusive lock and a recorded GPU baseline; this
box idles at ~32%.

**9. The lottery giant is a constant function, not a player.** It returns move
24 for every board (float64 delta 6.66e-16); 115 of 135 conv layers never left
initialization. It is policy-only, so it can never be searched. This **voids
every number that used it as an opponent or anchor**, including league stage-6's
10% slice and parts of `benchmark_suite` and `RESULT_M2.md`.

---

## 12. Rejected work -- do not rebuild these

Each was measured or attempted and closed. The number is why.

| idea | verdict |
|---|---|
| transposition DAG | 3.1% ceiling, measured before building |
| within-search symmetry folding | exactly 0.0% |
| dedicated inference server | GPU never starved; wave 8 -> 64 buys +9% |
| fused native terminal probe | +1.48 ms/move, three independent refusals |
| broad native tree port | superseded -- selection is now 3.5% of a move |
| target transformation / value-gap thresholds | no threshold can sort good changes from bad |
| refereeing individual move changes | structurally dead; play the match |
| raising teacher simulation count | distills worse, measured |
| tactical/correctness fixes for distillation | powered null, 0.97% recovery |
| `torch.compile` | requires Triton; no Windows build |
| architecture search (depth/width/residual) | all arms tie at equal wall clock |
| SWA / EMA for checkpoint wobble | failed to capture a real 0.080 span |
| c_puct sweep at 50 sims | 1.5 already optimal; no free win |

**The stopping rule, adopted 2026-09-04.** The engine-optimization branch is
closed. The remaining individual opportunities are mostly below the
reproducibility floor of the throughput metric, and at that point an ordinary
A/B stops being evidence and becomes ceremony.

> **No new engine optimization unless it has either a measured 5-10%+
> recoverable ceiling, or a proof of semantic identity plus a measurement method
> capable of resolving its expected throughput gain.**

That is why the 26.9 ms descent redundancy is archived despite being fully
priced: +3.3% against an estimator that replicates to 1.7-6.1% cannot be
confirmed by the uninstrumented A/B that would have to confirm it.

---

## 13. The engine is frozen for the training phase

`pocket_filter` is the chassis. **It does not move while models train.**

This is a scientific requirement, not tidiness. If the search implementation
keeps changing while networks train, a stronger generation N+1 cannot be
attributed -- learning may have improved, or the chassis may have moved
underneath it, and no amount of after-the-fact analysis separates those.

Enforced by three layers in `tools/engine_registry.py`, each catching drift the
others cannot see:

1. **Resolved-config fingerprint** -- sha256 over every parameter that can
   change how a player plays, including code defaults no spec mentions.
2. **Checkpoint sha256** -- catches a retrained or overwritten `.pt`.
3. **Engine source sha256** over `agents/mcts.py`, `agents/agent_base.py`,
   `agents/neural_net_agent_3.py`, `engine/rules.py`, `engine/game.py`. Hard
   failure for an anchor **and for the deployment baseline**; warning for a
   candidate.

Recovery from drift is a tag, not a repair:

    git worktree add ../uttt-anchor arena-1s-baseline

`--allow-anchor-drift` exists to make an override visible in the logs, not to
make it acceptable.

---

## 14. Open work

**Only one thing is open, and it is not engine plumbing.** See `ROADMAP.md`.

The learning loop has a repaired promotion gate that has never run. The teacher
lineage is parked at gen 22 with 564,480 games since the last promotion and a
teaching signal (`mcts_edge` 0.811) that says there is still plenty to distill.
The question the next program answers is whether the machine can be left
training for a week and come back stronger.

One genuinely open question carries over from the engine work and is recorded
rather than resolved: **why is a distilled student weaker than its teacher on a
fixed heuristic while beating it head-to-head?** The non-transitivity is
reproducible and deterministic. It was worked around, not explained.
