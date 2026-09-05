# Toward the Smallest, Strongest Ultimate Tic-Tac-Toe Player

> # SUPERSEDED 2026-09-04 by `ROADMAP.md`
>
> **This is no longer the direction.** It is kept as a record of what was
> predicted in July 2026, which is worth reading against what the measurements
> then returned -- several of its central proposals were built and found to be
> worth nothing.
>
> | this document proposed | measured outcome |
> |---|---|
> | efficient graph search (transpositions) | **3.1% ceiling** -- rejected before building |
> | symmetry exploitation in search | **exactly 0.0%** -- rejected |
> | a substantially stronger offline teacher | more teacher search distills **worse** (student h2h 0.411) |
> | exact tactical propagation | shipped, and recovers **0.97%** of the distillation penalty |
> | a small structure-aware network | confirmed -- 172k beats 6.77M at equal clock |
> | ~150-200 simulations as the sweet spot | obsolete; the agent now runs thousands under a clock |
>
> The one proposal that paid was the small network, and even that paid for a
> different reason than the one given here: the win came from the 1x1-squeeze
> head, not from structure awareness, and shrinking the network bought only
> 1.24x more search.
>
> Current direction: `ROADMAP.md`. Current facts: `CURRENT_STATE.md`.

> Status: strategy / vision document. Recorded 2026-07-18 at the owner's
> request to capture direction. NOT yet actioned -- do not implement from this
> without an explicit go-ahead. Builds on the gen-13 search-value measurement
> (loss_logs/sims_sweep_gen13.log; companion to STRENGTH_NEXT.md). Non-ASCII
> punctuation was normalized to ASCII (em-dash -> "--", smart quotes ->
> straight quotes) per repo convention; wording is otherwise unchanged.

## Executive conclusion

The Generation 13 experiments establish one major fact: the neural network
cannot be evaluated independently from the search algorithm that actually uses
it.

Against Gregory at depth three, the raw Generation 13 network scores only 39%.
The same network combined with Monte Carlo Tree Search scores:

| Search budget | Win rate vs. Gregory (d3) | Gain |
|---|---|---|
| 0 simulations | 39.0% | -- |
| 50 simulations | 63.0% | +24.0 |
| 100 simulations | 67.2% | +4.2 |
| 200 simulations | 75.7% | +8.5 |
| 400 simulations | 77.8% | +2.1 |

The immediate operational conclusion is that 50 simulations substantially
underuses the network. Approximately 150-200 simulations appears to be the
current practical sweet spot against Gregory, subject to browser latency. The
increase from 200 to 400 simulations is too small relative to the observed
statistical noise to establish a meaningful improvement.

The deeper conclusion, however, is not that the neural network has definitively
become the bottleneck. This experiment shows only that the current combination
of network, MCTS implementation, search parameters, and opponent gains little
from doubling the budget beyond 200 simulations. The plateau could be caused by
network capacity, value calibration, policy concentration, exploration
parameters, tree implementation, opponent characteristics, or some combination
of them.

The strongest final player is therefore unlikely to be merely a larger network
or merely a larger search. It should be a compact hybrid:

> A highly optimized game engine, a small structure-aware policy/value network,
> efficient graph search with exact tactical propagation, a compact opening and
> endgame knowledge layer, and an offline teacher substantially stronger than
> the deployed student.

That is the most credible route toward simultaneously minimizing model size and
maximizing playing strength.

---

## 1. What the current experiments establish

### Search is part of the product

The promotion gate currently evaluates a single network inference, while the
deployed Hard player uses the network as a policy and value oracle inside MCTS.
Those are materially different agents.

The 24-point swing from 39% to 63% demonstrates that Gregory is particularly
effective at exploiting tactical errors in the raw policy. Search repairs many
of those errors. A gate that evaluates only the raw network can reject a network
that is substantially stronger when used in its intended configuration.

AlphaZero-style systems are explicitly built around this partnership: the
network guides search, and search improves the action distribution produced by
the network. The relevant unit of evaluation is the combined player, not either
component in isolation.

The promotion system should retain raw-network tests because they provide useful
diagnostic information, but the decisive promotion result should come from the
deployed agent.

### Two hundred simulations is an operational knee, not a proven ceiling

The current evidence supports the following restrained interpretation:

- Search is enormously valuable between zero and 50 simulations.
- Search remains valuable through approximately 200 simulations.
- The additional value between 200 and 400 simulations is unresolved at 300
  games.
- Two hundred simulations is currently the best supported default for strength
  per unit of search.
- The experiment does not prove that 200 simulations is the optimal
  training-teacher budget.
- The experiment does not prove that network capacity is the sole cause of the
  plateau.

A 400-simulation teacher may still produce better policy targets, more accurate
values, or more useful tactical examples even when the same network gains little
by using 400 simulations during deployment. Teacher quality must ultimately be
measured by the strength of the student trained from it.

### Gregory has become a limited ruler

A 78% win rate is still measurable, but Gregory is no longer sufficient as the
primary benchmark. A single opponent can also produce misleading results because
an agent may be unusually strong or weak against one style.

The benchmark panel now needs:

1. Strong external agents.
2. Historical network checkpoints.
3. Several independently designed search agents.
4. Tactical regression positions.
5. Separate first-player and second-player results.

CodinGame remains useful because its Ultimate Tic-Tac-Toe arena contains
thousands of submitted agents and explicitly supports MCTS- and minimax-based
approaches. It provides a substantially more diverse competitive population than
the internal three-agent panel.

---

## 2. Define "smallest possible" correctly

There is no single definition of smallest. At least four costs matter:

- Download size.
- Runtime memory.
- Neural-network evaluations per move.
- Wall-clock latency on the weakest supported device.

These costs are related but not interchangeable. A smaller network can be slower
if its operations are poorly supported by the browser backend. A tiny model
loaded through a general-purpose inference runtime can also produce a larger
total download than a somewhat larger model with a custom inference kernel.

The development objective should therefore be a Pareto frontier rather than
"minimize parameter count":

> Maximize playing strength under a specified move-time and memory limit, and
> then minimize binary size without producing a statistically detectable loss of
> strength.

All serious comparisons should be performed at two budgets:

- Equal wall-clock time.
- Equal neural-network evaluation count.

Simulation count alone becomes misleading when comparing different networks or
search implementations. Terminal nodes, transposition reuse, batched
evaluations, and exact solvers can make one simulation substantially cheaper or
more informative than another. Research on Monte Carlo Graph Search similarly
reports results in neural-network evaluations because that better reflects the
principal computational expense.

---

## 3. The ideal deployed architecture

### A. A bitboard game engine

The engine should be almost entirely separate from the neural layer. It should
provide:

- Compact bitboard representation.
- Constant-time or near-constant-time legal-move generation.
- Incremental local-board and global-board win detection.
- Fast make-and-unmake operations.
- Zobrist or equivalent state hashing.
- Canonical handling of the active destination board.
- Exact terminal detection.

Ultimate Tic-Tac-Toe has substantial exploitable structure. There is no reason
to make the network relearn whether three marks form a line, whether a local
board is closed, or whether a move is legal.

All eight rotations and reflections should be handled consistently. Training
data can be augmented under these symmetries, and search states can optionally
be canonicalized before transposition lookup. Policy outputs must then be
transformed back to the original orientation.

### B. A hierarchical network rather than a generic image network

The board is not merely a flat 9-by-9 image. It is nine repeated 3-by-3 games
connected through a second 3-by-3 game. The network should reflect that
hierarchy.

A compact candidate architecture would contain:

1. A shared microboard encoder applied identically to each of the nine local
   boards.
2. Exact local features such as board status, immediate win squares, immediate
   blocks, and remaining legal cells.
3. A small global encoder operating over the nine microboard embeddings.
4. A policy head combining global context with each cell's local
   representation.
5. A value or win/draw/loss head.
6. A legal-move mask applied outside the network.

Weight sharing across the local boards can reduce parameter count while imposing
the correct inductive bias. The network should not require separate filters to
learn the same local pattern in nine different locations.

Rather than guessing the correct size, train a controlled ladder -- for example
approximately 10,000, 25,000, 50,000 and 100,000 parameters. Evaluate every
model at equal time and equal evaluation budgets. The desired model is the
smallest one on the strength frontier, not necessarily the smallest model that
trains successfully.

A second architecture worth testing is a microboard lookup encoder. Every local
board can be represented by a compact ternary state, so a shared learned
embedding or exact feature table can replace several convolutional layers. This
may be especially effective in WebAssembly, but its memory and cache behavior
must be benchmarked against a small MLP.

### C. A custom integer inference path

The final student should be tested under quantization-aware training, beginning
with signed INT8 weights and activations. Lower precision should be considered
only after INT8 is stable. Quantization can substantially reduce model storage
and inference cost, but the relevant acceptance test is playing strength with
search -- not policy accuracy in isolation. Post-training quantization research
demonstrates that low-precision inference can preserve model accuracy in
suitable architectures, although this must be validated specifically for the
UTTT network.

For a very small network, a generated custom WASM/SIMD inference implementation
may be preferable to shipping a general neural-runtime dependency. The runtime
itself can otherwise dwarf the model.

---

## 4. Improve search before assuming the network must grow

The current MCTS should remain the baseline, but it should not be treated as the
only viable search method.

### Tree reuse and transpositions

The search tree should survive between moves. After the opponent responds, the
matching child becomes the new root rather than discarding all prior
computation.

Identical states reached through different move orders should share information
through a transposition table or directed acyclic search graph. Monte Carlo
Graph Search was developed specifically to allow this information flow and has
demonstrated lower memory use and fewer redundant network evaluations than an
isolated tree.

### Proven-result propagation

Search nodes should support exact WIN, LOSS and DRAW states. When every response
to a move has been proven losing, the parent can be proven winning without
spending additional neural evaluations.

A terminal solver can:

- Stop searching already proven branches.
- Prefer the shortest forced win.
- Prefer the longest resistance in a forced loss.
- Override uncertain network values with exact results.
- Supply higher-quality training examples.

These mechanisms have been integrated successfully into neural-guided graph
search and are directly applicable to a deterministic perfect-information game.

### Gumbel search at small budgets

The current PUCT implementation should be compared with Gumbel AlphaZero search
at 16, 32, 50, 100 and 200 neural evaluations. Gumbel planning was designed in
part to produce better policy improvement when only a small number of
simulations is available. That makes it particularly relevant to a small
in-browser player.

This test could shift the curve left: a better search algorithm at 50 or 100
evaluations may match the current 200-simulation result.

### Alpha-beta and proof-number search

Ultimate Tic-Tac-Toe is deterministic, has a relatively constrained branching
factor through much of the game, and contains tactical forced sequences. A
neural-guided negamax or principal-variation search should therefore be
implemented as a serious competitor to MCTS.

The test set should include:

- PUCT MCTS.
- Gumbel/sequential-halving search.
- Iterative-deepening alpha-beta with a transposition table.
- Proof-number search for proving forcing lines.
- A hybrid that invokes exact tactical search inside MCTS.

Whichever method produces the greatest strength per millisecond should become
the deployed search. Algorithmic identity is not a product requirement.

### Adaptive search rather than a fixed number

Not every position deserves 200 simulations. The engine should allocate
additional time when:

- The policy distribution has high entropy.
- The best two root moves have similar values.
- The raw network and shallow search disagree.
- Immediate local or global threats exist.
- The position is near a proven tactical result.

It should stop early when one move has a stable, decisive lead. A time-limited
adaptive search can play stronger than a fixed-simulation search at the same
average latency.

---

## 5. Training the strongest small student

The deployed network should be small. The training teacher does not need to be.

A larger network, deeper search, an ensemble, exact tactical solvers, and
accumulated opening or endgame knowledge can all be used offline to train a
compact student. Distillation work in other deterministic board games has shown
that a specialized compact evaluator can remain highly competitive under limited
CPU resources.

### Hard-position replay

The most valuable new dataset is likely not another indiscriminate block of
ordinary self-play. It is a replay buffer of positions where:

- Gregory defeated the deployed agent.
- Raw policy and MCTS policy disagree substantially.
- Increasing search changes the selected move.
- The value prediction changes sharply after one move.
- The agent misses a forced win or fails to prevent a forced loss.
- Historical generations disagree.
- External opponents expose a repeatable weakness.

These states should be reanalyzed with the strongest available teacher and
sampled more frequently during training.

### Search-control curriculum

Ordinary self-play begins every game at the initial position, which can
overrepresent common openings and underrepresent rare but decisive midgame
states. Go-Exploit improves AlphaZero-style training by starting some
trajectories from archived states of interest, thereby generating more
independent and diverse value targets. A UTTT version should sample from
tactical failures, uncertain states, rare destination-board patterns and
external games.

### Progressive teacher budgets

Low simulation counts can be used early in training when the network itself is
weak. The teacher budget can increase as the network improves. MiniZero
experiments found that progressive simulation schedules can allocate training
computation more efficiently than using the final search budget throughout the
entire run.

The present 200-simulation teacher should therefore remain the default while two
controlled tests are run:

- Students trained from 200-simulation targets.
- Students trained from a mixture of 200- and 400-simulation reanalysis targets.

The question is not whether the 400-simulation teacher beats the 200-simulation
teacher directly. The question is whether its student becomes stronger enough to
justify the additional generation cost.

### Training-only auxiliary heads

A small network can be encouraged to learn the game's hierarchy through
auxiliary outputs that are discarded at deployment:

- Outcome of each local board.
- Immediate local win and block squares.
- Macroboard threat lines.
- Whether the next destination board is constrained or free.
- Estimated distance to terminal.
- Search uncertainty or policy disagreement.

These heads add no deployed model cost after the shared representation has been
trained.

### Historical and external opponents

Self-play should include a mixture of:

- Current best checkpoint.
- Several historical checkpoints.
- Gregory and other deterministic anchors.
- External ladder games.
- Search variants with different tactical styles.

This produces a more diverse strategy distribution and reduces the risk that the
agent becomes exceptionally strong against its own family while retaining
exploitable blind spots.

---

## 6. Add exact knowledge where it is cheaper than learning

Under the standard rules studied in the literature, Ultimate Tic-Tac-Toe is
known to be a first-player win. Researchers have proved the existence of a
winning first-player strategy, bounded an optimal win to at most 43 moves, and
identified the first two moves of any optimal strategy. This is not yet
equivalent to having a compact, complete strategy available for every reachable
position, but it changes the ultimate target: perfect play is theoretically
definable.

A truly ultimate player should exploit exact knowledge in three regions.

### Opening layer

Store proven or extremely high-confidence opening moves and continuations. An
opening book eliminates search variability in the most frequently encountered
states and consumes little space if represented as a compact trie or hashed
state-to-move table.

The book should be generated from strong search and checked against the exact
rule variant used by the site. Details such as what happens when a player is
directed to a completed local board must match precisely.

### Midgame neural search

Use the compact network and adaptive search where exhaustive proof remains
expensive. This is the region in which learned strategic evaluation provides the
most leverage.

### Endgame solver or tablebase

When few legal cells remain, switch from approximate evaluation to exact
minimax, proof-number search, or a compressed tablebase. The threshold should be
based on measured solve time rather than a fixed move number.

Solved endgame positions can also be fed back into training as perfectly labeled
policy and value targets.

Over time, the exact opening book can grow forward while the tablebase grows
backward. The neural-search region between them becomes progressively smaller.

---

## 7. Replace the promotion gate

Every candidate should be measured in four ways.

### Raw-network diagnostics

Record policy loss, value error, tactical-suite accuracy and raw head-to-head
play. These diagnose whether training is improving the representation.

### Deployed-agent matches

Evaluate the network with the exact search configuration intended for release.
This should determine promotion.

### Fixed-cost comparisons

Run at:

- Equal wall-clock time.
- Equal neural-network evaluations.
- The production browser configuration.

This separates model quality from model speed.

### Paired statistical evaluation

Each opening should be played in both directions, with player roles reversed
where possible. Report:

- Wins, draws and losses separately.
- Mean game score.
- Paired score difference.
- Confidence interval from paired bootstrapping or an equivalent method.
- First-player and second-player performance.
- Tactical-suite regressions.

Promotion should require a positive result against a composite panel, not merely
victory over one incumbent or one handcrafted opponent.

The existing seeded results should be reanalyzed pairwise. That may clarify
whether the unusual 100-to-200 jump is a genuine effect or opening-level
variance.

---

## 8. Recommended development order

### Immediate

1. Change the promotion gate so that deployed net-plus-search strength is
   decisive.
2. Benchmark 150 and 200 simulations in real browser WASM on weak mobile
   hardware.
3. Deploy approximately 150-200 simulations if latency remains acceptable.
4. Preserve the complete search curve and seeded game records.
5. Enter the current player into an external ladder.

### Highest-value search work

6. Add tree reuse and a transposition table.
7. Add exact terminal-result propagation.
8. Compare PUCT, Gumbel search and neural alpha-beta at equal time.
9. Add adaptive time allocation.
10. Record root entropy, Q-value gaps, average depth, unique states and neural
    evaluations.

### Highest-value training work

11. Build a hard-position replay archive.
12. Train a hierarchical, weight-shared architecture ladder.
13. Compare 200-simulation targets with limited 400-simulation reanalysis.
14. Add archived-state curriculum sampling.
15. Distill the strongest teacher into the smallest statistically equivalent
    student.

### Compression and exactness

16. Apply structured pruning and INT8 quantization-aware training.
17. Replace a general inference runtime with a specialized WASM kernel if it
    reduces total size.
18. Generate a compact opening book.
19. Add an exact late-game solver and begin accumulating tablebase entries.
20. Repeat the architecture and search sweep after every major algorithmic
    change.

---

## Final position

The current data does not say, "Make the network bigger and stop thinking about
search." It says:

- The raw-network gate is evaluating the wrong agent.
- Search is currently the largest demonstrated source of strength.
- Approximately 200 simulations is the present operational sweet spot against
  Gregory.
- The search plateau has not yet been causally attributed.
- Gregory is no longer a sufficiently discriminating primary benchmark.
- Training-teacher search and deployment search must be evaluated separately.
- The smallest strongest player will be a specialized hybrid, not a pure neural
  network.

The most promising final design is:

> A tiny hierarchical INT8 policy/value network, trained by a much stronger
> offline teacher; a bitboard engine; Gumbel or PUCT graph search with tree
> reuse and proven-result propagation; adaptive per-position search; a compact
> exact opening layer; and an exact late-game solver or tablebase.

The network should supply generalization. Search should correct uncertainty.
Exact logic should handle everything that can be solved more cheaply than it can
be learned. That combination offers the best plausible route to the smallest
deployable player that is also as close to perfect Ultimate Tic-Tac-Toe play as
the available computation permits.
