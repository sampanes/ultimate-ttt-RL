# ultimate-ttt-RL
From the ground up, build an ultimate Tic Tac Toe RL arena to get the best AI

**[Play now](https://sampanes.github.io/ultimate-ttt-RL/play/)** &nbsp;|&nbsp; [About](https://sampanes.github.io/ultimate-ttt-RL/) &nbsp;|&nbsp; [HuggingFace Space](https://huggingface.co/spaces/ratherbeembed/ultimate-tic-tac-toe)

---

## Quickstart

Zero-to-training, copy-paste top to bottom. By the end you'll have verified your
GPU, sanity-checked the game engine, trained a self-play agent, and played against
it. Every command is run from the repository root with the virtual environment
active.

> **Prerequisites:** Python 3.10+, Git, and (recommended) an NVIDIA GPU with a
> current driver -- run `nvidia-smi` and confirm it prints your card and a CUDA
> version. CPU-only works but training is dramatically slower.

### 1. Clone the repository

```bash
git clone https://github.com/sampanes/ultimate-ttt-RL.git
cd ultimate-ttt-RL
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch (CUDA build), then the remaining dependencies

Do **not** run a bare `pip install torch` -- PyPI may serve a CPU-only wheel. Select
the install command for your CUDA version at
<https://pytorch.org/get-started/locally/> (check your version with `nvidia-smi`):

```bash
# Example for CUDA 12.8 -- matches requirements.txt (torch 2.7.1+cu128)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 4. Confirm the GPU is live -- and watch it work

```bash
python -m tests.env_check both
```

Prints your Python / CUDA / cuDNN / GPU details, then runs a CPU-vs-GPU
matrix-multiply benchmark. This is the fastest way to confirm the card is actually
being driven (compare the reported matmuls/sec between CPU and GPU).

### 5. Sanity-check the game engine

```bash
python -m tests.rule_test
```

Expect `[OK] All tests passed.` -- this validates the Ultimate Tic-Tac-Toe rules:
mini-board win detection, the "play in the board your opponent sent you to"
constraint, and legal-move generation.

### 6. Train an agent (the GPU-intensive step)

Kick off a fresh Actor-Critic league run. `--parallel` is the number of games
batched through the network at once -- raise it to push the GPU harder, lower it if
you hit out-of-memory:

```bash
python -m scripts.train_league --parallel 64 --curriculum --seed_model "" --chunks 5 --chunk_games 1000
```

You'll get a live progress bar per chunk showing loss, win-rate (WR), and ELO. The
agent begins from random weights, warms up against a random opponent (curriculum
stage 0), and faces progressively stronger opponents as it improves. Checkpoints
are written to `models/league_pg/`, and new ELO highs are archived under
`models/league_pg/archive/`.

> **`--seed_model ""` is required for a genuine fresh start.** Without it the
> trainer attempts to seed weights from a prior checkpoint that does not exist in a
> clean clone and exits with `FileNotFoundError`. Once you have your own
> checkpoints, drop `--seed_model ""` and add `--resume` to continue from the latest.

> **`--parallel 64`** is the empirically confirmed best for an RTX 3080. Larger values
> (128, 256) reduce gradient update frequency and stall training; 512 OOMs.

### 7. Play against your trained agent (terminal)

Once training has produced an archive checkpoint, point the human-vs-AI script at
one of the files it created:

```bash
# Use whatever file exists under models/league_pg/archive/
python -m scripts.play_human --checkpoint models/league_pg/archive/archive_0000_NeuralNetAgentPG.pt
python -m scripts.play_human --checkpoint models/league_pg/archive/archive_0000_NeuralNetAgentPG.pt --side O
```

> Trained weights (`*.pt`) are **gitignored**, so a fresh clone ships with none. You
> must complete step 6 before `play_human`, the `pg_best` agent, or any
> checkpoint-based tournament command will work.

### 8. Run the multi-agent arena (the bots that compete for the top spot)

Step 6 trains a *single* agent up a curriculum. The **arena** is the headline
event: a whole **population** of differently-shaped networks training in parallel,
playing each other and a shared opponent pool, ranked by ELO on a live ladder. Weak
agents that stagnate are **retired**; the strong ones are **cloned-and-mutated** or
replaced by fresh random architectures -- so the field keeps evolving.

```bash
# Spawn 4 random-architecture agents and train continuously (Ctrl+C to stop).
python -m arena.train_arena --agents 4 --games 1000 --batch 64
```

What you get: each agent gets its own randomized architecture (conv depth 2-5, FC
widths 128-1024), a per-agent ELO that floats as it trains, a periodic
**calibration round-robin** that re-ranks the whole population head-to-head, a
**hall of fame** for anything that breaks ELO 1300, and stagnation-based
retirement/respawn. State is written to `models/arena/arena_state.json`, which the
dashboard (next step) reads live.

### 9. Watch the dashboard -- line go up [^]

In a **second terminal** (leave the arena training in the first), launch the
dashboard. It reads `arena_state.json` on every request, so it reflects the live
arena:

```bash
python -m arena.gui_server --port 5050
# then open http://127.0.0.1:5050/
```

The dashboard shows the ELO ladder, per-agent ELO history, an event feed
(spawns/retires/new-bests/calibrations), and controls to pause/resume the run or
spawn/retire agents on the fly -- the trainer picks those changes up automatically.

On Windows, `start-arena.bat` launches the dashboard in a named window and
`stop-arena.bat` tears it down.

### 10. (Optional) Watch it from your phone via Tailscale

The dashboard binds `0.0.0.0` by default, so with [Tailscale](https://tailscale.com)
running on both the training machine and your phone (same tailnet), you can watch
the ladder climb from anywhere. Two ways:

```bash
# A) Direct -- zero extra setup. Find this machine's Tailscale IP:
tailscale ip -4
# then on your phone (Tailscale connected) open:  http://<that-100.x.y.z-ip>:5050/

# B) Tailscale Serve -- nicer HTTPS URL + MagicDNS name, no port juggling:
tailscale serve --bg 5050
# Tailscale prints an https://<machine>.<tailnet>.ts.net URL -- open that on your phone.
# (Exact flag syntax varies by Tailscale version; see `tailscale serve --help`.)
```

> On the first launch Windows may prompt to allow Python through the firewall --
> allow it on **private** networks so the tailnet/LAN can reach the port. Use
> `--host 127.0.0.1` if you'd rather keep the server localhost-only (e.g. when
> fronting exclusively with `tailscale serve`).

> **Optional native engine:** a C++ engine lives in `engine/cpp/` for faster
> rollouts. It is entirely optional -- if it isn't built, the project automatically
> falls back to the pure-Python engine (you'll see a one-line notice at startup).

---

## Command Reference

### Head-to-Head (quick 2-agent test)

```bash
# Default: nn vs random, 500 games
python -m scripts.head_to_head_test

# Custom matchup
python -m scripts.head_to_head_test --a1 pg_best --a2 lottery_no_touchy --games 2000
python -m scripts.head_to_head_test --a1 pg_best --a2 random --games 1000
```

---

### Reproducible checkpoint benchmark

The fail-closed suite resolves Arena architecture from state, validates the
checkpoint exactly, alternates colors over frozen legal openings, and writes
JSON plus Markdown provenance:

```bash
python -m scripts.benchmark_suite \
  --candidate arena:21@hof \
  --anchors lottery,nn_big8,winblock,center,first \
  --candidate-sims 0,25,100 \
  --oracle-sims 400 \
  --openings standard \
  --out results/arena-21
```

Use `arena:<id>@latest` for the newest retained version, or pass a version-1
checkpoint manifest as documented at the top of
`scripts/benchmark_suite.py`. Add `--preflight-only` to validate a command
without starting games.

---

### Tournament (round-robin with standings + ELO)

```bash
# All major named agents vs each other
python -m scripts.run_tournament --agents pg_best lottery_no_touchy nn_big_8 random first --games 2000 --elo

# With version override (@latest loads newest version_XX.pt from that agent's model dir)
python -m scripts.run_tournament --agents new_cnn@latest lottery@latest nn2@latest nn_big_8@latest --games 5000 --elo

# Best arena agent vs named agents
python -m scripts.run_tournament --arena --agents pg_best lottery_no_touchy --games 1000 --elo

# Top 3 arena agents vs each other
python -m scripts.run_tournament --arena 3 --games 1000 --elo

# Top 5 arena agents + the lottery giant
python -m scripts.run_tournament --arena 5 --agents lottery_no_touchy --games 500 --elo

# Save results to JSON
python -m scripts.run_tournament --arena 3 --agents pg_best --games 2000 --elo --out results/latest.json

# List all available named agent keys
python -m scripts.run_tournament --list
```

**`--arena N`** reads `models/arena/arena_state.json`, picks top-N by current ELO (active agents first, falls back to retired by `best_elo` if needed).
Omit `N` to get just the single best (`--arena` == `--arena 1`).

---

### League Training (single actor-critic agent)

```bash
# Fresh start, small network, curriculum on, parallel games
# (--seed_model "" is required from a clean clone -- see note below)
python -m scripts.train_league --parallel 64 --curriculum --seed_model "" --chunks 200 --chunk_games 2000

# Resume from last checkpoint
python -m scripts.train_league --parallel 128 --curriculum --resume --chunks 200

# Resume from a specific checkpoint
python -m scripts.train_league --parallel 128 --curriculum \
  --resume_checkpoint models/league_pg/archive/archive_0003_NeuralNetAgentPG.pt \
  --chunks 200 --chunk_games 2000

# Different network sizes (small=default, medium, large)
python -m scripts.train_league --parallel 128 --curriculum --network medium --chunks 100
python -m scripts.train_league --parallel 128 --curriculum --network large  --chunks 100

# Eval only -- no training
python -m scripts.train_league --eval \
  --eval_checkpoint models/league_pg/archive/archive_0010_NeuralNetAgentPG.pt \
  --eval_opponent lottery_no_touchy --eval_games 500

# Debug: watch 3 games move-by-move, then exit
python -m scripts.train_league --debug_games
```

| Flag | Default | Description |
|---|---|---|
| `--chunks` | 10 | Number of training chunks |
| `--chunk_games` | 1000 | Games per chunk |
| `--parallel` | 0 | Batch size for parallel games (0 = sequential; 64 confirmed best for RTX 3080) |
| `--network` | `small` | `small` / `medium` / `large` |
| `--curriculum` | off | Auto-advance opponent difficulty stages |
| `--lr` | 1e-4 | Learning rate |
| `--resume` | off | Continue from latest checkpoint in `--model_dir` |
| `--seed_model` | `models/big_8_layer/.../best.pt` | Checkpoint to seed weights from. **Pass `""` for a fresh start** -- the default path does not exist in a clean clone and will raise `FileNotFoundError`. |
| `--keep_versions` | 5 | How many `version_XX.pt` files to keep on disk |

**Network sizes:**

| Name | Conv channels | FC hidden sizes |
|---|---|---|
| `small` | [32, 64, 64] | [256, 512, 256] |
| `medium` | [64, 128, 128, 256] | [512, 512, 256] |
| `large` | [64, 128, 256, 256] | [512, 1024, 512, 256] |

---

### Arena Training (multi-agent evolutionary population)

Runs multiple agents simultaneously. Agents with stagnant ELO are retired and replaced with clones or random spawns.

```bash
# Fresh start, 3 random agents, run forever
python -m arena.train_arena

# Resume (auto-detected if state file exists)
python -m arena.train_arena

# Tune population and chunk size
python -m arena.train_arena --agents 5 --max_active 5 --games 1500 --batch 64

# Seed fresh agents from an existing checkpoint
python -m arena.train_arena --seed_model models/league_pg/archive/archive_0010_NeuralNetAgentPG.pt

# Run a fixed number of chunks then stop
python -m arena.train_arena --chunks 500

# Custom stagnation threshold (chunks before retirement)
python -m arena.train_arena --retire 50
```

| Flag | Default | Description |
|---|---|---|
| `--agents` | 3 | Initial agent count (fresh start only) |
| `--max_active` | 5 | Max simultaneously active agents |
| `--games` | 1000 | Games per chunk |
| `--batch` | 32 | Parallel batch size |
| `--retire` | 30 | Stagnation threshold (chunks with no ELO improvement) |
| `--lr` | 1e-4 | Learning rate |
| `--chunks` | 0 | Total chunks to run (0 = forever) |
| `--seed_model` | `` | Weights to seed fresh agents from |
| `--state` | `models/arena/arena_state.json` | Population state file |

Pause/resume without killing the process: create/delete `models/arena/arena.pause`.

---

### Gated Training (promote-on-threshold)

```bash
python -m scripts.train_gated \
  --agent nn_big_8 \
  --train_opponent self \
  --chunk_games 50000 \
  --chunks 20 \
  --eval_games 5000 \
  --promote_threshold 0.52 \
  --min_vs_first 0.95 \
  --min_vs_random 0.90
```

---

### Flexible CNN Training (quick experiments)

```bash
python -m scripts.trainer_flexible_cnn --games 100_000 --overwrite --opponent random
```

---

### Play vs the AI (terminal)

```bash
python -m scripts.play_human --checkpoint models/league_pg/archive/archive_0010_NeuralNetAgentPG.pt
python -m scripts.play_human --checkpoint models/league_pg/archive/archive_0010_NeuralNetAgentPG.pt --side O
```

---

### Visualize Weight Changes Between Checkpoints

```bash
python -m scripts.visualize_weights \
  --f1 models/new_cnn/256-512-1024-512-128-81/version_05.pt \
  --f2 models/new_cnn/256-512-1024-512-128-81/version_00.pt \
  --heatmap
```

---

### Live Metrics (browser plot during training)

```bash
python -m http.server 8000
# Then open: http://localhost:8000/gui/live_plot/live_metrics_plot.html
```

---

## Named Agents

| Key | Description |
|---|---|
| `random` | Uniformly random legal move |
| `first` | Always picks the first legal move |
| `pg_best` | Latest league-trained Actor-Critic checkpoint |
| `lottery_no_touchy` | Giant frozen CNN (conv=[512]x135, fc=[4096...256]) |
| `nn_big_8` | 8-layer FC network, best/latest checkpoint |
| `new_cnn` | Small CNN (conv=[32,64,64]) |
| `lottery` | Medium CNN (conv=[64,128,256,256]) |
| `nn2` | FC network, version 01 |
| `huge_net` | Frozen eval-only large FC network (fresh weights unless a checkpoint exists) |

Use `@latest` or `@NN` suffixes in tournament specs to override checkpoints:
```
pg_best@latest       # newest version_XX.pt in that agent's model_dir
nn2@05               # version_05.pt
nn2@models/path.pt   # explicit path
```

---

## Project Structure

```
ultimate-ttt-RL/
+-- agents/          # Agent classes, AGENT_FACTORIES registry
+-- arena/           # Multi-agent evolutionary training loop + GUI server
+-- engine/          # Game logic, rules, GameState
+-- cli/             # CLI for human play
+-- scripts/         # Training scripts, tournament, eval, visualization
+-- models/          # Saved checkpoints (gitignored)
|   +-- arena/       # Arena agent checkpoints + arena_state.json
|   +-- league_pg/   # League training checkpoints + archive/
+-- gui/             # Web GUI assets
+-- tests/           # Test suite
+-- requirements.txt
```

---

## Roadmap

Current gated execution order: **[SHIP_PLAN.md](SHIP_PLAN.md)**.

- [x] Build base GameState class with legal move logic
- [x] CLI playable version of the game
- [x] Enforce mini-board constraints and win conditions
- [x] Add reward system and RL agent skeleton
- [x] Self-play training loop with matchmaking
- [x] Actor-Critic (PG) agent with league training and curriculum
- [x] Multi-agent arena with evolutionary population management
- [x] Round-robin tournament with ELO ratings
- [x] GUI to watch live games
- [x] PUCT MCTS implementation with sign/perspective tests
- [x] One-command, fail-closed benchmark suite for arbitrary checkpoints
- [ ] Independently benchmark the Arena pocket and strength finalists
- [ ] Quantized browser player with TypeScript/WASM search
- [ ] Search-trained oracle champion and Hugging Face deployment

---

## License

MIT
