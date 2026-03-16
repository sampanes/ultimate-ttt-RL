# ultimate-ttt-RL
From the ground up, build an ultimate Tic Tac Toe RL arena to get the best AI

## Setup

### 1. Create Virtual Environment

Windows:
```
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Requirements

This project uses PyTorch with GPU acceleration.

**IMPORTANT**: Do **not** blindly install `torch` from PyPI, or you may get the CPU-only version.

Instead, visit https://pytorch.org/get-started/locally/  
Select your OS, Python version, and most importantly: the correct CUDA version that matches your system.  
(Tip: run `nvidia-smi` in your terminal to see your installed CUDA version — for example, mine shows CUDA 12.8.)

Then run the install command it gives you, like:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

After PyTorch is installed, continue with the rest of the requirements:

```
python -m pip install -r requirements.txt
```

### 2.5 Check your GPU (from repo root)
```
python -m tests.env_check
``` 
or
```
python -m tests.env_check both
``` 

### 3. Run CLI (from repo root)
```
python -m cli.play
```

### 4. Run Training (can assign new NN shapes in flexible NN, weights save to new folder)
```
python -m scripts.trainer_flexible_cnn --games 100_000 --overwrite --opponent random
```

To reduce overfitting to one rival, train against a mixed opponent pool sampled each game:
```
python -m scripts.trainer_flexible_cnn --games 100_000 --opponent-pool random,first,nn2 --autosave-every 5000 --keep-last 10
```
(`--autosave-every` is useful for unattended overnight/weekend runs.)
and if you want live updates on http://localhost:8000/gui/live_plot/live_metrics_plot.html
```
python -m http.server 8000
```


To prune weak checkpoints from battle results with a confirmation prompt:
```
python -m scripts.prune_losers_by_battle --model-dir models/new_cnn/256-512-1024-512-128-81 --max-models 12 --games-per-pair 8 --delete-below 0.40
```
(The script prints winners/losers and requires typing `YES` exactly before deleting loser checkpoints.)

### 4. Look at weight updates, see if some layers are changing more than others
```
python -m scripts.visualize_weights --f1 models\new_cnn\256-512-1024-512-128-81\version_05.pt --f2 models\new_cnn\256-512-1024-512-128-81\version_00.pt --heatmap
```
don't forget to use "copy relative path"

### 4. Run Battles
```
python -m scripts.head_to_head_test
```

## Project Structure

```
ultimate-ttt-RL/
├── engine/          # Game logic and GameState class
├── cli/             # CLI interface for human-vs-human or bot-vs-human
├── agents/          # RL agents and training logic (WIP)
├── gui/             # Web or desktop GUI (optional)
├── notebooks/       # Jupyter notebooks for training/experiments
├── tests/           # Test suite
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Build base GameState class with legal move logic
- [x] CLI playable version of the game
- [x] Enforce mini-board constraints and win conditions
- [x] Add reward system and RL agent skeleton
- [x] Self-play training loop with matchmaking
- [ ] Basic GUI to visualize games
- [ ] Agent vs Agent tournaments and tracking

## License

MIT

## Super-agent mode (immortal + dormant + wake-on-surpass)

If your population is flatlining from repeatedly cycling weak checkpoints, use the new **super-agent arena loop**.

### What it does
- Keeps one immortal checkpoint at `models/super_agent/1024-2048-4096-2048-1024-512-81/super_agent.pt`.
- Never prunes/deletes that file.
- Evaluates your latest `new_cnn` challenger checkpoint against the super-agent.
- If challenger score is below threshold, super-agent stays dormant (no training burn).
- If challenger crosses threshold, super-agent wakes and trains hard against a curriculum that can include `random`, `first`, `last`, `best`, `second_best`, `best_plus_random`, plus any fixed arena agents.
- Uses the same training arena + metrics log + GUI live plot flow (`loss_logs/metrics_log.jsonl` and `gui/live_plot/live_metrics_plot.html`).

### Run it (brand-new user quickstart)
1. Install dependencies from Setup.
2. (Optional but recommended) Train challenger snapshots first:
   ```
   python -m scripts.trainer_flexible_cnn --games 100_000 --opponent-pool random,first,nn2 --autosave-every 5000 --keep-last 10
   ```
3. Start the live plot server in another terminal:
   ```
   python -m http.server 8000
   ```
4. Run the super-agent gate + wake loop:
   ```
   python -m scripts.super_agent_arena --challenger-model-dir models/new_cnn/256-512-1024-512-128-81 --gate-games 60 --wake-threshold 0.55 --train-games 50000 --autosave-every 2000 --curriculum random,first,last,best,second_best,best_plus_random,nn,nn2
   ```
5. Open the metrics graph:
   - `http://localhost:8000/gui/live_plot/live_metrics_plot.html`

### Useful knobs
- `--wake-threshold`: raise (e.g. 0.60) to wake less often, lower (e.g. 0.52) to wake more often.
- `--gate-games`: increase for less noisy wake decisions.
- `--rank-recent` + `--rank-games-per-pair`: controls how "best" and "second_best" challengers are selected from recent checkpoints.
- `--curriculum`: set the wake training curriculum. Supports: `random`, `first`, `last`, `best`, `second_best`, `best_plus_random`, `nn`, `nn2`, `nn_big_8`, `new_cnn`, `lottery`, `super_agent`.

### Suggested anti-flatline workflow
- Train challengers with `--opponent-pool` instead of one static opponent.
- Use regular battle pruning for challenger directories only.
- Keep super-agent immutable + continuous (single immortal file) so your benchmark never resets.
- Periodically run head-to-head checks to monitor if challengers are truly surpassing instead of farming one niche opponent.
