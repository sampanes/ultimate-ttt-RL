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

### 4. Run Training
```
python -m scripts.neural_net_self_train --games 100_000 --resume
```

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