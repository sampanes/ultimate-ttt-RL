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
```
python -m pip install -r requirements.txt
```

### 3. Run CLI (from repo root)
```
python -m cli.play
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
- [ ] Enforce mini-board constraints and win conditions
- [ ] Add reward system and RL agent skeleton
- [ ] Self-play training loop with matchmaking
- [ ] Basic GUI to visualize games
- [ ] Agent vs Agent tournaments and tracking

## License

MIT