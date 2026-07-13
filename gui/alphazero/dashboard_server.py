"""AZ dashboard server: static viewer plus an on-demand live-game API.

Drop-in replacement for the bare `python -m http.server 7654` the dashboard
used before. Static behavior is identical -- it serves the repo root, so the
page still reads loss_logs/metrics_log.jsonl and
models/expert_iter_v2/state.json as plain files. Two JSON endpoints are added:

    GET /api/play?opp=teacher|winblock|random|gregory
        Plays ONE game on CPU -- the CURRENT student.pt (weights reloaded on
        every request, so it is always the newest checkpoint) vs the chosen
        opponent -- and returns the full move list plus mini-board-win /
        game-win events for the browser to animate. Raw argmax policy with the
        first 4 plies sampled for variety, student color randomized per game.
        Cost: one game is ~40-60 single-position CPU forward passes of the
        6.8M-param net, well under a second. torch is not imported until the
        first request, and nothing runs unless the button is clicked; the
        training run keeps the GPU entirely to itself (this uses 2 CPU
        threads, no CUDA).

    GET /api/health
        Liveness probe: {"ok": true, "brain": "cold"|"warm"}.

Viewer-only by design: the training run never imports this file, and this
file never writes to the training run's state. If this server dies mid-game,
training does not notice.

Run (from repo root; start_dashboard.bat wraps this):
    .venv\\Scripts\\python -m gui.alphazero.dashboard_server
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(REPO_ROOT, "models", "expert_iter_v2")
OPPONENTS = ("teacher", "winblock", "random", "gregory")
GREGORY_DEPTH = 3  # mirrors scripts.expert_iter --gregory_depth default

# Heavy machinery (torch, agents, engine) is imported lazily on the first
# /api/play so the static server is up instantly and a metrics-only viewer
# session never pays the torch import at all.
_brain = {"ready": False}
_brain_lock = threading.Lock()
# One game at a time: burst-clicking the button queues 429s, not CPU work.
_play_lock = threading.Lock()


def _ensure_brain():
    with _brain_lock:
        if _brain["ready"]:
            return _brain
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        import torch
        torch.set_num_threads(2)  # good citizen: the training run owns this box
        from scripts.train_alphazero import NETWORK_CONFIGS, _policy_move
        from agents.agent_base import ModelConfigCNN
        from agents.neural_net_agent_pg import NeuralNetAgentPG
        from agents.deterministics import WinBlockAgent
        from agents.random_agent import RandomAgent
        from agents.gregory import GregoryAgent
        from engine.game import GameState
        from engine.rules import rule_utl_valid_moves
        from engine.constants import X, O, DRAW
        _brain.update(
            torch=torch,
            NETWORK_CONFIGS=NETWORK_CONFIGS,
            policy_move=_policy_move,
            ModelConfigCNN=ModelConfigCNN,
            NeuralNetAgentPG=NeuralNetAgentPG,
            GameState=GameState,
            valid_moves=rule_utl_valid_moves,
            X=X, O=O, DRAW=DRAW,
            agents={
                "winblock": WinBlockAgent(),
                "random": RandomAgent(),
                "gregory": GregoryAgent(depth=GREGORY_DEPTH),
            },
            nets={},   # (kind, network) -> constructed agent; weights swapped per request
            ready=True,
        )
        return _brain


def _read_state():
    try:
        with open(os.path.join(MODEL_DIR, "state.json"), "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _load_net(b, kind, network):
    """Return (agent, checkpoint_payload) for student.pt / teacher.pt.

    Architecture is built once per (kind, network); the state_dict is
    reloaded from disk on EVERY call so the viewer always plays the weights
    the training loop saved most recently.
    """
    agent = b["nets"].get((kind, network))
    if agent is None:
        cfg = b["ModelConfigCNN"](**b["NETWORK_CONFIGS"][network],
                                  learning_rate=1e-3,
                                  label="dashboard_viewer",
                                  model_dir=MODEL_DIR,
                                  value_tanh=True)
        agent = b["NeuralNetAgentPG"](cfg=cfg, model_path=None)
        agent.model.to("cpu")
        agent.device = "cpu"
        b["nets"][(kind, network)] = agent
    path = os.path.join(MODEL_DIR, kind + ".pt")
    payload = b["torch"].load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    agent.model.load_state_dict(state_dict)
    # v2 nets are tanh-headed; the flag only matters if a value is read, but a
    # stale False here is the exact poisoning class RESULT_M2_5.md documents.
    agent.model.value_tanh = True
    agent.model.eval()
    return agent, payload


def _mini_of(cell):
    return (cell // 9 // 3) * 3 + (cell % 9) // 3


def _forced_mini(cell):
    # The mini the NEXT player is sent to (same mapping the engine uses).
    return (cell // 9 % 3) * 3 + (cell % 9) % 3


def _play_game(b, opp_name):
    """One student-vs-opponent game; returns the animation payload."""
    meta = _read_state()
    network = meta.get("network", "arena22")
    student, _ = _load_net(b, "student", network)
    stu_fn = lambda s, mn: b["policy_move"](student.model, "cpu", s, 4, mn)

    teacher_gen = meta.get("teacher_gen")
    if opp_name == "teacher":
        teacher, tpayload = _load_net(b, "teacher", network)
        opp_fn = lambda s, mn: b["policy_move"](teacher.model, "cpu", s, 4, mn)
        gen = tpayload.get("gen", teacher_gen)
        opp_label = "teacher gen %s (raw)" % gen
    else:
        agent = b["agents"][opp_name]
        opp_fn = lambda s, mn: agent.select_move(s)
        opp_label = {
            "winblock": "win/block bot",
            "random": "random mover",
            "gregory": "gregory (depth-%d minimax)" % GREGORY_DEPTH,
        }[opp_name]

    X, O, DRAW = b["X"], b["O"], b["DRAW"]

    def sym(v):
        if v == X:
            return "X"
        if v == O:
            return "O"
        if v == DRAW:
            return "D"
        return None

    gs = b["GameState"]()
    student_side = X if random.random() < 0.5 else O
    prev_minis = list(gs.mini_winners)
    moves = []
    move_num = 0
    while not gs.is_over() and move_num < 200:
        is_student = gs.player == student_side
        side = gs.player
        fn = stu_fn if is_student else opp_fn
        move = int(fn(gs, move_num))
        valid = b["valid_moves"](gs.board, gs.last_move, gs.mini_winners)
        if move not in valid:
            raise RuntimeError("agent %r produced illegal move %d"
                               % (opp_name if not is_student else "student", move))
        gs.make_move(move)
        mini = _mini_of(move)
        mini_won = None
        if gs.mini_winners[mini] != prev_minis[mini]:
            mini_won = sym(gs.mini_winners[mini])
            prev_minis[mini] = gs.mini_winners[mini]
        moves.append({
            "cell": move,
            "side": sym(side),
            "student": is_student,
            "mini": mini,
            "mini_won": mini_won,
            "next_mini": _forced_mini(move),
        })
        move_num += 1

    if gs.winner == DRAW:
        result = "draw"
    elif gs.winner == student_side:
        result = "student"
    else:
        result = "opp"
    return {
        "ok": True,
        "opp": opp_name,
        "opp_label": opp_label,
        "student_side": sym(student_side),
        "teacher_gen": teacher_gen,
        "plies": len(moves),
        "result": result,
        "winner_side": sym(gs.winner),
        "moves": moves,
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=REPO_ROOT, **kwargs)

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            return self._json(200, {
                "ok": True,
                "brain": "warm" if _brain["ready"] else "cold",
            })
        if parsed.path == "/api/play":
            return self._api_play(parsed)
        return super().do_GET()

    def _api_play(self, parsed):
        query = urllib.parse.parse_qs(parsed.query)
        opp = (query.get("opp") or ["winblock"])[0]
        if opp not in OPPONENTS:
            return self._json(400, {"error": "opp must be one of: %s"
                                             % ", ".join(OPPONENTS)})
        if not _play_lock.acquire(blocking=False):
            return self._json(429, {"error": "a game is already in flight -- "
                                             "wait for it to finish"})
        try:
            t0 = time.perf_counter()
            b = _ensure_brain()
            with b["torch"].no_grad():
                out = _play_game(b, opp)
            out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
            return self._json(200, out)
        except FileNotFoundError as e:
            return self._json(503, {"error": "checkpoint not found: %s" % e})
        except Exception as e:  # surface the reason to the panel, keep serving
            return self._json(500, {"error": "%s: %s" % (type(e).__name__, e)})
        finally:
            _play_lock.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=7654)
    ap.add_argument("--bind", default="", help="bind address (default: all interfaces)")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), DashboardHandler)
    print("AZ dashboard server on port %d (serving %s)" % (args.port, REPO_ROOT))
    print("  dashboard: http://localhost:%d/gui/alphazero/" % args.port)
    print("  game API:  http://localhost:%d/api/play?opp=winblock" % args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
