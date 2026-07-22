"""AZ dashboard server: static viewer plus an on-demand live-game API.

Drop-in replacement for the bare `python -m http.server 7654` the dashboard
used before. Static behavior is identical -- it serves the repo root, so the
page still reads loss_logs/metrics_log.jsonl and
models/expert_iter_v2/state.json as plain files. JSON endpoints are added:

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

    GET  /api/control
        Run status: {"state": "running"|"stopped"|"stopping"|"starting",
        "teacher_gen": N, "last": <last op result or null>}.
    POST /api/control?action=stop|start
        Pause / resume the training run. This NEVER kills a process or writes
        a checkpoint itself -- it only shells out to the same graceful scripts
        you run by hand: stop_goat.bat (Ctrl+C, so the trainer's finally block
        saves resume state -- the best agent is never lost) and start_goat.bat
        (--resume, idempotent). Runs the batch in a background thread and
        returns 202 immediately; status flips running<->stopping<->stopped as
        the graceful stop (up to ~2 min) completes. 409 if an op is already in
        flight, if stopping an already-stopped run, or starting a running one.

Never force-kills, never touches the training run's state directly: the static
files and /api/play are read-only, and control only invokes the graceful
start/stop scripts. If this server dies, training does not notice.

Run (from repo root; start_dashboard.bat wraps this):
    .venv\\Scripts\\python -m gui.alphazero.dashboard_server
"""

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(REPO_ROOT, "models", "expert_iter_v2")
METRICS_PATH = os.path.join(REPO_ROOT, "loss_logs", "metrics_log.jsonl")
OPPONENTS = ("teacher", "winblock", "random", "gregory")

# Default cap for /api/metrics. The raw log grows one row every few seconds
# forever; the charts downsample to a few hundred points anyway, so serving
# every row just re-parses tens of MB on the client every poll (fatal on a
# phone). ~6000 rows keeps the full span AND every teacher generation while
# the payload stays a couple of MB.
METRICS_CAP = 6000
METRICS_TAIL_FULL = 500   # most-recent rows kept at full resolution
GREGORY_DEPTH = 3  # mirrors scripts.expert_iter --gregory_depth default

# Run control shells out ONLY to these two existing graceful scripts -- never a
# force-kill, never a checkpoint write -- so a pause cannot lose the best agent.
STOP_BAT = os.path.join(REPO_ROOT, "stop_goat.bat")
START_BAT = os.path.join(REPO_ROOT, "start_goat.bat")
# Give the spawned batch its own hidden console: nothing flashes on screen, and
# there is zero console cross-talk with this server (send_ctrl_c.py inside
# stop_goat.bat detaches and re-attaches to the goat-train console on its own).
_CREATE_NO_WINDOW = 0x08000000

# Heavy machinery (torch, agents, engine) is imported lazily on the first
# /api/play so the static server is up instantly and a metrics-only viewer
# session never pays the torch import at all.
_brain = {"ready": False}
_brain_lock = threading.Lock()
# One game at a time: burst-clicking the button queues 429s, not CPU work.
_play_lock = threading.Lock()

# Run control state. One pause/resume in flight at a time; the worker thread
# clears op when the graceful script returns. Nothing here touches a checkpoint.
_control = {"op": None, "since": 0.0, "last": None}  # op: None|'stopping'|'starting'
_control_lock = threading.Lock()
# Small TTL cache so many watchers (phone + desktop) don't each spawn a wmic.
_alive_cache = {"t": 0.0, "val": False}
_ALIVE_TTL = 2.5


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


def _bounded_metrics_lines(lines, cap, tail_full=METRICS_TAIL_FULL):
    """Downsample metric rows to at most `cap`, order preserved.

    Keeps the most recent `tail_full` rows at full resolution (so live cards,
    throughput and the recent chart tail are exact) and uniformly samples the
    older history to fill the rest. With ~2000 rows per generation, uniform
    sampling to 6000 still leaves every teacher generation represented, so the
    generations chart and the long trend lines are intact -- only the historic
    resolution drops. This is a pure viewer bound; the on-disk log is untouched.
    """
    n = len(lines)
    if n <= cap:
        return lines
    if tail_full >= cap:
        return lines[n - cap:]
    tail = lines[n - tail_full:]
    head = lines[:n - tail_full]
    budget = cap - tail_full
    step = len(head) / float(budget)
    idxs = sorted({int(i * step) for i in range(budget)})
    return [head[i] for i in idxs] + tail


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


# ==== run control (pause / resume) =========================================

def _goat_alive_raw():
    """One uncached process check: True/False, or None on a transient error.

    Uses the same commandline match stop_goat.bat waits on, so status reflects
    the exact truth the stopper uses -- not a window-title guess.
    """
    try:
        out = subprocess.run(
            ["wmic", "process", "where",
             "name='python.exe' and (commandline like '%goat_supervisor%' "
             "or commandline like '%scripts.expert_iter%')",
             "get", "processid"],
            capture_output=True, text=True, timeout=15,
            creationflags=_CREATE_NO_WINDOW)
        return any(tok.strip().isdigit() for tok in out.stdout.split())
    except Exception:
        return None


def _goat_alive():
    """Cached liveness (a couple of seconds) so many watchers don't each wmic."""
    now = time.time()
    if now - _alive_cache["t"] < _ALIVE_TTL:
        return _alive_cache["val"]
    val = _goat_alive_raw()
    if val is None:
        val = _alive_cache["val"]  # transient wmic hiccup: keep the last truth
    _alive_cache.update(t=now, val=val)
    return val


def _control_status():
    with _control_lock:
        op = _control["op"]
        last = _control["last"]
    # While an op runs, report it; the process check is meaningless mid-cutover.
    state = op if op else ("running" if _goat_alive() else "stopped")
    return {"state": state, "teacher_gen": _read_state().get("teacher_gen"),
            "last": last}


def _run_control_worker(kind):
    """Run the graceful start/stop script, then clear the op.

    stop and start are NOT symmetric on Windows:
      - stop_goat.bat is synchronous (Ctrl+C, then a bounded wait) and returns
        only once the run is fully down -- safe to capture output and wait on.
      - start_goat.bat spawns a DETACHED training daemon. That daemon inherits
        this process's stdout handle, so capturing output would block until the
        WHOLE run exits. So for start we send output to DEVNULL (the run logs
        itself to loss_logs) and hold the 'starting' state until the supervisor
        is actually detected -- giving the UI a clean starting -> running flip
        instead of a 4-minute false 'starting'.
    """
    result = {"kind": kind}
    try:
        if kind == "stop":
            proc = subprocess.run(
                ["cmd", "/c", STOP_BAT], cwd=REPO_ROOT,
                capture_output=True, text=True, timeout=240,
                creationflags=_CREATE_NO_WINDOW)
            result["rc"] = proc.returncode
            tail = (proc.stdout or "")[-400:].strip()
            if tail:
                result["out"] = tail
        else:
            subprocess.run(
                ["cmd", "/c", START_BAT], cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=200, creationflags=_CREATE_NO_WINDOW)
            up = False
            for _ in range(30):          # ~30s for the supervisor to appear
                if _goat_alive_raw() is True:
                    up = True
                    break
                time.sleep(1)
            result["rc"] = 0 if up else -1
            if not up:
                result["error"] = "launched, but no training process detected yet"
    except subprocess.TimeoutExpired:
        result["rc"] = -1
        result["error"] = "control script timed out"
    except Exception as e:  # never let the worker die silently
        result["rc"] = -1
        result["error"] = "%s: %s" % (type(e).__name__, e)
    result["when"] = time.time()
    with _control_lock:
        _control["op"] = None
        _control["last"] = result
    _alive_cache["t"] = 0.0  # force a fresh process check on the next status


def _start_control(kind):
    """Kick off a pause/resume in the background. Returns (http_code, payload)."""
    alive = _goat_alive()  # outside the lock: wmic is slow, don't block status
    with _control_lock:
        if _control["op"] is not None:
            return 409, {"error": "a %s is already in progress" % _control["op"]}
        if kind == "stop" and not alive:
            return 409, {"error": "training is not running"}
        if kind == "start" and alive:
            # start_goat.bat would stop-then-restart a healthy run; refuse so a
            # stray click can never bounce a good run mid-generation.
            return 409, {"error": "training is already running"}
        op = "stopping" if kind == "stop" else "starting"
        _control["op"] = op
        _control["since"] = time.time()
    threading.Thread(target=_run_control_worker, args=(kind,), daemon=True).start()
    return 202, {"ok": True, "op": op}


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
        if parsed.path == "/api/control":
            return self._json(200, _control_status())
        if parsed.path == "/api/metrics":
            return self._api_metrics(parsed)
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/control":
            query = urllib.parse.parse_qs(parsed.query)
            action = (query.get("action") or [""])[0]
            # Drain any request body so the socket closes cleanly on 4xx too.
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            if action not in ("stop", "start"):
                return self._json(400, {"error": "action must be stop or start"})
            code, payload = _start_control(action)
            return self._json(code, payload)
        return self._json(404, {"error": "not found"})

    def _api_metrics(self, parsed):
        """Bounded newline-JSON metrics for the viewer.

        Same row format the page already parses -- just downsampled so the
        payload stays small as the log grows without bound. The true (undown-
        sampled) row count rides in the X-Total-Rows header so the page can
        still show the real iteration count.
        """
        query = urllib.parse.parse_qs(parsed.query)
        try:
            cap = int((query.get("cap") or [str(METRICS_CAP)])[0])
        except ValueError:
            cap = METRICS_CAP
        cap = max(200, min(cap, 50000))
        try:
            with open(METRICS_PATH, "r", encoding="utf-8", errors="replace") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
        except OSError:
            lines = []
        total = len(lines)
        body = "\n".join(_bounded_metrics_lines(lines, cap)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Total-Rows", str(total))
        self.end_headers()
        self.wfile.write(body)

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
