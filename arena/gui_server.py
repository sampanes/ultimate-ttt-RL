"""
arena/gui_server.py -- Standalone Flask server for the Arena Control GUI.

Usage:
    python -m arena.gui_server [--state models/arena/arena_state.json] [--host 0.0.0.0] [--port 5050] [--games 1000]

Reads arena_state.json on every request (no caching).
Serves the GUI from gui/arena/templates + gui/arena/static.
"""

import argparse
import json
import os
import threading
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, render_template, request

from arena.arch_configs import arch_label as _arch_label
from arena.population_manager import PopulationManager

# ---------------------------------------------------------------------------
# Globals (set by main() before app.run)
# ---------------------------------------------------------------------------

_state_path: str = "models/arena/arena_state.json"
_pause_path: str = "models/arena/arena.pause"
_metrics_path: str = "loss_logs/metrics_log.jsonl"
_games_per_chunk: int = 1000
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Flask app -- template/static dirs point to gui/arena/
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_TEMPLATE_DIR = os.path.join(_ROOT, "gui", "arena", "templates")
_STATIC_DIR = os.path.join(_ROOT, "gui", "arena", "static")

app = Flask(__name__, template_folder=_TEMPLATE_DIR, static_folder=_STATIC_DIR)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_state() -> dict | None:
    """Read state JSON fresh from disk. Returns None if absent."""
    if not os.path.exists(_state_path):
        return None
    with open(_state_path) as f:
        return json.load(f)


def _compute_uptime(start_time_iso: str) -> str:
    if not start_time_iso:
        return "--"
    try:
        start = datetime.fromisoformat(start_time_iso)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        elapsed = int((datetime.now(tz=timezone.utc) - start).total_seconds())
        h, rem = divmod(max(elapsed, 0), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "--"


def _arch_label_from_dict(a: dict) -> str:
    return _arch_label(a.get("conv_channels", []), a.get("fc_hidden_sizes", []))


_STATUS_MAP = {"running": "Running", "paused": "Paused", "stopped": "Stopped"}


def _set_status_in_json(new_status: str):
    """Immediately write a new status into arena_state.json (caller holds _lock)."""
    if not os.path.exists(_state_path):
        return
    with open(_state_path) as f:
        data = json.load(f)
    data["status"] = new_status
    with open(_state_path, "w") as f:
        json.dump(data, f, indent=2)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with _lock:
        state = _load_state()

    if state is None:
        return jsonify({
            "status": "Stopped",
            "uptime": "--",
            "agents": [],
            "total_games": 0,
            "best_elo": 0,
            "retirement_threshold": 30,
        })

    agents_raw = state.get("agents", [])

    # Sort: active by ELO desc, then retired
    active = sorted(
        [a for a in agents_raw if a["status"] == "active"],
        key=lambda a: -a["elo"],
    )
    retired = [a for a in agents_raw if a["status"] != "active"]

    agents_out = []
    rank = 1
    for a in active:
        agents_out.append({
            "id": a["id"],
            "name": a["name"],
            "arch": _arch_label_from_dict(a),
            "elo": round(a["elo"]),
            "best_elo": round(a.get("best_elo", a["elo"])),
            "status": a["status"],
            "chunks_stagnant": a.get("stagnant_count", 0),
            "rank": rank,
        })
        rank += 1
    for a in retired:
        agents_out.append({
            "id": a["id"],
            "name": a["name"],
            "arch": _arch_label_from_dict(a),
            "elo": round(a["elo"]),
            "best_elo": round(a.get("best_elo", a["elo"])),
            "status": a["status"],
            "chunks_stagnant": a.get("stagnant_count", 0),
            "rank": None,
        })

    all_best = [a.get("best_elo", a["elo"]) for a in agents_raw]
    best_elo = round(max(all_best)) if all_best else 0

    return jsonify({
        "status": _STATUS_MAP.get(state.get("status", "stopped"), "Stopped"),
        "uptime": _compute_uptime(state.get("start_time", "")),
        "agents": agents_out,
        "total_games": state.get("total_chunks_done", 0) * _games_per_chunk,
        "best_elo": best_elo,
        "retirement_threshold": state.get("retirement_threshold", 30),
    })


@app.route("/api/events")
def api_events():
    with _lock:
        state = _load_state()
    if state is None:
        return jsonify({"events": []})
    return jsonify({"events": state.get("events", [])})


@app.route("/api/history")
def api_history():
    with _lock:
        state = _load_state()

    if state is None:
        return jsonify({"history": {}})

    history = {
        a["name"]: a.get("elo_history", [])
        for a in state.get("agents", [])
    }
    return jsonify({"history": history})


@app.route("/api/metrics")
def api_metrics():
    """Raw JSONL of the single-net training metrics log (loss/epsilon/winrate).

    Returned as text/plain so the client can reuse its exact parse -> moving-average ->
    downsample path and its cheap length-based "has it grown?" check. This is a
    *different* data source from the arena state: it's written by a single-net
    training run, so it's empty (returns "") when only a league run is active.
    """
    if not os.path.exists(_metrics_path):
        return Response("", mimetype="text/plain")
    with open(_metrics_path, "r") as f:
        return Response(f.read(), mimetype="text/plain")


@app.route("/api/control/pause", methods=["POST"])
def api_pause():
    with _lock:
        os.makedirs(os.path.dirname(_pause_path) or ".", exist_ok=True)
        open(_pause_path, "w").close()
        _set_status_in_json("paused")
    return jsonify({"success": True})


@app.route("/api/control/resume", methods=["POST"])
def api_resume():
    with _lock:
        if os.path.exists(_pause_path):
            os.remove(_pause_path)
        _set_status_in_json("running")
    return jsonify({"success": True})


@app.route("/api/control/retire/<int:agent_id>", methods=["POST"])
def api_retire(agent_id):
    with _lock:
        pm = PopulationManager(_state_path)
        loaded = pm.load()
        if not loaded:
            return jsonify({"success": False, "error": "State file not found"}), 404
        try:
            pm.retire_agent(agent_id, reason="manual")
        except KeyError as e:
            return jsonify({"success": False, "error": str(e)}), 404
    return jsonify({"success": True})


@app.route("/api/control/spawn", methods=["POST"])
def api_spawn():
    body = request.get_json(force=True) or {}
    name = body.get("name") or None
    mode_raw = body.get("mode", "random")
    depth = body.get("depth", None)
    elo = body.get("elo", None)
    if elo is not None:
        try:
            elo = float(elo)
        except (TypeError, ValueError):
            elo = None

    # Support "random_3" shorthand from the GUI dropdown
    if mode_raw.startswith("random"):
        mode = "random"
        if depth is None:
            parts = mode_raw.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                depth = int(parts[1])
    else:
        mode = mode_raw  # "clone_best"

    with _lock:
        pm = PopulationManager(_state_path)
        loaded = pm.load()
        if not loaded:
            return jsonify({"success": False, "error": "State file not found"}), 404
        try:
            record = pm.spawn_agent(mode=mode, depth=depth, name=name, elo=elo)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    return jsonify({
        "success": True,
        "agent": {
            "id": record.id,
            "name": record.name,
            "arch": record.arch_label,
            "elo": round(record.elo),
            "status": record.status,
        },
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    global _state_path, _pause_path, _metrics_path, _games_per_chunk

    parser = argparse.ArgumentParser(description="Arena Control GUI server")
    parser.add_argument(
        "--state", default="models/arena/arena_state.json",
        help="Path to arena_state.json (default: models/arena/arena_state.json)",
    )
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Interface to bind. Default 0.0.0.0 = reachable on your LAN/tailnet "
             "(hit the machine's Tailscale IP from your phone). Use 127.0.0.1 to "
             "restrict to localhost (e.g. when fronting with `tailscale serve`).",
    )
    parser.add_argument(
        "--games", type=int, default=1000,
        help="Games per chunk -- used to compute total_games stat (default: 1000)",
    )
    parser.add_argument(
        "--metrics", default="loss_logs/metrics_log.jsonl",
        help="Path to the single-net training metrics log shown on the Training tab "
             "(default: loss_logs/metrics_log.jsonl)",
    )
    args = parser.parse_args()

    _state_path = args.state
    _pause_path = os.path.join(os.path.dirname(os.path.abspath(args.state)), "arena.pause")
    _metrics_path = args.metrics
    _games_per_chunk = args.games

    if args.host in ("0.0.0.0", "::"):
        print(f"[Arena GUI] Local:    http://127.0.0.1:{args.port}")
        print(f"[Arena GUI] Network:  http://<this-machine-ip>:{args.port}")
        print(f"[Arena GUI] Tailnet:  http://<tailscale-ip>:{args.port}   "
              f"(find it with: tailscale ip -4)")
    else:
        print(f"[Arena GUI] http://{args.host}:{args.port}")
    print(f"[Arena GUI] State:   {os.path.abspath(_state_path)}")
    print(f"[Arena GUI] Metrics: {os.path.abspath(_metrics_path)}  (Training tab)")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
