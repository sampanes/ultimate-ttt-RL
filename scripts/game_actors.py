"""S4: multiprocess game actors for self-play generation.

WHY (measured, RESULT_S1.md sections 5c/5e): generation is 85% of block
wall-clock and only ~9% of a game is GPU -- roughly 0.7s of GPU inside a 7.77s
game. The rest is per-sim Python (tree bookkeeping, tensor building, pybind
crossings). The box has 24 logical cores and the trainer uses 1-2. So the lever
is not batching the GPU harder (batch 12 already costs the same as batch 1);
it is running whole games in parallel PROCESSES. Threads cannot: the work is
Python bytecode holding the GIL.

FIRST CUT (this module): N independent workers, each owning whole games end to
end with its own model replica and its own batch-12 forwards. No shared eval
server -- that is the second cut, once this proves out (it would push toward the
measured batch-256 sweet spot of 78.6k pos/s).

DISTRIBUTION-PRESERVING, NOT BYTE-IDENTICAL. The main process still draws every
opponent-slice tag from its own RNG stream and hands the tag to a worker, so the
opponent mix is drawn exactly as the sequential loop draws it. What changes is
per-game stochasticity (Dirichlet noise, temperature sampling), which each worker
draws from its own seeded stream. Games therefore differ from a sequential run
with the same seed; the DISTRIBUTION they are sampled from does not. Gate this
on a timing A/B plus panels landing inside noise, never on byte equality.

VRAM: each worker pays its own CUDA context (~300 MB) plus a 26 MB model replica.
At 8 workers that is ~2.6 GB on top of the trainer's 4.1 GB of 10.2 GiB. Workers
only generate; the train step runs in the main process while workers idle.
"""

import multiprocessing as mp
import os
import queue as _queue
import random
import time

import numpy as np


# Sentinel task kinds.
_TASK_GAME = "game"
_TASK_RELOAD = "reload"
_TASK_STOP = "stop"


def _build_worker_agent(cfg):
    """Construct a worker-local teacher replica. Mirrors expert_iter._make_agent.

    Imported lazily: under spawn (the only start method on Windows) this runs in
    a fresh interpreter, and we want torch/CUDA initialized in the CHILD, never
    inherited from the parent.
    """
    import torch

    torch.set_float32_matmul_precision("high")
    # One torch thread per worker. Without this, N workers x an OpenMP pool
    # sized to 24 cores oversubscribes the box and runs slower than sequential.
    torch.set_num_threads(1)

    from agents.agent_base import ModelConfigCNN
    from agents.neural_net_agent_pg import NeuralNetAgentPG
    from scripts.train_alphazero import NETWORK_CONFIGS

    mcfg = ModelConfigCNN(**NETWORK_CONFIGS[cfg["network"]],
                          learning_rate=cfg["lr"], label="expert_iter_actor",
                          model_dir=cfg["model_dir"], value_tanh=cfg["value_tanh"])
    agent = NeuralNetAgentPG(cfg=mcfg, model_path=None)
    agent.model.to(cfg["device"])
    agent.device = cfg["device"]
    agent.model.eval()
    return agent


def _load_weights(agent, weights_path, device):
    """Load teacher weights AND the runtime value_tanh flag into a replica.

    The flag is not optional bookkeeping. load_state_dict copies weights only,
    so an actor that reloaded a promoted teacher while keeping a stale
    value_tanh=False would feed unbounded pre-tanh values into its MCTS and
    quietly generate poisoned targets -- the exact failure expert_iter's
    _promote_teacher docstring records (values in [-1.66, 3.27] for ~12h).
    Every promoted teacher is tanh, and _save_teacher persists the flag, so we
    take it from the payload rather than from the actor's frozen startup cfg.
    """
    import torch

    payload = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        sd = payload["state_dict"]
        if "value_tanh" in payload:
            agent.model.value_tanh = payload["value_tanh"]
    else:
        sd = payload
    agent.model.load_state_dict(sd)
    agent.model.eval()


def _worker_loop(rank, cfg, task_q, result_q):
    """One actor process: build a replica, then play whole games on demand."""
    import torch

    from agents.deterministics import WinBlockAgent
    from agents.gregory import GregoryAgent
    from agents.random_agent import RandomAgent
    from scripts.train_alphazero import collect_game

    agent = _build_worker_agent(cfg)
    _load_weights(agent, cfg["weights_path"], cfg["device"])

    heur = WinBlockAgent()
    rnd = RandomAgent()
    greg = (GregoryAgent(depth=cfg["greg_mix_depth"])
            if cfg["greg_mix_depth"] else None)

    result_q.put(("ready", rank, None))

    with torch.no_grad():
        while True:
            kind, payload = task_q.get()

            if kind == _TASK_STOP:
                return

            if kind == _TASK_RELOAD:
                # Teacher swapped (promotion). Reload from disk rather than
                # shipping a 26 MB state_dict down every worker's pipe.
                _load_weights(agent, payload, cfg["device"])
                result_q.put(("reloaded", rank, None))
                continue

            tag, seed = payload
            # Per-game seeding: distinct across workers AND across games, so no
            # two actors ever walk the same Dirichlet/temperature stream.
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)

            opponent_fn = None
            opponent_game = False
            if tag == "heur":
                opponent_fn = lambda s: heur.select_move(s)
                opponent_game = True
            elif tag == "rnd":
                opponent_fn = lambda s: rnd.select_move(s)
            elif tag == "greg":
                opponent_fn = lambda s: greg.select_move(s)

            t0 = time.perf_counter()
            exs, winner, gstats = collect_game(
                model=agent.model,
                device=cfg["device"],
                n_sims=cfg["teacher_sims"],
                c_puct=1.5,
                dir_alpha=cfg["dir_alpha"],
                dir_eps=cfg["dir_eps"],
                wave_size=64,
                temperature_moves=cfg["temperature_moves"],
                use_tactics=True,
                use_mini_tactics=(cfg["mini_tactic_opp"] and opponent_game),
                opponent_fn=opponent_fn,
                value_blend=cfg["value_blend"],
            )
            result_q.put(("game", rank, {
                "tag": tag,
                "examples": exs,
                "winner": winner,
                "stats": gstats,
                "secs": time.perf_counter() - t0,
            }))


class GameActorPool:
    """Persistent pool of self-play actor processes.

    Persistent on purpose: a worker pays ~300 MB of CUDA context and a model
    build at startup, which would dwarf a 16-game block if respawned per block.
    """

    def __init__(self, n_actors, cfg, start_timeout=180.0):
        self.n_actors = n_actors
        self.cfg = dict(cfg)
        # Windows has no fork; be explicit so behavior matches across platforms.
        self._ctx = mp.get_context("spawn")
        self._task_q = self._ctx.Queue()
        self._result_q = self._ctx.Queue()
        self._procs = []
        self._closed = False

        for rank in range(n_actors):
            p = self._ctx.Process(target=_worker_loop,
                                  args=(rank, self.cfg, self._task_q,
                                        self._result_q),
                                  daemon=True)
            p.start()
            self._procs.append(p)

        # Block until every actor has its CUDA context and weights up, so the
        # first block's timing is not polluted by ~10s of torch import per proc.
        ready = 0
        deadline = time.time() + start_timeout
        while ready < n_actors:
            if time.time() > deadline:
                self.close()
                raise RuntimeError(
                    f"only {ready}/{n_actors} actors came up within "
                    f"{start_timeout:.0f}s")
            try:
                kind, _rank, _ = self._result_q.get(timeout=5.0)
                if kind == "ready":
                    ready += 1
            except _queue.Empty:
                self._assert_alive()

    def _assert_alive(self):
        for i, p in enumerate(self._procs):
            if not p.is_alive():
                raise RuntimeError(
                    f"actor {i} died (exitcode {p.exitcode}) -- see its "
                    f"traceback above; the pool cannot continue")

    def reload_weights(self, weights_path):
        """Point every actor at freshly saved teacher weights (promotion)."""
        for _ in range(self.n_actors):
            self._task_q.put((_TASK_RELOAD, weights_path))
        done = 0
        while done < self.n_actors:
            kind, _rank, _ = self._result_q.get(timeout=180.0)
            if kind == "reloaded":
                done += 1

    def play_block(self, tags, seeds, timeout=1800.0):
        """Play one game per tag across the pool; return results in tag order.

        `tags` is drawn by the CALLER from its own RNG, so the opponent mix is
        sampled exactly as the sequential loop samples it.
        """
        assert len(tags) == len(seeds)
        for tag, seed in zip(tags, seeds):
            self._task_q.put((_TASK_GAME, (tag, seed)))

        out = []
        deadline = time.time() + timeout
        while len(out) < len(tags):
            try:
                kind, _rank, res = self._result_q.get(timeout=10.0)
            except _queue.Empty:
                self._assert_alive()
                if time.time() > deadline:
                    raise RuntimeError(
                        f"block timed out with {len(out)}/{len(tags)} games")
                continue
            if kind == "game":
                out.append(res)
        return out

    def close(self):
        if self._closed:
            return
        self._closed = True
        for _ in self._procs:
            try:
                self._task_q.put((_TASK_STOP, None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def build_actor_cfg(args, device, weights_path, value_tanh):
    """Freeze the generation parameters an actor needs. Must be picklable."""
    return {
        "network": args.network,
        "device": device,
        "value_tanh": value_tanh,
        "model_dir": args.model_dir,
        "lr": args.lr,
        "weights_path": os.path.abspath(weights_path),
        "teacher_sims": args.teacher_sims,
        "dir_alpha": args.dir_alpha,
        "dir_eps": args.dir_eps,
        "temperature_moves": args.temperature_moves,
        "mini_tactic_opp": args.mini_tactic_opp,
        "value_blend": args.value_blend,
        "greg_mix_depth": args.greg_mix_depth if args.greg_mix > 0 else 0,
    }
