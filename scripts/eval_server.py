"""S5: shared eval server for multiprocess self-play generation.

WHY (measured, RESULT_S4.md section 3): the S4 first cut runs N actor PROCESSES
that each own a CUDA context and fire tiny batch-12 forwards. A consumer 3080
has no MPS, so those N contexts serialize -- GPU pegged at 95% on context-switch
thrash while ~9 of 24 cores idle. That cut cannot pass ~2.6-2.8x. RESULT_S1 5b
measured the way out: ONE context doing batch-256 forwards is 6.0x more
efficient per position (78.6k vs 13.1k pos/s).

THIS module: one EvalServer process owns the ONLY CUDA model and answers every
actor's forward from a single context, batching all in-flight requests into one
forward. The actors become pure-CPU: they build planes and walk the tree, but
hold NO model and NO CUDA context, so they are cheap and we can run many more
than 8. VRAM drops from ~8x326 MB to a single context, and the idle cores plus
the 6x batch headroom become the next multiple.

DISTRIBUTION-PRESERVING, exactly as S4. The main process still draws every
opponent-slice tag from its own RNG and hands tags to actors; only per-game
noise (Dirichlet, temperature) moves to per-actor seeded streams. Batching
across actors is numerically faithful: the model runs in eval() so conv / linear
/ BN(running-stats) are row-independent -- a batched forward equals per-request
forwards within floating-point noise (guarded by a parity test).

TRANSPORT: mp.Queue carrying numpy arrays, NOT shared memory. At the real
message rate (a few hundred to low-thousands of batched forwards/sec aggregate,
because each forward serves many actors) mp.Queue is comfortably fast and far
simpler/safer on Windows than named shared memory. If an A/B ever shows queue
overhead mattering, the RemoteModel seam below is the single place to swap in a
shared-memory slot (which also pairs with S8's fill_planes for zero-copy).

WEIGHT RELOAD is a SINGLE site now: only the server reloads at promotion, via
game_actors._load_weights (which takes value_tanh from the payload -- the
poisoning guard). Actors are stateless with respect to weights, so the N-site
reload hazard of the first cut is gone.
"""

import multiprocessing as mp
import os
import queue as _queue
import random
import time

import numpy as np

from scripts.game_actors import _build_worker_agent, _load_weights


# Sentinel task kinds (actor task queue).
_TASK_GAME = "game"
_TASK_STOP = "stop"

# Control kinds (server control queue).
_CTRL_RELOAD = "reload"
_CTRL_STOP = "stop"

# Forward request kind (actor -> server request queue).
_REQ_FWD = "fwd"

# How long an actor waits on a forward reply before deciding the server is dead.
_FWD_TIMEOUT = 120.0
# Server idle poll: how often it re-checks the control queue while no forward is
# pending. Only fires when generation is idle, so it adds no per-forward latency.
_SERVER_POLL = 0.1


class RemoteModel:
    """Drop-in for the net inside an actor's MCTS: forwards go to the server.

    Exposes ONLY forward_both, the single GPU seam MCTS uses (agents/mcts.py).
    Matches forward_both's contract exactly, including the batch-1 squeeze, so
    MCTS and collect_game are byte-for-byte unaware they are talking to another
    process.
    """

    def __init__(self, rank, req_q, reply_q):
        self.rank = rank
        self._req_q = req_q
        self._reply_q = reply_q

    def forward_both(self, x):
        import torch

        # x is a CPU tensor: (7,9,9), (1,7,9,9), or (K,7,9,9). Normalize to 2D.
        if x.dim() == 3:
            x = x.unsqueeze(0)
        k = x.shape[0]
        # .contiguous() so the numpy view we ship is a tight (K,7,9,9) block.
        self._req_q.put((_REQ_FWD, self.rank, x.contiguous().numpy()))
        try:
            logits_np, values_np = self._reply_q.get(timeout=_FWD_TIMEOUT)
        except _queue.Empty:
            raise RuntimeError(
                f"actor {self.rank}: eval server did not answer a forward "
                f"within {_FWD_TIMEOUT:.0f}s -- assume the server died")
        logits = torch.from_numpy(logits_np)
        values = torch.from_numpy(values_np)
        # forward_both squeezes iff the batch is 1 (input (7,9,9) or (1,...)).
        if k == 1:
            return logits.squeeze(0), values.squeeze(0)
        return logits, values


def _server_loop(cfg, req_q, reply_qs, ctrl_q, ctrl_reply_q, ready_q):
    """The one GPU owner. Batch every in-flight forward through a single context."""
    import torch

    torch.set_float32_matmul_precision("high")
    agent = _build_worker_agent(cfg)                 # on cfg["device"] (cuda)
    _load_weights(agent, cfg["weights_path"], cfg["device"])
    model = agent.model
    device = cfg["device"]

    ready_q.put("ready")

    with torch.no_grad():
        while True:
            # 1) Control has priority and is only sent between blocks (no forward
            #    in flight), so draining it first is race-free.
            try:
                kind, payload = ctrl_q.get_nowait()
                if kind == _CTRL_STOP:
                    return
                if kind == _CTRL_RELOAD:
                    _load_weights(agent, payload, device)
                    model = agent.model
                    ctrl_reply_q.put("reloaded")
                    continue
            except _queue.Empty:
                pass

            # 2) Block for one forward request, then greedily grab everything
            #    already queued so the batch grows with the actor count. The
            #    timeout only fires when idle -> lets the control check above run.
            try:
                first = req_q.get(timeout=_SERVER_POLL)
            except _queue.Empty:
                continue
            pending = [first]
            while True:
                try:
                    pending.append(req_q.get_nowait())
                except _queue.Empty:
                    break

            # 3) One concatenated forward for the whole wave of requests.
            arrays = [arr for (_, _, arr) in pending]
            xs = torch.from_numpy(np.concatenate(arrays, axis=0)).to(device)
            logits, values = model.forward_both(xs)
            if logits.dim() == 1:      # sumK == 1: forward_both squeezed it
                logits = logits.unsqueeze(0)
                values = values.unsqueeze(0)
            logits_np = logits.cpu().numpy()
            values_np = values.cpu().numpy()

            # 4) Scatter each actor its own row-slice.
            off = 0
            for (_, rank, arr) in pending:
                k = arr.shape[0]
                reply_qs[rank].put((logits_np[off:off + k].copy(),
                                    values_np[off:off + k].copy()))
                off += k


def _actor_loop(rank, cfg, task_q, result_q, req_q, reply_q):
    """One pure-CPU actor: play whole games, sending every forward to the server.

    No model, no CUDA context here. collect_game builds its MCTS with device
    'cpu' (planes build on CPU) and a RemoteModel, so all GPU work funnels to the
    single server process.
    """
    import torch

    torch.set_num_threads(1)

    from agents.deterministics import WinBlockAgent
    from agents.gregory import GregoryAgent
    from agents.random_agent import RandomAgent
    from scripts.train_alphazero import collect_game

    model = RemoteModel(rank, req_q, reply_q)

    heur = WinBlockAgent()
    rnd = RandomAgent()
    greg = (GregoryAgent(depth=cfg["greg_mix_depth"])
            if cfg["greg_mix_depth"] else None)

    result_q.put(("ready", rank, None))

    while True:
        kind, payload = task_q.get()

        if kind == _TASK_STOP:
            return

        # NOTE: no reload task here -- the SERVER reloads weights. Actors never
        # touch weights, which is why the value_tanh poisoning site is now one.

        tag, seed = payload
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
            model=model,
            device="cpu",
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


class EvalServerActorPool:
    """Persistent pool: 1 GPU eval server + N pure-CPU actors.

    Same duck-typed interface as game_actors.GameActorPool (play_block,
    reload_weights, close, context manager), so expert_iter selects between them
    with a single flag and the generation loop is unchanged.
    """

    def __init__(self, n_actors, cfg, start_timeout=180.0):
        self.n_actors = n_actors
        self.cfg = dict(cfg)
        self._ctx = mp.get_context("spawn")

        self._task_q = self._ctx.Queue()            # parent -> actors
        self._result_q = self._ctx.Queue()          # actors -> parent
        self._req_q = self._ctx.Queue()             # actors -> server (forwards)
        self._reply_qs = [self._ctx.Queue()         # server -> each actor
                          for _ in range(n_actors)]
        self._ctrl_q = self._ctx.Queue()            # parent -> server (reload/stop)
        self._ctrl_reply_q = self._ctx.Queue()      # server -> parent
        self._server_ready_q = self._ctx.Queue()

        self._procs = []
        self._closed = False

        # Server first, so it owns the GPU context before actors start asking.
        self._server = self._ctx.Process(
            target=_server_loop,
            args=(self.cfg, self._req_q, self._reply_qs, self._ctrl_q,
                  self._ctrl_reply_q, self._server_ready_q),
            daemon=True)
        self._server.start()

        for rank in range(n_actors):
            p = self._ctx.Process(
                target=_actor_loop,
                args=(rank, self.cfg, self._task_q, self._result_q,
                      self._req_q, self._reply_qs[rank]),
                daemon=True)
            p.start()
            self._procs.append(p)

        # Barrier: server up (has its CUDA context + weights), then every actor.
        deadline = time.time() + start_timeout
        try:
            self._server_ready_q.get(timeout=start_timeout)
        except _queue.Empty:
            self.close()
            raise RuntimeError(
                f"eval server did not come up within {start_timeout:.0f}s")

        ready = 0
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
        if not self._server.is_alive():
            raise RuntimeError(
                f"eval server died (exitcode {self._server.exitcode}) -- see "
                f"its traceback above; the pool cannot continue")
        for i, p in enumerate(self._procs):
            if not p.is_alive():
                raise RuntimeError(
                    f"actor {i} died (exitcode {p.exitcode}) -- see its "
                    f"traceback above; the pool cannot continue")

    def reload_weights(self, weights_path):
        """Point the server (only) at freshly saved teacher weights (promotion).

        Called between blocks, when no forward is in flight, so it cannot race a
        batched forward.
        """
        self._ctrl_q.put((_CTRL_RELOAD, os.path.abspath(weights_path)))
        try:
            self._ctrl_reply_q.get(timeout=180.0)
        except _queue.Empty:
            self._assert_alive()
            raise RuntimeError("eval server did not confirm a weight reload")

    def play_block(self, tags, seeds, timeout=1800.0):
        """Play one game per tag across the actors; return results (completion order).

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
        # Stop actors first (they may be mid-forward), then the server.
        for _ in self._procs:
            try:
                self._task_q.put((_TASK_STOP, None))
            except Exception:
                pass
        try:
            self._ctrl_q.put((_CTRL_STOP, None))
        except Exception:
            pass
        for p in self._procs:
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()
        if self._server is not None:
            self._server.join(timeout=10.0)
            if self._server.is_alive():
                self._server.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
