import argparse
import contextlib
import io
import json
import os
import random
import shutil
import time
from datetime import datetime

import torch
torch.set_float32_matmul_precision('high')

from agents.neural_net_agent_pg import NeuralNetAgentPG
from agents.agent_base import ModelConfigCNN, get_random_x_o
from arena.population_manager import PopulationManager, AgentRecord
from engine.game import GameState
from engine.constants import DRAW
from scripts.league_manager import LeagueManager
from scripts.trainer_base import find_latest_checkpoint, next_version, format_elapsed
from scripts.train_league import run_chunk_parallel, prune_versions

DEFAULT_STATE = "models/arena/arena_state.json"


def _find_best_archive_model(arena_base: str):
    """Scan all agent league.json files; return model_path with highest ELO."""
    best_elo, best_path = -float("inf"), None
    if not os.path.isdir(arena_base):
        return None
    for entry in os.listdir(arena_base):
        league_json = os.path.join(arena_base, entry, "league.json")
        if not os.path.isfile(league_json):
            continue
        try:
            with open(league_json) as f:
                entries = json.load(f)
            for e in entries:
                if e.get("elo", 0) > best_elo and os.path.isfile(e.get("model_path", "")):
                    best_elo, best_path = e["elo"], e["model_path"]
        except Exception:
            continue
    return best_path


HOF_DIR = "models/arena/hall_of_fame"
HOF_MIN_ELO = 1300


def _update_hall_of_fame(record: AgentRecord, checkpoint_path: str):
    """Copy checkpoint to hall_of_fame if it's a new best above HOF_MIN_ELO.

    File is named  <name>_elo<elo:.0f>.pt  -- overwrites any previous entry
    for the same agent so the directory stays tidy (one file per agent).
    """
    if record.elo < HOF_MIN_ELO:
        return
    os.makedirs(HOF_DIR, exist_ok=True)
    safe_name = record.name.lower().replace(" ", "-")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "-_")
    dest = os.path.join(HOF_DIR, f"{safe_name}_elo{record.elo:.0f}.pt")
    # Remove any previous entry for this agent before writing the new one.
    for existing in os.listdir(HOF_DIR):
        if existing.startswith(safe_name + "_elo") and existing.endswith(".pt"):
            os.remove(os.path.join(HOF_DIR, existing))
    shutil.copy2(checkpoint_path, dest)
    print(f"  [HoF] {record.name} ELO={record.elo:.0f} -> {dest}")


def _elo_to_stage(elo: float) -> int:
    if elo >= 1400: return 6
    if elo >= 1300: return 5
    if elo >= 1200: return 4
    if elo >= 1100: return 3
    return 2


# ---------------------------------------------------------------------------
# ArenaLeagueManager -- injects 15% cross-arena opponents
# ---------------------------------------------------------------------------

class ArenaLeagueManager(LeagueManager):
    """LeagueManager that replaces 15% of opponent slots with other active arena agents."""

    def __init__(self, population, model_dir, arena_agents_fn, exclude_id):
        super().__init__(population=population, model_dir=model_dir)
        self._arena_agents_fn = arena_agents_fn  # () -> {id: NeuralNetAgentPG}
        self._exclude_id = exclude_id

    def sample_opponent(self, active_agent):
        if random.random() < 0.15:
            others = [
                a for aid, a in self._arena_agents_fn().items()
                if aid != self._exclude_id
            ]
            if others:
                return self._make_self_play_clone(random.choice(others))
        return super().sample_opponent(active_agent)


# ---------------------------------------------------------------------------
# Helpers to build agents / leagues from records
# ---------------------------------------------------------------------------

def _build_agent(record: AgentRecord, lr: float) -> NeuralNetAgentPG:
    cfg = ModelConfigCNN(
        conv_channels=record.conv_channels,
        fc_hidden_sizes=record.fc_hidden_sizes,
        learning_rate=lr,
        label="arena",
        model_dir=record.model_dir,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        agent = NeuralNetAgentPG(cfg=cfg, model_path=None)
    agent.elo = record.elo
    return agent


def _load_into_memory(
    record: AgentRecord,
    nn_agents: dict,
    leagues: dict,
    nn_agents_fn,
    lr: float,
    seed_model: str,
):
    os.makedirs(record.model_dir, exist_ok=True)
    agent = _build_agent(record, lr=lr)

    latest_ver = find_latest_checkpoint(record.model_dir)
    if latest_ver is not None:
        ckpt_path = os.path.join(record.model_dir, f"version_{latest_ver:02d}.pt")
        agent.load(ckpt_path)
        agent.model.train()
    elif record.seed_model_dir:
        # Clone-best spawn: seed from source agent's latest checkpoint + perturb
        src_ver = find_latest_checkpoint(record.seed_model_dir)
        if src_ver is not None:
            src_path = os.path.join(record.seed_model_dir, f"version_{src_ver:02d}.pt")
            agent.seed_from_checkpoint(src_path)
            with torch.no_grad():
                for p in agent.model.parameters():
                    p.data.add_(torch.randn_like(p) * 0.01)
            print(f"  Seeded {record.name} from {record.seed_model_dir} (clone+perturb)")
    else:
        # Random spawn: seed from best archived model across all agents, then seed_model fallback
        arena_base = os.path.dirname(os.path.abspath(record.model_dir))
        best_path = _find_best_archive_model(arena_base)
        if best_path:
            agent.seed_from_checkpoint(best_path)
            print(f"  Seeded {record.name} from best archive: {best_path}")
        elif seed_model and os.path.isfile(seed_model):
            agent.seed_from_checkpoint(seed_model)

    # ELO in record is authoritative (may differ from checkpoint)
    agent.elo = record.elo

    archive_dir = os.path.join(record.model_dir, "archive")
    league = ArenaLeagueManager(
        population=[agent],
        model_dir=archive_dir,
        arena_agents_fn=nn_agents_fn,
        exclude_id=record.id,
    )
    league_json = os.path.join(record.model_dir, "league.json")
    if os.path.isfile(league_json):
        league.load_archive(league_json)

    nn_agents[record.id] = agent
    leagues[record.id] = league
    print(f"  Loaded {record.name} ({record.arch_label}) ELO={record.elo:.0f}")


# ---------------------------------------------------------------------------
# Disk sync -- pick up agents spawned externally (e.g. via GUI)
# ---------------------------------------------------------------------------

def _sync_new_agents(pm, nn_agents, leagues, nn_agents_fn, lr, seed_model):
    """Merge any agents added to arena_state.json by the GUI server since last save."""
    try:
        with open(pm.state_path) as f:
            disk = json.load(f)
    except Exception:
        return
    our_ids = {r.id for r in pm.agents}
    added = False
    for ad in disk.get("agents", []):
        if ad["id"] not in our_ids:
            rec = AgentRecord(**ad)
            pm.agents.append(rec)
            if rec.status == "active":
                print(f"  [sync] Picked up GUI-spawned agent: {rec.name} ({rec.arch_label})")
                _load_into_memory(rec, nn_agents, leagues, nn_agents_fn, lr, seed_model)
            added = True
    if added:
        pm._next_id = max((r.id for r in pm.agents), default=0) + 1


# ---------------------------------------------------------------------------
# Calibration tournament
# ---------------------------------------------------------------------------

def _play_eval_game(agent_a, agent_b) -> tuple:
    """Play one game between two agents in eval mode. No training data retained.
    Returns (winner: int, side_a: int)."""
    game = GameState()
    side_a = get_random_x_o()

    while not game.is_over():
        if game.player == side_a:
            move = agent_a.select_move(game)
        else:
            move = agent_b.select_move(game)
        game.make_move(move)

    # Discard any trajectory data that may have accumulated during the game.
    agent_a.clear_history()
    agent_b.clear_history()

    return game.winner, side_a


def _run_calibration_tournament(
    pm: PopulationManager,
    nn_agents: dict,
    games_per_pair: int = 40,
):
    """Round-robin evaluation tournament between all active agents.

    Runs `games_per_pair` games for every unique matchup, then incrementally
    updates each agent's ELO using K=16 (half of training K to avoid over-
    correction from a small sample).  ELO changes are written back to both the
    AgentRecord and the in-memory NeuralNetAgentPG so the training loop's next
    tick() call starts from calibrated values.
    """
    active = pm.active_agents()
    if len(active) < 2:
        return

    eligible = [r for r in active if r.id in nn_agents]
    if len(eligible) < 2:
        return

    n_pairs = len(eligible) * (len(eligible) - 1) // 2
    print(f"\n{'='*70}")
    print(
        f"[CALIBRATION] Round-robin: {len(eligible)} agents, "
        f"{n_pairs} pair(s) x {games_per_pair} games = {n_pairs * games_per_pair} games"
    )
    print(f"{'='*70}")

    # Switch to eval mode so no gradients or trajectories accumulate.
    for rec in eligible:
        nn_agents[rec.id].set_eval(True)

    # Working ELO table -- updated incrementally as matchups resolve.
    elos = {r.id: r.elo for r in eligible}
    totals = {r.id: [0, 0, 0] for r in eligible}  # [wins, losses, draws]

    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            rec_a, rec_b = eligible[i], eligible[j]
            agent_a = nn_agents[rec_a.id]
            agent_b = nn_agents[rec_b.id]

            w_a = w_b = draws = 0
            for _ in range(games_per_pair):
                winner, side_a = _play_eval_game(agent_a, agent_b)
                if winner == side_a:
                    w_a += 1
                elif winner == DRAW:
                    draws += 1
                else:
                    w_b += 1

            # Apply ELO updates for each individual game outcome (K=16).
            for _ in range(w_a):
                ea = 1.0 / (1.0 + 10 ** ((elos[rec_b.id] - elos[rec_a.id]) / 400.0))
                delta = 16.0 * (1.0 - ea)
                elos[rec_a.id] += delta
                elos[rec_b.id] -= delta
            for _ in range(w_b):
                eb = 1.0 / (1.0 + 10 ** ((elos[rec_a.id] - elos[rec_b.id]) / 400.0))
                delta = 16.0 * (1.0 - eb)
                elos[rec_b.id] += delta
                elos[rec_a.id] -= delta

            totals[rec_a.id][0] += w_a;  totals[rec_a.id][1] += w_b;  totals[rec_a.id][2] += draws
            totals[rec_b.id][0] += w_b;  totals[rec_b.id][1] += w_a;  totals[rec_b.id][2] += draws
            print(f"  {rec_a.name} vs {rec_b.name}: {w_a}W-{w_b}L-{draws}D")

    # Write calibrated ELOs back and restore train mode.
    for rec in eligible:
        rec.elo = elos[rec.id]
        nn_agents[rec.id].elo = elos[rec.id]
        nn_agents[rec.id].set_eval(False)

    ranked = sorted(eligible, key=lambda r: r.elo, reverse=True)
    print("[CALIBRATION] Recalibrated ELOs:")
    for rec in ranked:
        t = totals[rec.id]
        print(f"  {rec.name}: ELO={rec.elo:.0f}  ({t[0]}W/{t[1]}L/{t[2]}D)")

    pm.add_event(
        "calibration", "[*]",
        "Calibration: " + ", ".join(f"{r.name}={r.elo:.0f}" for r in ranked),
    )
    pm.save()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Arena multi-agent training loop")
    ap.add_argument("--state",      type=str,   default=DEFAULT_STATE)
    ap.add_argument("--chunks",     type=int,   default=0,    help="Total chunks to run (0 = forever)")
    ap.add_argument("--games",      type=int,   default=1000, help="Games per chunk")
    ap.add_argument("--batch",      type=int,   default=32,   help="Parallel batch size")
    ap.add_argument("--agents",     type=int,   default=3,    help="Initial agent count (fresh start only)")
    ap.add_argument("--retire",     type=int,   default=30,   help="Stagnation threshold (chunks)")
    ap.add_argument("--max_active", type=int,   default=5)
    ap.add_argument("--seed_model", type=str,   default="",   help="Seed weights for fresh agents")
    ap.add_argument("--lr",         type=float, default=1e-4)
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load or initialize PopulationManager                             #
    # ------------------------------------------------------------------ #
    pm = PopulationManager(
        state_path=args.state,
        retirement_threshold=args.retire,
        max_active=args.max_active,
    )

    is_resume = pm.load()

    if not is_resume:
        pm.status = "running"
        pm.start_time = datetime.now().isoformat(timespec="seconds")
        for _ in range(args.agents):
            pm.spawn_agent(mode="random")
        pm.add_event("start", "[>>]", f"Arena started fresh with {args.agents} agents")
        pm.save()
        print(f"Fresh start: spawned {args.agents} agents")
    else:
        if pm.status == "stopped":
            pm.status = "running"
            pm.add_event("start", ">", "Arena resumed (was stopped)")
            pm.save()
        print(f"Resumed arena: {len(pm.active_agents())} active agents, {pm.total_chunks_done} chunks done")

    # ------------------------------------------------------------------ #
    # 2. Build in-memory agents and leagues                               #
    # ------------------------------------------------------------------ #
    nn_agents: dict = {}  # id -> NeuralNetAgentPG
    leagues:   dict = {}  # id -> ArenaLeagueManager

    def nn_agents_fn():
        return nn_agents

    for record in pm.active_agents():
        _load_into_memory(record, nn_agents, leagues, nn_agents_fn, args.lr, args.seed_model)

    # ------------------------------------------------------------------ #
    # 3. Main loop                                                        #
    # ------------------------------------------------------------------ #
    state_dir = os.path.dirname(os.path.abspath(args.state))
    pause_path = os.path.join(state_dir, "arena.pause")

    # Track calibration milestones (every 50 total chunks).
    # Initialised from the current count so a resume doesn't re-run immediately.
    last_calib_milestone = pm.total_chunks_done // 50

    try:
        while True:
            # --- Pause check ---
            if os.path.exists(pause_path):
                pm.status = "paused"
                pm.save()
                print("Arena paused. Waiting...")
                while os.path.exists(pause_path):
                    time.sleep(5)
                pm.status = "running"
                pm.save()
                print("Arena resumed.")

            # Snapshot of active records for this round
            active_records = list(pm.active_agents())
            for record in active_records:
                # Pick up any agents spawned via the GUI before each tick/save
                _sync_new_agents(pm, nn_agents, leagues, nn_agents_fn, args.lr, args.seed_model)

                # May have been retired earlier in this same round
                if record.status != "active":
                    continue

                # Ensure agent is in memory (e.g. spawned mid-round)
                if record.id not in nn_agents:
                    _load_into_memory(record, nn_agents, leagues, nn_agents_fn, args.lr, args.seed_model)

                agent = nn_agents[record.id]
                league = leagues[record.id]
                league.population = [agent]

                chunk_label = pm.total_chunks_done + 1
                print(f"\n{'='*70}")
                print(f"[{record.name}] Chunk {chunk_label}  ELO={record.elo:.0f}  arch={record.arch_label}")
                print(f"{'='*70}")

                t0 = time.time()
                wins, losses, draws, avg_loss = run_chunk_parallel(
                    agent, league, chunk_label, args.games, args.batch
                )
                elapsed = time.time() - t0

                total = wins + losses + draws
                wr = wins / total if total else 0.0
                print(
                    f"\n[{record.name}] W={wins} L={losses} D={draws} "
                    f"WR={100*wr:.1f}% loss={avg_loss:.4f} "
                    f"ELO={agent.elo:.0f} {format_elapsed(elapsed)}"
                )

                pm.add_event(
                    "chunk", "[#]",
                    f"{record.name} chunk {chunk_label}: W={wins} L={losses} D={draws} "
                    f"WR={100*wr:.1f}% ELO={agent.elo:.0f}"
                )

                # Save versioned checkpoint
                version_path = next_version(record.model_dir)
                agent.save(version_path, verbose=False)
                print(f"  Checkpoint: {version_path}")
                prune_versions(record.model_dir, keep=5)

                # Hall of fame: track single best checkpoint per agent above ELO floor
                if agent.elo > record.best_elo:
                    _update_hall_of_fame(record, version_path)

                # Advance curriculum stage based on ELO
                new_stage = _elo_to_stage(agent.elo)
                if new_stage != league.curriculum_stage:
                    print(f"  [{record.name}] Stage {league.curriculum_stage}->{new_stage} (ELO={agent.elo:.0f})")
                    league.set_stage(new_stage)

                # LR decay at half the stagnation threshold (fires once at the exact crossing)
                half_thresh = args.retire // 2
                if record.stagnant_count == half_thresh and record.status == "active":
                    for pg in agent.optimizer.param_groups:
                        pg["lr"] *= 0.5
                    new_lr = agent.optimizer.param_groups[0]["lr"]
                    print(f"  [{record.name}] LR halved -> {new_lr:.2e} (stagnant for {record.stagnant_count} chunks)")

                # Sync again -- the chunk may have taken minutes; GUI could have spawned
                # an agent during that time. Without this, pm.tick()'s pm.save() would
                # overwrite arena_state.json and clobber the GUI-spawned agent.
                _sync_new_agents(pm, nn_agents, leagues, nn_agents_fn, args.lr, args.seed_model)

                # Tick (handles stagnation, retire, spawn; increments total_chunks_done)
                known_ids = {r.id for r in pm.agents}
                pm.tick(record, new_elo=agent.elo)

                # Check stop condition inside the round so --chunks is honoured precisely
                if args.chunks > 0 and pm.total_chunks_done >= args.chunks:
                    break

                # Load any agents spawned by tick
                for new_record in pm.agents:
                    if new_record.id not in known_ids and new_record.status == "active":
                        print(f"  New agent spawned: {new_record.name} ({new_record.arch_label})")
                        _load_into_memory(new_record, nn_agents, leagues, nn_agents_fn, args.lr, args.seed_model)

                # Evict retired agents from memory
                active_ids = {r.id for r in pm.active_agents()}
                for dead_id in list(nn_agents.keys()):
                    if dead_id not in active_ids:
                        print(f"  Evicting retired agent id={dead_id} from memory")
                        del nn_agents[dead_id]
                        del leagues[dead_id]

            # --- Calibration tournament every 50 total chunks ---
            # Must live INSIDE `while True:` so it actually fires during a run. It
            # previously sat after the loop and, under the default --chunks 0
            # (run-forever), was never reached -- so population ELOs were never
            # cross-calibrated and the "best spot" ranking drifted on uncalibrated
            # per-agent floats alone.
            calib_milestone = pm.total_chunks_done // 50
            if calib_milestone > last_calib_milestone and len(pm.active_agents()) >= 2:
                last_calib_milestone = calib_milestone
                _run_calibration_tournament(pm, nn_agents)

            # --- Stop condition ---
            if args.chunks > 0 and pm.total_chunks_done >= args.chunks:
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    pm.status = "stopped"
    pm.add_event("stop", "[STOP]", f"Stopped at {pm.total_chunks_done} total chunks")
    pm.save()

    active = pm.active_agents()
    print(f"\nArena stopped. Total chunks: {pm.total_chunks_done}")
    if active:
        best = max(active, key=lambda r: r.elo)
        print(f"Best active: {best.name}  ELO={best.elo:.0f}  arch={best.arch_label}")


if __name__ == "__main__":
    main()
