# scripts/train_league.py
import torch
torch.set_float32_matmul_precision('high')  # enables TF32 on Ampere+ GPUs (RTX 3080)
import argparse
import copy
import os
import time
from dataclasses import dataclass, field as dc_field
from datetime import datetime

from agents import get_agent
from agents.neural_net_agent_pg import NeuralNetAgentPG
from agents.agent_base import ModelConfigCNN
from engine.game import GameState
from engine.constants import X, O, DRAW
from engine.rules import rule_utl_valid_moves
from agents.agent_base import get_random_x_o
from scripts.league_manager import LeagueManager
from scripts.trainer_base import (
    format_elapsed, next_version, find_latest_checkpoint,
    append_metrics, clear_metrics_log,
)

import torch.nn.functional as F
from tqdm import trange


CHUNKS = 10
GAMES_PER_CHUNK = 1000
DEFAULT_ELO = 1000.0

NETWORK_CONFIGS = {
    "small":  dict(conv_channels=[32, 64, 64],         fc_hidden_sizes=[256, 512, 256]),
    "medium": dict(conv_channels=[64, 128, 128, 256],  fc_hidden_sizes=[512, 512, 256]),
    "large":  dict(conv_channels=[64, 128, 256, 256],  fc_hidden_sizes=[512, 1024, 512, 256]),
}

''' # TODO consider allowing random network configurations rather than just the same three in case we find new better architectures by accident.
def get_random_config( depth: int ):
    return {
        "conv_channels": [random.choice([32, 64, 128, 256]) for _ in range(depth)],
        "fc_hidden_sizes": [random.choice([128, 256, 512, 1024]) for _ in range(random.randint(2, 4))]
    }
'''


BIG8_BEST = r"models\big_8_layer\256-512-1024-2048-2048-1024-512-256-81\best.pt"
TS = "%Y-%m-%d %H:%M:%S"


def prune_versions(model_dir: str, keep: int):
    """Delete the oldest version_XX.pt files in model_dir, keeping the `keep` most recent."""
    import re
    entries = []
    for fname in os.listdir(model_dir):
        m = re.match(r"version_(\d+)\.pt$", fname)
        if m:
            entries.append((int(m.group(1)), os.path.join(model_dir, fname)))
    entries.sort(key=lambda t: t[0])
    to_delete = entries[:-keep] if len(entries) > keep else []
    for _, path in to_delete:
        os.remove(path)
        print(f"  Deleted old checkpoint: {path}")


def print_summary(start_dt: datetime, end_dt: datetime, chunks_done: int,
                  total_chunks: int, active: "NeuralNetAgentPG",
                  best_elo: float, league: "LeagueManager",
                  run_peak_elo: float = float("-inf")):
    elapsed = (end_dt - start_dt).total_seconds()
    print(f"\n{'='*70}")
    print(f"  Start          : {start_dt.strftime(TS)}")
    print(f"  End            : {end_dt.strftime(TS)}")
    print(f"  Elapsed        : {format_elapsed(elapsed)}")
    print(f"  Chunks done    : {chunks_done}/{total_chunks}")
    print(f"  Final ELO      : {active.elo:.0f}")
    # Run-realized peak (what best.pt holds) vs the global archive high-water mark.
    # For a run that never climbs above the seed default, best_elo stays at that floor
    # while run_peak_elo reflects the strongest weights the run actually produced.
    if run_peak_elo > float("-inf"):
        print(f"  Peak ELO (run) : {run_peak_elo:.0f}  (kept in best.pt if --save_best)")
    print(f"  Best ELO (h2o) : {best_elo:.0f}  (global archive high-water mark)")
    print(f"  Archive entries: {len(league.archive)}")
    print(f"{'='*70}")


_STAGE_THRESHOLDS = {0: 0.60, 1: 0.60, 2: 0.55, 3: 0.55, 4: 0.55, 5: 0.55}  # stage -> win-rate needed to advance; absent = final


def entropy_coef_for_stage(stage: int) -> float:
    """Phase 0b: decay the entropy bonus from 0.05 (explore) at stage 0 to 0.01
    (exploit) by stage >= 6. A fixed 0.05 keeps penalizing commitment to clearly
    correct moves once the agent is strong. Linear in stage, clamped to [0.01, 0.05].
    Derived from league.curriculum_stage, so non-curriculum runs get a fixed value
    matching whatever stage they sit at (stage 0 -> unchanged 0.05)."""
    stage = max(0, min(stage, 6))
    return 0.05 - (0.05 - 0.01) * (stage / 6)


def quick_eval_vs_random(active: NeuralNetAgentPG, games: int = 100) -> float:
    """Play `games` vs RandomAgent in eval mode (no gradient accumulation); return win rate."""
    from agents.random_agent import RandomAgent
    active.set_eval(True)
    opponent = RandomAgent()
    wins = 0
    for _ in range(games):
        winner, active_side = play_one_game(active, opponent)
        if winner == active_side:
            wins += 1
        active.clear_history()  # discard trajectory -- no learning
    active.set_eval(False)
    return wins / games


def maybe_advance_stage(active: NeuralNetAgentPG, league: "LeagueManager",
                        eval_games: int = 300) -> bool:
    """Eval vs random; if win rate meets threshold, advance stage. Returns True if advanced."""
    stage = league.curriculum_stage
    threshold = _STAGE_THRESHOLDS.get(stage)
    if threshold is None:
        return False  # final stage, no advancement

    games = 500 if stage >= 4 else eval_games
    wr = quick_eval_vs_random(active, games=games)
    print(f"  Curriculum eval vs random: {100*wr:.1f}% (stage {stage}, need {100*threshold:.0f}%)")

    if wr >= threshold:
        league.set_stage(stage + 1)
        print(f"  *** Stage advance: {stage} -> {stage + 1} ***")
        return True
    return False


def run_eval(checkpoint: str, opponent_key: str, games: int, network: str = "small",
             tactical: bool = False):
    """Load a PG checkpoint and play evaluation games (no learning)."""
    net = NETWORK_CONFIGS[network]
    cfg = ModelConfigCNN(
        **net,
        learning_rate=1e-4,
        label="league_pg",
        model_dir="models/league_pg",
    )
    agent = NeuralNetAgentPG(cfg=cfg, model_path=checkpoint, tactical=tactical)
    agent.set_eval(True)
    if tactical:
        print("Eval with 1-ply tactical lookahead ENABLED.")

    opponent = get_agent(opponent_key)
    getattr(opponent, "set_eval", lambda _: None)(True)

    wins = losses = draws = 0
    t = trange(
        games,
        desc="Eval",
        unit="game",
        bar_format="{desc}: {percentage:.0f}%|{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for _ in t:
        game = GameState()
        agent_side = get_random_x_o()
        getattr(opponent, "clear_history", lambda: None)()

        while not game.is_over():
            if game.player == agent_side:
                move = agent.select_move(game)
            else:
                move = opponent.select_move(game)
            valid, _ = game.make_move(move)
            if not valid:
                raise ValueError(f"Invalid move {move}")

        if game.winner == agent_side:
            wins += 1
        elif game.winner == DRAW:
            draws += 1
        else:
            losses += 1

        n = wins + losses + draws
        t.set_description(f"Eval | W={wins} L={losses} D={draws} WR={100*wins/n:.1f}%")

    total = wins + losses + draws
    print(f"\nEval vs {opponent_key} over {total} games:")
    print(f"  Wins   : {wins:>4}  ({100*wins/total:.1f}%)")
    print(f"  Losses : {losses:>4}  ({100*losses/total:.1f}%)")
    print(f"  Draws  : {draws:>4}  ({100*draws/total:.1f}%)")


def play_one_game(active: NeuralNetAgentPG, opponent) -> int:
    """
    Play a single full game. active is the learning agent; opponent is frozen.
    Returns game.winner (X, O, or DRAW).
    """
    game = GameState()
    active_side = get_random_x_o()

    active.clear_history()
    if hasattr(opponent, "clear_history"):
        opponent.clear_history()

    # pending accumulates shaping from opponent moves until the next active move
    pending = 0.0

    while not game.is_over():
        mover = game.player
        mini_before = game.mini_winners[:]

        if mover == active_side:
            move = active.select_move(game)
        else:
            move = opponent.select_move(game)

        # _valid_before = rule_utl_valid_moves(game.board, game.last_move, game.mini_winners)
        # assert move in _valid_before, (
        #     f"Move {move} not in valid moves {_valid_before} "
        #     f"(last_move={game.last_move}, player={game.player})"
        # ) game.make_move already validates

        valid, _ = game.make_move(move)
        if not valid:
            raise ValueError(f"Invalid move {move} played")

        # shaping: +0.3 per mini-board won by active, -0.3 per mini-board won by opponent
        shaping = 0.0
        for i in range(9):
            new = game.mini_winners[i]
            if new != mini_before[i] and new in (X, O):
                shaping += 0.3 if new == active_side else -0.3

        if mover == active_side:
            # active just moved: record accumulated pending + this move's shaping
            active.last_rewards.append(pending + shaping)
            pending = 0.0
        else:
            # opponent just moved: defer shaping to next active reward slot
            pending += shaping

    # terminal reward -- add to last active reward slot (covers games ending on opponent's turn)
    if game.winner == active_side:
        terminal = 1.0
    elif game.winner == DRAW:
        terminal = -0.1
    else:
        terminal = -1.0

    if active.last_rewards:
        active.last_rewards[-1] += terminal + pending
    # else: active never moved (shouldn't occur in UTTT)

    if active.log_probs:
        assert len(active.last_rewards) == len(active.log_probs), (
            f"last_rewards ({len(active.last_rewards)}) != log_probs ({len(active.log_probs)})"
        )

    return game.winner, active_side


@dataclass
class Trajectory:
    log_probs:   list
    values:      list
    rewards:     list
    entropies:   list
    winner:      int
    active_side: int
    opponent:    object   # kept for ELO updates; not used by learn_from_trajectories
    # Collect-then-recompute (THROUGHPUT.md Part C): detached per-move inputs, populated only
    # when ParallelGameRunner.run(collect_inputs=True). Empty by default -> the in-graph
    # learn_from_trajectories path ignores them and stays byte-identical.
    states:      list = dc_field(default_factory=list)
    valids:      list = dc_field(default_factory=list)
    actions:     list = dc_field(default_factory=list)


@dataclass
class _GameSlot:
    game:        GameState
    active_side: int
    opponent:    object
    log_probs:   list  = dc_field(default_factory=list)
    values:      list  = dc_field(default_factory=list)
    rewards:     list  = dc_field(default_factory=list)
    entropies:   list  = dc_field(default_factory=list)
    pending:     float = 0.0
    done:        bool  = False
    # collect-then-recompute capture (only filled under collect_inputs=True)
    states:      list  = dc_field(default_factory=list)
    valids:      list  = dc_field(default_factory=list)
    actions:     list  = dc_field(default_factory=list)


def _shaping(mini_after: list, mini_before: list, active_side: int) -> float:
    s = 0.0
    for i in range(9):
        new = mini_after[i]
        if new != mini_before[i] and new in (X, O):
            s += 0.3 if new == active_side else -0.3
    return s


def _terminal(winner: int, active_side: int) -> float:
    if winner == active_side:
        return 1.0
    if winner == DRAW:
        return -0.1
    return -1.0


def _opponent_key(opp):
    """Weight-identity key for grouping batched opponents within one run() call.

    Opponents sharing this key are guaranteed weight-identical AND same-tactical, so one
    of them can run the group's batched forward for all. The league tags self-play clones
    'clone' and archive copies 'archive:<path>'; shared anchor objects (nn_big8, lottery)
    have no tag and fall back to their model's id (a shared object -> one group). Grouping
    only ever happens inside a single run() call, where every 'clone' copies the same
    active weights, so 'clone' is unambiguous."""
    tac = getattr(opp, "tactical", False)
    k = getattr(opp, "weight_key", None)
    if k is not None:
        return (k, tac)
    model = getattr(opp, "model", None)
    return (id(model) if model is not None else id(opp), tac)


class ParallelGameRunner:
    def __init__(self, active: NeuralNetAgentPG, opponents: list):
        self.active = active
        self.opponents = opponents

    def run(self, collect_inputs: bool = False, batch_opponents: bool = False) -> list:
        """Run all games to completion, batching active-agent forward passes.
        Returns one Trajectory per game.

        collect_inputs=True additionally stores the detached per-move (state, valid_moves,
        action) on each slot/Trajectory so the collect-then-recompute learn path
        (THROUGHPUT.md Part C) can rebuild forward passes. Default False = byte-identical to
        the in-graph path (no extra capture; Trajectory.states/valids/actions stay empty).

        batch_opponents=True groups the opponent forward passes the same way the active
        step is already batched: opponents that expose batch_select_moves_eval (all the NN
        opponents -- clones, archives, nn_big8, lottery) are grouped by weight-identity and
        run one batched argmax forward per group; stochastic / non-NN opponents (random,
        deterministics, MixedAgent) stay in a per-slot loop in slot order. Because NN-eval
        is deterministic argmax and consumes no RNG, this is byte-identical in outcome to
        the per-slot default -- verify_opponent_batch_parity.py certifies it. Default False
        = the original unbatched opponent loop."""
        slots = []
        for opp in self.opponents:
            if hasattr(opp, "clear_history"):
                opp.clear_history()
            slots.append(_GameSlot(
                game=GameState(),
                active_side=get_random_x_o(),
                opponent=opp,
            ))

        while not all(s.done for s in slots):
            # --- batched active step ---
            active_slots = [s for s in slots if not s.done and s.game.player == s.active_side]
            if active_slots:
                mini_befores = [s.game.mini_winners[:] for s in active_slots]
                if collect_inputs:
                    actions, log_probs, values, entropies, states, valids = \
                        self.active.batch_select_moves(
                            [s.game for s in active_slots], return_inputs=True)
                else:
                    actions, log_probs, values, entropies = self.active.batch_select_moves(
                        [s.game for s in active_slots]
                    )
                for j, slot in enumerate(active_slots):
                    if actions[j] is None:
                        # empty valid moves -- game is over but not yet marked done
                        slot.done = True
                        continue
                    slot.game.make_move(actions[j])
                    shape = _shaping(slot.game.mini_winners, mini_befores[j], slot.active_side)
                    slot.log_probs.append(log_probs[j])
                    slot.values.append(values[j])
                    slot.entropies.append(entropies[j])
                    slot.rewards.append(slot.pending + shape)
                    if collect_inputs:
                        # Same index as log_probs/values just appended -> the (state,action)<->
                        # reward alignment is by construction; verify_recompute_parity.py
                        # certifies it numerically.
                        slot.states.append(states[j])
                        slot.valids.append(valids[j])
                        slot.actions.append(actions[j])
                    slot.pending = 0.0
                    if slot.game.is_over():
                        slot.done = True
                        slot.rewards[-1] += _terminal(slot.game.winner, slot.active_side)

            # --- opponent step ---
            # also catches slots that just had an active move and whose turn flipped to opponent
            opp_slots = [
                s for s in slots
                if not s.done and s.game.player != s.active_side
            ]
            # mini_winners snapshot BEFORE any opponent move. Slots are independent, so
            # capturing all up front == capturing each right before its own move.
            mini_befores = {id(s): s.game.mini_winners[:] for s in opp_slots}

            moves = {}
            if batch_opponents:
                # Group NN opponents (deterministic argmax, RNG-free) by weight identity;
                # resolve stochastic / non-NN opponents in place, in slot order, so their
                # random stream is identical to the unbatched path.
                groups = {}   # key -> [slots]
                reps = {}     # key -> representative opponent for the group
                for s in opp_slots:
                    opp = s.opponent
                    if hasattr(opp, "batch_select_moves_eval"):
                        key = _opponent_key(opp)
                        groups.setdefault(key, []).append(s)
                        reps.setdefault(key, opp)
                    else:
                        moves[id(s)] = opp.select_move(s.game)
                for key, members in groups.items():
                    group_moves = reps[key].batch_select_moves_eval([m.game for m in members])
                    for m, mv in zip(members, group_moves):
                        moves[id(m)] = mv
            else:
                for s in opp_slots:
                    moves[id(s)] = s.opponent.select_move(s.game)

            for slot in opp_slots:
                move = moves[id(slot)]
                if move is None:
                    # No legal move for the opponent (pathological -- the game is
                    # effectively over). Cannot occur in normal play; guarded so a
                    # None never reaches make_move.
                    slot.done = True
                    continue
                slot.game.make_move(move)
                slot.pending += _shaping(slot.game.mini_winners, mini_befores[id(slot)], slot.active_side)
                if slot.game.is_over():
                    slot.done = True
                    if slot.rewards:
                        slot.rewards[-1] += _terminal(slot.game.winner, slot.active_side) + slot.pending
                    slot.pending = 0.0

        return [
            Trajectory(
                log_probs=s.log_probs,
                values=s.values,
                rewards=s.rewards,
                entropies=s.entropies,
                winner=s.game.winner,
                active_side=s.active_side,
                opponent=s.opponent,
                states=s.states,
                valids=s.valids,
                actions=s.actions,
            )
            for s in slots
        ]


def run_chunk_parallel(active: NeuralNetAgentPG, league: LeagueManager,
                       chunk_idx: int, n_games: int, batch_size: int, gamma: float = 0.99,
                       log_metrics: bool = False, value_coef: float = 0.5,
                       recompute: bool = False, minibatch_size: int = 0,
                       batch_opponents: bool = False):
    active.set_eval(False)
    entropy_coef = entropy_coef_for_stage(league.curriculum_stage)

    wins = losses = draws = 0
    total_loss = 0.0
    loss_count = 0
    games_done = 0
    n_batches = (n_games + batch_size - 1) // batch_size

    t = trange(
        n_batches,
        desc=f"Chunk {chunk_idx}",
        unit="batch",
        bar_format="{desc}: {percentage:.0f}%|{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for _ in t:
        this_batch = min(batch_size, n_games - games_done)

        opponents = []
        for _ in range(this_batch):
            opp = league.sample_opponent(active)
            if hasattr(opp, "set_eval"):
                opp.set_eval(True)
            opponents.append(opp)

        trajectories = ParallelGameRunner(active, opponents).run(
            collect_inputs=recompute, batch_opponents=batch_opponents)

        if recompute:
            # Collect-then-recompute (Part C): decouples #gradient-steps from the batch size.
            loss = active.learn_from_trajectories_recompute(
                trajectories, gamma=gamma, entropy_coef=entropy_coef,
                value_coef=value_coef, minibatch_size=minibatch_size)
        else:
            loss = active.learn_from_trajectories(trajectories, gamma=gamma, entropy_coef=entropy_coef,
                                                  value_coef=value_coef)
        if loss is not None:
            total_loss += loss
            loss_count += 1

        for traj in trajectories:
            if traj.winner == traj.active_side:
                wins += 1
                league.update_elo(active, traj.opponent)
            elif traj.winner == DRAW:
                draws += 1
            else:
                losses += 1
                league.update_elo(traj.opponent, active)

        games_done += this_batch
        n = wins + losses + draws
        wr = wins / n if n else 0.0
        if log_metrics and loss is not None:
            append_metrics(loss, 0.0, wr,
                           stage=league.curriculum_stage,
                           elo=getattr(active, 'elo', DEFAULT_ELO),
                           value_loss=getattr(active, 'last_value_loss', None),
                           explained_var=getattr(active, 'last_explained_var', None))   # epsilon N/A for PG; running WR + curriculum stage + learner ELO + value-head quality
        avg_loss = total_loss / loss_count if loss_count else 0.0
        t.set_description(
            f"Chunk {chunk_idx} | loss={avg_loss:.4f} WR={100*wr:.1f}%"
            f" ELO={getattr(active, 'elo', DEFAULT_ELO):.0f} games={games_done}"
        )

    return wins, losses, draws, total_loss / loss_count if loss_count else 0.0


def run_chunk(active: NeuralNetAgentPG, league: LeagueManager, chunk_idx: int, games: int,
              gamma: float = 0.99, fix_0c: bool = False, log_metrics: bool = False,
              value_coef: float = 0.5):
    active.set_eval(False)  # train mode
    entropy_coef = entropy_coef_for_stage(league.curriculum_stage)

    wins = losses = draws = 0
    total_loss = 0.0
    loss_count = 0

    t = trange(
        games,
        desc=f"Chunk {chunk_idx}",
        unit="game",
        bar_format="{desc}: {percentage:.0f}%|{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for game_num, _ in enumerate(t):
        opponent = league.sample_opponent(active)
        if hasattr(opponent, "set_eval"):
            opponent.set_eval(True)

        winner, active_side = play_one_game(active, opponent)

        loss = active.learn(gamma=gamma, update=(game_num % 8 == 0),
                            entropy_coef=entropy_coef, fix_0c=fix_0c, value_coef=value_coef)
        if loss is not None:
            total_loss += loss
            loss_count += 1

        # ELO -- only meaningful when opponent carries .elo
        if winner == active_side:
            wins += 1
            league.update_elo(active, opponent)
        elif winner == DRAW:
            draws += 1
        else:
            losses += 1
            league.update_elo(opponent, active)

        n = wins + losses + draws
        wr = wins / n if n else 0.0
        if log_metrics and loss is not None:
            append_metrics(loss, 0.0, wr,
                           stage=league.curriculum_stage,
                           elo=getattr(active, 'elo', DEFAULT_ELO),
                           value_loss=getattr(active, 'last_value_loss', None),
                           explained_var=getattr(active, 'last_explained_var', None))   # epsilon N/A for PG; running WR + curriculum stage + learner ELO + value-head quality
        avg_loss = total_loss / loss_count if loss_count else 0.0
        t.set_description(
            f"Chunk {chunk_idx} | loss={avg_loss:.4f} WR={100*wr:.1f}% ELO={getattr(active, 'elo', DEFAULT_ELO):.0f}"
        )

    return wins, losses, draws, total_loss / loss_count if loss_count else 0.0


def run_debug_games(active: NeuralNetAgentPG, opponent, n_games: int = 3):
    """Play n_games, printing every move, board state, and winner. No training."""
    from engine.constants import PLAYER_MAP
    active.set_eval(True)
    getattr(opponent, "set_eval", lambda _: None)(True)

    for g in range(1, n_games + 1):
        game = GameState()
        active_side = get_random_x_o()
        active.clear_history()
        getattr(opponent, "clear_history", lambda: None)()

        print(f"\n{'#'*50}")
        print(f"# Game {g}  --  active={PLAYER_MAP[active_side]}")
        print(f"{'#'*50}")

        move_num = 0
        while not game.is_over():
            move_num += 1
            if game.player == active_side:
                move = active.select_move(game)
                actor = f"ACTIVE ({PLAYER_MAP[active_side]})"
            else:
                move = opponent.select_move(game)
                actor = f"opponent ({PLAYER_MAP[game.player]})"

            game.make_move(move)
            print(f"  Move {move_num:>3}: {actor} -> {move}")
            game.print_board()

        winner_str = PLAYER_MAP.get(game.winner, str(game.winner))
        if game.winner == active_side:
            result = "ACTIVE wins"
        elif game.winner == 3:  # DRAW
            result = "Draw"
        else:
            result = "opponent wins"
        print(f">>> Game {g} over -- winner: {winner_str} ({result})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks",          type=int,   default=CHUNKS,          help="Number of training chunks")
    ap.add_argument("--chunk_games",     type=int,   default=GAMES_PER_CHUNK, help="Games per chunk")
    ap.add_argument("--model_dir",       type=str,   default="models/league_pg", help="Root dir for checkpoints")
    ap.add_argument("--temperature",     type=float, default=1.0,             help="Softmax temperature for sampling")
    ap.add_argument("--lr",              type=float, default=1e-4,            help="Learning rate")
    ap.add_argument("--seed_model",      type=str,   default=r"models\big_8_layer\256-512-1024-2048-2048-1024-512-256-81\best.pt",
                    help="Path to a checkpoint to seed weights from before training. Pass empty string to skip.")
    ap.add_argument("--eval",            action="store_true",                 help="Run evaluation only, no training.")
    ap.add_argument("--eval_checkpoint", type=str,   default="",             help="PG checkpoint to evaluate (required with --eval).")
    ap.add_argument("--eval_games",      type=int,   default=200,             help="Number of eval games.")
    ap.add_argument("--eval_opponent",   type=str,   default="nn_big_8",      help="Agent key to evaluate against.")
    ap.add_argument("--resume",          action="store_true",                 help="Continue from the last checkpoint in --model_dir.")
    ap.add_argument("--resume_checkpoint", type=str,   default="",           help="Explicit checkpoint path to resume from (overrides --resume auto-find).")
    ap.add_argument("--keep_versions",   type=int,   default=5,               help="Keep only the N most recent version_XX.pt files.")
    ap.add_argument("--curriculum",      action="store_true",                 help="Enable curriculum training: start at stage 0 and auto-advance.")
    ap.add_argument("--debug_games",     action="store_true",                 help="Play 3 observed games, print every move and board state, then exit.")
    ap.add_argument("--parallel",        type=int,   default=0,               help="Parallel batch size for game runner. 0 = sequential (default).")
    ap.add_argument("--network",         type=str,   default="small",         choices=["small", "medium", "large"], help="Network size (default: small).")
    ap.add_argument("--fix_0c",          action=argparse.BooleanOptionalAction, default=True, help="Phase 0c: use the corrected sequential learn() (no per-game adv-norm; .mean() actor/entropy -- mirrors the batched path). DEFAULT ON (validated: RESULT_0c -- breaks the 1300-1550 plateau, peak 1728). Pass --no-fix_0c to reproduce the old broken curve. Sequential path only (--parallel 0); a no-op for --parallel>0, which already uses the corrected batched path.")
    ap.add_argument("--no_metrics",      action="store_true",                 help="Disable writing loss_logs/metrics_log.jsonl. By default a run clears and streams loss/win-rate there so the dashboard Training tab (python -m arena.gui_server) shows it live.")
    ap.add_argument("--save_best",       action=argparse.BooleanOptionalAction, default=True, help="Track the run's strongest weights in <model_dir>/best.pt (separate from the rolling version_NN.pt and the archive -- immune to pruning and the 50-cap eviction). A run-owned floor is written at startup so best.pt always exists and matches this run's architecture, then it's overwritten on each new REALIZED-ELO peak. DEFAULT ON. Pass --no-save_best to disable.")
    ap.add_argument("--patience",        type=int,   default=0,               help="Early-stop: stop if no new per-stage ELO peak for N consecutive chunks (0 = disabled, default). The baseline is the run's OWN first measured ELO (seeded from -inf, not the 1000 default), so a fresh run dipping below 1000 isn't falsely flagged. The counter is rebased when the curriculum stage advances (a harder pool legitimately lowers ELO), so it catches intra-stage forgetting, not curriculum drift.")
    ap.add_argument("--restore_best",    action=argparse.BooleanOptionalAction, default=True, help="When --patience fires, reload best.pt before exiting so the run ends on its peak weights (requires --save_best). Only restores a best.pt THIS run produced (never a stale/foreign leftover) and is load-guarded, so it can't crash the run. No effect without --patience. DEFAULT ON. Pass --no-restore_best to disable.")
    ap.add_argument("--value_coef",      type=float, default=0.5,             help="Weight on the value (critic) loss in the combined objective: loss = actor + value_coef*value - entropy_coef*entropy. Default 0.5 (unchanged). Sweep 0.25/0.5/1.0 to confirm the value-head weight after the 0c .mean() switch (RESULT_0c ask #3).")
    ap.add_argument("--recompute",       action=argparse.BooleanOptionalAction, default=False, help="Batched path only (--parallel>0): use the collect-then-recompute learn step (THROUGHPUT.md Part C) instead of the in-graph learn_from_trajectories. Stores detached (state,action,reward) during self-play and runs single-epoch minibatch SGD with fresh forwards, so the number of gradient steps is set by --minibatch_size, not the self-play batch (--parallel) -- fixes the 'big batch starves updates' stall AND the OOM. DEFAULT OFF (byte-identical to today). VALIDATE FIRST: python -m scripts.verify_recompute_parity (must PASS) + home_batch --phase recompute, before trusting it in a long run.")
    ap.add_argument("--minibatch_size",  type=int,   default=0,               help="With --recompute: SGD minibatch size over collected transitions (0 = one minibatch = the whole self-play batch = a single full-batch step, numerically equivalent to the in-graph path). Smaller = more gradient steps per self-play batch (the point of the decouple). No effect without --recompute.")
    ap.add_argument("--batch_opponents", action=argparse.BooleanOptionalAction, default=False, help="Batched path only (--parallel>0): batch the OPPONENT forward passes too, not just the active agent's. NN opponents (clones, archives, nn_big8, lottery) are deterministic argmax at eval time, so grouping them by weight and running one batched forward per group is byte-identical in outcome to the per-slot loop -- it just removes an unbatched Python-driven forward per opponent move. DEFAULT OFF. VALIDATE FIRST: python -m scripts.verify_opponent_batch_parity (must PASS) before trusting it in a long run.")
    ap.add_argument("--seed",            type=int,   default=None,            help="Seed torch/numpy/random for reproducible runs (weight init, opponent sampling, action sampling). Default None = unseeded (existing behavior). Use for N-seed repeats so a good seed can be reproduced; note CUDA kernels are not fully deterministic even when seeded.")
    ap.add_argument("--value_tanh",      action=argparse.BooleanOptionalAction, default=False, help="Apply tanh to the value head output (calibrates to [-1, 1]). Default OFF for backward compat with existing checkpoints. Enable for new AlphaZero runs: fixes the MCTS value-scale mismatch (see agents/mcts.py docstring). Existing checkpoints are incompatible -- start fresh or retrain.")
    ap.add_argument("--tactical",        action="store_true",                 help="Eval only (--eval): enable 1-ply tactical lookahead (take an immediate win / avoid an immediate loss) on top of the policy argmax. Lets you measure the lookahead's strength gain. No effect on training.")
    args = ap.parse_args()

    if args.seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seeded RNGs with {args.seed} (torch/numpy/random).")

    if args.eval:
        if not args.eval_checkpoint:
            ap.error("--eval requires --eval_checkpoint <path>")
        run_eval(args.eval_checkpoint, args.eval_opponent, args.eval_games, network=args.network,
                 tactical=args.tactical)
        return

    if args.debug_games:
        net = NETWORK_CONFIGS[args.network]
        cfg = ModelConfigCNN(
            **net,
            learning_rate=args.lr,
            label="league_pg",
            model_dir=args.model_dir,
            value_tanh=args.value_tanh,
        )
        active = NeuralNetAgentPG(cfg=cfg, model_path=None, temperature=args.temperature)
        if args.seed_model:
            active.seed_from_checkpoint(args.seed_model)
        from agents.random_agent import RandomAgent
        run_debug_games(active, RandomAgent())
        return

    os.makedirs(args.model_dir, exist_ok=True)

    net = NETWORK_CONFIGS[args.network]
    cfg = ModelConfigCNN(
        **net,
        learning_rate=args.lr,
        label="league_pg",
        model_dir=args.model_dir,
        value_tanh=args.value_tanh,
    )

    active = NeuralNetAgentPG(cfg=cfg, model_path=None, temperature=args.temperature)
    active.elo = float(DEFAULT_ELO)

    league = LeagueManager(population=[active], model_dir=os.path.join(args.model_dir, "archive"))

    if args.resume_checkpoint or args.resume:
        league_json = league.league_json

        if args.resume_checkpoint:
            resume_path = args.resume_checkpoint
            if not os.path.isfile(resume_path):
                ap.error(f"--resume_checkpoint: file not found: {resume_path}")
        else:
            latest_ver = find_latest_checkpoint(args.model_dir)
            if latest_ver is None:
                print(f"  Warning: --resume passed but no version_XX.pt found in {args.model_dir}. Starting fresh.")
                if args.seed_model and os.path.isfile(args.seed_model):
                    active.seed_from_checkpoint(args.seed_model)
                elif args.seed_model:
                    print(f"  Seed model not found ({args.seed_model}) -- starting from random weights.")
                resume_path = None
            else:
                resume_path = os.path.join(args.model_dir, f"version_{latest_ver:02d}.pt")

        if resume_path:
            active.load(resume_path)
            active.model.train()

            if os.path.isfile(league_json):
                league.load_archive(league_json)

            if league.archive:
                active.elo = max(e.elo for e in league.archive)

            print(
                f"  Resuming from {resume_path} | "
                f"ELO: {active.elo:.0f} | "
                f"Archive entries: {len(league.archive)}"
            )
    else:
        seed = args.seed_model
        if seed and not os.path.isfile(seed):
            print(f"[!]  Seed model not found: {seed}\n"
                  f"    Starting from RANDOM weights instead. Pass --seed_model \"\" to "
                  f"silence this, or point it at a checkpoint that exists.")
            seed = ""
        if seed:
            active.seed_from_checkpoint(seed)
        else:
            print("No seed model provided -- starting from random weights.")

    # Curriculum: --curriculum starts at stage 0 and auto-advances on win-rate.
    # Without it, a *fresh* (random-weight) agent would otherwise train at the
    # LeagueManager default stage 2 (50% self-play + a half-strength nn_big_8) -- a
    # brutal cold start with almost no win signal. So drop fresh starts to stage 0
    # too; only an explicit resume (a real checkpoint that can hold its own) keeps
    # the stage-2 default.
    if args.curriculum:
        league.set_stage(0)
        print("Curriculum training enabled -- starting at stage 0.")
    elif not (args.resume or args.resume_checkpoint):
        league.set_stage(0)
        print("Fresh start without --curriculum -- beginning at stage 0 to avoid a "
              "cold-start wall (pass --curriculum for automatic stage advancement).")

    log_metrics = not args.no_metrics
    if log_metrics:
        clear_metrics_log()

    best_elo = active.elo
    start_dt = datetime.now()
    print(f"Starting league training: {args.chunks} chunks x {args.chunk_games:,} games")
    print(f"Started            : {start_dt.strftime(TS)}")
    print(f"Model dir          : {args.model_dir}")
    if args.parallel > 0:
        # ASCII-only: this can print to a redirected/cp1252 stdout (see engine/game.py import banner).
        if args.recompute:
            mb = args.minibatch_size if args.minibatch_size > 0 else "full-batch"
            print(f"Learn path         : collect-then-recompute (Part C), minibatch={mb} "
                  f"[validate via: python -m scripts.verify_recompute_parity]")
        if not args.fix_0c:
            print(f"[note] --no-fix_0c with --parallel>0 has no effect -- the batched path is "
                  f"always corrected; 0c only governs the sequential (--parallel 0) learn().")
    else:
        if args.recompute:
            print(f"[note] --recompute has no effect with --parallel 0 (sequential path); it "
                  f"governs the batched (--parallel>0) learn step only.")
        print(f"Phase 0c fix       : {'ON (corrected sequential learn())' if args.fix_0c else 'OFF (old broken curve -- --no-fix_0c)'}")
    if log_metrics:
        print(f"Live metrics       : loss_logs/metrics_log.jsonl  "
              f"(watch the dashboard Training tab; --no_metrics to disable)")
    print()

    consecutive_collapse_chunks = 0
    chunks_since_best = 0              # consecutive chunks with no new per-stage ELO peak (for --patience)
    # Patience baseline seeds from the run's OWN realized trajectory, not the seed-default ELO
    # (1000). Seeding from 1000 made a fresh run that legitimately dips below 1000 early look like
    # "no new peak" every chunk and false-fire --patience at exactly chunk == patience. -inf means
    # chunk 1's measured ELO always becomes the first baseline.  (RESULT_STRENGTH_RUN bug 2)
    stage_best_elo = float("-inf")    # per-stage ELO baseline, rebased on curriculum advance (drift-safe)
    run_peak_elo = float("-inf")      # run's REALIZED peak ELO -- drives best.pt; decoupled from the
                                      # 1000-floored archive high-water mark so a sub-1000 run still
                                      # keeps its strongest weights.
    best_path = os.path.join(args.model_dir, "best.pt")
    best_saved = False                # did THIS run write best.pt? --restore_best refuses to load a
                                      # stale/foreign file it didn't produce.  (RESULT_STRENGTH_RUN bug 1)
    # Lay down a run-owned floor immediately so best.pt always exists, always matches this run's
    # architecture, and --restore_best can never grab a leftover checkpoint from an earlier run.
    if args.save_best:
        active.save(best_path, verbose=False)
        best_saved = True
        print(f"Best-model floor   : {best_path} (ELO {active.elo:.0f})")
    chunks_done = 0
    try:
        for chunk_idx in range(1, args.chunks + 1):
            stage_before = league.curriculum_stage
            stage_str = f"  stage={league.curriculum_stage}" if args.curriculum else ""
            print(f"\n{'='*70}")
            print(f"Chunk {chunk_idx}/{args.chunks}  (ELO before: {active.elo:.0f}){stage_str}")
            print(f"{'='*70}")

            t0 = time.time()
            if args.parallel > 0:
                wins, losses, draws, avg_loss = run_chunk_parallel(
                    active, league, chunk_idx, args.chunk_games, args.parallel,
                    gamma=0.99, log_metrics=log_metrics, value_coef=args.value_coef,
                    recompute=args.recompute, minibatch_size=args.minibatch_size,
                    batch_opponents=args.batch_opponents
                )
            else:
                wins, losses, draws, avg_loss = run_chunk(
                    active, league, chunk_idx, args.chunk_games,
                    gamma=0.99, fix_0c=args.fix_0c, log_metrics=log_metrics,
                    value_coef=args.value_coef
                )
            elapsed = time.time() - t0
            chunks_done = chunk_idx

            total_games = wins + losses + draws
            wr = wins / total_games if total_games else 0.0
            print(
                f"\nChunk {chunk_idx} done | W={wins} L={losses} D={draws} "
                f"WR={100*wr:.1f}% | avg_loss={avg_loss:.4f} | "
                f"ELO={active.elo:.0f} | {format_elapsed(elapsed)}"
            )

            if args.curriculum:
                maybe_advance_stage(active, league)

            # Global ELO high-water mark -- drives the ARCHIVE only. Stays a pure max() seeded from
            # the seed-default ELO so archiving is byte-identical when no new flag is set.
            if active.elo > best_elo:
                best_elo = active.elo
                entry = league.add_to_archive(active, elo=active.elo)
                print(f"  New ELO high {best_elo:.0f} -- archived to {entry.model_path}")

            # Track the run's REALIZED peak ELO (seeded from -inf, so always the run's own
            # trajectory -- not the 1000-floored archive mark). best.pt follows it under --save_best,
            # so a run whose ELO never climbs above the seed default still keeps its strongest
            # weights instead of leaving best.pt at the floor. (Caveat: across curriculum stages
            # ELO isn't strictly comparable; this keeps the max-ELO checkpoint, which is the
            # give-back-recovery the feature was built for.)
            if active.elo > run_peak_elo:
                run_peak_elo = active.elo
                if args.save_best:
                    active.save(best_path, verbose=False)
                    best_saved = True
                    print(f"  Best model saved: {best_path} (peak ELO {run_peak_elo:.0f})")

            # Early-stop patience uses a SEPARATE per-stage baseline so a curriculum advance
            # (harder pool -> legitimately lower ELO) isn't mistaken for "no progress". On a
            # stage advance: rebase the baseline to the new stage and don't penalize this chunk.
            if league.curriculum_stage > stage_before:
                stage_best_elo = active.elo
                chunks_since_best = 0
            elif active.elo > stage_best_elo:
                stage_best_elo = active.elo
                chunks_since_best = 0
            else:
                chunks_since_best += 1

            # Collapse detection: restore from archive if draws dominate for 2 consecutive chunks
            dr = draws / total_games if total_games else 0.0
            if dr > 0.40 and wr < 0.30:
                consecutive_collapse_chunks += 1
            else:
                consecutive_collapse_chunks = 0

            if consecutive_collapse_chunks >= 2:
                if league.archive:
                    restore_entry = league.archive[-1]
                    print(f"\n  *** COLLAPSE DETECTED ***")
                    print(f"  Draw rate {100*dr:.0f}% and WR {100*wr:.0f}% for 2 consecutive chunks.")
                    print(f"  Restoring weights from: {restore_entry.model_path} (ELO {restore_entry.elo:.0f})")
                    active.load(restore_entry.model_path)
                    active.model.train()
                    active.elo = restore_entry.elo
                    consecutive_collapse_chunks = 0
                    # A restore is a fresh baseline (like a stage advance) -- don't let the
                    # pre-restore low ELO keep counting against --patience and early-stop a
                    # run that just recovered.
                    stage_best_elo = active.elo
                    chunks_since_best = 0
                else:
                    print(f"\n  *** COLLAPSE DETECTED (draw rate {100*dr:.0f}%, WR {100*wr:.0f}%) -- no archive to restore from ***")

            # Early stop on plateau (opt-in via --patience > 0). The counter is drift-aware
            # (per-stage baseline rebased on stage advance, above), so this catches intra-stage
            # forgetting -- the 1728->1550 give-back -- not curriculum hardening.
            if args.patience > 0 and chunks_since_best >= args.patience:
                print(f"\n  *** EARLY STOP ***")
                print(f"  No new ELO peak for {chunks_since_best} chunks "
                      f"(stage peak {stage_best_elo:.0f}, global peak {best_elo:.0f}, patience {args.patience}).")
                if args.restore_best:
                    # Only restore a best.pt THIS run produced -- never a stale/foreign leftover
                    # (loading a small-net best.pt into a medium-net run crashed on shape mismatch).
                    # Restore the ELO that matches those weights (run_peak_elo), not the archive
                    # high-water mark. Guard the load so a restore hiccup can't kill a long run.
                    if best_saved and os.path.isfile(best_path):
                        try:
                            active.load(best_path)
                            active.model.train()
                            active.elo = run_peak_elo if run_peak_elo > float("-inf") else best_elo
                            print(f"  Restored peak weights from {best_path} (ELO {active.elo:.0f}).")
                        except RuntimeError as e:
                            print(f"  --restore_best: could not load {best_path} ({e}); "
                                  f"keeping current weights.")
                    elif not best_saved:
                        print(f"  --restore_best set but this run never wrote best.pt "
                              f"(needs --save_best) -- keeping current weights.")
                    else:
                        print(f"  --restore_best set but {best_path} not found "
                              f"-- keeping current weights.")
                # The break below skips the per-chunk version save, so persist the final state
                # (restored-peak with --restore_best, else the plateau weights) once, here.
                final_path = next_version(args.model_dir)
                active.save(final_path)
                print(f"  Final checkpoint saved: {final_path}")
                chunks_done = chunk_idx
                break

            # Always save a versioned checkpoint per chunk
            version_path = next_version(args.model_dir)
            active.save(version_path)
            print(f"  Checkpoint saved: {version_path}")
            prune_versions(args.model_dir, keep=args.keep_versions)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    print_summary(start_dt, datetime.now(), chunks_done, args.chunks, active, best_elo, league,
                  run_peak_elo=run_peak_elo)


if __name__ == "__main__":
    main()
