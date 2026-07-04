"""AlphaZero-style self-play training for Ultimate Tic-Tac-Toe.

Core loop per iteration:
  1. Self-play: play `--games_per_iter` games using MCTS with Dirichlet noise.
     Each move records (board_tensor, visit_pi, outcome_z).
     A `--opp_mix` slice of games is played vs diverse opponents (past-self
     pool / win-block bot / random) instead of the net's own twin -- the
     league-training insight; pure twin self-play is draw-heavy and narrow.
     With `--tactics` (default on), provable ultimate win-in-1 moves become
     sharp one-hot targets (no search) and moves that hand the opponent an
     immediate game win are zeroed out of targets (engine/tactics.py).
  2. Add examples to a fixed-size replay buffer.
  3. Train: sample random minibatches, optimize
       policy_loss (cross-entropy vs visit pi) + value_coef * value_loss (MSE vs z).
  4. Checkpoint + log metrics (+ wall-clock-paced gauntlet eval, see Gauntlet).

Background (2026-07-04): the first long M4 run plateaued -- diagnostics showed
the net was fitting its MCTS visit targets near-optimally, but the targets
themselves were close to uniform (weak value head -> unfocused search -> soft
targets -> uniform policy, a self-sealing loop). The tactics injection, the
diverse-opponent slice, dir_eps 0.25->0.15, and temperature_moves 20->10 all
exist to break that loop. models/alphazero_m4_flat holds the plateaued run.

Policy target: MCTS visit count distribution (not the sampled action).
Value target: game outcome from that position's player perspective
              (+1 win, -1 loss, 0 draw).

Relationship to train_league.py:
  train_league uses policy-gradient actor-critic from *sampled* game rollouts.
  train_alphazero uses MCTS-improved *planned* policy targets. The two are
  complementary: league-training produces the initial policy; AZ refines it with
  search. Start from a league checkpoint via --checkpoint.

value_tanh recommendation:
  Use --value_tanh. Without tanh the value head is unbounded (trained on shaped
  returns), which means terminal values (+/-1) and leaf values use different scales
  -- the known MCTS quality caveat in agents/mcts.py. --value_tanh fixes this for
  new runs. Start from a *fresh* model or a checkpoint trained with --value_tanh;
  loading an untanhed checkpoint and immediately enabling --value_tanh produces
  garbage (the value head output range changes). A warning is printed if omitted.

Examples:
  # Fresh AZ run, small net, verify it runs before long commitment:
  python -m scripts.train_alphazero --network small --iters 3 --games_per_iter 10

  # Full fresh run (M4b config):
  python -m scripts.train_alphazero \\
    --network medium --n_sims 300 --wave_size 64 \\
    --games_per_iter 50 --iters 0

  # CAUTION: --checkpoint with an UNTANHED checkpoint (e.g. league best.pt)
  # under the value_tanh default produces garbage -- pass --no-value_tanh
  # for legacy checkpoints, or start fresh.
"""

import argparse
import collections
import json
import os
import random
import re
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agents.agent_base import ModelConfigCNN, board_to_tensor_from_gamestate
from agents.deterministics import WinBlockAgent
from agents.random_agent import RandomAgent
from agents.neural_net_agent_pg import NeuralNetAgentPG
from agents.mcts import MCTS, MCTSAgent
from engine.game import GameState
from engine.constants import X, O, DRAW
from engine.rules import rule_utl_valid_moves
from engine.tactics import winning_moves, losing_moves
from scripts.train_league import NETWORK_CONFIGS, prune_versions, DEFAULT_ELO
from scripts.trainer_base import append_metrics, clear_metrics_log

# A single training example produced by self-play.
# x     : (7, 9, 9) float32 board tensor (cpu)
# pi    : (81,) float32 visit distribution (sums to 1 over legal moves)
# z     : float -- game outcome from that position's player perspective (+1/-1/0)
Example = Tuple[torch.Tensor, np.ndarray, float]


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self._buf: collections.deque = collections.deque(maxlen=maxlen)

    def extend(self, examples: List[Example]):
        self._buf.extend(examples)

    def sample(self, n: int) -> List[Example]:
        return random.sample(self._buf, min(n, len(self._buf)))

    def __len__(self):
        return len(self._buf)


def _tactical_target(state, valid):
    """If an immediate ULTIMATE win exists, return (one-hot-ish pi, move to play).

    Ground truth from engine/tactics.py -- no search needed, and the sharp
    target injects provable signal into otherwise soft MCTS visit targets.
    Mini-board wins are deliberately NOT forced (they can be bad in UTTT --
    see the tactics module docstring); those must be learned from outcomes.
    """
    wins = winning_moves(state, valid)
    if not wins:
        return None, None
    pi = np.zeros(81, dtype=np.float32)
    for m in wins:
        pi[m] = 1.0 / len(wins)
    return pi, random.choice(wins)


def _filter_losing(pi, state, valid):
    """Zero out moves that hand the opponent an immediate game win; renormalize.

    Provably-bad moves (depth-2 ultimate loss) should carry zero target mass.
    If EVERY legal move loses, the position is lost -- leave pi untouched.
    """
    losers = losing_moves(state, valid)
    if not losers or len(losers) >= len(valid):
        return pi
    pi = pi.copy()
    for m in losers:
        pi[m] = 0.0
    s = pi.sum()
    if s > 1e-8:
        return pi / s
    safe = [m for m in valid if m not in set(losers)]
    pi = np.zeros(81, dtype=np.float32)
    for m in safe:
        pi[m] = 1.0 / len(safe)
    return pi


@torch.no_grad()
def collect_game(model, device, n_sims: int, c_puct: float,
                 dir_alpha: float, dir_eps: float,
                 wave_size: int, temperature_moves: int,
                 use_tactics: bool = True,
                 opponent_fn=None) -> Tuple[List[Example], int, dict]:
    """Play one training game; returns (examples, winner, stats).

    stats: tac_wins (win-in-1 shortcuts taken), tac_dodges (positions where
    the losing-move filter actually changed pi), moves (total game plies).

    Pure self-play when opponent_fn is None (both sides use MCTS, both sides'
    positions are recorded). With opponent_fn, the net takes a random color,
    the opponent plays its own policy, and ONLY the net's positions are
    recorded (opponent moves have no MCTS target behind them). Diverse
    opponents widen the position/outcome distribution the way the league did
    for PG training -- twin self-play alone is draw-heavy and narrow.
    """
    mcts = MCTS(model, device, n_sims=n_sims, c_puct=c_puct,
                add_dirichlet_at_root=True, dir_alpha=dir_alpha, dir_eps=dir_eps,
                wave_size=wave_size)

    net_side = None
    if opponent_fn is not None:
        net_side = X if random.random() < 0.5 else O

    state = GameState()
    trajectory = []  # (tensor, pi, player)
    move_num = 0
    tac_wins = 0
    tac_dodges = 0

    while not state.is_over():
        if net_side is not None and state.player != net_side:
            state.make_move(opponent_fn(state))
            move_num += 1
            continue

        valid = rule_utl_valid_moves(state.board, state.last_move, state.mini_winners)

        if use_tactics:
            forced_pi, forced_move = _tactical_target(state, valid)
            if forced_pi is not None:
                tac_wins += 1
                x = board_to_tensor_from_gamestate(state, v_computed=valid).cpu()
                trajectory.append((x, forced_pi, state.player))
                state.make_move(forced_move)
                move_num += 1
                continue

        pi, _ = mcts.search(state)
        if use_tactics:
            filtered = _filter_losing(pi, state, valid)
            if filtered is not pi:  # _filter_losing returns pi unchanged when nothing to zero
                tac_dodges += 1
            pi = filtered

        x = board_to_tensor_from_gamestate(state, v_computed=valid).cpu()
        trajectory.append((x, pi.copy(), state.player))

        if move_num < temperature_moves:
            # Proportional sampling with temperature=1 for early moves (exploration).
            pi_sum = pi.sum()
            if pi_sum > 0:
                probs = pi / pi_sum
            else:
                probs = np.ones(81) / 81
            move = int(np.random.choice(81, p=probs))
        else:
            move = int(np.argmax(pi))

        state.make_move(move)
        move_num += 1

    # Assign outcome z from each position's player perspective.
    winner = state.winner
    examples: List[Example] = []
    for x, pi, player in trajectory:
        if winner == DRAW:
            z = 0.0
        elif winner == player:
            z = 1.0
        else:
            z = -1.0
        examples.append((x, pi, z))

    stats = {"tac_wins": tac_wins, "tac_dodges": tac_dodges, "moves": move_num}
    return examples, winner, stats


def train_on_examples(model, optimizer, examples: List[Example],
                      value_coef: float, device: str) -> Tuple[float, float, float]:
    """Run one gradient step on a batch. Returns (total_loss, policy_loss, value_loss)."""
    model.train()

    xs = torch.stack([e[0] for e in examples]).to(device)       # (B, 7, 9, 9)
    pis = torch.tensor(np.stack([e[1] for e in examples]),
                       dtype=torch.float32, device=device)       # (B, 81)
    zs = torch.tensor([e[2] for e in examples],
                      dtype=torch.float32, device=device)        # (B,)

    policy_logits, values = model.forward_both(xs)               # (B,81), (B,)
    if policy_logits.dim() == 1:
        policy_logits = policy_logits.unsqueeze(0)
        values = values.unsqueeze(0)

    # Policy: cross-entropy between visit distribution and log_softmax of logits.
    # Using KL-divergence form: -sum(pi * log_softmax) = CE(pi, logits).
    log_probs = F.log_softmax(policy_logits, dim=-1)        # (B, 81)
    policy_loss = -(pis * log_probs).sum(dim=-1).mean()     # scalar

    # Value: MSE between predicted value and outcome z.
    value_loss = F.mse_loss(values, zs)

    loss = policy_loss + value_coef * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()


def _eval_winrate(agent: NeuralNetAgentPG, n_games: int = 50) -> float:
    """Quick score vs random (draws = 0.5) to track learning progress."""
    rand = RandomAgent()
    wins = 0.0
    for g in range(n_games):
        state = GameState()
        agent_side = X if g % 2 == 0 else O
        agent.set_eval(True)
        while not state.is_over():
            mover = agent if state.player == agent_side else rand
            move = mover.select_move(state)
            state.make_move(move)
        if state.winner == agent_side:
            wins += 1.0
        elif state.winner == DRAW:
            wins += 0.5
    agent.set_eval(False)
    return wins / n_games


class OpponentPool:
    """Diverse opponents for a slice of self-play games (the league insight:
    a population of different enemies beats training against your own twin).

    Mix: past-self snapshots (raw policy + ultimate win/block filter),
    WinBlockAgent (punishes hanging mini-boards), RandomAgent (keeps the
    distribution wide). Snapshots are in-memory cpu state_dicts, refreshed
    every `pool_every` iterations, capped at `cap` (oldest kept as an anchor,
    second-oldest evicted first to preserve a strength spread).
    """

    def __init__(self, cfg, device, cap: int, sample_moves: int = 6):
        self.device = device
        self.cap = cap
        self.sample_moves = sample_moves
        self.snaps = []  # list of (iteration, cpu state_dict)
        self.heur = WinBlockAgent()
        self.rand = RandomAgent()
        self.shell = NeuralNetAgentPG(cfg=cfg, model_path=None)
        self.shell.model.to(device)
        self.shell.device = device
        self.shell.model.eval()

    def maybe_add(self, model, iteration: int, every: int):
        if every <= 0 or iteration % every != 0:
            return
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.snaps.append((iteration, sd))
        if len(self.snaps) > self.cap:
            self.snaps.pop(1)

    def _net_move(self, state, move_num: int) -> int:
        valid = rule_utl_valid_moves(state.board, state.last_move, state.mini_winners)
        wins = winning_moves(state, valid)
        if wins:
            return random.choice(wins)
        losers = losing_moves(state, valid)
        if losers and len(losers) < len(valid):
            valid = [m for m in valid if m not in set(losers)]
        return _policy_move_from_valid(self.shell.model, self.device, state, valid,
                                       self.sample_moves, move_num)

    def sample_opponent_fn(self):
        """Returns (label, fn(state) -> move) for one game."""
        r = random.random()
        if self.snaps and r < 0.5:
            it, sd = random.choice(self.snaps)
            self.shell.model.load_state_dict(sd)
            counter = {"n": 0}
            def fn(state):
                mv = self._net_move(state, counter["n"])
                counter["n"] += 1
                return mv
            return f"pool_i{it}", fn
        if r < 0.8:
            return "winblock", lambda s: self.heur.select_move(s)
        return "random", lambda s: self.rand.select_move(s)


@torch.no_grad()
def _policy_move_from_valid(model, device, state, valid, sample_moves: int,
                            move_num: int) -> int:
    """Raw-net move restricted to `valid`: masked policy, sampled early.

    First `sample_moves` plies are sampled from the softmax (so repeated games
    between deterministic nets differ); after that, argmax.
    """
    x = board_to_tensor_from_gamestate(state, v_computed=None).unsqueeze(0).to(device)
    logits, _ = model.forward_both(x)
    logits = logits.reshape(-1)
    mask = torch.full((81,), float("-inf"), device=logits.device)
    for m in valid:
        mask[m] = 0.0
    logits = logits + mask
    if move_num < sample_moves:
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())
    return int(torch.argmax(logits).item())


def _policy_move(model, device, state, sample_moves: int, move_num: int) -> int:
    """Raw-net move over all legal moves (gauntlet protocol)."""
    valid = rule_utl_valid_moves(state.board, state.last_move, state.mini_winners)
    return _policy_move_from_valid(model, device, state, valid, sample_moves, move_num)


def _play_match(move_fn_a, move_fn_b, n_games: int) -> float:
    """Score of A over `n_games` with alternating colors. Draws count 0.5.

    move_fn(state, move_num) -> int. Returns score in [0, 1]; 0.5 = parity.
    """
    score = 0.0
    for g in range(n_games):
        a_side = X if g % 2 == 0 else O
        state = GameState()
        move_num = 0
        while not state.is_over():
            fn = move_fn_a if state.player == a_side else move_fn_b
            state.make_move(fn(state, move_num))
            move_num += 1
        if state.winner == a_side:
            score += 1.0
        elif state.winner == DRAW:
            score += 0.5
    return score / n_games


class Gauntlet:
    """Wall-clock-paced eval vs stationary + historical opponents.

    Every `every_min` minutes (checked between iterations) the current raw net
    plays quick matches against:
      * anchor    -- the run's day-one weights (saved once, kept across resume)
      * past self -- a rolling snapshot refreshed every `lookback_min` minutes
      * win/block -- WinBlockAgent (mini-board win/block, else random): a
                     stationary 'attentive beginner' yardstick
    and, every `mcts_probe_every`-th gauntlet, an MCTS(probe_sims) wrapper over
    the CURRENT net plays the raw net ('mcts_edge'). Search edge near 1.0 means
    the raw policy still leaves most of the search gain on the table; drifting
    toward 0.5 means the policy has internalized what search finds.

    All scores count draws as 0.5. Raw-net games sample the first
    `sample_moves` plies for variety, argmax after.
    """

    def __init__(self, agent, cfg, device, model_dir, every_min, games,
                 lookback_min, sample_moves, mcts_probe_every, mcts_probe_games,
                 mcts_probe_sims, wave_size, start_iteration):
        self.agent = agent
        self.device = device
        self.every_min = every_min
        self.games = games
        self.lookback_min = lookback_min
        self.sample_moves = sample_moves
        self.mcts_probe_every = mcts_probe_every
        self.mcts_probe_games = mcts_probe_games
        self.mcts_probe_sims = mcts_probe_sims
        self.wave_size = wave_size

        self.anchor_path = os.path.join(model_dir, "gauntlet_anchor.pt")
        self.past_path = os.path.join(model_dir, "gauntlet_past.pt")
        self.heur = WinBlockAgent()
        self.last_run_t = 0.0     # 0 -> first gauntlet fires after the first iteration
        self.n_gauntlets = 0

        # Reusable opponent shell: weights swapped via load_state_dict (no deepcopy).
        self.opp = NeuralNetAgentPG(cfg=cfg, model_path=None)
        self.opp.model.to(device)
        self.opp.device = device
        self.opp.model.eval()

        if os.path.isfile(self.anchor_path):
            payload = torch.load(self.anchor_path, map_location=device)
            self._anchor_sd = payload["state_dict"]
        else:
            self._anchor_sd = {k: v.detach().clone()
                               for k, v in agent.model.state_dict().items()}
            torch.save({"state_dict": self._anchor_sd, "iter": start_iteration},
                       self.anchor_path)
            print(f"[gauntlet] day-one anchor saved (iter {start_iteration})")

        self._past_sd = None
        self._past_iter = None
        self._past_t = None
        if os.path.isfile(self.past_path):
            payload = torch.load(self.past_path, map_location=device)
            self._past_sd = payload["state_dict"]
            self._past_iter = payload.get("iter")
            self._past_t = payload.get("t", time.time())

    def _net_fn(self, model):
        return lambda s, mn: _policy_move(model, self.device, s, self.sample_moves, mn)

    def _refresh_past(self, iteration):
        self._past_sd = {k: v.detach().clone()
                         for k, v in self.agent.model.state_dict().items()}
        self._past_iter = iteration
        self._past_t = time.time()
        torch.save({"state_dict": self._past_sd, "iter": iteration, "t": self._past_t},
                   self.past_path)

    @torch.no_grad()
    def maybe_run(self, iteration):
        """Returns a dict of gauntlet metrics, or None if not due yet."""
        if self.every_min <= 0:
            return None
        if time.time() - self.last_run_t < self.every_min * 60:
            return None
        t0 = time.perf_counter()
        res = {}
        cur_fn = self._net_fn(self.agent.model)

        # 1. stationary heuristic yardstick
        res["wr_heur"] = round(_play_match(
            cur_fn, lambda s, mn: self.heur.select_move(s), self.games), 4)

        # 2. day-one anchor
        self.opp.model.load_state_dict(self._anchor_sd)
        res["wr_anchor"] = round(_play_match(
            cur_fn, self._net_fn(self.opp.model), self.games), 4)

        # 3. rolling past self
        if self._past_sd is not None:
            self.opp.model.load_state_dict(self._past_sd)
            res["wr_past"] = round(_play_match(
                cur_fn, self._net_fn(self.opp.model), self.games), 4)
            res["past_iter"] = self._past_iter
        if self._past_sd is None or time.time() - self._past_t >= self.lookback_min * 60:
            self._refresh_past(iteration)

        # 4. search-edge probe (MCTS over current net vs raw current net)
        if self.mcts_probe_every > 0 and self.n_gauntlets % self.mcts_probe_every == 0:
            probe = MCTSAgent(self.agent, n_sims=self.mcts_probe_sims,
                              temperature=0.0, wave_size=self.wave_size)
            res["mcts_edge"] = round(_play_match(
                lambda s, mn: probe.select_move(s), cur_fn,
                self.mcts_probe_games), 4)

        self.n_gauntlets += 1
        res["gauntlet_secs"] = round(time.perf_counter() - t0, 1)
        self.last_run_t = time.time()
        return res


def _find_latest_version(model_dir):
    """Returns (path, index) of the newest version_NNN.pt, or (None, -1)."""
    best_idx, best_path = -1, None
    if os.path.isdir(model_dir):
        for name in os.listdir(model_dir):
            m = re.fullmatch(r"version_(\d+)\.pt", name)
            if m and int(m.group(1)) > best_idx:
                best_idx = int(m.group(1))
                best_path = os.path.join(model_dir, name)
    return best_path, best_idx


def main():
    ap = argparse.ArgumentParser(description="AlphaZero-style self-play training.")
    ap.add_argument("--checkpoint", type=str, default="",
                    help="Starting weights (.pt). Leave empty for fresh random init. "
                         "Load a league checkpoint to jump-start policy quality.")
    ap.add_argument("--network", type=str, default="small", choices=list(NETWORK_CONFIGS),
                    help="Network architecture (small/medium/large).")
    ap.add_argument("--value_tanh", action=argparse.BooleanOptionalAction, default=True,
                    help="Apply tanh to value head. DEFAULT ON (2026-07-04). "
                         "Calibrates output to [-1,1], fixing the MCTS value-scale mismatch. "
                         "Start from a fresh model or a --value_tanh checkpoint; pass "
                         "--no-value_tanh only for legacy untanhed checkpoints.")
    ap.add_argument("--iters", type=int, default=0,
                    help="Training iterations (0 = run forever until Ctrl+C).")
    ap.add_argument("--games_per_iter", type=int, default=50,
                    help="Self-play games per iteration.")
    ap.add_argument("--n_sims", type=int, default=100,
                    help="MCTS simulations per move during self-play.")
    ap.add_argument("--wave_size", type=int, default=64,
                    help="Batched leaf-eval wave size. DEFAULT 64 (gated 2026-07-02: ~20x "
                         "self-play throughput vs 1 on RTX 3080, monotonic 1<4<8<16<32<64 -- "
                         "RESULT_PERF_BENCH.md). 1 = serial leaf eval.")
    ap.add_argument("--c_puct", type=float, default=1.5)
    ap.add_argument("--dir_alpha", type=float, default=0.3,
                    help="Dirichlet alpha for root noise (~0.3 for UTTT).")
    ap.add_argument("--dir_eps", type=float, default=0.15,
                    help="Dirichlet mixing weight. DEFAULT 0.15 (2026-07-04, was 0.25: "
                         "the M4 plateau diagnostic showed near-uniform visit targets; "
                         "less root noise keeps targets sharper).")
    ap.add_argument("--temperature_moves", type=int, default=10,
                    help="Number of moves to sample proportionally (exploration); "
                         "after this, use argmax. DEFAULT 10 (2026-07-04, was 20: "
                         "UTTT games run ~40-60 plies; 20 sampled plies made "
                         "self-play too random).")
    ap.add_argument("--tactics", action=argparse.BooleanOptionalAction, default=True,
                    help="Inject provable ultimate win-in-1 targets and zero out "
                         "hand-opponent-the-game moves in training targets "
                         "(engine/tactics.py ground truth). DEFAULT ON.")
    ap.add_argument("--opp_mix", type=float, default=0.30,
                    help="Fraction of games per iteration played vs a diverse opponent "
                         "(past-self pool / win-block bot / random) instead of pure "
                         "self-play. Only the net's own positions are recorded. "
                         "0 = pure self-play. DEFAULT 0.30 (2026-07-04).")
    ap.add_argument("--pool_every", type=int, default=25,
                    help="Snapshot current weights into the opponent pool every N iters.")
    ap.add_argument("--pool_cap", type=int, default=10,
                    help="Max snapshots in the opponent pool (oldest kept as anchor).")
    ap.add_argument("--buffer_size", type=int, default=100_000,
                    help="Maximum replay buffer size (examples, not games).")
    ap.add_argument("--train_steps", type=int, default=100,
                    help="Gradient steps per iteration.")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="SGD batch size drawn from the replay buffer.")
    ap.add_argument("--value_coef", type=float, default=1.0,
                    help="Weight on value loss. 1.0 = equal weight (AlphaZero default).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model_dir", type=str, default="models/alphazero")
    ap.add_argument("--keep_versions", type=int, default=5,
                    help="Number of version_NNN.pt files to keep.")
    ap.add_argument("--save_best", action=argparse.BooleanOptionalAction, default=True,
                    help="Save models/alphazero/best.pt whenever eval winrate improves. DEFAULT ON. Pass --no-save_best to disable.")
    ap.add_argument("--eval_games", type=int, default=40,
                    help="Games vs random per eval (0 = skip eval).")
    ap.add_argument("--eval_every", type=int, default=5,
                    help="Evaluate every N iterations.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from --model_dir: newest version_NNN.pt weights, "
                         "run_state.json counters, resume.pt optimizer+buffer if present. "
                         "Does NOT clear the metrics log. Mutually exclusive with --checkpoint.")
    ap.add_argument("--gauntlet_every_min", type=float, default=5.0,
                    help="Wall-clock minutes between gauntlet evals (0 = off). Checked "
                         "between iterations, so effective cadence is max(this, iter time).")
    ap.add_argument("--gauntlet_games", type=int, default=20,
                    help="Games per gauntlet opponent (raw-net matches, seconds each).")
    ap.add_argument("--gauntlet_lookback_min", type=float, default=30.0,
                    help="Age of the rolling 'past self' snapshot in minutes.")
    ap.add_argument("--gauntlet_sample_moves", type=int, default=6,
                    help="Plies sampled from the policy at the start of each gauntlet "
                         "game (variety between deterministic nets); argmax after.")
    ap.add_argument("--mcts_probe_every", type=int, default=4,
                    help="Run the MCTS-edge probe every Nth gauntlet (0 = never). "
                         "Costlier than the raw-net matches.")
    ap.add_argument("--mcts_probe_games", type=int, default=10)
    ap.add_argument("--mcts_probe_sims", type=int, default=64)
    ap.add_argument("--device", type=str, default="",
                    help="'cpu' / 'cuda' (default: auto).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed torch/numpy/random (weight init, self-play sampling).")
    ap.add_argument("--no_metrics", action="store_true",
                    help="Skip writing to loss_logs/metrics_log.jsonl.")
    args = ap.parse_args()

    if not args.value_tanh:
        print("[!] --value_tanh not set. Value head is unbounded (trained on shaped returns).")
        print("    MCTS will mix that scale with clean +/-1 terminal values.")
        print("    For best AlphaZero quality, use --value_tanh on a fresh model.")

    if args.seed is not None:
        import random as _r
        _r.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seeded RNGs: {args.seed}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.model_dir, exist_ok=True)

    net = NETWORK_CONFIGS[args.network]
    cfg = ModelConfigCNN(
        **net,
        learning_rate=args.lr,
        label="alphazero",
        model_dir=args.model_dir,
        value_tanh=args.value_tanh,
    )
    if args.resume and args.checkpoint:
        ap.error("--resume and --checkpoint are mutually exclusive")

    agent = NeuralNetAgentPG(cfg=cfg, model_path=None)
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            ap.error(f"--checkpoint not found: {args.checkpoint}")
        agent.seed_from_checkpoint(args.checkpoint)
        print(f"Seeded weights from {args.checkpoint}")

    agent.model.to(device)
    agent.device = device
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=args.lr)

    buffer = ReplayBuffer(args.buffer_size)
    version_idx = 0
    best_wr = 0.0
    start_iteration = 0
    games_total = 0

    run_state_path = os.path.join(args.model_dir, "run_state.json")
    resume_path = os.path.join(args.model_dir, "resume.pt")

    if args.resume:
        ver_path, ver_idx = _find_latest_version(args.model_dir)
        if ver_path is None:
            ap.error(f"--resume: no version_NNN.pt found in {args.model_dir}")
        agent.seed_from_checkpoint(ver_path)
        print(f"Resumed weights from {ver_path}")
        if os.path.isfile(run_state_path):
            with open(run_state_path) as f:
                rs = json.load(f)
            start_iteration = rs["iteration"]
            version_idx = rs["version_idx"]
            best_wr = rs.get("best_wr", 0.0)
            games_total = rs.get("games_total", 0)
            print(f"Resumed counters: iter={start_iteration} version={version_idx} "
                  f"best_wr={best_wr:.3f} games={games_total}")
        else:
            # Legacy run (no run_state.json): estimate from the version index.
            start_iteration = ver_idx + 1
            version_idx = ver_idx + 1
            games_total = start_iteration * args.games_per_iter
            print(f"[!] no run_state.json -- estimated iter={start_iteration}, "
                  f"games={games_total} from {os.path.basename(ver_path)}")
        if os.path.isfile(resume_path):
            # weights_only=False: our own file, contains numpy arrays (buffer pis)
            # + optimizer state, which the torch>=2.6 safe loader rejects.
            # map_location cpu: buffer examples live on cpu (train_on_examples
            # stacks them with fresh cpu examples); optimizer.load_state_dict
            # re-homes its state to the params' device on its own.
            payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            try:
                optimizer.load_state_dict(payload["optimizer"])
                print("Resumed optimizer state")
            except (ValueError, KeyError) as e:
                print(f"[!] optimizer state incompatible, starting fresh: {e}")
            if "buffer_x" in payload:
                xs = payload["buffer_x"]
                pis = payload["buffer_pi"]
                zs = payload["buffer_z"]
                buffer.extend([(xs[i], pis[i], float(zs[i]))
                               for i in range(xs.shape[0])])
                print(f"Resumed replay buffer: {len(buffer)} examples")
        else:
            print("[!] no resume.pt -- replay buffer starts empty (refills in a few iters)")

    if not args.no_metrics and not args.resume:
        clear_metrics_log()

    pool = OpponentPool(cfg=cfg, device=device, cap=args.pool_cap,
                        sample_moves=args.gauntlet_sample_moves)

    gauntlet = Gauntlet(
        agent=agent, cfg=cfg, device=device, model_dir=args.model_dir,
        every_min=args.gauntlet_every_min, games=args.gauntlet_games,
        lookback_min=args.gauntlet_lookback_min,
        sample_moves=args.gauntlet_sample_moves,
        mcts_probe_every=args.mcts_probe_every,
        mcts_probe_games=args.mcts_probe_games,
        mcts_probe_sims=args.mcts_probe_sims,
        wave_size=args.wave_size,
        start_iteration=start_iteration,
    )

    print(f"AlphaZero training | net={args.network} | device={device} | "
          f"n_sims={args.n_sims} wave={args.wave_size} | value_tanh={args.value_tanh}")
    print(f"  games_per_iter={args.games_per_iter} | train_steps={args.train_steps} | "
          f"batch_size={args.batch_size} | buffer_size={args.buffer_size}")
    print(f"  tactics={'on' if args.tactics else 'off'} | opp_mix={args.opp_mix:g} "
          f"(pool: every {args.pool_every} iters, cap {args.pool_cap}) | "
          f"dir_eps={args.dir_eps:g} | temp_moves={args.temperature_moves}")
    if args.gauntlet_every_min > 0:
        print(f"  gauntlet: every {args.gauntlet_every_min:g} min | "
              f"{args.gauntlet_games} games/opponent | "
              f"past-self lookback {args.gauntlet_lookback_min:g} min | "
              f"MCTS probe every {args.mcts_probe_every} gauntlets")

    def _save_resume_state():
        """Persist optimizer + replay buffer for --resume (called on exit)."""
        payload = {"optimizer": optimizer.state_dict()}
        if len(buffer) > 0:
            examples = list(buffer._buf)
            payload["buffer_x"] = torch.stack([e[0] for e in examples])
            payload["buffer_pi"] = np.stack([e[1] for e in examples])
            payload["buffer_z"] = np.array([e[2] for e in examples], dtype=np.float32)
        torch.save(payload, resume_path)
        print(f"Saved resume state ({len(buffer)} buffered examples) -> {resume_path}")

    iteration = start_iteration
    try:
        while args.iters == 0 or iteration < args.iters:
            t0 = time.perf_counter()

            # --- Self-play (with a diverse-opponent slice) ---
            agent.model.eval()
            new_examples: List[Example] = []
            sp_draws = 0
            tac_wins = tac_dodges = moves_total = 0
            for g in range(args.games_per_iter):
                opponent_fn = None
                if args.opp_mix > 0 and random.random() < args.opp_mix:
                    _, opponent_fn = pool.sample_opponent_fn()
                exs, winner, gstats = collect_game(
                    model=agent.model,
                    device=device,
                    n_sims=args.n_sims,
                    c_puct=args.c_puct,
                    dir_alpha=args.dir_alpha,
                    dir_eps=args.dir_eps,
                    wave_size=args.wave_size,
                    temperature_moves=args.temperature_moves,
                    use_tactics=args.tactics,
                    opponent_fn=opponent_fn,
                )
                new_examples.extend(exs)
                tac_wins += gstats["tac_wins"]
                tac_dodges += gstats["tac_dodges"]
                moves_total += gstats["moves"]
                if winner == DRAW:
                    sp_draws += 1

            # Mean entropy of this iteration's policy targets: the M4a plateau's
            # smoking gun was near-uniform targets (max = ln(81) ~ 4.39 nats).
            # Falling entropy = sharpening targets = the search is decisive.
            pi_ent = float('nan')
            if new_examples:
                pis = np.stack([e[1] for e in new_examples])
                pi_ent = float(-(pis * np.log(pis + 1e-12)).sum(axis=1).mean())

            buffer.extend(new_examples)
            pool.maybe_add(agent.model, iteration, args.pool_every)

            # --- Train ---
            total_loss = policy_loss_sum = value_loss_sum = 0.0
            steps_done = 0
            for _ in range(args.train_steps):
                if len(buffer) < args.batch_size:
                    break
                batch = buffer.sample(args.batch_size)
                tl, pl, vl = train_on_examples(
                    agent.model, optimizer, batch,
                    value_coef=args.value_coef, device=device,
                )
                total_loss += tl
                policy_loss_sum += pl
                value_loss_sum += vl
                steps_done += 1
            agent.model.eval()

            avg_loss   = total_loss / max(steps_done, 1)
            avg_pol    = policy_loss_sum / max(steps_done, 1)
            avg_val    = value_loss_sum / max(steps_done, 1)
            elapsed    = time.perf_counter() - t0

            # --- Checkpoint ---
            version_path = os.path.join(args.model_dir, f"version_{version_idx:03d}.pt")
            agent.save(version_path)
            prune_versions(args.model_dir, args.keep_versions)
            version_idx += 1

            # --- Eval ---
            wr = float('nan')
            if args.eval_games > 0 and (iteration % args.eval_every == 0):
                wr = _eval_winrate(agent, args.eval_games)
                if args.save_best and wr > best_wr:
                    best_wr = wr
                    best_path = os.path.join(args.model_dir, "best.pt")
                    agent.save(best_path)

            # --- Gauntlet (wall-clock paced) ---
            gres = gauntlet.maybe_run(iteration)

            # --- Metrics ---
            games_total += args.games_per_iter
            gpi = args.games_per_iter
            extra = {
                "sp_draws": round(sp_draws / gpi, 3),
                "tac_w": round(tac_wins / gpi, 2),
                "tac_d": round(tac_dodges / gpi, 2),
                "pi_ent": round(pi_ent, 3) if np.isfinite(pi_ent) else None,
                "avg_len": round(moves_total / gpi, 1),
            }
            if gres:
                extra.update(gres)
            if not args.no_metrics:
                append_metrics(
                    loss=avg_loss,
                    epsilon=float('nan'),
                    winrate=wr,  # NaN on non-eval iters -> null in the jsonl
                    value_loss=avg_val,
                    t=time.time(),
                    policy_loss=avg_pol,
                    games_total=games_total,
                    buffer=len(buffer),
                    extra=extra,
                )

            wr_str = f"{wr*100:.1f}%" if not (isinstance(wr, float) and np.isnan(wr)) else "--"
            print(
                f"iter {iteration+1:4d} | "
                f"buf={len(buffer):6d} | "
                f"loss={avg_loss:.4f} (pol={avg_pol:.4f} val={avg_val:.4f}) | "
                f"wr_vs_rand={wr_str} | "
                f"draws={sp_draws}/{args.games_per_iter} | "
                f"ent={pi_ent:.2f} | "
                f"tac={tac_wins}/{tac_dodges} | "
                f"{elapsed:.1f}s"
            )
            if gres:
                parts = [f"heur={gres['wr_heur']*100:.0f}%",
                         f"day1={gres['wr_anchor']*100:.0f}%"]
                if "wr_past" in gres:
                    parts.append(f"past(i{gres['past_iter']})={gres['wr_past']*100:.0f}%")
                if "mcts_edge" in gres:
                    parts.append(f"mcts_edge={gres['mcts_edge']*100:.0f}%")
                print(f"     gauntlet | {' | '.join(parts)} | {gres['gauntlet_secs']}s")

            iteration += 1

            # --- Run state (cheap, every iteration; enables --resume) ---
            with open(run_state_path, "w") as f:
                json.dump({"iteration": iteration, "version_idx": version_idx,
                           "best_wr": best_wr, "games_total": games_total}, f)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving final checkpoint...")
        agent.save(os.path.join(args.model_dir, "final.pt"))
        print("Saved.")
    finally:
        try:
            _save_resume_state()
        except Exception as e:  # never let the resume save mask the real exit
            print(f"[!] failed to save resume state: {e}")


if __name__ == "__main__":
    main()
