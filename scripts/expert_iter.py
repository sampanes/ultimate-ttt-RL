"""Expert iteration: distill a strong MCTS teacher into a student net.

WHY (2026-07-04): two AlphaZero-from-scratch runs (M4a/M4b) flatlined or
regressed. Root cause was NOT the paradigm alone -- agents/mcts.py had an
inverted virtual-loss sign plus an unclamped wave size that together made
wave-batched search WEAKER than the raw policy it wrapped, poisoning every
visit target (both fixed, see agents/mcts.py + agents/test_mcts.py). But even
with search fixed, bootstrapping from random weights at ~1 game/s is
data-starved. This script starts from PROVEN strength instead:

    teacher = MCTS(teacher_sims) over the independently selected Arena-22 HOF
    checkpoint (RESULT_M2.md), with a fixed slice of games against WinBlockAgent
    to cover the repo's measured tactical blind spot, plus a slice against
    RandomAgent so the student trains on blunder-created positions (the fixed
    random panel regressed and stalled promotion without it). An opt-in
    --greg_mix slice against a shallow GregoryAgent (depth kept strictly below
    the d3 ruler) covers the minimax-style distribution gap -- the gregory(d3)
    panel crawls at +1-2 pts/gen while winblock moves at +4 (STRENGTH_NEXT S1).

Loop, forever (Ctrl+C safe, --resume picks up where it left off):
  1. GENERATE: teacher self-play games via train_alphazero.collect_game
     (Dirichlet root noise + early-move temperature for diversity, tactical
     win-in-1 / losing-move ground truth applied to targets). Positions are
     appended to an in-RAM window AND written to disk shards for resume. In the
     WinBlock opponent slice, local mini-board win/block targets are injected to
     cover the measured blind spot that Arena-22 search still misses.
  2. TRAIN: student (fresh net, tanh value head) supervised on the window:
     CE on visit distributions + MSE on game outcomes.
  3. GATE (wall-clock paced, dashboard-friendly): student raw vs random /
     WinBlockAgent / teacher raw, plus the search invariant (MCTS over
     student must score >= 0.5 vs raw student -- below that, halt and debug).
  4. PROMOTE: only when the RAW student beats the RAW teacher on fixed,
     color-swapped openings, improves the absolute WinBlock score, preserves
     random-opponent strength, and has trained on enough current-teacher games.
     A depth-limited alpha-beta anchor (gregory, the one panel opponent
     independent of the training gene pool) is measured on every check as the
     long-horizon ruler; its no-regression gate self-arms at --gregory_arm.
     Old-generation replay is discarded on promotion and the LR decay clock
     restarts (--lr_gen_reset): each generation is a fresh distillation
     problem, so late generations are not stranded at the lr_min floor.
     This optimizes the actual inference-only artifact instead of a noisy
     search-vs-search proxy.

Metrics go to loss_logs/metrics_log.jsonl -- the AZ dashboard
(gui/alphazero/index.html) renders them unchanged: 'winrate' = student raw
vs random, wr_heur = vs win/block bot, wr_anchor = vs CURRENT TEACHER raw,
mcts_edge = search invariant, teacher_gen = promotions so far.

Run (cmd, from repo root; start_goat.bat / stop_goat.bat wrap this):
    set CUBLAS_WORKSPACE_CONFIG=:4096:8
    .venv\\Scripts\\python -m scripts.expert_iter --resume
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from agents.agent_base import ModelConfigCNN
from engine.constants import X, O, DRAW
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from agents.deterministics import WinBlockAgent
from agents.gregory import GregoryAgent
from agents.mcts import MCTS
from agents.neural_net_agent_pg import NeuralNetAgentPG
from agents.random_agent import RandomAgent
from scripts.train_alphazero import (NETWORK_CONFIGS, ReplayBuffer,
                                     _policy_move, collect_game,
                                     train_on_examples)
from scripts.trainer_base import append_metrics, clear_metrics_log, format_elapsed


# --------------------------------------------------------------------------- #
# Move-function builders (gauntlet protocol: sample first plies for variety)
# --------------------------------------------------------------------------- #

def _raw_fn(model, device, sample_moves=6):
    return lambda s, mn: _policy_move(model, device, s, sample_moves, mn)


def _search_fn(model, device, sims, sample_moves=6):
    mcts = MCTS(model, device, n_sims=sims, c_puct=1.5,
                add_dirichlet_at_root=False, wave_size=64)

    def f(state, move_num):
        pi, _ = mcts.search(state)
        if move_num < sample_moves:
            s = pi.sum()
            p = pi / s if s > 0 else np.ones(81) / 81
            return int(np.random.choice(81, p=p))
        return int(np.argmax(pi))
    return f


def _agent_fn(agent):
    return lambda s, mn: agent.select_move(s)


def _opponent_slice(r, opp_mix, rnd_mix, greg_mix=0.0):
    """Map a uniform draw r in [0, 1) to an opponent-slice tag (or None).

    Slice order is fixed [winblock | random | gregory | self-play], so
    enabling --greg_mix never reshuffles which r values land in the existing
    slices, and greg_mix=0.0 is arithmetically identical to the pre-S1 layout.
    """
    if r < opp_mix:
        return "heur"
    if r < opp_mix + rnd_mix:
        return "rnd"
    if r < opp_mix + rnd_mix + greg_mix:
        return "greg"
    return None


# --------------------------------------------------------------------------- #
# Shard store: every generation block is persisted so --resume never loses data
# --------------------------------------------------------------------------- #

class ShardStore:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _path(self, idx):
        return os.path.join(self.data_dir, f"shard_{idx:05d}.pt")

    def write(self, idx, examples, teacher_gen):
        xs = torch.stack([e[0] for e in examples])
        pis = torch.from_numpy(np.stack([e[1] for e in examples]))
        zs = torch.tensor([e[2] for e in examples], dtype=torch.float32)
        torch.save({"x": xs, "pi": pis, "z": zs, "teacher_gen": teacher_gen},
                   self._path(idx))

    def load_window(self, last_idx, max_examples, teacher_gen=None):
        """Newest-first reload into (examples, count) until max_examples."""
        examples = []
        idx = last_idx
        while idx >= 0 and len(examples) < max_examples:
            p = self._path(idx)
            if os.path.isfile(p):
                d = torch.load(p, map_location="cpu", weights_only=False)
                shard_gen = int(d.get("teacher_gen", 0))
                if teacher_gen is not None and shard_gen != teacher_gen:
                    # Generations are monotonic. Once the newest-first scan
                    # reaches an older teacher, no earlier shard is admissible.
                    if shard_gen < teacher_gen:
                        break
                    idx -= 1
                    continue
                for i in range(d["x"].shape[0]):
                    examples.append((d["x"][i], d["pi"][i].numpy(),
                                     float(d["z"][i])))
            idx -= 1
        examples.reverse()   # oldest first, so the deque evicts oldest
        return examples


# --------------------------------------------------------------------------- #
# Checkpoint shells
# --------------------------------------------------------------------------- #

def _make_agent(network, device, value_tanh, model_dir, lr):
    cfg = ModelConfigCNN(**NETWORK_CONFIGS[network], learning_rate=lr,
                         label="expert_iter", model_dir=model_dir,
                         value_tanh=value_tanh)
    a = NeuralNetAgentPG(cfg=cfg, model_path=None)
    a.model.to(device)
    a.device = device
    a.model.eval()
    return a


def _save_teacher(path, model, value_tanh, gen):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in
                               model.state_dict().items()},
                "value_tanh": value_tanh, "gen": gen}, path)


def _promote_teacher(teacher, student):
    """Swap the promoted student's weights into the live teacher; return tanh flag.

    load_state_dict copies weights only, NOT the runtime value_tanh flag, and
    promoted students always have a tanh value head. A stale False here feeds
    unbounded pre-tanh values into MCTS generation -- value-scale poisoning the
    raw-argmax promote gate cannot detect (measured on the first v2 run: the
    live gen-2 teacher emitted values in [-1.66, 3.27] instead of [-1, 1] for
    ~12h of generation until a restart healed it from saved metadata).
    """
    teacher.model.load_state_dict(student.model.state_dict())
    teacher.model.value_tanh = True
    return True


# --------------------------------------------------------------------------- #
# Fixed inference-only evaluation
# --------------------------------------------------------------------------- #

def _eval_openings(n_games, seed=20260705):
    """Build a reproducible diverse opening panel (two colors per opening)."""
    pairs = max(1, (int(n_games) + 1) // 2)
    rng = random.Random(seed)
    openings = []
    for i in range(pairs):
        state = GameState()
        moves = []
        target_plies = 4 + (i % 7)  # cover early positions at plies 4..10
        for _ in range(target_plies):
            valid = rule_utl_valid_moves(
                state.board, state.last_move, state.mini_winners)
            if not valid or state.is_over():
                break
            move = rng.choice(valid)
            ok, _ = state.make_move(move)
            if not ok:
                raise RuntimeError(f"opening generator produced illegal move {move}")
            moves.append(move)
        if not state.is_over():
            openings.append(tuple(moves))
    if not openings:
        raise RuntimeError("failed to generate non-terminal evaluation openings")
    return openings


def _play_fixed_match(move_fn_a, move_fn_b, n_games, seed=20260705):
    """Inference-only score on fixed openings with colors swapped.

    The RNG is reset per game so stochastic anchors such as WinBlockAgent have
    reproducible fallbacks. Candidate move functions used for promotion are raw
    argmax (no sampling, no MCTS).
    """
    openings = _eval_openings(n_games, seed)
    py_state = random.getstate()
    np_state = np.random.get_state()
    score = 0.0
    played = 0
    try:
        for opening_idx, opening in enumerate(openings):
            for a_side in (X, O):
                if played >= n_games:
                    break
                game_seed = seed + opening_idx * 2 + (0 if a_side == X else 1)
                random.seed(game_seed)
                np.random.seed(game_seed & 0xFFFFFFFF)
                state = GameState()
                for move in opening:
                    ok, _ = state.make_move(move)
                    if not ok:
                        raise RuntimeError(
                            f"fixed opening contains illegal move {move}")
                move_num = len(opening)
                while not state.is_over():
                    fn = move_fn_a if state.player == a_side else move_fn_b
                    move = fn(state, move_num)
                    valid = rule_utl_valid_moves(
                        state.board, state.last_move, state.mini_winners)
                    if not isinstance(move, (int, np.integer)) or move not in valid:
                        raise RuntimeError(
                            f"evaluation agent returned illegal move {move}")
                    state.make_move(int(move))
                    move_num += 1
                if state.winner == a_side:
                    score += 1.0
                elif state.winner == DRAW:
                    score += 0.5
                played += 1
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return score / played


def _decayed_lr(base_lr, steps_total, half_life_steps, lr_min):
    """Stateless exponential LR decay derived from lifetime optimizer steps.

    Keyed off the persisted steps_total so it survives --resume without
    scheduler state. half_life_steps <= 0 disables decay.
    """
    if half_life_steps <= 0:
        return base_lr
    return max(lr_min, base_lr * 0.5 ** (steps_total / half_life_steps))


def _promotion_decision(head_to_head, heur_score, random_score,
                        best_heur, best_random, threshold,
                        absolute_margin, random_tolerance,
                        gregory_score=None, best_gregory=None,
                        gregory_tolerance=0.03, gregory_arm=0.10):
    """Return (promote, failed gates) for an inference-objective promotion.

    The gregory gate is a no-regression check against a depth-limited
    alpha-beta anchor -- the one panel opponent that is fully independent of
    the training gene pool (GRADING_AND_ORACLE.md). It self-arms: below
    gregory_arm the score is noise-floor territory (raw nets currently score
    ~0.03 vs depth 3) and regressions there are meaningless, so it is
    report-only until the lineage first reaches gregory_arm.
    """
    failed = []
    if head_to_head < threshold:
        failed.append("head_to_head")
    if heur_score < min(1.0, best_heur + absolute_margin):
        failed.append("winblock")
    if random_score < best_random - random_tolerance:
        failed.append("random_regression")
    if (gregory_score is not None and best_gregory is not None
            and best_gregory >= gregory_arm
            and gregory_score < best_gregory - gregory_tolerance):
        failed.append("gregory_regression")
    return not failed, failed


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Expert iteration: MCTS teacher -> student distillation.")
    ap.add_argument("--network", choices=list(NETWORK_CONFIGS), default="arena22")
    ap.add_argument("--teacher_ckpt", type=str,
                    default=os.path.join("models", "arena", "hall_of_fame",
                                         "06-27-26_elo1864.pt"),
                    help="Gen-0 teacher weights (ignored on --resume with state).")
    ap.add_argument("--teacher_tanh", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Gen-0 teacher has a tanh value head (league best.pt "
                         "does NOT; promoted students always do).")
    ap.add_argument("--teacher_sims", type=int, default=200,
                    help="MCTS sims per teacher move during generation.")
    ap.add_argument("--games_per_block", type=int, default=16,
                    help="Teacher self-play games per generate/train block.")
    ap.add_argument("--train_steps", type=int, default=100,
                    help="SGD steps per block.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_half_life_steps", type=int, default=25_000,
                    help="Halve the learning rate every this many optimizer "
                         "steps (stateless: derived from the persisted "
                         "steps_total, so it survives --resume; 0 disables). "
                         "Constant lr floored both v1 and early v2 at a "
                         "policy-CE plateau.")
    ap.add_argument("--lr_min", type=float, default=1e-4,
                    help="Learning-rate floor for the decay schedule.")
    ap.add_argument("--lr_gen_reset", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Key the LR decay clock to steps since the last "
                         "promotion instead of lifetime steps. Each generation "
                         "is a fresh distillation problem (the buffer is "
                         "cleared on promotion, so there is no old data to "
                         "protect); a global clock strands late generations "
                         "at the lr_min floor and stretches every gen longer "
                         "than the last for reasons unrelated to capability.")
    ap.add_argument("--value_coef", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=200_000,
                    help="Max positions in the training window.")
    ap.add_argument("--min_window", type=int, default=5_000,
                    help="Do not train until this many positions exist.")
    ap.add_argument("--dir_eps", type=float, default=0.10,
                    help="Dirichlet mix at the teacher's search root.")
    ap.add_argument("--dir_alpha", type=float, default=0.3)
    ap.add_argument("--temperature_moves", type=int, default=10)
    ap.add_argument("--opp_mix", type=float, default=0.35,
                    help="Fraction of expert games played against WinBlockAgent "
                         "to cover the repo's measured tactical blind spot.")
    ap.add_argument("--mini_tactic_opp", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="In WinBlock opponent games, replace the MCTS target "
                         "with a local mini-board win/block target when one is "
                         "available. DEFAULT ON; affects only the opponent slice.")
    ap.add_argument("--rnd_mix", type=float, default=0.15,
                    help="Fraction of expert games played against RandomAgent. "
                         "Pure teacher/WinBlock games never visit the sloppy "
                         "positions random opponents create, so the raw student "
                         "regressed ~0.045 on the fixed random panel (v2 checks "
                         "1-4: promote_random 0.71-0.74 vs the 0.75 floor) and "
                         "the promotion gate correctly stalled. Teacher search "
                         "still labels every training position in this slice.")
    ap.add_argument("--greg_mix", type=float, default=0.0,
                    help="Fraction of expert games played against a shallow "
                         "GregoryAgent (--greg_mix_depth). The one measured "
                         "distribution gap: the net never sees minimax-style "
                         "play, so the gregory(d3) panel crawls at +1-2 "
                         "pts/gen vs winblock's +4 (STRENGTH_NEXT S1). "
                         "OPT-IN, default 0. Suggested first segment: 0.10, "
                         "donated equally from --opp_mix and --rnd_mix so "
                         "pure self-play stays at 0.50.")
    ap.add_argument("--greg_mix_depth", type=int, default=2,
                    help="Depth of the training-mix gregory. Must stay BELOW "
                         "--gregory_depth: training against the ruler burns "
                         "the instrument (STRENGTH_NEXT rule 1).")
    ap.add_argument("--gate_min", type=float, default=5.0,
                    help="Minutes between gate evals (raw matches + edge probe).")
    ap.add_argument("--gate_games", type=int, default=40)
    ap.add_argument("--edge_sims", type=int, default=64,
                    help="Sims for the search-invariant probe.")
    ap.add_argument("--edge_games", type=int, default=20)
    ap.add_argument("--promote_min", type=float, default=30.0,
                    help="Minimum minutes between inference promotion checks.")
    ap.add_argument("--promote_games", type=int, default=300,
                    help="Games per promotion panel. Must keep sampling noise "
                         "well under the 0.02-0.03 promotion margins: 40 games "
                         "has SE ~0.08 (coin-flip gates, ~1 false promotion "
                         "per day); 300 has SE ~0.03. Raw inference, so a "
                         "3-panel check still costs well under a minute.")
    ap.add_argument("--promote_thresh", type=float, default=0.55,
                    help="Raw student score vs raw teacher on fixed openings.")
    ap.add_argument("--min_promote_games", type=int, default=1000,
                    help="Minimum new expert games between teacher changes.")
    ap.add_argument("--promote_margin", type=float, default=0.02,
                    help="Required absolute WinBlock score improvement.")
    ap.add_argument("--random_tolerance", type=float, default=0.03,
                    help="Maximum allowed regression against random.")
    ap.add_argument("--gregory_depth", type=int, default=3,
                    help="Depth of the alpha-beta anchor measured at every "
                         "promotion check. Gregory is the one panel opponent "
                         "independent of the training gene pool; its score is "
                         "the long-horizon honest ruler for when the WinBlock "
                         "and random anchors saturate. Raw nets currently "
                         "score ~0.03 vs depth 3 (a 300-game panel costs "
                         "~40s), so there is years of headroom.")
    ap.add_argument("--gregory_tolerance", type=float, default=0.03,
                    help="Maximum allowed regression against gregory once "
                         "the gate has armed.")
    ap.add_argument("--gregory_arm", type=float, default=0.10,
                    help="The gregory no-regression gate arms only once "
                         "best_gregory first reaches this score; below it "
                         "the panel is noise-floor and report-only.")
    ap.add_argument("--model_dir", type=str,
                    default=os.path.join("models", "expert_iter_v2"))
    ap.add_argument("--blocks", type=int, default=0, help="0 = run forever.")
    ap.add_argument("--resume", action="store_true",
                    help="Continue from model_dir state if present, else fresh.")
    ap.add_argument("--no_metrics", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if not 0.0 <= args.opp_mix <= 1.0:
        ap.error("--opp_mix must be in [0, 1]")
    if not 0.0 <= args.rnd_mix <= 1.0 or args.opp_mix + args.rnd_mix > 1.0:
        ap.error("--rnd_mix must be in [0, 1] and --opp_mix + --rnd_mix <= 1")
    if (not 0.0 <= args.greg_mix <= 1.0
            or args.opp_mix + args.rnd_mix + args.greg_mix > 1.0):
        ap.error("--greg_mix must be in [0, 1] and "
                 "--opp_mix + --rnd_mix + --greg_mix <= 1")
    if args.greg_mix > 0 and args.greg_mix_depth >= args.gregory_depth:
        ap.error("--greg_mix_depth must be below --gregory_depth: never train "
                 "against the honest ruler (STRENGTH_NEXT rule 1)")
    if args.promote_games < 2 or args.gate_games < 2:
        ap.error("--promote_games and --gate_games must be at least 2")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.model_dir, exist_ok=True)
    state_path = os.path.join(args.model_dir, "state.json")
    teacher_path = os.path.join(args.model_dir, "teacher.pt")
    student_path = os.path.join(args.model_dir, "student.pt")
    resume_path = os.path.join(args.model_dir, "resume.pt")
    store = ShardStore(os.path.join(args.model_dir, "data"))

    resuming = args.resume and os.path.isfile(state_path)
    saved_state = None
    if resuming:
        with open(state_path) as f:
            saved_state = json.load(f)
        saved_network = saved_state.get("network")
        if saved_network is not None and saved_network != args.network:
            raise RuntimeError(
                f"resume architecture mismatch: state uses {saved_network!r}, "
                f"command requested {args.network!r}")

    # ---- teacher ----------------------------------------------------------
    if resuming:
        payload = torch.load(teacher_path, map_location="cpu",
                             weights_only=False)
        teacher_tanh = payload["value_tanh"]
        teacher_gen = payload["gen"]
        teacher = _make_agent(args.network, device, teacher_tanh,
                              args.model_dir, args.lr)
        teacher.model.load_state_dict(payload["state_dict"])
    else:
        teacher_tanh = args.teacher_tanh
        teacher_gen = 0
        teacher = _make_agent(args.network, device, teacher_tanh,
                              args.model_dir, args.lr)
        # STRICT load: a partially-transferred gen-0 teacher would silently
        # generate garbage expert data. Arch mismatch must be a hard error.
        payload = torch.load(args.teacher_ckpt, map_location="cpu",
                             weights_only=False)
        sd = payload["state_dict"] if (isinstance(payload, dict)
                                       and "state_dict" in payload) else payload
        teacher.model.load_state_dict(sd)
        _save_teacher(teacher_path, teacher.model, teacher_tanh, teacher_gen)
        print(f"Gen-0 teacher seeded from {args.teacher_ckpt}")
    teacher.model.eval()

    # ---- student (always tanh: clean [-1,1] value for search) -------------
    student = _make_agent(args.network, device, True, args.model_dir, args.lr)
    optimizer = torch.optim.Adam(student.model.parameters(), lr=args.lr)

    block = 0
    shard_idx = -1
    games_total = 0
    steps_total = 0
    games_since_promotion = 0
    steps_since_promotion = 0
    buffer = ReplayBuffer(args.window)

    if resuming:
        st = saved_state
        block = st["block"]
        shard_idx = st["shard_idx"]
        games_total = st["games_total"]
        steps_total = st["steps_total"]
        games_since_promotion = st.get(
            "games_since_promotion", games_total)
        steps_since_promotion = st.get("steps_since_promotion")
        if steps_since_promotion is None:
            # Pre-schema-3 state never tracked the per-generation step clock.
            # Estimate it from the games ratio so the current generation
            # resumes mid-decay instead of at the global floor.
            steps_since_promotion = int(
                steps_total * games_since_promotion / max(1, games_total))
        teacher_gen = st["teacher_gen"]
        sd = torch.load(student_path, map_location=device, weights_only=False)
        student.model.load_state_dict(sd["state_dict"])
        if os.path.isfile(resume_path):
            opt_sd = torch.load(resume_path, map_location="cpu",
                                weights_only=False)
            optimizer.load_state_dict(opt_sd["optimizer"])
        buffer.extend(store.load_window(
            shard_idx, args.window, teacher_gen=teacher_gen))
        print(f"Resumed: block {block}, {games_total} games, "
              f"{len(buffer)} positions reloaded, teacher gen {teacher_gen}")
    else:
        # Preserve the strongest known policy at step zero. Starting the student
        # from random weights threw away the verified Arena policy and spent the
        # first phase relearning it. Tanh changes only the value output; policy
        # logits remain byte-identical and the value head is then recalibrated.
        student.model.load_state_dict(teacher.model.state_dict())
        if not args.no_metrics:
            clear_metrics_log()

    heur = WinBlockAgent()
    rnd = RandomAgent()
    greg = GregoryAgent(depth=args.gregory_depth)
    # Training-mix gregory is a SEPARATE, shallower instance -- the d3 ruler
    # above must never appear in the training data (STRENGTH_NEXT rule 1).
    greg_train = (GregoryAgent(depth=args.greg_mix_depth)
                  if args.greg_mix > 0 else None)
    raw_teacher = _raw_fn(teacher.model, device, sample_moves=0)
    best_heur = (saved_state.get("best_heur") if saved_state else None)
    best_random = (saved_state.get("best_random") if saved_state else None)
    best_gregory = (saved_state.get("best_gregory") if saved_state else None)
    if saved_state and saved_state.get("promote_panel") != args.promote_games:
        # Scores from different panel sizes are not comparable; rebase the
        # promotion baselines against the current teacher on the new panel.
        print(f"[!] promotion panel changed "
              f"({saved_state.get('promote_panel')} -> {args.promote_games} "
              f"games) -- recomputing baselines vs current teacher")
        best_heur = None
        best_random = None
        best_gregory = None
    if best_heur is None:
        best_heur = _play_fixed_match(
            raw_teacher, _agent_fn(heur), args.promote_games, seed=3301)
    if best_random is None:
        best_random = _play_fixed_match(
            raw_teacher, _agent_fn(rnd), args.promote_games, seed=4401)
    if best_gregory is None:
        best_gregory = _play_fixed_match(
            raw_teacher, _agent_fn(greg), args.promote_games, seed=8801)
    print(f"Inference promotion baseline | winblock={best_heur:.3f} | "
          f"random={best_random:.3f} | gregory(d{args.gregory_depth})="
          f"{best_gregory:.3f}")
    last_gate_t = 0.0      # fire the first gate after the first block
    last_promote_t = time.time()
    t_start = time.time()

    def save_all():
        student.save(student_path, verbose=False)
        torch.save({"optimizer": optimizer.state_dict()}, resume_path)
        with open(state_path, "w") as f:
            json.dump({"block": block, "shard_idx": shard_idx,
                       "games_total": games_total, "steps_total": steps_total,
                       "teacher_gen": teacher_gen,
                       "games_since_promotion": games_since_promotion,
                       "steps_since_promotion": steps_since_promotion,
                       "best_heur": best_heur,
                       "best_random": best_random,
                       "best_gregory": best_gregory,
                       "promote_panel": args.promote_games,
                       "network": args.network,
                       "schema_version": 3}, f)

    print(f"Expert iteration on {device} | network={args.network} | "
          f"teacher_sims={args.teacher_sims} | window={args.window}")

    try:
        while args.blocks == 0 or block < args.blocks:
            t0 = time.perf_counter()

            # ---- 1. GENERATE ------------------------------------------------
            teacher.model.eval()
            new_examples = []
            draws = 0
            moves_total = 0
            opponent_games = 0
            rnd_games = 0
            greg_games = 0
            mini_tac_wins = 0
            mini_tac_blocks = 0
            with torch.no_grad():
                for _ in range(args.games_per_block):
                    opponent_fn = None
                    opponent_game = False
                    tag = _opponent_slice(random.random(), args.opp_mix,
                                          args.rnd_mix, args.greg_mix)
                    if tag == "heur":
                        opponent_fn = lambda s: heur.select_move(s)
                        opponent_games += 1
                        opponent_game = True
                    elif tag == "rnd":
                        # Random-opponent slice: mini tactics stay OFF here --
                        # the point is teacher-search targets on blunder-created
                        # positions, not the WinBlock motif.
                        opponent_fn = lambda s: rnd.select_move(s)
                        rnd_games += 1
                    elif tag == "greg":
                        # Gregory slice (STRENGTH_NEXT S1): teacher-search
                        # targets on positions a minimax-style opponent forces.
                        # Mini tactics stay OFF, same reasoning as the random
                        # slice.
                        opponent_fn = lambda s: greg_train.select_move(s)
                        greg_games += 1
                    exs, winner, gstats = collect_game(
                        model=teacher.model,
                        device=device,
                        n_sims=args.teacher_sims,
                        c_puct=1.5,
                        dir_alpha=args.dir_alpha,
                        dir_eps=args.dir_eps,
                        wave_size=64,   # MCTS clamps to n_sims // 16 internally
                        temperature_moves=args.temperature_moves,
                        use_tactics=True,
                        use_mini_tactics=(
                            args.mini_tactic_opp and opponent_game),
                        opponent_fn=opponent_fn,
                    )
                    new_examples.extend(exs)
                    moves_total += gstats["moves"]
                    mini_tac_wins += gstats.get("mini_tac_wins", 0)
                    mini_tac_blocks += gstats.get("mini_tac_blocks", 0)
                    if winner == DRAW:
                        draws += 1
            games_total += args.games_per_block
            games_since_promotion += args.games_per_block
            shard_idx += 1
            store.write(shard_idx, new_examples, teacher_gen)
            buffer.extend(new_examples)

            pis = np.stack([e[1] for e in new_examples])
            pi_ent = float(-(pis * np.log(pis + 1e-12)).sum(axis=1).mean())
            gen_secs = time.perf_counter() - t0

            # ---- 2. TRAIN ---------------------------------------------------
            t_train = time.perf_counter()
            avg_loss = avg_pol = avg_val = float("nan")
            lr_clock = (steps_since_promotion if args.lr_gen_reset
                        else steps_total)
            cur_lr = _decayed_lr(args.lr, lr_clock,
                                 args.lr_half_life_steps, args.lr_min)
            if len(buffer) >= args.min_window:
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
                tl = pl = vl = 0.0
                for _ in range(args.train_steps):
                    batch = buffer.sample(args.batch_size)
                    l, p, v = train_on_examples(student.model, optimizer, batch,
                                                value_coef=args.value_coef,
                                                device=device)
                    tl += l
                    pl += p
                    vl += v
                avg_loss = tl / args.train_steps
                avg_pol = pl / args.train_steps
                avg_val = vl / args.train_steps
                steps_total += args.train_steps
                steps_since_promotion += args.train_steps
            train_secs = time.perf_counter() - t_train
            student.model.eval()

            # ---- 3. GATE (wall-clock paced) ----------------------------------
            extra = {
                "sp_draws": round(draws / args.games_per_block, 3),
                "pi_ent": round(pi_ent, 3),
                "avg_len": round(moves_total / args.games_per_block, 1),
                "teacher_gen": teacher_gen,
                "opponent_games": opponent_games,
                "rnd_games": rnd_games,
                "greg_games": greg_games,
                "mini_tac_wins": mini_tac_wins,
                "mini_tac_blocks": mini_tac_blocks,
                "lr": round(cur_lr, 8),
                # S0: where block time actually goes -- the decision data for
                # the cross-game generation batching call (STRENGTH_NEXT S4).
                "gen_secs": round(gen_secs, 1),
                "train_secs": round(train_secs, 1),
            }
            wr_random = float("nan")
            gate_line = ""
            with torch.no_grad():
                if time.time() - last_gate_t >= args.gate_min * 60:
                    tg0 = time.perf_counter()
                    stu_fn = _raw_fn(student.model, device, sample_moves=0)
                    wr_random = _play_fixed_match(
                        stu_fn, _agent_fn(rnd), args.gate_games, seed=1101)
                    extra["wr_heur"] = round(_play_fixed_match(
                        stu_fn, _agent_fn(heur), args.gate_games, seed=2201), 4)
                    extra["wr_anchor"] = round(_play_fixed_match(
                        stu_fn, _raw_fn(teacher.model, device, sample_moves=0),
                        args.gate_games, seed=5501), 4)
                    extra["mcts_edge"] = round(_play_fixed_match(
                        _search_fn(student.model, device, args.edge_sims,
                                   sample_moves=0),
                        stu_fn, args.edge_games, seed=6601), 4)
                    extra["gauntlet_secs"] = round(time.perf_counter() - tg0, 1)
                    last_gate_t = time.time()
                    gate_line = (f"     gate | rand={wr_random*100:.0f}% | "
                                 f"heur={extra['wr_heur']*100:.0f}% | "
                                 f"vs_teacher={extra['wr_anchor']*100:.0f}% | "
                                 f"edge={extra['mcts_edge']*100:.0f}% | "
                                 f"{extra['gauntlet_secs']}s")
                    if extra["mcts_edge"] < 0.5:
                        gate_line += "\n     [!] edge < 0.5 -- search losing " \
                                     "to raw student; investigate if persistent"

                # ---- 4. PROMOTE? --------------------------------------------
                if (time.time() - last_promote_t >= args.promote_min * 60
                        and len(buffer) >= args.min_window
                        and games_since_promotion >= args.min_promote_games
                        and not np.isnan(avg_loss)):
                    tp0 = time.perf_counter()
                    stu_raw = _raw_fn(
                        student.model, device, sample_moves=0)
                    teacher_raw = _raw_fn(
                        teacher.model, device, sample_moves=0)
                    score = _play_fixed_match(
                        stu_raw, teacher_raw, args.promote_games, seed=7701)
                    promote_heur = _play_fixed_match(
                        stu_raw, _agent_fn(heur),
                        args.promote_games, seed=3301)
                    promote_random = _play_fixed_match(
                        stu_raw, _agent_fn(rnd),
                        args.promote_games, seed=4401)
                    promote_greg = _play_fixed_match(
                        stu_raw, _agent_fn(greg),
                        args.promote_games, seed=8801)
                    last_promote_t = time.time()
                    extra["promote_score"] = round(score, 4)
                    extra["promote_heur"] = round(promote_heur, 4)
                    extra["promote_random"] = round(promote_random, 4)
                    extra["promote_gregory"] = round(promote_greg, 4)
                    promote, failed = _promotion_decision(
                        score, promote_heur, promote_random,
                        best_heur, best_random, args.promote_thresh,
                        args.promote_margin, args.random_tolerance,
                        gregory_score=promote_greg,
                        best_gregory=best_gregory,
                        gregory_tolerance=args.gregory_tolerance,
                        gregory_arm=args.gregory_arm)
                    if promote:
                        teacher_gen += 1
                        teacher_tanh = _promote_teacher(teacher, student)
                        _save_teacher(teacher_path, teacher.model,
                                      teacher_tanh, teacher_gen)
                        best_heur = promote_heur
                        best_random = max(best_random, promote_random)
                        best_gregory = max(best_gregory, promote_greg)
                        games_since_promotion = 0
                        # New generation = new distillation problem; restart
                        # the LR decay clock with the buffer.
                        steps_since_promotion = 0
                        # Old-policy targets are off-policy supervision after a
                        # teacher change. The prior 200k window spanned many
                        # generations and made current-shard error far worse
                        # than the reported replay loss.
                        buffer.clear()
                        extra["teacher_gen"] = teacher_gen
                        print(f"[**] INFERENCE PROMOTION: raw student beat raw "
                              f"teacher {score*100:.0f}% | winblock "
                              f"{promote_heur*100:.0f}% | random "
                              f"{promote_random*100:.0f}% | gregory "
                              f"{promote_greg*100:.0f}% -> gen {teacher_gen}")
                    else:
                        print(f"     no promotion ({','.join(failed)}): "
                              f"head={score*100:.0f}% | "
                              f"winblock={promote_heur*100:.0f}% "
                              f"(best {best_heur*100:.0f}%) | "
                              f"random={promote_random*100:.0f}% | "
                              f"gregory={promote_greg*100:.0f}% | "
                              f"{time.perf_counter()-tp0:.0f}s")

            # ---- metrics + state --------------------------------------------
            if not args.no_metrics:
                append_metrics(
                    loss=avg_loss,
                    epsilon=float("nan"),
                    winrate=wr_random,
                    value_loss=avg_val,
                    t=time.time(),
                    policy_loss=avg_pol,
                    games_total=games_total,
                    buffer=len(buffer),
                    extra=extra,
                )
            block += 1
            save_all()

            elapsed = time.perf_counter() - t0
            loss_str = f"{avg_loss:.4f}" if not np.isnan(avg_loss) else "warmup"
            print(f"block {block:5d} | gen {teacher_gen} | "
                  f"games={games_total:6d} | buf={len(buffer):6d} | "
                  f"loss={loss_str} | ent={pi_ent:.2f} | "
                  f"draws={draws}/{args.games_per_block} | "
                  f"gen {gen_secs:.0f}s / total {elapsed:.0f}s")
            if gate_line:
                print(gate_line)

    except KeyboardInterrupt:
        print("\n[!] Interrupted -- saving state...")
    finally:
        save_all()
        print(f"State saved to {args.model_dir} "
              f"({format_elapsed(time.time() - t_start)} this session). "
              f"Relaunch with --resume to continue.")


if __name__ == "__main__":
    main()
