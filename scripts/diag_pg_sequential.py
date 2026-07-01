# scripts/diag_pg_sequential.py
"""Phase-0c diagnostic -- dump the numeric structure of the SEQUENTIAL learn() path.

WHY
  scripts/train_league.py's run_chunk() uses NeuralNetAgentPG.learn() (the per-game
  sequential path). Three numeric issues are suspected there; the *batched*
  learn_from_trajectories() path is already fine. Two of the three need REAL
  magnitudes to fix confidently -- this script captures them. Run it at a torch box
  and send the output back (see HOME_RUN.md for the full one-trip runbook):

      python -m scripts.diag_pg_sequential --games 1000 > diag_0c.txt
      # (40 is plenty too; 1000 just tightens the ratios -- the table is capped so the
      #  output stays ~60 lines either way. Or: python scripts/diag_pg_sequential.py ...)

  No checkpoint needed -- random weights are fine, because we measure the STRUCTURE of
  the loss terms (how they scale with game length, whether normalization flattens the
  win signal), not playing strength. Nothing is trained; learn() is never called.

WHAT IT CONFIRMS  (see DIAGNOSE_0C.md for how each number maps to the fix)
  (2) per-game advantage normalization (learn():~106) zeros the win signal: advantage
      is normalized to mean 0 / std 1 *within a single game*, removing the constant
      win/loss component that should drive the update.
  (3) actor_loss/entropy use .sum() while value_loss uses .mean() (learn():108-115):
      the actor term scales with game length T while the value term does not, so the
      value head is under-weighted ~T and the effective LR scales with T.
  (1) perspective channel (board_to_tensor_from_gamestate): X/O live on absolute
      channels (0/1/4) with turn on ch2, so a position and its color-swap are
      different inputs. (Confirmation printout only -- the fix is authorable from code;
      no home data needed.)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import statistics as st

import torch
import torch.nn.functional as F

from agents.neural_net_agent_pg import NeuralNetAgentPG
from agents.agent_base import ModelConfigCNN, board_to_tensor_from_gamestate
from agents.random_agent import RandomAgent
from engine.game import GameState, _CPP_ENGINE, _cpp_build
from engine.rules import rule_utl_valid_moves
from scripts.train_league import play_one_game, NETWORK_CONFIGS


def diagnose_cpp_engine():
    """Env check: is the C++ engine actually live on THIS box?

    engine.game already prints a [OK]/[!] line at import time (it leaks to the top of
    diag_0c.txt). This restates it as a labeled, grep-able block so the saved output
    answers the question definitively -- no guessing from whether build/ is gitignored.
    Reporting `active` AND `build dir exists` separately catches the sneaky case: a
    .pyd built against a different Python ABI exists on disk but fails to import, so
    it silently falls back to Python (built != loadable).
    """
    print("\n=== C++ ENGINE (env check) ===")
    built = os.path.isdir(_cpp_build)
    print(f"  active (loaded)   : {_CPP_ENGINE}")
    print(f"  expected build dir: {_cpp_build}")
    print(f"  build dir exists  : {built}")
    if _CPP_ENGINE:
        print("  -> GameState is C++-backed (faster make_move). NOTE clone()/MCTS")
        print("     rollouts still run in Python (the known half-done part) -- search")
        print("     is NOT accelerated yet. Since it's live, run engine/cpp/test_engine.py")
        print("     once for a Python/C++ parity check (correctness insurance, not speed).")
    elif built:
        print("  -> build dir EXISTS but the module did NOT load (see the [!] import line")
        print("     at the top: likely a stale .pyd / wrong Python ABI / missing runtime).")
        print("     Rebuild against this interpreter. Falls back to Python meanwhile --")
        print("     does NOT affect 0c diagnostic/validation correctness, only speed.")
    else:
        print("  -> not built on this box; pure-Python fallback. Does NOT affect 0c")
        print("     correctness, only speed. To build: cmake + pybind11 via")
        print("     engine/cpp/CMakeLists.txt (needs a C++17 compiler).")


def diagnose_perspective():
    """Item 1: print the input-tensor channel layout for X-to-move vs O-to-move."""
    print("\n=== PERSPECTIVE CHANNEL (item 1) -- confirmation only ===")
    labels = ["0:X-pos(abs)", "1:O-pos(abs)", "2:turn(+1=X,-1=O)", "3:valid",
              "4:mini-win(abs)", "5:last-move", "6:bias"]
    gs = GameState()
    tX = board_to_tensor_from_gamestate(gs)                       # X to move (fresh board)
    moves = rule_utl_valid_moves(gs.board, gs.last_move, gs.mini_winners)
    gs.make_move(moves[0])
    tO = board_to_tensor_from_gamestate(gs)                       # now O to move
    print(f"  tensor shape = {tuple(tX.shape)}")
    print(f"  {'channel':>18} {'X-to-move mean':>16} {'O-to-move mean':>16}")
    for c, lab in enumerate(labels):
        print(f"  {lab:>18} {tX[c].mean().item():>16.4f} {tO[c].mean().item():>16.4f}")
    print("  -> ch2 flips sign with turn; X/O sit on absolute ch0/1/4. A color-swapped")
    print("     mirror is a DIFFERENT input. Fix = canonicalize to side-to-move (put")
    print("     'my' pieces on one channel, 'theirs' on another; ch2 becomes constant).")


def diagnose_numeric(active, opp, games, gamma, max_rows=50):
    """Items 2 & 3: replicate learn()'s math per game and report term magnitudes.

    The per-game table is capped at `max_rows` so a big run (e.g. --games 1000)
    stays paste-friendly; the AGGREGATE below always uses every game played. More
    games => tighter ratio estimates and a fuller game-length (T) distribution, at
    no cost to the read.
    """
    print(f"\n=== NUMERIC TRIO (items 2 & 3) -- gamma={gamma}, games={games} ===")
    print(f"{'g':>3} {'T':>3} {'term':>6} {'adv_mean':>9} {'adv_std':>8} "
          f"{'|act_sum|':>10} {'|act_mean|':>11} {'val_loss':>9} {'sum/val':>9} {'mean/val':>9}")

    Ts, act_sums, act_means, val_losses, sum_ratios, mean_ratios = [], [], [], [], [], []
    shapes_printed = False
    printed = 0

    for g in range(games):
        try:
            play_one_game(active, opp)                            # populates active.* buffers
        except Exception as e:                                    # noqa: BLE001 -- diagnostic, keep going
            print(f"{g:>3}  play_one_game error: {e}")
            active.clear_history()
            continue

        T = len(active.last_rewards)
        if T == 0 or not active.log_probs:
            active.clear_history()
            continue

        # --- replicate learn() EXACTLY (neural_net_agent_pg.py:92-115) ---
        returns = torch.zeros(T, dtype=torch.float32, device=active.device)
        G = 0.0
        for t in reversed(range(T)):
            G = active.last_rewards[t] + gamma * G
            returns[t] = G
        log_probs = torch.stack(active.log_probs)
        values = torch.stack(active.values)
        adv_raw = returns - values.detach()

        adv_norm = adv_raw
        if T > 1:                                                 # the per-game norm (item 2)
            adv_norm = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

        actor_sum = -(log_probs * adv_norm).sum()                 # what learn() uses today (item 3)
        actor_mean = -(log_probs * adv_norm).mean()               # what the fix would switch to
        value_loss = F.mse_loss(values, returns)

        if not shapes_printed:
            print(f"  [shapes] log_probs={tuple(log_probs.shape)} values={tuple(values.shape)} "
                  f"returns={tuple(returns.shape)} adv={tuple(adv_raw.shape)}  "
                  f"(if adv is 2-D that's a broadcasting bug worth flagging)")
            shapes_printed = True

        adv_mean = adv_raw.float().mean().item()
        adv_std = adv_raw.float().std(unbiased=False).item()
        term = float(active.last_rewards[-1])
        vl = value_loss.item()
        a_s, a_m = abs(actor_sum.item()), abs(actor_mean.item())
        sr = a_s / vl if vl > 1e-9 else float('nan')
        mr = a_m / vl if vl > 1e-9 else float('nan')

        if printed < max_rows:
            print(f"{g:>3} {T:>3} {term:>6.1f} {adv_mean:>9.3f} {adv_std:>8.3f} "
                  f"{a_s:>10.3f} {a_m:>11.3f} {vl:>9.3f} {sr:>9.2f} {mr:>9.2f}")
            printed += 1

        Ts.append(T); act_sums.append(a_s); act_means.append(a_m); val_losses.append(vl)
        if vl > 1e-9:
            sum_ratios.append(sr); mean_ratios.append(mr)
        active.clear_history()

    if not Ts:
        print("  (no games produced trajectories -- check the agent/runner)")
        return

    if len(Ts) > printed:
        print(f"  ... table truncated to first {printed} games with trajectories; "
              f"aggregates below use all {len(Ts)} ...")

    print("\n--- AGGREGATE ---")
    print(f"  T (game length): min={min(Ts)} mean={st.mean(Ts):.1f} max={max(Ts)}")
    print(f"  mean |actor_sum| = {st.mean(act_sums):.3f}   "
          f"mean |actor_mean| = {st.mean(act_means):.3f}   "
          f"mean value_loss = {st.mean(val_losses):.3f}")
    if sum_ratios:
        print(f"  mean actor_sum/value_loss  = {st.mean(sum_ratios):.2f}   "
              f"(item 3: large + T-growing => value head under-weighted ~T)")
        print(f"  mean actor_mean/value_loss = {st.mean(mean_ratios):.2f}   "
              f"(the .mean() variant the fix would switch to)")
        if len(sum_ratios) >= 4:
            qs = st.quantiles(sum_ratios, n=4)   # p25 / p50 / p75 -- robust to game-length outliers
            print(f"  actor_sum/value_loss p25/p50/p75 = {qs[0]:.2f} / {qs[1]:.2f} / {qs[2]:.2f}")
    if len(Ts) > 2:                                               # crude T-vs-actor_sum trend
        med = st.median(Ts)
        lo = [a for a, t in zip(act_sums, Ts) if t <= med]
        hi = [a for a, t in zip(act_sums, Ts) if t > med]
        if lo and hi:
            print(f"  |actor_sum| short games (T<={med:.0f})={st.mean(lo):.3f}  vs  "
                  f"long games (T>{med:.0f})={st.mean(hi):.3f}")
            print(f"  (item 3: under .sum() the long-game term should be clearly larger)")
    print("\n  adv_mean column: under per-game norm it sits ~0.000 every game (item 2 --")
    print("  the win/loss sign in the *raw* return is being normalized away).")


def main():
    ap = argparse.ArgumentParser(description="Phase-0c sequential-PG numeric diagnostic.")
    ap.add_argument("--games", type=int, default=40, help="Games to sample (default 40).")
    ap.add_argument("--network", type=str, default="small", choices=["small", "medium", "large"])
    ap.add_argument("--gamma", type=float, default=0.99, help="Discount used in the replicated math.")
    ap.add_argument("--max_rows", type=int, default=50,
                    help="Cap the per-game table to this many rows (aggregates still use all games).")
    args = ap.parse_args()

    net = NETWORK_CONFIGS[args.network]
    cfg = ModelConfigCNN(**net, learning_rate=1e-4, label="diag_pg", model_dir="models/diag_pg")
    active = NeuralNetAgentPG(cfg=cfg, model_path=None)            # random weights
    active.set_eval(False)                                        # train mode: records log_probs/values/entropies
    opp = RandomAgent()

    print("Phase-0c diagnostic -- sequential learn() numeric structure")
    print(f"network={args.network}  games={args.games}  gamma={args.gamma}  (random weights, nothing trained)")
    try:
        diagnose_cpp_engine()
    except Exception as e:                                        # noqa: BLE001
        print(f"  C++ engine check error: {e}")
    try:
        diagnose_perspective()
    except Exception as e:                                        # noqa: BLE001
        print(f"  perspective check error: {e}")
    diagnose_numeric(active, opp, args.games, args.gamma, max_rows=args.max_rows)
    print("\nDone. Send this whole output back (e.g. save with '> diag_0c.txt').")


if __name__ == "__main__":
    main()
