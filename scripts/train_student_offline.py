"""Train ONE student architecture on a frozen expert corpus.

The expensive half of expert iteration -- running MCTS-200 self-play to make
(state, improved_policy, outcome) targets -- does not depend on the student at
all. So a corpus generated once with `expert_iter --generate_only` can train any
number of architectures, and that is what makes an architecture A/B cheap.

The A/B property this script guarantees: data order and the dihedral symmetry
sequence come from dedicated RNGs seeded by --seed, NOT from the global torch
RNG that weight init consumes. Two architectures run at the same --seed
therefore see byte-identical batches in byte-identical order, so any difference
in the panel is attributable to the architecture and nothing else.

Usage:
    python -m scripts.train_student_offline --corpus models/corpus_gen22 \
        --arch modern --steps 40000 --out models/ab_arch/modern.pt
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from agents.agent_base import ModelConfigCNN
from agents.neural_net_agent_3 import ConvNet
from scripts.train_alphazero import apply_dihedral_symmetry, policy_value_loss
from scripts.expert_iter import _decayed_lr, ShardStore

torch.set_float32_matmul_precision("high")


# --------------------------------------------------------------------------- #
# The A/B: two architectures at a matched parameter budget (~921k).
#
# Three arms, because parameters and latency are DIFFERENT budgets and
# param-matching alone would rig the test. Measured 1-thread CPU batch-32
# forward (the closest proxy for ONNX Runtime Web on WASM):
#
#   plain       921,026 params   6.20 ms   the architecture this repo has always
#               used: plain Conv2d+ReLU, no norm, no skips, heads that flatten
#               the whole conv output into a Linear. 94% of its parameters land
#               in those two head Linears; only 6% reach the conv tower.
#   modern      921,688 params  54.28 ms   PARAMETER-matched. Stem + 5 residual
#               blocks at width 96, GroupNorm, 1x1-squeezed heads. 91% of its
#               parameters are in the tower -- 14x the spatial capacity per
#               parameter, but 8.8x the compute. Answers "better per byte of
#               download?"
#   modern_w32  141,656 params   6.11 ms   LATENCY-matched. Same recipe at width
#               32, stem + 3 blocks. Same speed as plain, 6.5x fewer parameters.
#               Answers "better per millisecond?" -- the deployment question.
#   squeeze     ~155,000 params  ~6.2 ms   DECOMPOSITION arm. Plain convs, NO
#               residual and NO norm, but with the 1x1-squeezed heads. Isolates
#               how much of any win comes from the two-line head fix alone
#               versus from the residual tower, which is a real rewrite.
# --------------------------------------------------------------------------- #
AB_ARCHS = {
    "plain": dict(conv_channels=[64, 48, 48, 16], fc_hidden_sizes=[384]),
    "modern": dict(conv_channels=[96] * 6, fc_hidden_sizes=[256],
                   residual=True, norm="group", head_squeeze=2),
    "modern_w32": dict(conv_channels=[32] * 4, fc_hidden_sizes=[256],
                       residual=True, norm="group", head_squeeze=2),
    "squeeze": dict(conv_channels=[56] * 4, fc_hidden_sizes=[256],
                    head_squeeze=2),
}


def build_model(arch, device):
    cfg = ModelConfigCNN(value_tanh=True, model_dir="models/_offline", **arch)
    return ConvNet(cfg).to(device)


def load_corpus(corpus_dir, max_examples, verbose=True):
    """Load every shard into three contiguous tensors (oldest shard first)."""
    store = ShardStore(os.path.join(corpus_dir, "data"))
    files = sorted(f for f in os.listdir(store.data_dir)
                   if f.startswith("shard_") and f.endswith(".pt"))
    if not files:
        raise SystemExit(f"no shards in {store.data_dir}")

    xs, pis, zs, total = [], [], [], 0
    for name in files:
        d = torch.load(os.path.join(store.data_dir, name), map_location="cpu",
                       weights_only=False)
        xs.append(d["x"])
        pis.append(d["pi"])
        zs.append(d["z"])
        total += d["x"].shape[0]
        if max_examples and total >= max_examples:
            break

    X = torch.cat(xs).float()
    PI = torch.cat(pis).float()
    Z = torch.cat(zs).float()
    if max_examples and X.shape[0] > max_examples:
        X, PI, Z = X[:max_examples], PI[:max_examples], Z[:max_examples]
    if verbose:
        gb = (X.numel() * 4 + PI.numel() * 4 + Z.numel() * 4) / 1e9
        print(f"corpus: {X.shape[0]:,} examples from {len(files)} shards "
              f"({gb:.2f} GB)")
    return X, PI, Z


def main():
    ap = argparse.ArgumentParser(
        description="Train one student architecture on a frozen expert corpus.")
    ap.add_argument("--corpus", type=str, default="models/corpus_gen22")
    ap.add_argument("--arch", type=str, default="modern", choices=list(AB_ARCHS),
                    help="Named A/B architecture (see AB_ARCHS).")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr_min", type=float, default=1e-4)
    ap.add_argument("--lr_half_life_steps", type=int, default=15000)
    ap.add_argument("--value_coef", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234,
                    help="Seeds data order + symmetry. MUST match across the "
                         "A/B arms; weight init is deliberately NOT tied to it.")
    ap.add_argument("--init_seed", type=int, default=0,
                    help="Seed torch before building the model, making weight "
                         "init reproducible and SHAREABLE across arms. 0 keeps "
                         "the historical behaviour (uninitialised global RNG, "
                         "so init differs run to run). Required for a paired "
                         "A/B where the arms must start from identical weights.")
    ap.add_argument("--max_examples", type=int, default=0, help="0 = all")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data_device", type=str, default="cuda",
                    help="Hold the corpus here; falls back to cpu on OOM.")
    ap.add_argument("--log_every", type=int, default=1000)
    # ---- weight averaging ------------------------------------------------
    # The live run's panel scores wobble across a 0.080-wide band with a lag-1
    # autocorrelation of only +0.120 -- consecutive checkpoints are nearly
    # independent draws, and which one you happen to save is worth more than
    # any architecture in RESULT_ARCH_AB.md. Averaging weights reduces that
    # wobble instead of selecting a lucky point inside it, and costs nothing at
    # inference: same architecture, same parameter count, same latency.
    ap.add_argument("--swa_from", type=int, default=0,
                    help="Step at which the averaging phase starts. 0 = off. "
                         "Snapshots are taken from here on and their mean is "
                         "written as the SWA checkpoint.")
    ap.add_argument("--swa_lr", type=float, default=0.0,
                    help="Constant LR to hold during the averaging phase. 0 "
                         "keeps the decay schedule -- but by then the LR has "
                         "decayed to the floor, the weights barely move, and "
                         "there is no wobble left to average. A constant LR "
                         "here is the standard SWA recipe and the only way "
                         "this measures anything.")
    ap.add_argument("--snapshot_every", type=int, default=2500,
                    help="Snapshot cadence during the averaging phase.")
    ap.add_argument("--ema_decay", type=float, default=0.0,
                    help="Track an exponential moving average of the weights "
                         "with this decay (0.999 ~ a 1000-step window). "
                         "0 = off. Written as the EMA checkpoint.")
    args = ap.parse_args()

    device = args.device
    X, PI, Z = load_corpus(args.corpus, args.max_examples)
    n = X.shape[0]

    if args.data_device == "cuda" and device == "cuda":
        try:
            X, PI, Z = X.to("cuda"), PI.to("cuda"), Z.to("cuda")
            print("corpus resident on GPU")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("[!] corpus does not fit on GPU -- streaming from CPU")

    if args.init_seed:
        torch.manual_seed(args.init_seed)
        torch.cuda.manual_seed_all(args.init_seed)
    model = build_model(AB_ARCHS[args.arch], device)
    nparams = sum(p.numel() for p in model.parameters())
    ntower = sum(p.numel() for p in model.conv_layers.parameters())
    print(f"arch={args.arch} params={nparams:,} "
          f"(conv tower {ntower:,} = {ntower / nparams * 100:.1f}%)")
    if args.init_seed:
        # Cheap fingerprint so a paired run can be shown to have started from
        # the same weights, without shipping the whole state dict around.
        with torch.no_grad():
            fp = sum(float(p.double().sum()) for p in model.parameters())
        print(f"init_seed={args.init_seed} init_fingerprint={fp:.12f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Architecture-independent RNGs: this is what makes the A/B fair.
    idx_rng = np.random.RandomState(args.seed)
    sym_rng = random.Random(args.seed + 1)

    order = idx_rng.permutation(n)
    cursor = 0
    t0 = time.time()
    hist = []

    # Averaging accumulators. Buffers (and any non-float state) are copied
    # rather than averaged -- only floating-point weights are meaningful to
    # average, and this arch has no running stats anyway.
    ema = None
    if args.ema_decay > 0:
        ema = {k: v.detach().clone().float()
               for k, v in model.state_dict().items()}
    swa_sum, swa_count, snapshots = None, 0, []

    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > n:          # next epoch
            order = idx_rng.permutation(n)
            cursor = 0
        sel = torch.from_numpy(order[cursor:cursor + args.batch_size]).to(X.device)
        cursor += args.batch_size

        xs = X.index_select(0, sel).to(device, non_blocking=True)
        pis = PI.index_select(0, sel).to(device, non_blocking=True)
        zs = Z.index_select(0, sel).to(device, non_blocking=True)
        xs, pis = apply_dihedral_symmetry(xs, pis, sym_rng.randrange(8))

        lr = _decayed_lr(args.lr, step, args.lr_half_life_steps, args.lr_min)
        if args.swa_from and step >= args.swa_from and args.swa_lr > 0:
            lr = args.swa_lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        loss, pol, val = policy_value_loss(model, xs, pis, zs, args.value_coef)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if ema is not None:
            d = args.ema_decay
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
                    else:
                        ema[k] = v.detach().clone()

        if (args.swa_from and step >= args.swa_from
                and (step - args.swa_from) % args.snapshot_every == 0):
            with torch.no_grad():
                sd = {k: v.detach().clone()
                      for k, v in model.state_dict().items()}
            snapshots.append((step, sd))
            if swa_sum is None:
                swa_sum = {k: v.clone().float() for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    if v.dtype.is_floating_point:
                        swa_sum[k].add_(v.float())
                    else:
                        swa_sum[k] = v.clone().float()
            swa_count += 1

        if step % args.log_every == 0 or step == 1:
            rec = dict(step=step, loss=round(loss.item(), 5),
                       policy=round(pol.item(), 5), value=round(val.item(), 5),
                       lr=round(lr, 7), secs=round(time.time() - t0, 1))
            hist.append(rec)
            print(f"step {step:6d}/{args.steps} | loss {rec['loss']:.4f} "
                  f"| pol {rec['policy']:.4f} | val {rec['value']:.4f} "
                  f"| lr {lr:.2e} | {rec['secs']:.0f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    base = dict(value_tanh=True, arch_name=args.arch, arch=AB_ARCHS[args.arch],
                params=nparams, steps=args.steps, seed=args.seed,
                corpus=args.corpus, corpus_examples=n, history=hist)

    def _write(path, sd, kind):
        torch.save(dict(base, state_dict=sd, weights=kind), path)
        print(f"saved {path}  [{kind}]")

    _write(args.out, model.state_dict(), "final")
    stem = args.out[:-3] if args.out.endswith(".pt") else args.out

    ref = model.state_dict()
    for step, sd in snapshots:
        _write(f"{stem}.snap{step}.pt", sd, f"snapshot@{step}")
    if swa_count:
        swa = {k: (v / swa_count).to(ref[k].dtype) if ref[k].dtype.is_floating_point
               else ref[k] for k, v in swa_sum.items()}
        _write(f"{stem}.swa.pt", swa, f"swa(mean of {swa_count} snapshots)")
    if ema is not None:
        emad = {k: v.to(ref[k].dtype) for k, v in ema.items()}
        _write(f"{stem}.ema.pt", emad, f"ema(decay {args.ema_decay})")
    print(f"done ({nparams:,} params, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
