"""Export a trained agent to ONNX for M3 browser inference.

Produces two files in --out-dir:
  model.onnx       fp32 (always)
  model_int8.onnx  dynamic-int8 (with --quantize; requires onnxruntime)

Also writes/updates docs/models/model_config.json so the browser page
picks up the new model without any other changes. When swapping agents,
just re-run this script with the new checkpoint -- the JSON is the only
coupling between the Python side and the JS side.

Input tensor consumed by the browser (channel layout mirrors
agents/agent_base.py board_to_tensor_from_gamestate):
  name  : "input"
  shape : (1, 7, 9, 9)  float32  NCHW
  ch 0  : X piece positions
  ch 1  : O piece positions
  ch 2  : current player uniform plane (+1 if X to move, -1 if O)
  ch 3  : legal move mask
  ch 4  : mini-board winners (+1 X, -1 O, 0 draw/empty)
  ch 5  : last move indicator
  ch 6  : bias (all 1.0)

Outputs:
  "policy_logits" : (1, 81) float32  -- raw unnormalised logits
  "value"         : (1,)    float32  -- to-move expected return; unbounded for
                    legacy shaped-return nets, tanh-bounded to [-1, 1] when the
                    checkpoint was trained with a tanh value head (value_tanh)

Usage (home box, needs torch + onnxruntime):
    python -m scripts.export_onnx --candidate arena:21@hof --quantize
    python -m scripts.export_onnx --checkpoint models/league_pg/best.pt --network medium
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from scripts.benchmark_suite import resolve_candidate, _build_base_candidate
from scripts.train_league import NETWORK_CONFIGS
from agents.agent_base import ModelConfigCNN
from agents.neural_net_agent_pg import NeuralNetAgentPG


# -----------------------------------------------------------------------
# ONNX export wrapper
# -----------------------------------------------------------------------

class _ExportWrapper(torch.nn.Module):
    """Thin wrapper that gives the ONNX graph stable output names and shapes."""
    def __init__(self, convnet):
        super().__init__()
        self.net = convnet

    def forward(self, x):
        policy, value = self.net.forward_both(x)
        # Guarantee (B,) for value regardless of squeeze behaviour.
        return policy, value.reshape(x.shape[0])


def _export_fp32(agent, out_path: Path) -> Path:
    wrapper = _ExportWrapper(agent.model).eval()
    dummy   = torch.zeros(1, 7, 9, 9, dtype=torch.float32)
    fp32    = out_path.parent / "model.onnx"

    torch.onnx.export(
        wrapper, dummy, str(fp32),
        input_names  = ["input"],
        output_names = ["policy_logits", "value"],
        dynamic_axes = {"input": {0: "batch"}, "policy_logits": {0: "batch"}, "value": {0: "batch"}},
        opset_version = 17,
        do_constant_folding = True,
    )
    kb = fp32.stat().st_size // 1024
    print(f"[OK] fp32 ONNX  -> {fp32}  ({kb} KB)")
    return fp32


def _quantize(fp32_path: Path) -> Path:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[!] onnxruntime not installed -- skipping quantization. "
              "pip install onnxruntime", file=sys.stderr)
        return fp32_path

    int8 = fp32_path.parent / "model_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8), weight_type=QuantType.QInt8)
    kb = int8.stat().st_size // 1024
    print(f"[OK] int8 ONNX  -> {int8}  ({kb} KB)")
    return int8


def _write_config(out_dir: Path, label: str, model_file: str, fp32_kb: int, int8_kb: int | None,
                  value_tanh: bool = False):
    """Write/update docs/models/model_config.json.

    This JSON is the ONLY coupling between the export pipeline and the browser.
    To swap the agent: re-run this script; only 'name', 'version', and 'file'
    change. The input/output tensor spec is fixed by the game encoding.
    """
    cfg = {
        "_swap_guide": (
            "To deploy a new model: run scripts/export_onnx.py --candidate <new> [--quantize]. "
            "This file is rewritten automatically. Only name/version/file ever change."
        ),
        "name":        label,
        "version":     "m3-r1",
        "description": "Policy-gradient self-play, 1-ply tactical overlay at eval.",
        "file":        f"./{model_file}",
        "fp32_kb":     fp32_kb,
        "int8_kb":     int8_kb,
        "value_tanh":  value_tanh,
        "input": {
            "name":  "input",
            "shape": [1, 7, 9, 9],
            "channels": [
                "x_pieces", "o_pieces", "current_player",
                "valid_moves", "mini_winners", "last_move", "bias"
            ]
        },
        "outputs": {
            "policy": "policy_logits",
            "value":  "value"
        }
    }
    cfg_path = out_dir / "model_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"[OK] config     -> {cfg_path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Export a candidate net to ONNX for M3 browser inference."
    )
    ap.add_argument("--candidate", type=str, default="",
                    help="Arena spec, e.g. 'arena:21@hof'. Reads arch from arena_state.json.")
    ap.add_argument("--checkpoint", type=str, default="",
                    help="Direct checkpoint path (use with --network).")
    ap.add_argument("--network", type=str, default="medium", choices=list(NETWORK_CONFIGS),
                    help="Architecture preset when using --checkpoint.")
    ap.add_argument("--out-dir", type=str, default="docs/models",
                    help="Output directory (default: docs/models/).")
    ap.add_argument("--quantize", action="store_true",
                    help="Also produce a dynamic int8 model (requires onnxruntime).")
    ap.add_argument("--value_tanh", action="store_true",
                    help="Bound the exported value head with tanh. Only for --checkpoint "
                         "mode; --candidate manifests declare value_tanh themselves.")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Torch device for export (cpu recommended).")
    args = ap.parse_args()

    if not args.candidate and not args.checkpoint:
        ap.error("Supply --candidate arena:N@hof  OR  --checkpoint path/to/net.pt")
    if args.candidate and args.checkpoint:
        ap.error("--candidate and --checkpoint are mutually exclusive")
    if args.candidate and args.value_tanh:
        ap.error("--value_tanh only applies to --checkpoint mode; put "
                 '"value_tanh": true in the candidate manifest instead')

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.candidate:
        spec  = resolve_candidate(args.candidate)
        agent = _build_base_candidate(spec, device)
        label = args.candidate
        value_tanh = spec.value_tanh
    else:
        if not os.path.isfile(args.checkpoint):
            ap.error(f"checkpoint not found: {args.checkpoint}")
        net_cfg = NETWORK_CONFIGS[args.network]
        cfg = ModelConfigCNN(
            **net_cfg, learning_rate=1e-4, label="export",
            model_dir=str(Path(args.checkpoint).parent), device=device,
            value_tanh=args.value_tanh,
        )
        agent = NeuralNetAgentPG(cfg=cfg, model_path=args.checkpoint)
        agent.set_eval(True)
        label = f"{Path(args.checkpoint).name} ({args.network})"
        value_tanh = args.value_tanh

    print(f"Exporting: {label}")
    fp32     = _export_fp32(agent, out_dir / "model.onnx")
    fp32_kb  = fp32.stat().st_size // 1024
    int8_kb  = None
    deployed = "model.onnx"

    if args.quantize:
        int8    = _quantize(fp32)
        int8_kb = int8.stat().st_size // 1024
        deployed = "model_int8.onnx"

    _write_config(out_dir, label, deployed, fp32_kb, int8_kb, value_tanh=value_tanh)
    print(f"\nDone. Browser will load: docs/models/{deployed}")


if __name__ == "__main__":
    main()
