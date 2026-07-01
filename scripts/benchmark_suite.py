"""Reproducible, fail-closed benchmark panel for policy/value checkpoints.

Examples:
    python -m scripts.benchmark_suite --candidate arena:21@hof \
        --anchors lottery,nn_big8,winblock,center,first \
        --candidate-sims 0,25,100 --oracle-sims 400 \
        --openings standard --out results/arena-21

    python -m scripts.benchmark_suite --candidate candidate.json \
        --anchors winblock,center,first --candidate-sims 0,25 \
        --oracle-sims 100 --out results/candidate

Manifest schema:
    {
      "schema_version": 1,
      "label": "candidate-name",
      "checkpoint": "relative/or/absolute.pt",
      "sha256": "optional expected digest",
      "architecture": {
        "type": "conv_policy_value_v1",
        "input_channels": 7,
        "conv_channels": [32, 64, 64],
        "fc_hidden_sizes": [256, 512, 256],
        "output_size": 81
      }
    }
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from agents import get_agent
from agents.agent_base import (
    Agent,
    ModelConfigCNN,
    board_to_tensor,
    board_to_tensor_from_gamestate,
)
from agents.mcts import MCTSAgent
from agents.neural_net_agent_2 import ConfigurableNN
from agents.neural_net_agent_3 import ConvNet
from agents.neural_net_agent_pg import NeuralNetAgentPG
from engine.constants import DRAW, O, X
from engine.game import GameState, _CPP_ENGINE
from engine.rules import rule_utl_valid_moves


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARENA_STATE = REPO_ROOT / "models" / "arena" / "arena_state.json"
DEFAULT_OPENINGS = REPO_ROOT / "benchmarks" / "openings_standard.json"
RULESET_ID = "ultimate-tic-tac-toe-repo-v1"
MANIFEST_ARCH_TYPE = "conv_policy_value_v1"

ANCHOR_ALIASES = {
    "lottery": "lottery_no_touchy",
    "lottery_no_touchy": "lottery_no_touchy",
    "nn_big8": "nn_big_8",
    "nn_big_8": "nn_big_8",
    "winblock": "winblock",
    "center": "center",
    "first": "first",
    "random": "random",
    "last": "last",
}

ANCHOR_CHECKPOINTS = {
    "lottery_no_touchy": (
        REPO_ROOT
        / "models"
        / "lottery"
        / "4096-3072-2048-1024-512-256-81"
        / "several_weeks_no_touchy.pt"
    ),
    "nn_big_8": (
        REPO_ROOT
        / "models"
        / "big_8_layer"
        / "256-512-1024-2048-2048-1024-512-256-81"
        / "best.pt"
    ),
}

class BenchmarkError(RuntimeError):
    """A preflight or gameplay condition that invalidates the benchmark."""


@dataclass(frozen=True)
class Architecture:
    conv_channels: list[int]
    fc_hidden_sizes: list[int]
    input_channels: int = 7
    output_size: int = 81
    type: str = MANIFEST_ARCH_TYPE

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], source: str) -> "Architecture":
        if not isinstance(raw, dict):
            raise BenchmarkError(f"{source}: architecture must be an object")
        arch_type = raw.get("type", MANIFEST_ARCH_TYPE)
        if arch_type != MANIFEST_ARCH_TYPE:
            raise BenchmarkError(
                f"{source}: unknown architecture type {arch_type!r}; "
                f"expected {MANIFEST_ARCH_TYPE!r}"
            )
        conv = raw.get("conv_channels")
        fc = raw.get("fc_hidden_sizes")
        if not _positive_int_list(conv) or not _positive_int_list(fc):
            raise BenchmarkError(
                f"{source}: conv_channels and fc_hidden_sizes must be non-empty "
                "lists of positive integers"
            )
        input_channels = raw.get("input_channels", 7)
        output_size = raw.get("output_size", 81)
        if input_channels != 7 or output_size != 81:
            raise BenchmarkError(
                f"{source}: unsupported input/output shape "
                f"({input_channels}, {output_size}); expected (7, 81)"
            )
        return cls(
            conv_channels=list(conv),
            fc_hidden_sizes=list(fc),
            input_channels=input_channels,
            output_size=output_size,
            type=arch_type,
        )


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    checkpoint: Path
    architecture: Architecture
    source: str
    expected_sha256: str | None = None
    arena_id: int | None = None
    arena_name: str | None = None
    arena_selector: str | None = None


@dataclass(frozen=True)
class Opening:
    id: str
    moves: list[int]


LOTTERY_CONV_CHANNELS = [512] * 135
LOTTERY_FC_HIDDEN_SIZES = [4096, 3072, 2048, 1024, 512, 256]
BIG8_HIDDEN_SIZES = [256, 512, 1024, 2048, 2048, 1024, 512, 256]


class _FlatPolicyConfig:
    input_size = 81
    output_size = 81

    def __init__(self, hidden_sizes: list[int]):
        self.hidden_sizes = hidden_sizes
        self.activation = torch.nn.functional.relu


def _positive_int_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value)
    )


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_arena_name(name: str) -> str:
    safe = name.lower().replace(" ", "-")
    safe = "".join(c for c in safe if c.isalnum() or c in "-_")
    return safe.strip("-_")


def _repo_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_json(path: Path, description: str) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise BenchmarkError(f"{description} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{description} is invalid JSON: {path}: {exc}") from exc


def _resolve_hof(record: dict[str, Any], state_path: Path) -> Path:
    safe_name = _safe_arena_name(str(record["name"]))
    hof_dir = state_path.parent / "hall_of_fame"
    matches = sorted(hof_dir.glob(f"{safe_name}_elo*.pt"))
    if not matches:
        raise BenchmarkError(
            f"arena:{record['id']}@hof: no Hall-of-Fame checkpoint matching "
            f"{safe_name}_elo*.pt in {hof_dir}"
        )
    if len(matches) != 1:
        names = ", ".join(p.name for p in matches)
        raise BenchmarkError(
            f"arena:{record['id']}@hof: expected exactly one checkpoint; found {names}"
        )
    return matches[0].resolve()


def _resolve_latest(record: dict[str, Any]) -> Path:
    model_dir = _repo_path(record["model_dir"])
    versions: list[tuple[int, Path]] = []
    for path in model_dir.glob("version_*.pt"):
        match = re.fullmatch(r"version_(\d+)\.pt", path.name)
        if match:
            versions.append((int(match.group(1)), path))
    if not versions:
        raise BenchmarkError(
            f"arena:{record['id']}@latest: no version_*.pt checkpoints in {model_dir}"
        )
    return max(versions, key=lambda item: item[0])[1].resolve()


def resolve_arena_candidate(spec: str, state_path: Path) -> CandidateSpec:
    match = re.fullmatch(r"arena:(\d+)@(hof|latest)", spec.strip(), re.IGNORECASE)
    if not match:
        raise BenchmarkError(
            f"invalid Arena candidate {spec!r}; expected arena:<id>@hof or "
            "arena:<id>@latest"
        )
    arena_id = int(match.group(1))
    selector = match.group(2).lower()
    state = _load_json(state_path, "Arena state")
    records = [record for record in state.get("agents", []) if record.get("id") == arena_id]
    if len(records) != 1:
        raise BenchmarkError(
            f"{spec}: expected exactly one Arena record for id {arena_id}; "
            f"found {len(records)}"
        )
    record = records[0]
    architecture = Architecture.from_mapping(record, f"Arena record {arena_id}")
    checkpoint = (
        _resolve_hof(record, state_path)
        if selector == "hof"
        else _resolve_latest(record)
    )
    return CandidateSpec(
        label=f"arena-{arena_id}-{record['name']}-{selector}",
        checkpoint=checkpoint,
        architecture=architecture,
        source=spec,
        arena_id=arena_id,
        arena_name=str(record["name"]),
        arena_selector=selector,
    )


def resolve_manifest_candidate(path: Path) -> CandidateSpec:
    raw = _load_json(path, "Candidate manifest")
    if raw.get("schema_version") != 1:
        raise BenchmarkError(
            f"{path}: unsupported or missing schema_version; expected 1"
        )
    label = raw.get("label")
    if not isinstance(label, str) or not label.strip():
        raise BenchmarkError(f"{path}: manifest label must be a non-empty string")
    checkpoint_raw = raw.get("checkpoint")
    if not isinstance(checkpoint_raw, str) or not checkpoint_raw.strip():
        raise BenchmarkError(f"{path}: manifest checkpoint must be a non-empty string")
    checkpoint = Path(checkpoint_raw).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = path.parent / checkpoint
    architecture = Architecture.from_mapping(raw.get("architecture"), str(path))
    expected = raw.get("sha256")
    if expected is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", str(expected)):
        raise BenchmarkError(f"{path}: sha256 must be exactly 64 hexadecimal characters")
    return CandidateSpec(
        label=label.strip(),
        checkpoint=checkpoint.resolve(),
        architecture=architecture,
        source=str(path),
        expected_sha256=str(expected).lower() if expected else None,
    )


def resolve_candidate(spec: str, state_path: Path = DEFAULT_ARENA_STATE) -> CandidateSpec:
    if spec.lower().startswith("arena:"):
        return resolve_arena_candidate(spec, state_path.resolve())
    manifest = _repo_path(spec)
    if manifest.suffix.lower() != ".json":
        raise BenchmarkError(
            f"candidate {spec!r} is neither an Arena selector nor a .json manifest"
        )
    return resolve_manifest_candidate(manifest)


def _unwrap_checkpoint(checkpoint: Any, path: Path) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise BenchmarkError(f"{path}: checkpoint is not a state-dict object")
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "initial_state_dict" in checkpoint:
        state = checkpoint["initial_state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict) or not state:
        raise BenchmarkError(f"{path}: checkpoint has no usable state_dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise BenchmarkError(f"{path}: state_dict must map string keys to tensors")
    return state


def validate_checkpoint(spec: CandidateSpec) -> dict[str, Any]:
    path = spec.checkpoint
    if not path.is_file():
        raise BenchmarkError(f"{spec.source}: checkpoint not found: {path}")
    digest = sha256_file(path)
    if spec.expected_sha256 and digest != spec.expected_sha256:
        raise BenchmarkError(
            f"{path}: SHA-256 mismatch; expected {spec.expected_sha256}, got {digest}"
        )

    cfg = ModelConfigCNN(
        conv_channels=spec.architecture.conv_channels,
        fc_hidden_sizes=spec.architecture.fc_hidden_sizes,
        input_channels=spec.architecture.input_channels,
        output_size=spec.architecture.output_size,
        learning_rate=1e-4,
        label="benchmark-preflight",
        model_dir=str(path.parent),
        device=torch.device("cpu"),
    )
    expected_model = ConvNet(cfg)
    expected = expected_model.state_dict()
    parameter_count = sum(parameter.numel() for parameter in expected_model.parameters())
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise BenchmarkError(f"{path}: checkpoint could not be loaded safely: {exc}") from exc
    state = _unwrap_checkpoint(checkpoint, path)

    if not any(key.startswith("value_head.") for key in state):
        raise BenchmarkError(
            f"{path}: policy-only checkpoint; a value head is required for the MCTS panel"
        )
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:6]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:6]}")
        if mismatched:
            shape_details = [
                f"{key}:{tuple(state[key].shape)}!={tuple(expected[key].shape)}"
                for key in mismatched[:6]
            ]
            details.append(f"shape_mismatch={shape_details}")
        raise BenchmarkError(
            f"{path}: checkpoint does not exactly match declared architecture: "
            + "; ".join(details)
        )
    del checkpoint, state, expected_model
    return {
        "path": _display_path(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "architecture": asdict(spec.architecture),
        "parameter_count": parameter_count,
    }


def _build_base_candidate(
    spec: CandidateSpec, device: torch.device, tactical: bool = False
) -> NeuralNetAgentPG:
    cfg = ModelConfigCNN(
        conv_channels=spec.architecture.conv_channels,
        fc_hidden_sizes=spec.architecture.fc_hidden_sizes,
        input_channels=spec.architecture.input_channels,
        output_size=spec.architecture.output_size,
        learning_rate=1e-4,
        label="benchmark",
        model_dir=str(spec.checkpoint.parent),
        device=device,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        agent = NeuralNetAgentPG(
            cfg=cfg, model_path=str(spec.checkpoint), tactical=tactical
        )
    agent.name = f"{spec.label}-{'tactical' if tactical else 'raw'}"
    agent.set_eval(True)
    return agent


def build_candidate_modes(
    spec: CandidateSpec,
    device: torch.device,
    simulation_ladder: Iterable[int],
    include_tactical: bool = True,
    c_puct: float = 1.5,
) -> dict[str, Any]:
    sims = list(simulation_ladder)
    if not sims or any(not isinstance(value, int) or value < 0 for value in sims):
        raise BenchmarkError("candidate simulation ladder must contain integers >= 0")
    if len(sims) != len(set(sims)):
        raise BenchmarkError("candidate simulation ladder contains duplicate values")
    modes: dict[str, Any] = {}
    raw = _build_base_candidate(spec, device, tactical=False)
    if 0 in sims:
        modes["raw"] = raw
    if include_tactical:
        modes["tactical"] = _build_base_candidate(spec, device, tactical=True)
    for count in sims:
        if count > 0:
            modes[f"mcts_{count}"] = MCTSAgent(
                raw, n_sims=count, c_puct=c_puct, temperature=0.0
            )
    if not modes:
        raise BenchmarkError("candidate mode selection produced no modes")
    return modes


def parse_anchor_names(raw: str) -> list[str]:
    requested = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not requested:
        raise BenchmarkError("at least one anchor is required")
    canonical = []
    for name in requested:
        try:
            canonical.append(ANCHOR_ALIASES[name])
        except KeyError as exc:
            supported = ", ".join(sorted(ANCHOR_ALIASES))
            raise BenchmarkError(f"unknown anchor {name!r}; supported: {supported}") from exc
    duplicates = sorted({name for name in canonical if canonical.count(name) > 1})
    if duplicates:
        raise BenchmarkError(
            "duplicate anchor label(s) after alias resolution: " + ", ".join(duplicates)
        )
    return canonical


def _code_provenance(path: Path) -> dict[str, Any]:
    return {
        "type": "code",
        "path": _display_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


class _FrozenConvPolicyAgent(Agent):
    """Strict loader for a legacy ConvNet policy checkpoint with no value head."""

    def __init__(
        self,
        label: str,
        checkpoint: Path,
        conv_channels: list[int],
        fc_hidden_sizes: list[int],
        device: torch.device,
    ):
        super().__init__(label)
        self.device = device
        cfg = ModelConfigCNN(
            conv_channels=conv_channels,
            fc_hidden_sizes=fc_hidden_sizes,
            learning_rate=1e-4,
            label=f"benchmark-anchor-{label}",
            model_dir=str(checkpoint.parent),
            device=device,
        )
        self.model = ConvNet(cfg).to(device)
        try:
            loaded = torch.load(
                checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
        except Exception as exc:
            raise BenchmarkError(f"anchor {label} could not be loaded: {exc}") from exc
        state = _unwrap_checkpoint(loaded, checkpoint)
        expected = self.model.state_dict()
        policy_keys = {key for key in expected if not key.startswith("value_head.")}
        missing = sorted(policy_keys - set(state))
        unexpected = sorted(set(state) - policy_keys)
        mismatched = sorted(
            key
            for key in policy_keys & set(state)
            if tuple(expected[key].shape) != tuple(state[key].shape)
        )
        if missing or unexpected or mismatched:
            raise BenchmarkError(
                f"anchor {label} does not match its frozen policy architecture: "
                f"missing={missing[:4]} unexpected={unexpected[:4]} "
                f"shape_mismatch={mismatched[:4]}"
            )
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def select_move(self, gamestate: GameState) -> int:
        valid = rule_utl_valid_moves(
            gamestate.board, gamestate.last_move, gamestate.mini_winners
        )
        x = board_to_tensor_from_gamestate(gamestate, v_computed=valid).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        logits = logits.squeeze(0) if logits.dim() == 2 else logits
        masked = torch.full_like(logits, float("-inf"))
        valid_t = torch.tensor(valid, dtype=torch.long, device=logits.device)
        masked.scatter_(0, valid_t, logits[valid_t])
        return int(torch.argmax(masked).item())


class _FrozenMLPPolicyAgent(Agent):
    """Strict, inference-only loader for a frozen flat-board policy."""

    def __init__(
        self,
        label: str,
        checkpoint: Path,
        hidden_sizes: list[int],
        device: torch.device,
    ):
        super().__init__(label)
        self.device = device

        self.model = ConfigurableNN(_FlatPolicyConfig(hidden_sizes)).to(device)
        try:
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise BenchmarkError(f"anchor {label} could not be loaded: {exc}") from exc
        state = _unwrap_checkpoint(loaded, checkpoint)
        expected = self.model.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatched = sorted(
            key
            for key in set(expected) & set(state)
            if tuple(expected[key].shape) != tuple(state[key].shape)
        )
        if missing or unexpected or mismatched:
            raise BenchmarkError(
                f"anchor {label} does not match its frozen policy architecture: "
                f"missing={missing[:4]} unexpected={unexpected[:4]} "
                f"shape_mismatch={mismatched[:4]}"
            )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def select_move(self, gamestate: GameState) -> int:
        valid = rule_utl_valid_moves(
            gamestate.board, gamestate.last_move, gamestate.mini_winners
        )
        x = board_to_tensor(gamestate.board).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        masked = torch.full_like(logits, float("-inf"))
        valid_t = torch.tensor(valid, dtype=torch.long, device=logits.device)
        masked.scatter_(0, valid_t, logits[valid_t])
        return int(torch.argmax(masked).item())


def build_anchors(
    names: list[str], device: torch.device
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    agents: dict[str, Any] = {}
    provenance: list[dict[str, Any]] = []
    for name in names:
        checkpoint = ANCHOR_CHECKPOINTS.get(name)
        if checkpoint is not None and not checkpoint.is_file():
            raise BenchmarkError(f"anchor {name}: checkpoint not found: {checkpoint}")
        if name == "lottery_no_touchy":
            agent = _FrozenConvPolicyAgent(
                name,
                checkpoint,
                LOTTERY_CONV_CHANNELS,
                LOTTERY_FC_HIDDEN_SIZES,
                device,
            )
        elif name == "nn_big_8":
            agent = _FrozenMLPPolicyAgent(
                name, checkpoint, BIG8_HIDDEN_SIZES, device
            )
        else:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    agent = get_agent(name)
            except Exception as exc:
                raise BenchmarkError(f"anchor {name} failed to load: {exc}") from exc
            if hasattr(agent, "model"):
                agent.model.to(device)
            if hasattr(agent, "device"):
                agent.device = device
        if hasattr(agent, "set_eval"):
            agent.set_eval(True)
        agents[name] = agent
        if checkpoint is not None:
            provenance.append(
                {
                    "label": name,
                    "type": "checkpoint",
                    "path": _display_path(checkpoint),
                    "sha256": sha256_file(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "class": type(agent).__name__,
                }
            )
        else:
            provenance.append(
                {
                    "label": name,
                    "class": type(agent).__name__,
                    **_code_provenance(REPO_ROOT / "agents" / (
                        "random_agent.py" if name == "random" else "deterministics.py"
                    )),
                }
            )
    return agents, provenance


def preflight_anchors(names: list[str]) -> list[dict[str, Any]]:
    """Validate frozen anchor bytes and shapes without allocating their full models."""
    checked = []
    for name in names:
        checkpoint = ANCHOR_CHECKPOINTS.get(name)
        if checkpoint is None:
            source = REPO_ROOT / "agents" / (
                "random_agent.py" if name == "random" else "deterministics.py"
            )
            checked.append({"label": name, **_code_provenance(source)})
            continue
        if not checkpoint.is_file():
            raise BenchmarkError(f"anchor {name}: checkpoint not found: {checkpoint}")
        try:
            loaded = torch.load(
                checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
        except Exception as exc:
            raise BenchmarkError(f"anchor {name} could not be loaded safely: {exc}") from exc
        state = _unwrap_checkpoint(loaded, checkpoint)
        with torch.device("meta"):
            if name == "lottery_no_touchy":
                cfg = ModelConfigCNN(
                    conv_channels=LOTTERY_CONV_CHANNELS,
                    fc_hidden_sizes=LOTTERY_FC_HIDDEN_SIZES,
                    learning_rate=1e-4,
                    label="benchmark-anchor-preflight",
                    model_dir=str(checkpoint.parent),
                    device=torch.device("meta"),
                )
                expected_all = ConvNet(cfg).state_dict()
                expected = {
                    key: value
                    for key, value in expected_all.items()
                    if not key.startswith("value_head.")
                }
            else:
                expected = ConfigurableNN(
                    _FlatPolicyConfig(BIG8_HIDDEN_SIZES)
                ).state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatched = sorted(
            key
            for key in set(expected) & set(state)
            if tuple(expected[key].shape) != tuple(state[key].shape)
        )
        if missing or unexpected or mismatched:
            raise BenchmarkError(
                f"anchor {name} does not match its frozen architecture: "
                f"missing={missing[:4]} unexpected={unexpected[:4]} "
                f"shape_mismatch={mismatched[:4]}"
            )
        checked.append(
            {
                "label": name,
                "type": "checkpoint",
                "path": _display_path(checkpoint),
                "sha256": sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
            }
        )
    return checked


def load_openings(spec: str, max_openings: int | None = None) -> tuple[list[Opening], dict[str, Any]]:
    path = DEFAULT_OPENINGS if spec == "standard" else _repo_path(spec)
    raw = _load_json(path, "Opening suite")
    if raw.get("schema_version") != 1:
        raise BenchmarkError(f"{path}: opening schema_version must be 1")
    if raw.get("ruleset") != RULESET_ID:
        raise BenchmarkError(
            f"{path}: opening ruleset {raw.get('ruleset')!r} does not match {RULESET_ID!r}"
        )
    entries = raw.get("openings")
    if not isinstance(entries, list) or not entries:
        raise BenchmarkError(f"{path}: openings must be a non-empty list")
    if max_openings is not None:
        if max_openings <= 0:
            raise BenchmarkError("--max-openings must be greater than zero")
        entries = entries[:max_openings]
    openings: list[Opening] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise BenchmarkError(f"{path}: opening {index} must be an object")
        opening_id = entry.get("id")
        moves = entry.get("moves")
        if not isinstance(opening_id, str) or not opening_id:
            raise BenchmarkError(f"{path}: opening {index} has no valid id")
        if opening_id in seen:
            raise BenchmarkError(f"{path}: duplicate opening id {opening_id!r}")
        if not isinstance(moves, list) or any(
            not isinstance(move, int) or isinstance(move, bool) for move in moves
        ):
            raise BenchmarkError(f"{path}: opening {opening_id!r} moves must be integers")
        game = GameState()
        for ply, move in enumerate(moves):
            valid, _ = game.make_move(move)
            if not valid:
                raise BenchmarkError(
                    f"{path}: opening {opening_id!r} has illegal move {move} at ply {ply}"
                )
        if game.is_over():
            raise BenchmarkError(f"{path}: opening {opening_id!r} is already terminal")
        seen.add(opening_id)
        openings.append(Opening(opening_id, list(moves)))
    return openings, {
        "name": raw.get("name", path.stem),
        "path": _display_path(path),
        "sha256": sha256_file(path),
        "count": len(openings),
        "ids": [opening.id for opening in openings],
        "ruleset": raw["ruleset"],
    }


def _stable_game_seed(
    seed: int, mode: str, opponent: str, opening: str, candidate_side: int
) -> int:
    text = f"{seed}|{mode}|{opponent}|{opening}|{candidate_side}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**63))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed % (2**63))


def configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _clear(agent: Any) -> None:
    clear = getattr(agent, "clear_history", None)
    if callable(clear):
        clear()
    inner = getattr(agent, "nn", None)
    inner_clear = getattr(inner, "clear_history", None)
    if callable(inner_clear):
        inner_clear()


def play_opening_game(
    candidate: Any,
    opponent: Any,
    opening: Opening,
    candidate_side: int,
    game_seed: int,
    candidate_label: str = "candidate",
    opponent_label: str = "opponent",
) -> int:
    seed_everything(game_seed)
    game = GameState()
    for move in opening.moves:
        valid, _ = game.make_move(move)
        if not valid:
            raise BenchmarkError(
                f"opening {opening.id}: prevalidated move {move} became illegal"
            )
    _clear(candidate)
    _clear(opponent)
    ply = len(opening.moves)
    while not game.is_over():
        mover = candidate if game.player == candidate_side else opponent
        mover_label = candidate_label if mover is candidate else opponent_label
        try:
            move = mover.select_move(game)
        except Exception as exc:
            raise BenchmarkError(
                f"{mover_label} failed selecting a move in opening {opening.id}, "
                f"ply {ply}: {exc}"
            ) from exc
        if isinstance(move, np.integer):
            move = int(move)
        if not isinstance(move, int) or isinstance(move, bool):
            raise BenchmarkError(
                f"{mover_label} returned non-integer move {move!r} in opening "
                f"{opening.id}, ply {ply}"
            )
        valid, _ = game.make_move(move)
        if not valid:
            raise BenchmarkError(
                f"{mover_label} returned illegal move {move} in opening "
                f"{opening.id}, ply {ply}"
            )
        ply += 1
    return game.winner


def wilson_score_interval(points: float, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    proportion = points / games
    denominator = 1.0 + z * z / games
    center = (proportion + z * z / (2.0 * games)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / games
            + z * z / (4.0 * games * games)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def run_pair(
    candidate: Any,
    opponent: Any,
    openings: list[Opening],
    seed: int,
    mode_label: str,
    opponent_label: str,
) -> dict[str, Any]:
    wins = draws = losses = 0
    started = time.perf_counter()
    for opening in openings:
        for candidate_side in (X, O):
            game_seed = _stable_game_seed(
                seed, mode_label, opponent_label, opening.id, candidate_side
            )
            winner = play_opening_game(
                candidate,
                opponent,
                opening,
                candidate_side,
                game_seed,
                candidate_label=mode_label,
                opponent_label=opponent_label,
            )
            if winner == candidate_side:
                wins += 1
            elif winner == DRAW:
                draws += 1
            else:
                losses += 1
    elapsed = time.perf_counter() - started
    games = wins + draws + losses
    points = wins + 0.5 * draws
    low, high = wilson_score_interval(points, games)
    return {
        "mode": mode_label,
        "opponent": opponent_label,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": games,
        "openings": len(openings),
        "score": points / games if games else 0.0,
        "score_ci95": [low, high],
        "seconds_per_game": elapsed / games if games else 0.0,
        "elapsed_seconds": elapsed,
    }


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
        except Exception:
            return "unknown"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": status not in ("", "unknown"),
    }


def _device_metadata(device: torch.device) -> dict[str, Any]:
    data = {
        "requested": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if device.type == "cuda":
        data["cuda"] = torch.version.cuda
        data["name"] = torch.cuda.get_device_name(device)
    else:
        data["name"] = platform.processor() or "CPU"
    return data


def _ruleset_metadata() -> dict[str, Any]:
    return {
        "id": RULESET_ID,
        "engine": "cpp-wrapper" if _CPP_ENGINE else "python",
        "files": {
            "engine/rules.py": sha256_file(REPO_ROOT / "engine" / "rules.py"),
            "engine/game.py": sha256_file(REPO_ROOT / "engine" / "game.py"),
            "engine/constants.py": sha256_file(REPO_ROOT / "engine" / "constants.py"),
        },
    }


def validate_report_payload(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "created_utc",
        "candidate",
        "anchors",
        "openings",
        "settings",
        "git",
        "device",
        "ruleset",
        "results",
        "total_elapsed_seconds",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise BenchmarkError("report payload missing metadata: " + ", ".join(missing))
    candidate_required = {"path", "sha256", "bytes", "architecture", "parameter_count"}
    candidate_missing = sorted(candidate_required - set(payload["candidate"]))
    if candidate_missing:
        raise BenchmarkError(
            "candidate report metadata missing: " + ", ".join(candidate_missing)
        )


def render_markdown(payload: dict[str, Any]) -> str:
    candidate = payload["candidate"]
    settings = payload["settings"]
    lines = [
        f"# Benchmark: {candidate['label']}",
        "",
        f"- Created UTC: `{payload['created_utc']}`",
        f"- Checkpoint: `{candidate['path']}`",
        f"- SHA-256: `{candidate['sha256']}`",
        f"- Bytes: `{candidate['bytes']:,}`",
        f"- Parameters: `{candidate['parameter_count']:,}`",
        f"- Architecture: conv `{candidate['architecture']['conv_channels']}`, "
        f"FC `{candidate['architecture']['fc_hidden_sizes']}`",
        f"- Git commit: `{payload['git']['commit']}` "
        f"({'dirty' if payload['git']['dirty'] else 'clean'})",
        f"- Device: `{payload['device']['requested']}` / `{payload['device']['name']}`",
        f"- Seed: `{settings['seed']}`",
        f"- Ruleset: `{payload['ruleset']['id']}`",
        f"- Openings: `{payload['openings']['name']}` "
        f"({payload['openings']['count']}, SHA-256 `{payload['openings']['sha256']}`)",
        f"- MCTS: `c_puct={settings['c_puct']}`, "
        f"candidate sims `{settings['candidate_sims']}`, "
        f"oracle sims `{settings['oracle_sims']}`",
        "",
        "| Candidate mode | Opponent | W-D-L | Score | 95% CI | sec/game |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        low, high = result["score_ci95"]
        lines.append(
            f"| {result['mode']} | {result['opponent']} | "
            f"{result['wins']}-{result['draws']}-{result['losses']} | "
            f"{result['score']:.3f} | {low:.3f}-{high:.3f} | "
            f"{result['seconds_per_game']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Total elapsed: `{payload['total_elapsed_seconds']:.2f}s`.",
            "",
            "The interval is a 95% Wilson interval applied to game score "
            "(win=1, draw=0.5, loss=0). MCTS is empirical, not proof of optimal play.",
            "",
            "## Anchor provenance",
            "",
        ]
    )
    for anchor in payload["anchors"]:
        lines.append(
            f"- `{anchor['label']}`: `{anchor['type']}` "
            f"`{anchor['path']}` SHA-256 `{anchor['sha256']}`"
        )
    return "\n".join(lines) + "\n"


def write_reports(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    validate_report_payload(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    markdown_path = output_dir / "report.md"
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(payload))
    return json_path, markdown_path


def _parse_sims(raw: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise BenchmarkError("--candidate-sims must be comma-separated integers") from exc
    if not values or any(value < 0 for value in values):
        raise BenchmarkError("--candidate-sims must contain one or more integers >= 0")
    if len(values) != len(set(values)):
        raise BenchmarkError("--candidate-sims contains duplicate values")
    return values


def _resolve_device(raw: str) -> torch.device:
    if raw:
        device = torch.device(raw)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BenchmarkError("CUDA was requested but is unavailable")
    return device


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible raw/tactical/MCTS benchmark panel."
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="arena:<id>@hof, arena:<id>@latest, or a checkpoint manifest JSON",
    )
    parser.add_argument(
        "--arena-state",
        default=str(DEFAULT_ARENA_STATE),
        help="Arena state JSON used by arena:<id> selectors",
    )
    parser.add_argument(
        "--anchors",
        default="lottery,nn_big8,winblock,center,first",
        help="Comma-separated frozen anchors",
    )
    parser.add_argument(
        "--candidate-sims",
        default="0,25,100",
        help="Candidate MCTS simulation ladder; 0 means raw policy",
    )
    parser.add_argument(
        "--oracle-sims",
        type=int,
        default=400,
        help="Deep MCTS opponent simulations; 0 disables the oracle",
    )
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument(
        "--tactical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the 1-ply tactical candidate mode (default: enabled)",
    )
    parser.add_argument(
        "--openings",
        default="standard",
        help="'standard' or a compatible opening-suite JSON",
    )
    parser.add_argument(
        "--max-openings",
        type=int,
        default=None,
        help="Use only the first N openings (intended for smoke tests)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="", help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Resolve and validate inputs without playing games or writing reports",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    started = time.perf_counter()
    state_path = _repo_path(args.arena_state)
    candidate_spec = resolve_candidate(args.candidate, state_path)
    candidate_metadata = validate_checkpoint(candidate_spec)
    candidate_metadata.update(
        {
            "label": candidate_spec.label,
            "source": candidate_spec.source,
            "arena_id": candidate_spec.arena_id,
            "arena_name": candidate_spec.arena_name,
            "arena_selector": candidate_spec.arena_selector,
        }
    )
    simulation_ladder = _parse_sims(args.candidate_sims)
    if args.oracle_sims < 0:
        raise BenchmarkError("--oracle-sims must be >= 0")
    if args.c_puct <= 0:
        raise BenchmarkError("--c-puct must be greater than zero")
    anchor_names = parse_anchor_names(args.anchors)
    openings, opening_metadata = load_openings(args.openings, args.max_openings)
    device = _resolve_device(args.device)

    print(f"Candidate: {candidate_spec.label}")
    print(f"Checkpoint: {candidate_metadata['path']}")
    print(f"SHA-256: {candidate_metadata['sha256']}")
    print(
        "Architecture: "
        f"conv={candidate_spec.architecture.conv_channels} "
        f"fc={candidate_spec.architecture.fc_hidden_sizes}"
    )
    print(f"Openings: {len(openings)} (two games each)")
    if args.preflight_only:
        for anchor in preflight_anchors(anchor_names):
            print(
                f"Anchor: {anchor['label']} "
                f"SHA-256={anchor['sha256']} bytes={anchor['bytes']}"
            )
        print("Preflight passed; no games were run.")
        return None

    configure_determinism()
    seed_everything(args.seed)
    candidate_modes = build_candidate_modes(
        candidate_spec,
        device,
        simulation_ladder,
        include_tactical=args.tactical,
        c_puct=args.c_puct,
    )
    anchors, anchor_provenance = build_anchors(anchor_names, device)
    if candidate_spec.label in anchors or candidate_spec.label in candidate_modes:
        raise BenchmarkError(f"duplicate label: {candidate_spec.label}")
    if args.oracle_sims > 0:
        raw_inner = next(
            (
                agent
                for label, agent in candidate_modes.items()
                if label == "raw"
            ),
            None,
        )
        if raw_inner is None:
            raw_inner = _build_base_candidate(candidate_spec, device, tactical=False)
        oracle_label = f"oracle_mcts_{args.oracle_sims}"
        if oracle_label in anchors:
            raise BenchmarkError(f"duplicate label: {oracle_label}")
        anchors[oracle_label] = MCTSAgent(
            raw_inner,
            n_sims=args.oracle_sims,
            c_puct=args.c_puct,
            temperature=0.0,
        )
        anchor_provenance.append(
            {
                "label": oracle_label,
                "type": "mcts-wrapper",
                "path": candidate_metadata["path"],
                "sha256": candidate_metadata["sha256"],
                "bytes": candidate_metadata["bytes"],
                "class": "MCTSAgent",
                "n_sims": args.oracle_sims,
                "c_puct": args.c_puct,
            }
        )

    results = []
    total_pairs = len(candidate_modes) * len(anchors)
    pair_index = 0
    for mode_label, candidate in candidate_modes.items():
        for opponent_label, opponent in anchors.items():
            pair_index += 1
            print(
                f"[{pair_index}/{total_pairs}] {mode_label} vs {opponent_label} "
                f"({len(openings) * 2} games)"
            )
            result = run_pair(
                candidate,
                opponent,
                openings,
                args.seed,
                mode_label,
                opponent_label,
            )
            results.append(result)
            low, high = result["score_ci95"]
            print(
                f"  {result['wins']}W {result['draws']}D {result['losses']}L "
                f"score={result['score']:.3f} CI={low:.3f}-{high:.3f} "
                f"{result['seconds_per_game']:.3f}s/game"
            )

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate_metadata,
        "anchors": anchor_provenance,
        "openings": opening_metadata,
        "settings": {
            "seed": args.seed,
            "candidate_sims": simulation_ladder,
            "include_tactical": args.tactical,
            "oracle_sims": args.oracle_sims,
            "c_puct": args.c_puct,
            "games_per_pair": len(openings) * 2,
            "opening_selector": args.openings,
            "max_openings": args.max_openings,
        },
        "git": _git_metadata(),
        "device": _device_metadata(device),
        "ruleset": _ruleset_metadata(),
        "results": results,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    json_path, markdown_path = write_reports(_repo_path(args.out), payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except BenchmarkError as exc:
        print(f"BENCHMARK FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
