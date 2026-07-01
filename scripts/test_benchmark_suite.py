"""Focused tests for scripts.benchmark_suite.

Run with:
    python -m scripts.test_benchmark_suite
    pytest scripts/test_benchmark_suite.py
"""

from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path

import torch

from agents.agent_base import Agent, ModelConfigCNN
from agents.mcts import MCTSAgent
from agents.neural_net_agent_3 import ConvNet
from engine.rules import rule_utl_valid_moves
from scripts.benchmark_suite import (
    Architecture,
    BenchmarkError,
    CandidateSpec,
    Opening,
    build_candidate_modes,
    parse_anchor_names,
    play_opening_game,
    resolve_candidate,
    run_pair,
    validate_checkpoint,
    validate_report_payload,
)


def _write_checkpoint(root: Path, architecture: Architecture, policy_only: bool = False) -> Path:
    cfg = ModelConfigCNN(
        conv_channels=architecture.conv_channels,
        fc_hidden_sizes=architecture.fc_hidden_sizes,
        input_channels=architecture.input_channels,
        output_size=architecture.output_size,
        model_dir=str(root),
        device=torch.device("cpu"),
    )
    state = ConvNet(cfg).state_dict()
    if policy_only:
        state = {key: value for key, value in state.items() if not key.startswith("value_head.")}
    path = root / "candidate.pt"
    torch.save({"state_dict": state}, path)
    return path


def _small_spec(root: Path, policy_only: bool = False) -> CandidateSpec:
    architecture = Architecture(conv_channels=[4], fc_hidden_sizes=[8])
    return CandidateSpec(
        label="tiny-test",
        checkpoint=_write_checkpoint(root, architecture, policy_only=policy_only),
        architecture=architecture,
        source="test",
    )


def _assert_raises(expected, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except expected:
        return
    raise AssertionError(f"Expected {expected.__name__}")


def test_arena_hof_resolution_uses_state_architecture():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        arena_dir = root / "models" / "arena"
        hof_dir = arena_dir / "hall_of_fame"
        hof_dir.mkdir(parents=True)
        checkpoint = hof_dir / "named-agent_elo1234.pt"
        checkpoint.write_bytes(b"checkpoint")
        state = {
            "agents": [
                {
                    "id": 7,
                    "name": "Named Agent",
                    "conv_channels": [4, 8],
                    "fc_hidden_sizes": [16],
                    "model_dir": str(arena_dir / "named-agent"),
                }
            ]
        }
        state_path = arena_dir / "arena_state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        resolved = resolve_candidate("arena:7@hof", state_path)
        assert resolved.checkpoint == checkpoint.resolve()
        assert resolved.architecture.conv_channels == [4, 8]
        assert resolved.architecture.fc_hidden_sizes == [16]


def test_arena_latest_resolution_selects_highest_numeric_version():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        arena_dir = root / "models" / "arena"
        model_dir = arena_dir / "agent"
        model_dir.mkdir(parents=True)
        (model_dir / "version_02.pt").write_bytes(b"old")
        newest = model_dir / "version_11.pt"
        newest.write_bytes(b"new")
        state = {
            "agents": [
                {
                    "id": 4,
                    "name": "Agent",
                    "conv_channels": [4],
                    "fc_hidden_sizes": [8],
                    "model_dir": str(model_dir),
                }
            ]
        }
        state_path = arena_dir / "arena_state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        resolved = resolve_candidate("arena:4@latest", state_path)
        assert resolved.checkpoint == newest.resolve()


def test_manifest_resolution_uses_declared_architecture_and_hash():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        spec = _small_spec(root)
        digest = hashlib.sha256(spec.checkpoint.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "label": "manifest-test",
            "checkpoint": spec.checkpoint.name,
            "sha256": digest,
            "architecture": {
                "type": "conv_policy_value_v1",
                "conv_channels": [4],
                "fc_hidden_sizes": [8],
            },
        }
        manifest_path = root / "candidate.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        resolved = resolve_candidate(str(manifest_path))
        assert resolved.checkpoint == spec.checkpoint.resolve()
        assert resolved.expected_sha256 == digest
        validate_checkpoint(resolved)


def test_manifest_architecture_and_checkpoint_validate_strictly():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        spec = _small_spec(root)
        metadata = validate_checkpoint(spec)
        assert metadata["sha256"]
        assert metadata["parameter_count"] > 0

        wrong = CandidateSpec(
            label=spec.label,
            checkpoint=spec.checkpoint,
            architecture=Architecture(conv_channels=[5], fc_hidden_sizes=[8]),
            source="wrong architecture test",
        )
        _assert_raises(BenchmarkError, validate_checkpoint, wrong)


def test_policy_only_checkpoint_fails_preflight():
    with tempfile.TemporaryDirectory() as tmp:
        spec = _small_spec(Path(tmp), policy_only=True)
        _assert_raises(BenchmarkError, validate_checkpoint, spec)


def test_search_wrapping_builds_raw_tactical_and_ladder():
    with tempfile.TemporaryDirectory() as tmp:
        spec = _small_spec(Path(tmp))
        validate_checkpoint(spec)
        modes = build_candidate_modes(
            spec, torch.device("cpu"), [0, 2], include_tactical=True
        )
        assert set(modes) == {"raw", "tactical", "mcts_2"}
        assert isinstance(modes["mcts_2"], MCTSAgent)


def test_duplicate_anchor_alias_fails_closed():
    _assert_raises(
        BenchmarkError, parse_anchor_names, "lottery,lottery_no_touchy"
    )


class _SeededRandomAgent(Agent):
    def select_move(self, gamestate):
        import random

        return random.choice(
            rule_utl_valid_moves(
                gamestate.board, gamestate.last_move, gamestate.mini_winners
            )
        )


class _FirstAgent(Agent):
    def select_move(self, gamestate):
        return rule_utl_valid_moves(
            gamestate.board, gamestate.last_move, gamestate.mini_winners
        )[0]


class _IllegalAgent(Agent):
    def select_move(self, gamestate):
        return 999


def test_same_seed_and_openings_reproduce_results():
    openings = [Opening("empty", []), Opening("short", [17, 34])]
    first = run_pair(
        _SeededRandomAgent(), _FirstAgent(), openings, 123, "candidate", "first"
    )
    second = run_pair(
        _SeededRandomAgent(), _FirstAgent(), openings, 123, "candidate", "first"
    )
    for key in ("wins", "draws", "losses", "games", "score", "score_ci95"):
        assert first[key] == second[key]


def test_illegal_move_fails_closed():
    _assert_raises(
        BenchmarkError,
        play_opening_game,
        _IllegalAgent(),
        _FirstAgent(),
        Opening("empty", []),
        1,
        0,
    )


def test_report_requires_provenance_metadata():
    _assert_raises(BenchmarkError, validate_report_payload, {"schema_version": 1})
    payload = {
        "schema_version": 1,
        "created_utc": "test",
        "candidate": {
            "path": "candidate.pt",
            "sha256": "0" * 64,
            "bytes": 1,
            "architecture": {},
            "parameter_count": 1,
        },
        "anchors": [],
        "openings": {},
        "settings": {},
        "git": {},
        "device": {},
        "ruleset": {},
        "results": [],
        "total_elapsed_seconds": 0.0,
    }
    validate_report_payload(payload)


def _run_all() -> bool:
    tests = [
        value
        for key, value in sorted(globals().items())
        if key.startswith("test_") and callable(value)
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return failed == 0


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
