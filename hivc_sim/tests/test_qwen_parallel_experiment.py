from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import qwen_parallel_experiment as qp


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_nvidia_smi_query_reports_stdout_stderr_and_exit_code(monkeypatch) -> None:
    monkeypatch.setattr(
        qp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout='Field "invalid.field" is not a valid field to query.\n',
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        qp._nvidia_smi_query([0], ["invalid.field"])

    message = str(exc_info.value)
    assert "exit=2" in message
    assert "invalid.field" in message
    assert "stderr=''" in message


def test_gpu_snapshot_uses_supported_software_thermal_slowdown_field(monkeypatch) -> None:
    captured_fields: list[str] = []

    def fake_query(gpu_ids, fields):
        assert gpu_ids == [0]
        captured_fields.extend(fields)
        return [
            {
                field: {
                    "index": "0",
                    "uuid": "GPU-test",
                    "name": "NVIDIA RTX A5000",
                    "compute_mode": "Default",
                    "memory.free": "24000",
                    "memory.used": "1",
                    "memory.total": "24564",
                    "temperature.gpu": "35",
                    "clocks_throttle_reasons.sw_thermal_slowdown": "Active",
                }.get(field, "Not Active")
                for field in fields
            }
        ]

    monkeypatch.setattr(qp, "_nvidia_smi_query", fake_query)

    snapshot = qp.get_gpu_snapshot([0])

    assert "clocks_throttle_reasons.sw_thermal_slowdown" in captured_fields
    assert "clocks_throttle_reasons.thermal" not in captured_fields
    assert snapshot[0]["thermal_throttle"] is True


def test_gpu_snapshot_does_not_treat_not_active_as_active(monkeypatch) -> None:
    def fake_query(gpu_ids, fields):
        values = {
            "index": "0",
            "uuid": "GPU-test",
            "name": "NVIDIA RTX A5000",
            "compute_mode": "Default",
            "memory.free": "24000",
            "memory.used": "1",
            "memory.total": "24564",
            "temperature.gpu": "35",
        }
        return [{field: values.get(field, "Not Active") for field in fields}]

    monkeypatch.setattr(qp, "_nvidia_smi_query", fake_query)

    snapshot = qp.get_gpu_snapshot([0])

    assert snapshot[0]["thermal_throttle"] is False
    assert snapshot[0]["hw_thermal_slowdown"] is False
    assert snapshot[0]["hw_slowdown"] is False


def test_compute_shards_even_split_two_gpus() -> None:
    shards = qp.compute_shards(["control"], seed=42, games=30, gpu_ids=[0, 1], workers_per_gpu=1)
    assert len(shards) == 2
    assert shards[0].gpu_id == 0
    assert shards[0].seed_start == 42
    assert shards[0].seed_count == 15
    assert shards[0].shard_id == "control-gpu0-seed42-56"
    assert shards[1].gpu_id == 1
    assert shards[1].seed_start == 57
    assert shards[1].seed_count == 15
    assert shards[1].shard_id == "control-gpu1-seed57-71"


def test_compute_shards_three_conditions_share_same_split() -> None:
    shards = qp.compute_shards(["control", "consulting", "hivc_d"], seed=42, games=30, gpu_ids=[0, 1], workers_per_gpu=1)
    assert len(shards) == 2
    for s in shards:
        assert set(s.conditions) == {"control", "consulting", "hivc_d"}
        assert s.seed_count == 15
    assert shards[0].seed_start == 42
    assert shards[1].seed_start == 57


def test_compute_shards_uneven_one_game_no_empty_shard() -> None:
    shards = qp.compute_shards(["control"], seed=42, games=1, gpu_ids=[0, 1], workers_per_gpu=1)
    assert len(shards) == 1
    assert shards[0].gpu_id == 0
    assert shards[0].seed_count == 1


def test_compute_shards_workers_per_gpu_two() -> None:
    shards = qp.compute_shards(["control"], seed=42, games=4, gpu_ids=[0, 1], workers_per_gpu=2)
    assert len(shards) == 4
    assert [s.gpu_id for s in shards] == [0, 0, 1, 1]
    assert [s.seed_start for s in shards] == [42, 43, 44, 45]


def test_counterbalanced_shard_rounds_are_deterministic_and_vary_by_seed_range() -> None:
    shards = qp.compute_shards(
        ["control", "consulting", "hivc_d"], seed=42, games=2,
        gpu_ids=[0, 1], workers_per_gpu=1,
    )
    rounds1 = qp.counterbalanced_shard_rounds(shards)
    rounds2 = qp.counterbalanced_shard_rounds(shards)
    assert len(rounds1) == 1
    assert [[s.shard_id for s in r] for r in rounds1] == [[s.shard_id for s in r] for r in rounds2]

    per_seed: dict[int, list[str]] = {}
    for s in shards:
        for cond, game_seed in s.tasks:
            per_seed.setdefault(game_seed, []).append(cond)
    assert set(per_seed) == {42, 43}
    assert all(set(order) == {"control", "consulting", "hivc_d"} for order in per_seed.values())
    assert per_seed[42] != per_seed[43]


def _make_row(condition: str, seed: int, turn: int) -> dict[str, str]:
    return {
        "condition": condition,
        "seed": str(seed),
        "turn": str(turn),
        "scenario_id": "test",
        "event": "none",
        "alpha_role_key": "a",
        "beta_role_key": "b",
        "alpha_persona": "p",
        "beta_persona": "p",
        "alpha_persona_params": "{}",
        "beta_persona_params": "{}",
        "state_before": "{}",
        "individual_actions": "A,A",
        "individual_reasons": '{"alpha": "", "beta": ""}',
        "discussion_turns": "0",
        "discussion_token_budget_used": "0",
        "discussion_transcript": "[]",
        "alpha_vote": "A",
        "alpha_vote_reason": "",
        "alpha_vote_message": "",
        "alpha_vote_ready": "true",
        "alpha_vote_raw": "",
        "alpha_vote_thinking": "",
        "beta_vote": "A",
        "beta_vote_reason": "",
        "beta_vote_message": "",
        "beta_vote_ready": "true",
        "beta_vote_raw": "",
        "beta_vote_thinking": "",
        "group_action": "A",
        "group_action_label": "A",
        "group_reason": "",
        "decision_rule": "consensus",
        "best_action": "A",
        "acceptable_actions": "A",
        "regret": "0.0",
        "q_values": "{}",
        "state_after": "{}",
        "outcome": "win",
        "terminal_score": "0.0",
        "decision_opportunity_count": "1",
        "decision_attempts": "1",
        "decision_attempt_index": "1",
        "free_discussion_message_count": "0",
        "decision_history": "[]",
        "fallback_used": "false",
        "fallback_priority_agent": "",
        "planned_route": "comms",
        "optimal_route": "comms",
        "route_switch": "false",
        "premature": "false",
        "role_specific_evidence": "{}",
        "alpha_evidence": "",
        "beta_evidence": "",
        "unanswered_question_count": "0",
        "question_response_latency": "nan",
        "forced_decision_with_open_question": "false",
        "forced_decision_reason": "",
    }


def test_merge_results_generates_master_csvs_and_report(tmp_path: Path) -> None:
    master_dir = tmp_path / "run1"
    master_dir.mkdir()
    shards_dir = master_dir / "shards"
    shards_dir.mkdir()

    shard = qp.compute_shards(["control"], seed=42, games=1, gpu_ids=[0], workers_per_gpu=1)[0]
    shard.shard_dir = shards_dir / shard.shard_id
    shard.shard_dir.mkdir()
    shard.status = "completed"
    shard.exit_code = 0

    cfg = {
        "conditions": ["control"],
        "games": 1,
        "seed": 42,
        "model_path": "/tmp/model",
    }
    manifest = {
        "config_hash": "config-hash",
        "git_sha": "git-hash",
        "persona_hash": None,
        "framework_info": {"python_version": "3.11"},
    }
    shard_manifest = {
        "config_hash": "config-hash",
        "git_sha": "git-hash",
        "persona_hash": None,
        "framework_info": {"python_version": "3.11"},
    }
    (shard.shard_dir / "shard_manifest.json").write_text(json.dumps(shard_manifest), encoding="utf-8")
    (shard.shard_dir / "value_manifest.json").write_text(
        json.dumps({"schema_version": "value-manifest-1", "frameworks": {"control": {}}, "game_profile_assignments": [{"seed": 42}]}),
        encoding="utf-8",
    )
    qp._write_csv(shard.shard_dir / "control_games.csv", [_make_row("control", 42, 1)])

    logger = qp.MasterLogger(master_dir)
    exit_code = qp._merge_results(master_dir, cfg, manifest, [shard], logger)
    assert exit_code == 0
    assert (master_dir / "control_games.csv").is_file()
    assert (master_dir / "all_games.csv").is_file()
    assert (master_dir / "summary.csv").is_file()
    assert (master_dir / "merge_report.json").is_file()
    assert (master_dir / "value_manifest.json").is_file()
    value_manifest = json.loads((master_dir / "value_manifest.json").read_text(encoding="utf-8"))
    assert value_manifest["game_profile_assignments"] == [{"seed": 42}]
    assert value_manifest["framework_ids"] == ["control"]

    report = json.loads((master_dir / "merge_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "merged"
    assert all(c["passed"] for c in report["checks"])
    assert report["row_counts"]["control"] == 1


def test_merge_results_fails_when_shard_missing_output(tmp_path: Path) -> None:
    master_dir = tmp_path / "run1"
    master_dir.mkdir()
    shards_dir = master_dir / "shards"
    shards_dir.mkdir()

    shard = qp.compute_shards(["control"], seed=42, games=1, gpu_ids=[0], workers_per_gpu=1)[0]
    shard.shard_dir = shards_dir / shard.shard_id
    shard.shard_dir.mkdir()
    shard.status = "failed"
    shard.exit_code = 1

    cfg = {
        "conditions": ["control"],
        "games": 1,
        "seed": 42,
        "model_path": "/tmp/model",
    }
    manifest = {
        "config_hash": "config-hash",
        "git_sha": "git-hash",
        "persona_hash": None,
        "framework_info": {},
    }
    logger = qp.MasterLogger(master_dir)
    exit_code = qp._merge_results(master_dir, cfg, manifest, [shard], logger)
    assert exit_code == 1
    report = json.loads((master_dir / "merge_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert not all(c["passed"] for c in report["checks"])


def test_merge_results_fails_when_duplicate_turn(tmp_path: Path) -> None:
    master_dir = tmp_path / "run1"
    master_dir.mkdir()
    shards_dir = master_dir / "shards"
    shards_dir.mkdir()

    shard = qp.compute_shards(["control"], seed=42, games=1, gpu_ids=[0], workers_per_gpu=1)[0]
    shard.shard_dir = shards_dir / shard.shard_id
    shard.shard_dir.mkdir()
    shard.status = "completed"
    shard.exit_code = 0

    cfg = {
        "conditions": ["control"],
        "games": 1,
        "seed": 42,
        "model_path": "/tmp/model",
    }
    manifest = {
        "config_hash": "config-hash",
        "git_sha": "git-hash",
        "persona_hash": None,
        "framework_info": {},
    }
    shard_manifest = {
        "config_hash": "config-hash",
        "git_sha": "git-hash",
        "persona_hash": None,
        "framework_info": {},
    }
    (shard.shard_dir / "shard_manifest.json").write_text(json.dumps(shard_manifest), encoding="utf-8")
    (shard.shard_dir / "value_manifest.json").write_text(
        json.dumps({"schema_version": "value-manifest-1", "frameworks": {"control": {}}}),
        encoding="utf-8",
    )
    qp._write_csv(shard.shard_dir / "control_games.csv", [_make_row("control", 42, 1), _make_row("control", 42, 1)])

    logger = qp.MasterLogger(master_dir)
    exit_code = qp._merge_results(master_dir, cfg, manifest, [shard], logger)
    assert exit_code == 1
    report = json.loads((master_dir / "merge_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    dup_check = next(c for c in report["checks"] if c["name"] == "no_duplicate_turn")
    assert not dup_check["passed"]
