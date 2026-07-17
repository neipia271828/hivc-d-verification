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


def test_power_limit_guard_applies_verifies_and_restores(monkeypatch) -> None:
    query_count = 0
    set_calls: list[tuple[int, float]] = []

    def fake_constraints(gpu_ids):
        nonlocal query_count
        query_count += 1
        current = 180.0 if query_count == 2 else 230.0
        return [
            {
                "gpu_id": gpu_id,
                "current_w": current,
                "default_w": 230.0,
                "min_w": 100.0,
                "max_w": 230.0,
            }
            for gpu_id in gpu_ids
        ]

    class FakeLogger:
        def log(self, _message):
            pass

    monkeypatch.setattr(qp, "get_gpu_power_constraints", fake_constraints)
    monkeypatch.setattr(
        qp, "_set_gpu_power_limit", lambda gpu_id, watts: set_calls.append((gpu_id, watts))
    )

    guard = qp.GpuPowerLimitGuard([0, 1], 180, FakeLogger())
    guard.apply()
    assert set_calls == [(0, 180.0), (1, 180.0)]
    assert all(item["applied_w"] == 180.0 for item in guard.devices)

    guard.restore()
    assert set_calls == [(0, 180.0), (1, 180.0), (1, 230.0), (0, 230.0)]
    assert all(item["restored"] is True for item in guard.devices)
    assert all(item["restored_w"] == 230.0 for item in guard.devices)


def test_power_limit_guard_rejects_out_of_range_before_setting(monkeypatch) -> None:
    monkeypatch.setattr(
        qp,
        "get_gpu_power_constraints",
        lambda gpu_ids: [
            {
                "gpu_id": 0,
                "current_w": 230.0,
                "default_w": 230.0,
                "min_w": 200.0,
                "max_w": 230.0,
            }
        ],
    )
    monkeypatch.setattr(
        qp,
        "_set_gpu_power_limit",
        lambda *_args: pytest.fail("範囲外の値を設定してはならない"),
    )
    guard = qp.GpuPowerLimitGuard([0], 180, SimpleNamespace(log=lambda _message: None))

    with pytest.raises(RuntimeError, match="許容範囲外"):
        guard.apply()


def test_power_limit_command_reports_permission_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        qp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=4,
            stdout="",
            stderr="Insufficient Permissions",
        ),
    )

    with pytest.raises(RuntimeError, match="管理者権限"):
        qp._set_gpu_power_limit(0, 180)


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


def test_compute_shards_keeps_games_as_paired_seeds_per_condition() -> None:
    conditions = ["control", "consulting", "hivc_d"]
    shards = qp.compute_shards(
        conditions, seed=42, games=100, gpu_ids=[0, 1], workers_per_gpu=1
    )

    assert [(s.seed_start, s.seed_count) for s in shards] == [(42, 50), (92, 50)]
    assert sum(s.seed_count for s in shards) == 100
    assert sum(len(s.tasks) for s in shards) == 300
    for shard in shards:
        per_seed: dict[int, list[str]] = {}
        for condition, game_seed in shard.tasks:
            per_seed.setdefault(game_seed, []).append(condition)
        assert len(per_seed) == shard.seed_count
        assert all(set(order) == set(conditions) for order in per_seed.values())


def test_master_manifest_separates_paired_seed_count_from_total_games(
    tmp_path: Path, monkeypatch
) -> None:
    conditions = ["control", "consulting", "hivc_d"]
    cfg = {
        "conditions": conditions,
        "games": 100,
        "seed": 42,
        "model_path": "/tmp/model",
        "role_file": None,
    }
    shards = qp.compute_shards(conditions, 42, 100, [0, 1], 1)
    monkeypatch.setattr(qp, "_git_sha", lambda: "git-hash")
    monkeypatch.setattr(qp, "_persona_file_hash", lambda config: None)
    monkeypatch.setattr(qp, "_framework_info", lambda: {})

    manifest = qp._create_master_manifest(
        SimpleNamespace(
            config="configs/experiment.yaml",
            workers_per_gpu=1,
            temperature_warning=80,
            temperature_stop_scheduling=83,
            resume=False,
        ),
        cfg,
        [0, 1],
        shards,
        tmp_path,
    )

    assert manifest["games"] == 100
    assert manifest["games_per_condition"] == 100
    assert manifest["total_condition_games"] == 300
    assert [(item["seed_start"], item["seed_count"]) for item in manifest["shards"]] == [
        (42, 50),
        (92, 50),
    ]


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


def test_launch_worker_passes_master_config_hash_and_clears_stale_pause(
    tmp_path: Path, monkeypatch
) -> None:
    shard = qp.compute_shards(["control"], seed=42, games=1, gpu_ids=[0], workers_per_gpu=1)[0]
    shard.shard_dir = tmp_path / shard.shard_id
    shard.pause_file = shard.shard_dir / qp.PAUSE_REQUEST_FILE
    shard.shard_dir.mkdir()
    shard.pause_file.write_text("thermal", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 1234

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProcess()

    monkeypatch.setattr(qp.subprocess, "Popen", fake_popen)
    cfg = {"conditions": ["control"], "games": 1, "seed": 42, "output_dir": "run"}
    qp._launch_worker(shard, cfg, SimpleNamespace(config=None))

    command = captured["cmd"]
    hash_index = command.index("--master-config-hash")
    assert len(command[hash_index + 1]) == 64
    assert not shard.pause_file.exists()


def test_resume_reuses_only_completed_shards() -> None:
    shards = qp.compute_shards(["control"], seed=42, games=2, gpu_ids=[0, 1], workers_per_gpu=1)
    existing = {
        "config_hash": "same-hash",
        "shards": [
            {"shard_id": shards[0].shard_id, "status": "completed", "exit_code": 0},
            {"shard_id": shards[1].shard_id, "status": "paused_thermal", "exit_code": 2},
        ],
    }
    manifest = {"config_hash": "same-hash", "resume": True}

    qp._apply_resume(existing, manifest, shards)

    assert shards[0].skip is True
    assert shards[0].status == "completed"
    assert shards[1].skip is False


def test_resume_rejects_mismatched_master_config() -> None:
    shards = qp.compute_shards(["control"], seed=42, games=1, gpu_ids=[0], workers_per_gpu=1)
    with pytest.raises(RuntimeError, match="config_hash"):
        qp._apply_resume(
            {"config_hash": "old", "shards": []},
            {"config_hash": "new", "resume": True},
            shards,
        )


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


def test_incomplete_shard_reports_partial_rows_without_full_seed_mismatch(tmp_path: Path) -> None:
    master_dir = tmp_path / "run-partial"
    master_dir.mkdir()
    shard = qp.compute_shards(["control"], seed=42, games=2, gpu_ids=[0], workers_per_gpu=1)[0]
    shard.shard_dir = master_dir / "shards" / shard.shard_id
    shard.shard_dir.mkdir(parents=True)
    shard.status = "paused_thermal"
    shard.exit_code = 2
    qp._write_csv(shard.shard_dir / "control_games.csv", [_make_row("control", 42, 1)])
    (shard.shard_dir / "shard_manifest.json").write_text(
        json.dumps(
            {
                "config_hash": "config-hash",
                "git_sha": "git-hash",
                "persona_hash": None,
                "framework_info": {},
            }
        ),
        encoding="utf-8",
    )

    cfg = {"conditions": ["control"], "games": 2, "seed": 42, "model_path": "/tmp/model"}
    manifest = {
        "config_hash": "config-hash",
        "git_sha": "git-hash",
        "persona_hash": None,
        "framework_info": {},
    }
    exit_code = qp._merge_results(master_dir, cfg, manifest, [shard], qp.MasterLogger(master_dir))

    assert exit_code == 1
    report = json.loads((master_dir / "merge_report.json").read_text(encoding="utf-8"))
    assert report["partial_results"] is True
    assert report["row_counts"] == {"control": 1}
    seed_check = next(c for c in report["checks"] if c["name"] == "condition_seed_set_match")
    assert seed_check == {
        "name": "condition_seed_set_match",
        "passed": None,
        "skipped": True,
        "reason": "shards_incomplete",
    }


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
