from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import pytest

from scripts.workflow_cli import (
    EXPECTED_ORIGIN,
    WorkflowError,
    _build_experiment_parser,
    _local_commit_commands,
    _parallel_runner_args,
    _remote_project_shell,
    _start_experiment_remote_command,
    _sync_remote_command,
    _validate_run_id,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_uv_project_exposes_workflow_commands() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"] == {
        "sync": "scripts.workflow_cli:sync_main",
        "experiment": "scripts.workflow_cli:experiment_main",
        "download": "scripts.workflow_cli:download_main",
        "visualize": "scripts.workflow_cli:visualize_main",
        "validate-smoke": "scripts.validate_experiment_preflight:main",
    }


def test_validate_run_id_rejects_paths() -> None:
    assert _validate_run_id("episode-20260714-120000") == "episode-20260714-120000"
    for invalid in ("../run", "a/b", "", "run name"):
        with pytest.raises(WorkflowError):
            _validate_run_id(invalid)


def test_remote_project_shell_expands_home_without_quoting_tilde() -> None:
    cfg = {"remote_project_dir": "~/projects/hivc-d-verification"}
    assert _remote_project_shell(cfg) == '"$HOME"/projects/hivc-d-verification'


def test_sync_command_requires_git_pull_and_matching_head() -> None:
    cfg = {"remote_project_dir": "~/projects/hivc-d-verification"}
    command = _sync_remote_command(cfg, "main", "abc123", "ssh-ed25519 AAAAtest test@example")
    assert "git pull --ff-only origin main" in command
    assert EXPECTED_ORIGIN in command
    assert '"$remote_head" != abc123' in command
    assert "git clone --quiet --branch main --single-branch" in command
    assert "git reset --hard origin/main" in command


def test_sync_stages_and_commits_dirty_worktree_by_default() -> None:
    commands = _local_commit_commands(
        " M scripts/workflow_cli.py",
        allow_dirty=False,
        message="chore: sync experiment workflow",
    )
    assert commands == [
        ["git", "add", "-A"],
        ["git", "commit", "-m", "chore: sync experiment workflow"],
    ]
    assert _local_commit_commands(
        " M scripts/workflow_cli.py",
        allow_dirty=True,
        message="unused",
    ) == []
    assert _local_commit_commands("", allow_dirty=False, message="unused") == []


def test_experiment_command_creates_isolated_run_artifacts() -> None:
    cfg = {
        "remote_project_dir": "~/projects/hivc-d-verification",
        "remote_venv": ".venv",
    }
    args = argparse.Namespace(
        experiment_config="configs/experiment.yaml",
        conditions=["hivc_d"],
        games=1,
        seed=42,
    )
    command, run_dir = _start_experiment_remote_command(cfg, args, "episode-test")
    assert run_dir.endswith("/episode-test")
    assert "--conditions hivc_d --games 1" in command
    assert "run.log" in command
    assert "exit_code" in command
    assert "stream.jsonl" in command
    assert "_discussion_json_contract" in command


def test_experiment_running_check_uses_recorded_pid_without_self_matching() -> None:
    cfg = {
        "remote_project_dir": "~/projects/hivc-d-verification",
        "remote_venv": ".venv",
    }
    args = argparse.Namespace(
        experiment_config="configs/experiment.yaml",
        conditions=["hivc_d"],
        games=1,
        seed=None,
    )
    command, _ = _start_experiment_remote_command(cfg, args, "episode-test")
    assert "pgrep -f" not in command
    assert "for pid_file in hivc_sim/results/turn_game/experiment/runs/*/pid" in command
    assert 'kill -0 "$active_pid"' in command
    assert '"/proc/$active_pid/cmdline"' in command
    assert '"$active_pid" != "$$"' in command


def test_parallel_runner_args_passes_parallel_and_gpu_options() -> None:
    cfg = {"remote_venv": ".venv"}
    args = argparse.Namespace(
        experiment_config="configs/experiment.yaml",
        conditions=["control", "consulting", "hivc_d"],
        games=30,
        seed=42,
        gpus=[0, 1],
        workers_per_gpu=1,
        temperature_warning=80,
        temperature_stop_scheduling=83,
        power_limit_w=180,
        resume=False,
        scientific_gate="not-applicable",
        role_value_mode="soft_value",
    )
    command = _parallel_runner_args(cfg, args, "hivc_sim/results/turn_game/experiment/runs/episode-test")
    assert "python" in command[0]
    assert "scripts/qwen_parallel_experiment.py" in command
    assert "--parallel" in command
    assert "--gpus" in command
    assert "0" in command
    assert "1" in command
    assert "--workers-per-gpu" not in command  # 既定値は省略
    assert "--temperature-warning" not in command  # 既定値は省略
    assert "--temperature-stop-scheduling" not in command  # 既定値は省略
    power_index = command.index("--power-limit-w")
    assert command[power_index + 1] == "180"
    assert "--resume" not in command
    mode_index = command.index("--role-value-mode")
    assert command[mode_index + 1] == "soft_value"


def test_experiment_parser_accepts_prescribed_condition_and_role_value_mode() -> None:
    args = _build_experiment_parser().parse_args(
        ["--conditions", "hivc_d_prescribed_v1", "--role-value-mode", "expertise_only", "--dry-run"]
    )
    assert args.conditions == ["hivc_d_prescribed_v1"]
    assert args.role_value_mode == "expertise_only"


def test_experiment_parser_accepts_temporary_power_limit() -> None:
    args = _build_experiment_parser().parse_args(
        ["--parallel", "--gpus", "0", "--power-limit-w", "180", "--dry-run"]
    )
    assert args.parallel is True
    assert args.power_limit_w == 180


def test_parallel_runner_args_passes_thermal_duty_cycle() -> None:
    cfg = {"remote_venv": ".venv"}
    args = argparse.Namespace(
        experiment_config="configs/experiment.yaml",
        conditions=["control", "consulting", "hivc_d"],
        games=1,
        seed=42,
        gpus=[0],
        workers_per_gpu=1,
        temperature_warning=80,
        temperature_stop_scheduling=83,
        thermal_duty_cycle=True,
        thermal_suspend_temperature=76,
        thermal_resume_temperature=68,
        power_limit_w=None,
        resume=False,
    )

    command = _parallel_runner_args(cfg, args, "runs/episode-test")

    assert "--thermal-duty-cycle" in command
    assert command[command.index("--thermal-suspend-temperature") + 1] == "76"
    assert command[command.index("--thermal-resume-temperature") + 1] == "68"


def test_stop_command_targets_workers_before_power_limit_orchestrator(monkeypatch) -> None:
    from scripts import workflow_cli

    cfg = {
        "remote_project_dir": "~/projects/hivc-d-verification",
        "remote_venv": ".venv",
    }
    captured: dict[str, str] = {}
    monkeypatch.setattr(workflow_cli, "_load_gpu_config", lambda _path: cfg)
    monkeypatch.setattr(workflow_cli, "_resolve_run_id", lambda _run_id: "episode-test")

    def fake_remote(_cfg, command, **_kwargs):
        captured["command"] = command
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(workflow_cli, "_remote", fake_remote)
    monkeypatch.setattr(
        "sys.argv", ["experiment", "--stop", "--run-id", "episode-test"]
    )

    workflow_cli.experiment_main()

    command = captured["command"]
    assert command.index('kill -TERM -"$worker_pid"') < command.index("orchestrator_pid=$(cat")
    assert command.index('kill -TERM -"$worker_pid"') < command.index('kill -CONT -"$worker_pid"')
    assert 'kill -TERM "$orchestrator_pid"' in command


def test_parallel_experiment_command_uses_parallel_runner_and_blocks_other_parallel_workers() -> None:
    cfg = {
        "remote_project_dir": "~/projects/hivc-d-verification",
        "remote_venv": ".venv",
    }
    args = argparse.Namespace(
        experiment_config="configs/experiment.yaml",
        conditions=["control", "consulting", "hivc_d"],
        games=30,
        seed=42,
        parallel=True,
        gpus=[0, 1],
        workers_per_gpu=1,
        temperature_warning=80,
        temperature_stop_scheduling=83,
        resume=False,
        scientific_gate="not-applicable",
    )
    command, run_dir = _start_experiment_remote_command(cfg, args, "episode-test")
    assert run_dir.endswith("/episode-test")
    assert "scripts/qwen_parallel_experiment.py" in command
    assert "--parallel" in command
    assert "--gpus 0 1" in command
    assert "stream.jsonl" not in command
    # worker プロセスも含め、他の並列実験を検出する
    assert "qwen_parallel_worker" in command


def test_parallel_resume_reuses_existing_run_directory() -> None:
    cfg = {
        "remote_project_dir": "~/projects/hivc-d-verification",
        "remote_venv": ".venv",
    }
    args = argparse.Namespace(
        experiment_config="configs/experiment.yaml",
        conditions=["control", "consulting", "hivc_d"],
        games=100,
        seed=42,
        parallel=True,
        gpus=[0, 1],
        workers_per_gpu=1,
        temperature_warning=80,
        temperature_stop_scheduling=83,
        resume=True,
        role_value_mode=None,
        role_file=None,
    )

    command, _ = _start_experiment_remote_command(cfg, args, "episode-existing")

    assert "--resume" in command
    assert "resume対象runが存在しません" in command
    assert "run IDが既に存在します" not in command
    assert "resumed_at" in command
    assert "exit_code" in command and ".pre-resume" in command


def test_large_experiment_requires_and_runs_scientific_smoke_gate() -> None:
    cfg = {
        "remote_project_dir": "~/projects/hivc-d-verification",
        "remote_venv": ".venv",
    }
    args = argparse.Namespace(
        experiment_config="configs/experiment.yaml",
        conditions=["hivc_d"],
        games=100,
        seed=42,
        parallel=False,
        resume=False,
        scientific_gate="required",
        smoke_run_id=None,
    )
    with pytest.raises(WorkflowError, match="--smoke-run-id"):
        _start_experiment_remote_command(cfg, args, "episode-full")

    args.smoke_run_id = "episode-smoke"
    command, _ = _start_experiment_remote_command(cfg, args, "episode-full")
    assert "scripts/validate_experiment_preflight.py" in command
    assert "episode-smoke --applicability required" in command
