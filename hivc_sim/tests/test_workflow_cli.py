from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import pytest

from scripts.workflow_cli import (
    EXPECTED_ORIGIN,
    WorkflowError,
    _local_commit_commands,
    _remote_project_shell,
    _start_experiment_remote_command,
    _sync_remote_command,
    _validate_run_id,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_uv_project_exposes_four_workflow_commands() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"] == {
        "sync": "scripts.workflow_cli:sync_main",
        "experiment": "scripts.workflow_cli:experiment_main",
        "download": "scripts.workflow_cli:download_main",
        "visualize": "scripts.workflow_cli:visualize_main",
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
