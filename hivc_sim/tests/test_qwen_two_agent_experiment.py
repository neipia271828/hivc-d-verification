from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.qwen_two_agent_experiment import _complete_direct_run, _prepare_direct_run


def test_direct_run_reserves_isolated_directory_stream_and_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "episode-visible"
    stream = run_dir / "stream.jsonl"

    run_id, live_path, metadata, metadata_path = _prepare_direct_run(
        run_dir, None, str(stream)
    )

    assert run_id == "episode-visible"
    assert live_path == str(stream)
    assert stream.exists() and stream.read_text(encoding="utf-8") == ""
    assert metadata["status"] == "running"
    assert metadata["run_id"] == "episode-visible"
    assert metadata["git_commit"]
    assert (run_dir / "run_id").read_text(encoding="utf-8").strip() == run_id

    (run_dir / "all_games.csv").write_text("condition\n", encoding="utf-8")
    (run_dir / "value_manifest.json").write_text("{}\n", encoding="utf-8")
    _complete_direct_run(
        run_dir,
        metadata,
        metadata_path,
        ["all_games.csv", "stream.jsonl", "value_manifest.json"],
    )
    completed = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["completed_at"]
    assert set(completed["artifacts"]) == {
        "all_games.csv",
        "stream.jsonl",
        "value_manifest.json",
    }


def test_direct_run_refuses_existing_artifacts_and_stream_reuse(tmp_path: Path) -> None:
    reused = tmp_path / "reused"
    reused.mkdir()
    (reused / "all_games.csv").write_text("old\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already contains run artifacts"):
        _prepare_direct_run(reused, "reused", None)

    workflow_dir = tmp_path / "workflow-run"
    workflow_dir.mkdir()
    (workflow_dir / "run_id").write_text("workflow-run\n", encoding="utf-8")
    (workflow_dir / "started_at").write_text("now\n", encoding="utf-8")
    (workflow_dir / "run.log").write_text("", encoding="utf-8")
    old_stream = workflow_dir / "stream.jsonl"
    old_stream.write_text('{"old":true}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="already contains run artifacts|refusing to reuse existing stream"):
        _prepare_direct_run(workflow_dir, "workflow-run", str(old_stream))


def test_direct_run_rejects_stream_outside_run_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the isolated output_dir"):
        _prepare_direct_run(tmp_path / "run", "run", str(tmp_path / "shared-stream.jsonl"))
