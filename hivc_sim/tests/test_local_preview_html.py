from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HTML_PATH = REPO_ROOT / "scripts" / "local_preview.html"


def _run_normalize_record(row: dict[str, object]) -> dict[str, object]:
    node_script = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('function safeJSON(str, fallback) {');
const end = html.indexOf('\nfunction parseSummaryCSV', start);
eval(html.slice(start, end));
process.stdout.write(JSON.stringify(normalizeRecord(JSON.parse(fs.readFileSync(0, 'utf8')))));
"""
    result = subprocess.run(
        ["node", "-e", node_script, str(HTML_PATH)],
        input=json.dumps(row),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required to test browser JavaScript")
def test_csv_parser_preserves_newlines_inside_quoted_fields() -> None:
    source = HTML_PATH.read_text(encoding="utf-8")
    start = source.index("function parseCSV(text) {")
    end = source.index("\nfunction normalizeRecord", start)
    node_script = """
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('function parseCSV(text) {');
const end = html.indexOf('\\nfunction normalizeRecord', start);
eval(html.slice(start, end));
process.stdout.write(JSON.stringify(parseCSV(fs.readFileSync(0, 'utf8'))));
"""
    csv_text = (
        "turn,alpha_vote_thinking,discussion_transcript\r\n"
        '0,"first line\nsecond line","[{""speaker"":""alpha""}]"\r\n'
        '1,"one line","[{""speaker"":""beta""}]"\r\n'
    )
    result = subprocess.run(
        ["node", "-e", node_script, str(HTML_PATH)],
        input=csv_text,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(result.stdout) == [
        {
            "turn": "0",
            "alpha_vote_thinking": "first line\nsecond line",
            "discussion_transcript": '[{"speaker":"alpha"}]',
        },
        {
            "turn": "1",
            "alpha_vote_thinking": "one line",
            "discussion_transcript": '[{"speaker":"beta"}]',
        },
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required to test browser JavaScript")
def test_legacy_row_keeps_missing_v_fields_as_not_recorded() -> None:
    row = _run_normalize_record({"condition": "control", "seed": "7", "turn": "2"})

    assert row["regret"] is None
    assert row["alpha_v_before"] is None
    assert row["v_proposals"] is None
    assert row["v_star"] is None
    assert row["v_star_status"] is None
    assert row["v_star_action_consistency"] is None


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required to test browser JavaScript")
def test_v_fields_are_parsed_without_cross_condition_fallback() -> None:
    first = _run_normalize_record(
        {
            "condition": "control",
            "seed": "13",
            "turn": "1",
            "alpha_v_before": '{"safety":0.8}',
            "v_proposals": '[{"id":"control-p1","act":"accept"}]',
            "v_star_id": "control-vstar",
            "v_star": '{"safety":0.7}',
            "v_star_status": "accepted",
            "alpha_vote_changed": "true",
            "v_star_action_consistency": "false",
        }
    )
    second = _run_normalize_record(
        {
            "condition": "hivcd",
            "seed": "13",
            "turn": "1",
            "v_star_id": "hivcd-vstar",
            "v_star": '{"mission":0.6}',
            "v_star_status": "accepted",
        }
    )

    assert first["v_star_id"] == "control-vstar"
    assert first["v_star"] == {"safety": 0.7}
    assert first["v_proposals"][0]["id"] == "control-p1"
    assert first["alpha_vote_changed"] is True
    assert first["v_star_action_consistency"] is False
    assert second["v_star_id"] == "hivcd-vstar"
    assert second["v_star"] == {"mission": 0.6}


def test_preview_lists_and_serves_value_manifest(tmp_path: Path) -> None:
    module_path = REPO_ROOT / "scripts" / "local_preview.py"
    spec = importlib.util.spec_from_file_location("local_preview_for_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    manifest = {"role_value_mode": "soft_value", "frameworks": []}
    (run_dir / "value_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    server = module.PreviewServer(tmp_path, 0, "127.0.0.1")

    runs = server._list_runs()
    assert runs[0]["has_value_manifest"] is True
    assert json.loads(server._read_file("run-1", "value_manifest.json")) == manifest


def test_preview_serves_merged_parallel_shard_value_manifests(tmp_path: Path) -> None:
    module_path = REPO_ROOT / "scripts" / "local_preview.py"
    spec = importlib.util.spec_from_file_location("local_preview_parallel_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run_dir = tmp_path / "parallel-run"
    for name, condition, seed in (("s1", "control", 42), ("s2", "hivc_d", 43)):
        shard_dir = run_dir / "shards" / name
        shard_dir.mkdir(parents=True)
        (shard_dir / "value_manifest.json").write_text(
            json.dumps({"frameworks": {condition: {}}, "game_entries": [{"seed": seed}]}),
            encoding="utf-8",
        )
    server = module.PreviewServer(tmp_path, 0, "127.0.0.1")
    assert server._list_runs()[0]["has_value_manifest"] is True
    merged = json.loads(server._read_value_manifest("parallel-run"))
    assert set(merged["frameworks"]) == {"control", "hivc_d"}
    assert {entry["seed"] for entry in merged["game_entries"]} == {42, 43}


def test_comparison_is_keyed_by_exact_seed_and_turn() -> None:
    source = HTML_PATH.read_text(encoding="utf-8")
    assert "row.seed === r.seed && row.turn === r.turn" in source
    assert "各行はそのcondition固有のV*です" in source
    assert "記録なし" in source
