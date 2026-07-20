from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.validate_experiment_preflight import validate_smoke_run

CRITERIA = ["oxygen", "power", "hull_damage", "flooding", "communication"]


def _v(weights: list[float]) -> str:
    return json.dumps({"ordered_criteria": CRITERIA, "weights": dict(zip(CRITERIA, weights))})


def _write_run(path: Path, rows: list[dict[str, object]], mode: str = "soft_value") -> None:
    path.mkdir()
    fields = sorted({key for row in rows for key in row})
    with (path / "all_games.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (path / "value_manifest.json").write_text(
        json.dumps({"role_value_mode": mode}), encoding="utf-8"
    )


def _passing_row() -> dict[str, object]:
    return {
        "condition": "hivc_d",
        "role_value_mode": "soft_value",
        "alpha_v_before": _v([0.4, 0.1, 0.2, 0.2, 0.1]),
        "beta_v_before": _v([0.1, 0.4, 0.2, 0.1, 0.2]),
        "v_alignment_required": "true",
        "v_proposals": json.dumps([{"proposal_id": "p1"}]),
        "v_star_status": "accepted",
        "v_star_id": "p1",
        "discussion_turns": "10",
        "invalid_discussion_output_count": "0",
    }


def test_scientific_preflight_passes_on_observed_smoke_behavior(tmp_path: Path) -> None:
    run_dir = tmp_path / "smoke-pass"
    _write_run(run_dir, [_passing_row()])

    report = validate_smoke_run(run_dir, "required")

    assert report["passed"] is True
    assert report["status"] == "passed"
    assert all(gate["passed"] for gate in report["gates"])


def test_scientific_preflight_reports_each_failed_behavior_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "smoke-fail"
    uniform = _v([0.2] * 5)
    row = _passing_row() | {
        "alpha_v_before": uniform,
        "beta_v_before": uniform,
        "v_alignment_required": "false",
        "v_proposals": "[]",
        "v_star_status": "unresolved",
        "v_star_id": "",
        "invalid_discussion_output_count": "1",
        "discussion_turns": "10",
    }
    _write_run(run_dir, [row])

    report = validate_smoke_run(run_dir, "required")
    failures = {gate["name"]: gate for gate in report["gates"] if not gate["passed"]}

    assert report["passed"] is False
    assert set(failures) == {
        "no_identical_copied_uniform_v",
        "v_alignment_required_observed",
        "required_v_proposal_denominator_observed",
        "accepted_v_star_observed",
        "invalid_discussion_output_rate_below_0_10",
    }
    assert failures["no_identical_copied_uniform_v"]["observed"]["copied_uniform_pairs"] == 1
    assert failures["invalid_discussion_output_rate_below_0_10"]["observed"] == 0.1


def test_scientific_preflight_explicit_legacy_not_applicable(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy"
    _write_run(run_dir, [{"condition": "control", "discussion_turns": "0"}], mode="legacy_hard")

    report = validate_smoke_run(run_dir, "not-applicable")

    assert report["passed"] is True
    assert report["status"] == "not_applicable"
    assert report["gates"] == []
