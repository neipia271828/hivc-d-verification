"""Validate scientific smoke artifacts before a large V-flow GPU experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
HIVC_SIM = REPO_ROOT / "hivc_sim"
if str(HIVC_SIM) not in sys.path:
    sys.path.insert(0, str(HIVC_SIM))

from turn_game_metrics import invalid_discussion_output_rate, v_process_metrics  # noqa: E402

APPLICABILITY = ("auto", "required", "not-applicable")
UNIFORM_WEIGHT = 0.2


def _read_rows(run_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    all_games = run_dir / "all_games.csv"
    paths = [all_games] if all_games.is_file() else sorted(run_dir.glob("*_games.csv"))
    rows: list[dict[str, str]] = []
    sources: list[str] = []
    for path in paths:
        if path.name == "summary.csv":
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
        sources.append(path.name)
    return rows, sources


def _json_mapping(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _weights(value: object) -> dict[str, float] | None:
    body = _json_mapping(value)
    raw = body.get("weights") if body else None
    if not isinstance(raw, dict) or len(raw) != 5:
        return None
    try:
        result = {str(key): float(number) for key, number in raw.items()}
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(number) for number in result.values()):
        return None
    return result


def _is_uniform(weights: dict[str, float]) -> bool:
    return all(math.isclose(value, UNIFORM_WEIGHT, rel_tol=0.0, abs_tol=1e-9) for value in weights.values())


def _to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _auto_applicable(run_dir: Path, rows: list[dict[str, str]]) -> bool:
    manifest_path = run_dir / "value_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        mode = manifest.get("role_value_mode")
        if mode is None and isinstance(manifest.get("experiment_config"), dict):
            mode = manifest["experiment_config"].get("role_value_mode")
        if mode:
            return True
    return any(str(row.get("role_value_mode", "")).strip() not in {"", "legacy_unmeasured"} for row in rows)


def validate_smoke_run(run_dir: str | Path, applicability: str = "auto") -> dict[str, Any]:
    """Return a diagnostic report. ``passed`` is false if any applicable gate fails."""
    if applicability not in APPLICABILITY:
        raise ValueError(f"unknown applicability: {applicability}")
    path = Path(run_dir)
    rows, sources = _read_rows(path) if path.is_dir() else ([], [])
    applicable = applicability == "required" or (
        applicability == "auto" and bool(rows) and _auto_applicable(path, rows)
    )
    if applicability == "not-applicable" or not applicable:
        return {
            "run_dir": str(path),
            "status": "not_applicable",
            "applicability": applicability,
            "passed": True,
            "row_count": len(rows),
            "sources": sources,
            "gates": [],
            "reason": "V-specific scientific gate explicitly or automatically not applicable",
        }

    measured_pairs = 0
    copied_uniform_pairs = 0
    for row in rows:
        alpha = _weights(row.get("alpha_v_before"))
        beta = _weights(row.get("beta_v_before"))
        if alpha is None or beta is None:
            continue
        measured_pairs += 1
        copied_uniform_pairs += int(alpha == beta and _is_uniform(alpha))

    v_rows = [row for row in rows if str(row.get("condition", "")).strip().startswith("hivc_d")]
    v_metrics = v_process_metrics(v_rows)
    alignment_required = sum(_to_bool(row.get("v_alignment_required")) for row in v_rows)
    proposal_denominator = int(v_metrics["v_proposal_rate_denominator"])
    accepted_v_star = sum(
        str(row.get("v_star_status", "")).strip().lower() == "accepted"
        and bool(str(row.get("v_star_id", "")).strip())
        for row in v_rows
    )
    invalid_rate = invalid_discussion_output_rate(rows)
    discussion_messages = sum(int(float(row.get("discussion_turns", 0) or 0)) for row in rows)

    def gate(name: str, passed: bool, diagnostic: str, observed: object) -> dict[str, Any]:
        return {"name": name, "passed": passed, "observed": observed, "diagnostic": diagnostic}

    gates = [
        gate(
            "no_identical_copied_uniform_v",
            measured_pairs > 0 and copied_uniform_pairs == 0,
            f"measured_pairs={measured_pairs}, copied_uniform_pairs={copied_uniform_pairs}",
            {"measured_pairs": measured_pairs, "copied_uniform_pairs": copied_uniform_pairs},
        ),
        gate(
            "v_alignment_required_observed",
            alignment_required > 0,
            f"hivc_d v_alignment_required=true turns={alignment_required}",
            alignment_required,
        ),
        gate(
            "required_v_proposal_denominator_observed",
            proposal_denominator > 0,
            f"v_proposal_rate denominator={proposal_denominator}",
            proposal_denominator,
        ),
        gate(
            "accepted_v_star_observed",
            accepted_v_star > 0,
            f"accepted V* rows with non-empty id={accepted_v_star}",
            accepted_v_star,
        ),
        gate(
            "invalid_discussion_output_rate_below_0_10",
            discussion_messages > 0 and math.isfinite(invalid_rate) and invalid_rate < 0.10,
            f"invalid_discussion_output_rate={invalid_rate!r}, discussion_messages={discussion_messages}",
            invalid_rate if math.isfinite(invalid_rate) else None,
        ),
    ]
    passed = bool(rows) and all(item["passed"] for item in gates)
    return {
        "run_dir": str(path),
        "status": "passed" if passed else "failed",
        "applicability": applicability,
        "passed": passed,
        "row_count": len(rows),
        "sources": sources,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU本実験前のsmoke科学的妥当性ゲート")
    parser.add_argument("run_dir", help="smoke runディレクトリ（CSV/value_manifestを含む）")
    parser.add_argument("--applicability", choices=APPLICABILITY, default="auto")
    parser.add_argument("--report", help="診断JSONの保存先")
    args = parser.parse_args()
    report = validate_smoke_run(args.run_dir, args.applicability)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
