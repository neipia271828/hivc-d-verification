from __future__ import annotations

import json
from pathlib import Path

import pytest

from profiles import (
    ProfileValidationError,
    Role,
    Value,
    canonical_sha256,
    load_profiles,
    resolve_profile_entry,
)


CRITERIA = ("oxygen", "power")


def role_data() -> dict:
    return {
        "id": "safety_operator",
        "label": "安全管理担当",
        "expertise_domains": ["oxygen"],
        "observation_scope": ["oxygen"],
        "responsibility": "観測事実を正確に伝える",
    }


def value_data(**overrides) -> dict:
    data = {
        "id": "safety_soft",
        "version": "1.0",
        "initial_priority_weights": {"oxygen": 3, "power": 1},
        "confidence": 0.6,
        "negotiable": True,
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    "field", ["priority_weights", "goal_focus", "notes", "concession_tendency"]
)
def test_role_rejects_value_and_negotiation_fields(field: str) -> None:
    data = role_data()
    data[field] = "forbidden"
    with pytest.raises(ProfileValidationError, match="forbidden"):
        Role.from_dict(data)


def test_role_rejects_unknown_alias_that_could_hide_action_instruction() -> None:
    data = role_data()
    data["operational_directive"] = "always choose A"
    with pytest.raises(ProfileValidationError, match="unknown fields.*operational_directive"):
        Role.from_dict(data)


def test_value_normalizes_weights_and_has_stable_hash() -> None:
    value = Value.from_dict(value_data(), criteria=CRITERIA)
    assert value.initial_priority_weights == {"oxygen": 0.75, "power": 0.25}
    assert value.ordered_criteria == ("oxygen", "power")
    assert value.sha256 == canonical_sha256(value.to_dict())


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"oxygen": 1}, "missing criteria"),
        ({"oxygen": 1, "power": 1, "unknown": 1}, "unknown criteria"),
        ({"oxygen": "high", "power": 1}, "must be numeric"),
        ({"oxygen": -1, "power": 2}, "must not be negative"),
        ({"oxygen": 0, "power": 0}, "greater than zero"),
    ],
)
def test_value_rejects_invalid_weights(weights: dict, message: str) -> None:
    with pytest.raises(ProfileValidationError, match=message):
        Value.from_dict(value_data(initial_priority_weights=weights), criteria=CRITERIA)


def test_non_negotiable_soft_value_requires_sensitivity_opt_in() -> None:
    hard = value_data(negotiable=False)
    with pytest.raises(ProfileValidationError, match="main comparison"):
        Value.from_dict(hard, criteria=CRITERIA)
    assert Value.from_dict(hard, criteria=CRITERIA, allow_hard_value=True).negotiable is False


def test_resolve_soft_value_and_expertise_only() -> None:
    base = {"role": role_data(), "persona": {"communication_style": "skeptical", "evidence_demand": 0.8}}
    soft = resolve_profile_entry(
        "agent_01", {**base, "value": value_data()}, "soft_value", criteria=CRITERIA
    )
    assert soft.value is not None
    assert soft.role_value_mode == "soft_value"

    expertise = resolve_profile_entry("agent_01", base, "expertise_only", criteria=CRITERIA)
    assert expertise.value is None
    with pytest.raises(ProfileValidationError, match="must not define"):
        resolve_profile_entry(
            "agent_01", {**base, "value": value_data()}, "expertise_only", criteria=CRITERIA
        )


def test_load_legacy_hard_keeps_explicit_mode_and_warns(tmp_path: Path) -> None:
    path = tmp_path / "role.json"
    path.write_text(
        json.dumps(
            {
                "agent_01": {
                    "role": "旧安全担当",
                    "priority_weights": {"oxygen": 0.7, "power": 0.3},
                    "communication_style": "skeptical",
                    "evidence_demand": 0.8,
                    "notes": "酸素を優先する",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="legacy_hard"):
        loaded = load_profiles(path, "legacy_hard", criteria=CRITERIA)
    profile = loaded["agent_01"]
    assert profile.role.schema_version == "legacy-1"
    assert profile.role_value_mode == "legacy_hard"
    assert profile.value is not None and profile.value.negotiable is False
    assert profile.warnings and "do not pool" in profile.warnings[0]


def test_repository_separated_profile_files_load_and_soft_weights_are_normalized() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    soft = load_profiles(repo_root / "configs/profiles_soft_value.yaml", "soft_value")
    expertise = load_profiles(repo_root / "configs/profiles_expertise_only.yaml", "expertise_only")
    assert set(soft) >= {"alpha", "beta"}
    assert set(expertise) >= {"alpha", "beta"}
    for profile in soft.values():
        assert profile.role.schema_version == "2.0"
        assert profile.value is not None and profile.value.negotiable is True
        assert sum(profile.value.initial_priority_weights.values()) == pytest.approx(1.0)
    assert all(profile.value is None for profile in expertise.values())
