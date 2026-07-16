"""Role / Persona / Value profiles used by the turn-game experiments.

The resolved representation deliberately keeps expertise (Role), presentation
(Persona), and provisional decision criteria (Value) separate.  Legacy role
files are supported only through the explicitly labelled ``legacy_hard`` mode;
loading one never silently changes its fixed preferences into a soft value.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
import warnings

try:  # YAML is an optional convenience; JSON remains fully supported.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - requirements.txt normally supplies it
    yaml = None


ROLE_VALUE_MODES = frozenset({"legacy_hard", "soft_value", "expertise_only"})
DEFAULT_VALUE_CRITERIA = (
    "oxygen",
    "power",
    "hull_damage",
    "flooding",
    "communication",
)
ROLE_FORBIDDEN_FIELDS = frozenset(
    {
        "priority_weights",
        "initial_priority_weights",
        "goal_focus",
        "notes",
        "risk_tolerance",
        "concession_tendency",
        "consensus_orientation",
        "dominance",
        "value",
    }
)
ROLE_ALLOWED_FIELDS = frozenset(
    {
        "id",
        "label",
        "expertise_domains",
        "observation_scope",
        "responsibility",
        "feasibility_constraints",
        "schema_version",
    }
)


class ProfileValidationError(ValueError):
    """Raised when a profile violates the Role/Value separation contract."""


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical UTF-8 JSON for ``value``."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_string(data: Mapping[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _string_list(data: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = data.get(name)
    if not isinstance(value, (list, tuple)) or not value:
        raise ProfileValidationError(f"{name} must be a non-empty list of strings")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ProfileValidationError(f"{name} must not contain empty items")
    return result


def _unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileValidationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ProfileValidationError(f"{name} must be between 0 and 1")
    return number


@dataclass(frozen=True)
class Role:
    id: str
    label: str
    expertise_domains: tuple[str, ...]
    observation_scope: tuple[str, ...]
    responsibility: str
    feasibility_constraints: tuple[str, ...] = ()
    schema_version: str = "2.0"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Role":
        if not isinstance(data, Mapping):
            raise ProfileValidationError("role must be an object")
        forbidden = sorted(ROLE_FORBIDDEN_FIELDS.intersection(data))
        if forbidden:
            raise ProfileValidationError(
                "Role contains value/negotiation fields that are forbidden: "
                + ", ".join(forbidden)
            )
        unknown = sorted(set(data) - ROLE_ALLOWED_FIELDS)
        if unknown:
            raise ProfileValidationError(
                "Role contains unknown fields; only expertise and observation fields are allowed: "
                + ", ".join(unknown)
            )
        constraints = data.get("feasibility_constraints", ())
        if not isinstance(constraints, (list, tuple)):
            raise ProfileValidationError("feasibility_constraints must be a list")
        return cls(
            id=_require_string(data, "id"),
            label=_require_string(data, "label"),
            expertise_domains=_string_list(data, "expertise_domains"),
            observation_scope=_string_list(data, "observation_scope"),
            responsibility=_require_string(data, "responsibility"),
            feasibility_constraints=tuple(str(item).strip() for item in constraints),
            schema_version=str(data.get("schema_version", "2.0")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Persona:
    communication_style: str
    evidence_demand: float
    concession_tendency: float = 0.5
    consensus_orientation: float = 0.5
    dominance: float = 0.5
    id: str = "neutral"
    version: str = "1.0"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Persona":
        data = data or {}
        if not isinstance(data, Mapping):
            raise ProfileValidationError("persona must be an object")
        return cls(
            id=str(data.get("id", "neutral")),
            version=str(data.get("version", "1.0")),
            communication_style=str(data.get("communication_style", "neutral")),
            evidence_demand=_unit_interval(data.get("evidence_demand", 0.5), "evidence_demand"),
            concession_tendency=_unit_interval(
                data.get("concession_tendency", 0.5), "concession_tendency"
            ),
            consensus_orientation=_unit_interval(
                data.get("consensus_orientation", 0.5), "consensus_orientation"
            ),
            dominance=_unit_interval(data.get("dominance", 0.5), "dominance"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Value:
    id: str
    version: str
    initial_priority_weights: dict[str, float]
    confidence: float
    negotiable: bool
    ordered_criteria: tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        criteria: tuple[str, ...] = DEFAULT_VALUE_CRITERIA,
        allow_hard_value: bool = False,
    ) -> "Value":
        if not isinstance(data, Mapping):
            raise ProfileValidationError("value must be an object")
        raw_weights = data.get("initial_priority_weights")
        if not isinstance(raw_weights, Mapping):
            raise ProfileValidationError("initial_priority_weights must be an object")

        expected = set(criteria)
        supplied = set(raw_weights)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing:
            raise ProfileValidationError("initial_priority_weights missing criteria: " + ", ".join(missing))
        if unknown:
            raise ProfileValidationError("initial_priority_weights contains unknown criteria: " + ", ".join(unknown))

        weights: dict[str, float] = {}
        for criterion in criteria:
            raw = raw_weights[criterion]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ProfileValidationError(f"weight for {criterion} must be numeric")
            number = float(raw)
            if not math.isfinite(number):
                raise ProfileValidationError(f"weight for {criterion} must be finite")
            if number < 0:
                raise ProfileValidationError(f"weight for {criterion} must not be negative")
            weights[criterion] = number
        total = sum(weights.values())
        if total <= 0:
            raise ProfileValidationError("initial_priority_weights total must be greater than zero")
        weights = {key: value / total for key, value in weights.items()}

        negotiable = data.get("negotiable", True)
        if not isinstance(negotiable, bool):
            raise ProfileValidationError("negotiable must be boolean")
        if not negotiable and not allow_hard_value:
            raise ProfileValidationError(
                "negotiable=false is excluded from the main comparison; "
                "set allow_hard_value=True for an explicitly labelled sensitivity analysis"
            )
        ordered = data.get("ordered_criteria", ())
        if ordered:
            if not isinstance(ordered, (list, tuple)) or set(ordered) != expected or len(ordered) != len(criteria):
                raise ProfileValidationError("ordered_criteria must contain every known criterion exactly once")
            ordered_tuple = tuple(str(item) for item in ordered)
        else:
            ordered_tuple = tuple(sorted(criteria, key=weights.get, reverse=True))
        return cls(
            id=_require_string(data, "id"),
            version=_require_string(data, "version"),
            initial_priority_weights=weights,
            confidence=_unit_interval(data.get("confidence"), "confidence"),
            negotiable=negotiable,
            ordered_criteria=ordered_tuple,
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedProfile:
    role: Role
    persona: Persona
    value: Value | None
    role_value_mode: str
    source_path: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.to_dict(),
            "persona": self.persona.to_dict(),
            "value": self.value.to_dict() if self.value else None,
            "role_value_mode": self.role_value_mode,
            "source_path": self.source_path,
            "warnings": list(self.warnings),
        }


def _legacy_profile(
    profile_id: str,
    data: Mapping[str, Any],
    source_path: str | None,
    criteria: tuple[str, ...],
) -> ResolvedProfile:
    warning = (
        f"{profile_id} loaded as schema_version=legacy-1 / role_value_mode=legacy_hard; "
        "do not pool this result with the soft_value main comparison"
    )
    warnings.warn(warning, UserWarning, stacklevel=3)
    weights = data.get("priority_weights")
    if not isinstance(weights, Mapping):
        raise ProfileValidationError(f"legacy profile {profile_id} has no priority_weights")
    expertise = tuple(str(key) for key in weights)
    role = Role(
        id=profile_id,
        label=str(data.get("role", profile_id)),
        expertise_domains=expertise,
        observation_scope=expertise,
        responsibility="担当領域の観測事実と専門的制約を正確に共有する",
        schema_version="legacy-1",
    )
    persona = Persona.from_dict(data)
    value = Value.from_dict(
        {
            "id": profile_id,
            "version": "legacy-1",
            "initial_priority_weights": weights,
            "confidence": 1.0,
            "negotiable": False,
        },
        criteria=criteria,
        allow_hard_value=True,
    )
    return ResolvedProfile(role, persona, value, "legacy_hard", source_path, (warning,))


def resolve_profile_entry(
    profile_id: str,
    data: Mapping[str, Any],
    role_value_mode: str,
    *,
    criteria: tuple[str, ...] = DEFAULT_VALUE_CRITERIA,
    source_path: str | None = None,
    allow_hard_value: bool = False,
) -> ResolvedProfile:
    """Validate and resolve one profile entry for the requested regime."""
    if role_value_mode not in ROLE_VALUE_MODES:
        raise ProfileValidationError(
            f"unknown role_value_mode {role_value_mode!r}; expected one of {sorted(ROLE_VALUE_MODES)}"
        )
    if not isinstance(data, Mapping):
        raise ProfileValidationError(f"profile {profile_id} must be an object")
    if role_value_mode == "legacy_hard":
        return _legacy_profile(profile_id, data, source_path, criteria)

    raw_role = data.get("role")
    if not isinstance(raw_role, Mapping):
        raise ProfileValidationError(f"profile {profile_id} requires a separated role object")
    role_data = dict(raw_role)
    role_data.setdefault("id", profile_id)
    role = Role.from_dict(role_data)
    persona = Persona.from_dict(data.get("persona"))
    raw_value = data.get("value")
    if role_value_mode == "expertise_only":
        if raw_value not in (None, {}):
            raise ProfileValidationError("expertise_only profiles must not define an explicit value")
        value = None
    else:
        if not isinstance(raw_value, Mapping):
            raise ProfileValidationError("soft_value profiles require a separated value object")
        value = Value.from_dict(
            raw_value, criteria=criteria, allow_hard_value=allow_hard_value
        )
    return ResolvedProfile(role, persona, value, role_value_mode, source_path)


def _read_profile_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML profile files")
        return yaml.safe_load(text)
    raise ProfileValidationError(f"unsupported profile file extension: {path.suffix}")


def load_profiles(
    path: str | Path,
    role_value_mode: str,
    *,
    criteria: tuple[str, ...] = DEFAULT_VALUE_CRITERIA,
    allow_hard_value: bool = False,
) -> dict[str, ResolvedProfile]:
    """Load a JSON/YAML mapping of agent IDs to resolved profiles."""
    file_path = Path(path).expanduser()
    loaded = _read_profile_file(file_path)
    if not isinstance(loaded, Mapping):
        raise ProfileValidationError("profile file root must be an object")
    # An optional envelope makes schema and mode explicit without breaking old files.
    entries = loaded.get("profiles", loaded)
    if not isinstance(entries, Mapping):
        raise ProfileValidationError("profiles must be an object")
    declared_mode = loaded.get("role_value_mode")
    if declared_mode is not None and declared_mode != role_value_mode:
        raise ProfileValidationError(
            f"profile file declares role_value_mode={declared_mode!r}, requested {role_value_mode!r}"
        )
    return {
        str(profile_id): resolve_profile_entry(
            str(profile_id),
            entry,
            role_value_mode,
            criteria=criteria,
            source_path=str(file_path),
            allow_hard_value=allow_hard_value,
        )
        for profile_id, entry in entries.items()
    }
