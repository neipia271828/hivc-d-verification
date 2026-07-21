"""REQUIREMENTS §7 の3条件（control / consulting / hivc_d）バッチ実験用 共有モジュール。

REQUIREMENTS §7.1 のマルチエージェント議論・意思決定機会を実装する。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# hivc_sim/turn_game.py を import テストでも解決できるよう、リポジトリルートから hivc_sim を sys.path に追加
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HIVC_SIM_PATH = _REPO_ROOT / "hivc_sim"
if str(_HIVC_SIM_PATH) not in sys.path:
    sys.path.insert(0, str(_HIVC_SIM_PATH))

from profiles import DEFAULT_VALUE_CRITERIA_SCHEMA, ROLE_VALUE_MODES  # noqa: E402
from turn_game import (
    ACTION_EFFECT_DESCRIPTIONS,
    ACTION_LABELS,
    ALL_ACTIONS,
    Action,
    EVENT_EFFECT_DESCRIPTIONS,
    EVENT_LABELS,
    TURN_START_BASE_EFFECT,
    acceptable_actions,
    best_action,
    estimate_q_values,
    initial_state,
    optimal_route,
    role_specific_evidence,
    route_of_action,
    preview_action_safety,
    preview_turn_start_state,
    step,
    terminal_score,
)

PRIORITY_LEVELS = ("high", "mid", "low")


def _qualitative_level(value: Any) -> str:
    """既存の0..1設定をモデル表示用の離散レベルへ変換する。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip().lower()
        return text if text in PRIORITY_LEVELS else "mid"
    if number >= 0.30:
        return "high"
    if number >= 0.15:
        return "mid"
    return "low"


def _qualitative_scalar_level(value: Any) -> str:
    """confidenceや傾向の0..1設定を三分割してモデル向けに離散化する。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip().lower()
        return text if text in PRIORITY_LEVELS else "mid"
    if number >= 2 / 3:
        return "high"
    if number >= 1 / 3:
        return "mid"
    return "low"


def _qualitative_value_profile(value: Any) -> Any:
    """Value profileからmodel-facingな数値重み・数値confidenceを除く。"""
    if not isinstance(value, dict):
        return value
    result = {key: item for key, item in value.items() if key not in {"initial_priority_weights", "confidence"}}
    weights = value.get("initial_priority_weights")
    if isinstance(weights, dict):
        result["initial_priority_levels"] = {
            str(key): _qualitative_level(item) for key, item in weights.items()
        }
    if "confidence" in value:
        result["confidence_level"] = _qualitative_scalar_level(value["confidence"])
    return result


# REQUIREMENTS §7: 条件間で差をつけるのは合意形成手順の指示のみ。
CONDITION_PROCEDURES: dict[str, str] = {
    "control": "",
    "consulting": (
        "【一般的な合意形成・リスク管理ガイド】\n"
        "このターンの自由議論と最終投票では、状況を整理し、選択肢のリスクと便益を比較し、実行前に確認する。\n"
        "このガイドはゲーム規則・可視情報・出力JSONの契約を変更しない。見えていない状態を推測で事実扱いせず、\n"
        "相手の役割固有情報が必要なときは質問して確認する。\n\n"
        "状況を整理する\n"
        "- まず、自分に可視な現在状態、イベント、資源残量、勝敗に近い危険を、具体的な値または根拠とともに短く共有する。\n"
        "- 相手の主張に使われた数値・イベント・因果関係が不明または矛盾する場合は、結論を急がず、何を確認したいかを一つ質問する。\n"
        "- 相手から質問を受けた場合は、先にその質問へ直接回答する。観測できない値は『不明』とし、推測で補わない。\n\n"
        "選択肢のリスクと便益を比較する\n"
        "- 次に、各行動案について、直近の敗北リスク、勝利条件への寄与、資源消費、次ターンに残る選択肢を比較する。\n"
        "- 相手と結論が異なる場合は、両案の利点・不利点と、どの事実が採否を分けるかを明示する。\n"
        "- 根拠が強い提案や新しい事実が示された場合は、当初案に固執せず、より安全または合理的な案へ譲歩する。\n\n"
        "実行前に確認する\n"
        "- 有力になった行動について、現在観測できる資源で実行可能か、直後に敗北条件を悪化させないか、\n"
        "  どの前提が未確認かを確認する。\n"
        "- 前提が不足する場合は、断定的な合意ではなく、確認後に採る条件付き提案を示す。\n"
        "- 行動案または最終投票の reason には、確認した事実、比較したリスクと便益、実行上の制約を簡潔に結び付ける。\n\n"
        "最終投票前チェック: (1) 相手の重要情報または質問を無視していないか、(2) 採用案の主な便益とリスクを説明できるか、\n"
        "(3) 提案行動が観測済みの制約と敗北条件に矛盾しないかを確認してから回答する。"
        "判断基準の測定は全条件で同じタイミング・JSON契約を使い、測定後に投票を変更しない。"
    ),
    "hivc_d": (
        "【HIVC-D 合意形成プロトコル：I → V → A】\n"
        "このターンの自由議論と最終投票では、必ず I、V、A の順で考える。\n"
        "このプロトコルはゲーム規則・可視情報・出力JSONの契約を変更しない。見えていない状態を推測で事実扱いせず、\n"
        "相手の役割固有情報が必要なときは質問して確認する。\n\n"
        "I（Information: 情報の共有）\n"
        "- まず、自分に可視な現在状態、イベント、資源残量、勝敗に近い危険を、具体的な値または根拠とともに短く共有する。\n"
        "- 相手の主張に使われた数値・イベント・因果関係が不明または矛盾する場合は、結論を急がず、何を確認したいかを一つ質問する。\n"
        "- 相手から質問を受けた場合は、先にその質問へ直接回答する。観測できない値は『不明』とし、推測で補わない。\n\n"
        "V（Value: 判断基準の整合）\n"
        "- 次に、各自が何を優先しているか（例: 直近の敗北回避、救助達成、資源温存）を事実判断と分けて明示する。\n"
        "- 優先順位が異なる場合は、当該ターンの観測事実に基づき、両者が受け入れられる共通基準 V* を提案する。\n"
        "  基準の具体的な順序を外部から補完せず、proposal_id、ordered_criteria、scope を明示する。\n"
        "- 各自が同一 proposal_id と同一内容へ accept した場合だけ V* が成立する。reject/counter や欠落は合意にしない。\n"
        "- V* に基づき、対立する行動案を比較し、どの根拠が採否を分けるかを述べる。自分の当初案に固執しない。\n\n"
        "A（Ability: 実行可能性の確認）\n"
        "- 最後に、V* で有力になった行動について、現在観測できる資源で実行可能か、直後に敗北条件を悪化させないか、\n"
        "  どの前提が未確認かを確認する。\n"
        "- 前提が不足する場合は、断定的な合意ではなく、確認後に採る条件付き提案を示す。\n"
        "- 行動案または最終投票の reason には、Iで確認した事実、V* の比較基準、Aで確認した制約を簡潔に結び付ける。\n\n"
        "最終投票前チェック: (1) 相手の重要情報または質問を無視していないか、(2) V* を一文で説明できるか、\n"
        "(3) 提案行動が観測済みの制約と敗北条件に矛盾しないかを確認してから回答する。"
    ),
    "hivc_d_prescribed_v1": (
        "【HIVC-D prescribed V* v1：感度分析】\n"
        "外部規定基準『直近の破局回避 → 勝利条件への寄与 → 次ターンの選択肢』を今ターンの比較順序として使う。\n"
        "これは自由なV整合ではなく、外部V*への適応条件として扱う。"
    ),
}

CONDITIONS: tuple[str, ...] = ("control", "consulting", "hivc_d", "hivc_d_prescribed_v1")


PERSONAS = {
    "alpha": "安全管理担当。酸素、浸水、船体損傷などの破局リスクを最優先で見る。",
    "beta": "通信・電力担当。救助要請に必要な通信復旧と電力維持を最優先で見る。",
}

DEFAULT_PERSONA_PARAMS: dict[str, dict[str, object]] = {
    "alpha": {
        "role": "安全管理担当",
        "priority_weights": {
            "oxygen": 0.30,
            "power": 0.10,
            "hull_damage": 0.25,
            "flooding": 0.25,
            "communication": 0.10,
        },
        "risk_tolerance": 0.20,
        "goal_focus": "survival_first",
        "communication_style": "cautious",
        "concession_tendency": 0.35,
        "evidence_demand": 0.70,
        "notes": "短期的な勝利よりも施設崩壊の回避を重視する。",
    },
    "beta": {
        "role": "通信・電力担当",
        "priority_weights": {
            "oxygen": 0.10,
            "power": 0.25,
            "hull_damage": 0.10,
            "flooding": 0.10,
            "communication": 0.45,
        },
        "risk_tolerance": 0.55,
        "goal_focus": "mission_first",
        "communication_style": "goal_directed",
        "concession_tendency": 0.45,
        "evidence_demand": 0.50,
        "notes": "勝利条件の達成速度を重視する。",
    },
}


class SpeechAct(str, Enum):
    EVIDENCE = "evidence"
    QUESTION_OBJECTION = "question_objection"
    TRADEOFF = "tradeoff"
    CONCESSION_INTEGRATION = "concession_integration"
    INFORMATION_REQUEST = "information_request"
    QUESTION = "question"


QUESTION_SPEECH_ACTS = frozenset({
    SpeechAct.QUESTION_OBJECTION,
    SpeechAct.INFORMATION_REQUEST,
    SpeechAct.QUESTION,
})


V_STAR_RESPONSE_TYPES = frozenset({"accept", "reject", "counter"})


def _read_declared_role_value_mode(path: Path) -> str | None:
    """Return the top-level role_value_mode declared in a JSON/YAML profile file."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            from profiles import yaml as _yaml

            if _yaml is None:
                return None
            data = _yaml.safe_load(text)
        else:
            return None
    except Exception:
        return None
    if isinstance(data, dict):
        return data.get("role_value_mode")
    return None


_DEFAULT_ROLE_FILES: dict[str, str | None] = {
    "soft_value": str(_REPO_ROOT / "configs" / "profiles_soft_value.yaml"),
    "expertise_only": str(_REPO_ROOT / "configs" / "profiles_expertise_only.yaml"),
    "legacy_hard": None,
}
_KNOWN_ROLE_FILES: set[str] = {p for p in _DEFAULT_ROLE_FILES.values() if p is not None}


def resolve_role_file_path(role_file: str | Path | None, role_value_mode: str | None) -> str | None:
    """Select a role_file compatible with the requested role_value_mode.

    If the requested mode is unknown, the original path is returned unchanged.
    If no path is given, the canonical default for the mode is used.
    If one of the known default files is supplied but does not match the mode,
    the correct default file is substituted automatically (this covers the
    common case of overriding only --role-value-mode via CLI).
    Custom files are left as-is; mismatches will be caught by load_profiles.
    """
    if role_value_mode not in ROLE_VALUE_MODES:
        return str(role_file) if role_file is not None else None
    default = _DEFAULT_ROLE_FILES.get(role_value_mode)
    if not role_file:
        return default
    path = Path(str(role_file)).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    normalized = str(path)
    if normalized in _KNOWN_ROLE_FILES:
        return default
    declared = _read_declared_role_value_mode(path)
    if declared is not None and declared != role_value_mode:
        # Keep custom files as the user supplied them; the validation error
        # from load_profiles will make a mismatch explicit.
        return normalized
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _profile_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def condition_order_for_seed(conditions: list[str], game_seed: int) -> list[str]:
    """Return a deterministic per-seed permutation to remove fixed-order confounds."""
    ordered = list(conditions)
    random.Random(game_seed ^ 0x5EEDC0DE).shuffle(ordered)
    return ordered


def build_value_manifest(
    config: dict[str, Any],
    personas: dict[str, str],
    persona_params: dict[str, dict[str, object] | None],
    role_keys: dict[str, str | None],
    *,
    role_value_mode: str,
    framework_ids: list[str] | tuple[str, ...],
    resolved_profiles: dict[str, Any] | None = None,
    runner_version: str = "v-flow-2",
) -> dict[str, Any]:
    """Build the immutable §9.2 run snapshot (the manifest is the authority)."""
    resolved_profiles = resolved_profiles or {}
    profile_source = config.get("role_file") or config.get("personas_file")
    source_path = Path(str(profile_source)).expanduser() if profile_source else None
    if source_path is not None and not source_path.is_absolute():
        source_path = _REPO_ROOT / source_path
    source_body: Any = None
    if source_path is not None and source_path.is_file():
        try:
            source_body = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            source_body = source_path.read_text(encoding="utf-8")

    roles: dict[str, Any] = {}
    persona_entries: dict[str, Any] = {}
    values: dict[str, Any] = {}
    negotiation_traits: dict[str, Any] = {}
    resolved_entries: dict[str, Any] = {}
    for agent in ("alpha", "beta"):
        resolved = resolved_profiles.get(agent)
        if resolved is not None and hasattr(resolved, "to_dict"):
            resolved_body = resolved.to_dict()
            resolved_entries[agent] = resolved_body
            role_body = resolved_body.get("role") or {}
            persona_body = resolved_body.get("persona") or {}
            value_body = resolved_body.get("value")
        else:
            params = persona_params.get(agent) or {}
            separated = params.get("_resolved_profile")
            if isinstance(separated, dict):
                role_body = separated.get("role") or {}
                persona_body = separated.get("persona") or {}
                value_body = separated.get("value")
            else:
                role_body = {"id": role_keys.get(agent), "label": personas.get(agent), "legacy_body": params}
                persona_body = {
                    "id": str(params.get("persona_id", f"{agent}-persona")),
                    "version": str(params.get("version", "legacy-1")),
                    "communication_style": params.get("communication_style"),
                    "evidence_demand": params.get("evidence_demand"),
                }
                value_body = None
                weights = params.get("priority_weights")
                if isinstance(weights, dict):
                    value_body = {
                        "id": str(params.get("value_profile_id", role_keys.get(agent) or f"{agent}-legacy-value")),
                        "version": str(params.get("version", "legacy-1")),
                        "initial_priority_weights": weights,
                        "negotiable": role_value_mode != "legacy_hard",
                    }
            resolved_entries[agent] = {
                "role": role_body,
                "persona": persona_body,
                "value": value_body,
                "role_value_mode": role_value_mode,
                "source_path": str(source_path) if source_path else None,
            }
        roles[agent] = {
            **role_body,
            "body": role_body,
            "sha256": _profile_sha256(role_body),
            "input_path": str(source_path) if source_path else None,
        }
        persona_entries[agent] = {
            **persona_body,
            "body": persona_body,
            "sha256": _profile_sha256(persona_body),
            "input_path": str(source_path) if source_path else None,
        }
        if value_body is not None:
            values[agent] = {
                **value_body,
                "body": value_body,
                "sha256": _profile_sha256(value_body),
                "input_path": str(source_path) if source_path else None,
            }
        negotiation_traits[agent] = {
            key: persona_body.get(key, (persona_params.get(agent) or {}).get(key, 0.5))
            for key in ("concession_tendency", "consensus_orientation", "dominance")
        }

    frameworks = {
        framework_id: {
            "id": framework_id,
            "version": "2" if framework_id == "hivc_d" else "1",
            "body": CONDITION_PROCEDURES.get(framework_id, ""),
            "sha256": _profile_sha256(CONDITION_PROCEDURES.get(framework_id, "")),
        }
        for framework_id in framework_ids
    }
    config_snapshot = json.loads(json.dumps(config, ensure_ascii=False, default=str))
    scenario_value = config.get("scenarios", config.get("scenario_id"))
    scenario_range = (
        {"scenarios": scenario_value}
        if scenario_value is not None
        else {"not_applicable_reason": "no_scenario_filter_specified"}
    )
    value_criteria_body = DEFAULT_VALUE_CRITERIA_SCHEMA.to_dict()
    return {
        "schema_version": "value-manifest-2",
        "value_criteria_schema": {
            **value_criteria_body,
            "body": value_criteria_body,
            "sha256": DEFAULT_VALUE_CRITERIA_SCHEMA.sha256,
        },
        "role_value_mode": role_value_mode,
        "role_profiles": roles,
        "persona_profiles": persona_entries,
        "value_profiles": values,
        "resolved_profiles": resolved_entries,
        "negotiation_traits": negotiation_traits,
        "frameworks": frameworks,
        "model": {
            "path_or_id": config.get("model_path"),
            "generation": {
                key: config.get(key)
                for key in ("max_new_tokens", "enable_thinking", "thinking_budget", "do_sample")
            },
        },
        "seed_range": {"start": config.get("seed"), "count": config.get("games")},
        "scenario_range": scenario_range,
        "experiment_config": config_snapshot,
        "experiment_config_sha256": _profile_sha256(config_snapshot),
        "profile_input": {"path": str(source_path) if source_path else None, "body": source_body},
        "git_commit": _git_commit(),
        "started_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "runner_version": runner_version,
    }


def write_value_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_profile_assignment(
    manifest: dict[str, Any],
    seed: int,
    personas: dict[str, str],
    persona_params: dict[str, dict[str, object] | None],
    role_keys: dict[str, str | None],
    condition: str | None = None,
) -> None:
    """Record the exact per-game assignment as an authoritative snapshot.

    固定プロファイルでもseed・conditionごとに明示的な割当レコードを生成する。
    同一seedのframework条件間ではrole_value_assignment_idを共有する。
    """
    agents: dict[str, Any] = {}
    for agent in ("alpha", "beta"):
        params = persona_params.get(agent) or {}
        resolved = params.get("_resolved_profile")
        body = resolved if isinstance(resolved, dict) else {
            "role_key": role_keys.get(agent), "persona": personas.get(agent), "legacy_body": params
        }
        agents[agent] = {
            "profile_key": role_keys.get(agent),
            "body": body,
            "sha256": _profile_sha256(body),
        }
    assignment_id = _role_value_assignment_id(personas, persona_params, role_keys, seed)
    entry: dict[str, Any] = {
        "role_value_assignment_id": assignment_id,
        "seed": seed,
        "agents": agents,
    }
    if condition is not None:
        entry["condition"] = condition
    manifest.setdefault("game_profile_assignments", []).append(entry)


def _normalize_v(
    value: Any,
    schema=DEFAULT_VALUE_CRITERIA_SCHEMA,
) -> dict[str, Any] | None:
    """Validate the model-facing V representation against the common ontology.

    新しいV測定は完全なcriteria集合と high/mid/low のpriority_levelsを必要とする。
    既存run・legacy backendからの数値weightsも読み取り互換のため受理する。
    """
    if not isinstance(value, dict):
        return None
    expected = set(schema.criteria)
    criteria = value.get("ordered_criteria")
    levels = value.get("priority_levels")
    weights = value.get("weights")
    confidence_level = value.get("confidence_level")
    confidence = value.get("confidence")
    if not isinstance(criteria, list) or not criteria or not all(isinstance(v, str) and v.strip() for v in criteria):
        return None
    ordered = [v.strip() for v in criteria]
    if set(ordered) != expected or len(ordered) != len(expected):
        return None
    if isinstance(levels, dict):
        normalized_levels = {str(k): str(v).strip().lower() for k, v in levels.items()}
        if set(normalized_levels) != expected or any(v not in PRIORITY_LEVELS for v in normalized_levels.values()):
            return None
        if confidence_level is not None and str(confidence_level).strip().lower() not in PRIORITY_LEVELS:
            return None
        result: dict[str, Any] = {"ordered_criteria": ordered, "priority_levels": normalized_levels}
        if confidence_level is not None:
            result["confidence_level"] = str(confidence_level).strip().lower()
        return result
    if not isinstance(weights, dict) or set(weights) != expected:
        return None
    try:
        numeric = {str(k): float(v) for k, v in weights.items()}
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(v) or v < 0 for v in numeric.values()) or sum(numeric.values()) <= 0:
        return None
    total = sum(numeric.values())
    normalized_weights = {k: v / total for k, v in numeric.items()}
    try:
        normalized_confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        return None
    if normalized_confidence is not None and not 0 <= normalized_confidence <= 1:
        return None
    result: dict[str, Any] = {"ordered_criteria": ordered, "weights": normalized_weights}
    if normalized_confidence is not None:
        result["confidence"] = normalized_confidence
    return result


def v_alignment_distance(first: dict[str, Any] | None, second: dict[str, Any] | None) -> float:
    """V差。新形式ではpriority levelの不一致率、legacyではL1距離を返す。"""
    if not first or not second:
        return float("nan")
    first_levels = first.get("priority_levels")
    second_levels = second.get("priority_levels")
    if isinstance(first_levels, dict) and isinstance(second_levels, dict):
        if set(first_levels) != set(second_levels) or not first_levels:
            return float("nan")
        return sum(first_levels[k] != second_levels[k] for k in first_levels) / len(first_levels)
    if not isinstance(first.get("weights"), dict) or not isinstance(second.get("weights"), dict):
        return float("nan")
    first_weights = first["weights"]
    second_weights = second["weights"]
    if set(first_weights) != set(second_weights):
        return float("nan")
    return float(sum(abs(float(first_weights[k]) - float(second_weights[k])) for k in first_weights))


def _top_criterion(v: dict[str, Any] | None) -> str | None:
    """Return the top criterion in a comparable form, or None."""
    if not v:
        return None
    ordered = v.get("ordered_criteria")
    if isinstance(ordered, (list, tuple)) and ordered:
        return str(ordered[0]).strip().casefold()
    weights = v.get("weights")
    if isinstance(weights, dict) and weights:
        return str(max(weights, key=weights.get)).strip().casefold()  # type: ignore[arg-type]
    return None


def v_alignment_required(
    alpha_action: Action | None,
    beta_action: Action | None,
    alpha_v: dict[str, Any] | None,
    beta_v: dict[str, Any] | None,
    threshold: float = 0.20,
) -> tuple[bool, list[str]]:
    """Action案が衝突した場合だけV整合を要求する。

    Role由来のpriority差は監査用distanceとして別途記録するが、それだけで
    V交渉を強制しない。異なるValueから同じActionへ到達することを許容する。
    """
    reasons: list[str] = []
    if alpha_action is not None and beta_action is not None and alpha_action != beta_action:
        reasons.append("action_before_mismatch")
    del alpha_v, beta_v, threshold
    return bool(reasons), reasons


def _role_value_assignment_id(
    personas: dict[str, str],
    persona_params: dict[str, dict[str, object] | None],
    role_keys: dict[str, str | None],
    seed: int,
) -> str:
    """同一seedの全条件で共通のrole-value割当ID。"""
    body = {
        "seed": seed,
        "alpha_role_key": role_keys.get("alpha"),
        "beta_role_key": role_keys.get("beta"),
        "alpha_persona_params": persona_params.get("alpha"),
        "beta_persona_params": persona_params.get("beta"),
    }
    return _profile_sha256(body)[:16]


def _normalize_requested_fields(value: Any) -> list[str]:
    """requested_fields を正規化したソート済みリストに変換する。

    None・空・無効型の場合は空リストを返す。
    """
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return []
    normalized: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip().lower()
            if text:
                normalized.append(text)
    return sorted(set(normalized))


def _question_signature(item: dict[str, Any]) -> tuple[str, str, str]:
    """質問の (speaker, addressed_to, normalized_requested_fields) signature。

    requested_fields が明示されている場合はそれを正規化して使用する。
    明示されていない場合は action + reason + message を正規化してフォールバックする。
    """
    speaker = str(item.get("speaker", "")).strip().lower()
    addressed_to = str(item.get("addressed_to", "")).strip().lower()
    requested_fields = _normalize_requested_fields(item.get("requested_fields"))
    if requested_fields:
        fields = _canonical_json({"requested_fields": requested_fields})
    else:
        action = str(item.get("action", "")).strip().lower()
        reason = str(item.get("reason", "")).strip().lower()
        message = str(item.get("message", "")).strip().lower()
        fields = _canonical_json({"action": action, "reason": reason, "message": message})
    return (speaker, addressed_to, fields)


def extract_json_v_measurement(response: str) -> tuple[dict[str, Any] | None, Action | None, str, str]:
    """Parse an independent v_before/action_before or post-vote v_after response."""
    try:
        payload = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError):
        return None, None, "", "invalid_json"
    if not isinstance(payload, dict):
        return None, None, "", "invalid_payload"
    raw_v = payload.get("v_before", payload.get("v_after"))
    v = _normalize_v(raw_v)
    action_text = str(payload.get("action_before", "")).strip().upper()
    action = Action(action_text) if action_text in {a.value for a in ALL_ACTIONS} else None
    reason = str(payload.get("reason_before", payload.get("reason_after", ""))).strip()
    error = "" if v is not None else "invalid_or_missing_v"
    return v, action, reason, error


def _normalize_v_proposal(
    value: Any,
    fallback_id: str,
    schema=DEFAULT_VALUE_CRITERIA_SCHEMA,
) -> dict[str, Any] | None:
    """Validate a V* proposal against the common ontology.

    順位のみの提案も許容するが、ordered_criteria は全criteriaを重複なく含む必要がある。
    priority_levels が含まれる場合も同じcriteria集合と high/mid/low 値を持つ必要がある。
    legacyの数値weightsも読み取り互換のため受理する。
    """
    if not isinstance(value, dict):
        return None
    expected = set(schema.criteria)
    proposal_id = str(value.get("proposal_id", value.get("id", fallback_id))).strip()
    criteria = value.get("ordered_criteria")
    if not proposal_id or not isinstance(criteria, list) or not criteria:
        return None
    ordered = [str(item).strip() for item in criteria]
    if set(ordered) != expected or len(ordered) != len(expected):
        return None
    scope = str(value.get("scope", "turn")).strip().lower()
    if scope not in {"turn", "game"}:
        return None
    proposal = {"proposal_id": proposal_id, "ordered_criteria": ordered, "scope": scope}
    if isinstance(value.get("priority_levels"), dict):
        levels = {str(k): str(v).strip().lower() for k, v in value["priority_levels"].items()}
        if set(levels) != expected or any(v not in PRIORITY_LEVELS for v in levels.values()):
            return None
        proposal["priority_levels"] = levels
    if isinstance(value.get("weights"), dict):
        weights = value["weights"]
        if set(weights) != expected:
            return None
        try:
            numeric = {str(k): float(v) for k, v in weights.items()}
        except (TypeError, ValueError):
            return None
        if any(not math.isfinite(v) or v < 0 for v in numeric.values()) or sum(numeric.values()) <= 0:
            return None
        total = sum(numeric.values())
        proposal["weights"] = {k: v / total for k, v in numeric.items()}
    return proposal


def parse_v_negotiation(payload: Any, speaker: str, message_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Parse proposal and accept/reject/counter without ever inferring acceptance.

    counter応答で counter_proposal と同時に self_accept=true が指定された場合、
    返す response に self_accept_for_counter_id を含める。
    呼び出し側はこのフラグを見て counter提案への明示的acceptを別途記録できる。
    """
    if not isinstance(payload, dict):
        return None, None
    proposal = _normalize_v_proposal(payload.get("v_proposal"), f"{speaker}-message-{message_id}")
    message_index = int(message_id) if str(message_id).isdigit() else 0
    if proposal is not None:
        proposal["message_index"] = message_index
    raw_response = payload.get("v_star_response")
    response: dict[str, Any] | None = None
    if isinstance(raw_response, dict):
        response_type = str(raw_response.get("response", raw_response.get("type", ""))).strip().lower()
        proposal_id = str(raw_response.get("proposal_id", "")).strip()
        if response_type in V_STAR_RESPONSE_TYPES and proposal_id:
            response = {"response": response_type, "proposal_id": proposal_id, "message_index": message_index}
            if response_type == "counter":
                counter = _normalize_v_proposal(raw_response.get("counter_proposal"), f"{speaker}-message-{message_id}-counter")
                if counter is not None:
                    counter["message_index"] = message_index
                    response["counter_proposal"] = counter
                    # counter出力で counter提案への明示的self-acceptが同時表現された場合
                    self_accept_flag = raw_response.get("self_accept")
                    if isinstance(self_accept_flag, bool) and self_accept_flag:
                        response["self_accept_for_counter_id"] = counter.get("proposal_id", "")
    return proposal, response


def resolve_v_star(proposals: list[dict[str, Any]], responses: dict[str, list[dict[str, Any]]]) -> tuple[str, str, dict[str, Any] | None, str]:
    """Return accepted only for explicit, matching acceptance by both agents."""
    def semantic(item: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in item.items() if k not in {"speaker", "message_id", "message_index"}}

    by_id: dict[str, dict[str, Any]] = {}
    ambiguous_ids: set[str] = set()
    for item in proposals:
        proposal_id = str(item.get("proposal_id"))
        if proposal_id in by_id and _canonical_json(semantic(by_id[proposal_id])) != _canonical_json(semantic(item)):
            ambiguous_ids.add(proposal_id)
        by_id[proposal_id] = item
    for proposal_id, proposal in reversed(list(by_id.items())):
        if proposal_id in ambiguous_ids:
            continue
        accepted_by: list[str] = []
        for agent in ("alpha", "beta"):
            proposal_index = int(proposal.get("message_index", 0))
            matching = [
                r for r in responses.get(agent, [])
                if r.get("proposal_id") == proposal_id and int(r.get("message_index", -1)) >= proposal_index
            ]
            if matching and matching[-1].get("response") == "accept":
                accepted_by.append(agent)
        if accepted_by == ["alpha", "beta"]:
            return "accepted", proposal_id, proposal, ""
    if not proposals:
        return "unresolved", "", None, "missing_v_proposal"
    if ambiguous_ids:
        return "unresolved", "", None, "proposal_id_content_mismatch"
    return "unresolved", "", None, "missing_matching_explicit_acceptance"


def ensure_unique_v_proposal_id(
    proposal: dict[str, Any],
    existing: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, str]:
    """同じIDが別内容に再利用された場合だけ、決定論的な一意IDへ修復する。"""
    original_id = str(proposal.get("proposal_id", "")).strip()
    semantic_keys = ("ordered_criteria", "priority_levels", "weights", "scope")
    semantic = {key: proposal.get(key) for key in semantic_keys if key in proposal}
    conflicting = any(
        str(item.get("proposal_id", "")).strip() == original_id
        and _canonical_json({key: item.get(key) for key in semantic_keys if key in item}) != _canonical_json(semantic)
        for item in existing
    )
    if not conflicting:
        return proposal, original_id, original_id
    digest = hashlib.sha256(_canonical_json(semantic).encode("utf-8")).hexdigest()[:10]
    speaker = str(proposal.get("speaker", "agent")).strip() or "agent"
    message_index = str(proposal.get("message_index", proposal.get("message_id", "0")))
    repaired_id = f"{original_id}--{speaker}-{message_index}-{digest}"
    return {**proposal, "proposal_id": repaired_id}, original_id, repaired_id


SPEECH_ACT_LABELS: dict[SpeechAct, str] = {
    SpeechAct.EVIDENCE: "根拠提示（状態・リスク・制約）",
    SpeechAct.QUESTION_OBJECTION: "質問・反論",
    SpeechAct.TRADEOFF: "トレードオフ比較",
    SpeechAct.CONCESSION_INTEGRATION: "譲歩案・統合案",
    SpeechAct.INFORMATION_REQUEST: "情報要請・要約",
    SpeechAct.QUESTION: "質問",
}


def get_role_entry(
    loaded: dict[str, object],
    requested_key: str | None,
    fallback_key: str,
    used_keys: set[str],
) -> tuple[str, dict[str, object] | str]:
    if requested_key:
        if requested_key not in loaded:
            raise KeyError(f"Role key not found in role file: {requested_key}")
        return requested_key, loaded[requested_key]  # type: ignore[index]
    if fallback_key in loaded and fallback_key not in used_keys:
        return fallback_key, loaded[fallback_key]  # type: ignore[index]
    for key in sorted(loaded):
        if key not in used_keys and isinstance(loaded[key], (dict, str)):
            return key, loaded[key]  # type: ignore[index]
    raise ValueError("No usable role entries found in role file.")


def apply_role_entry(
    agent_slot: str,
    role_key: str,
    role_entry: dict[str, object] | str,
    personas: dict[str, str],
    persona_params: dict[str, dict[str, object] | None],
    role_keys: dict[str, str | None],
) -> None:
    role_keys[agent_slot] = role_key
    if isinstance(role_entry, dict):
        params = dict(role_entry)
        params.setdefault("role_key", role_key)
        persona_params[agent_slot] = params
        personas[agent_slot] = str(params.get("role", PERSONAS[agent_slot]))
    else:
        persona_params[agent_slot] = None
        personas[agent_slot] = str(role_entry)


def load_personas(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, dict[str, object] | None], dict[str, str | None]]:
    """ペルソナを読み込む。

    random_persona が真の場合、role_file から重複なしで2エージェントをランダム選択する。
    その場合 args.random_seed があればそれを使い、なければゲーム seed とは独立に
    決定論的に選ぶ（再現性のため）。
    """
    personas = dict(PERSONAS)
    persona_params: dict[str, dict[str, object] | None] = {"alpha": None, "beta": None}
    role_keys: dict[str, str | None] = {"alpha": None, "beta": None}

    random_persona = getattr(args, "random_persona", False)
    random_seed = getattr(args, "random_seed", None)

    role_value_mode = getattr(args, "role_value_mode", None)
    role_file = resolve_role_file_path(
        getattr(args, "role_file", None) or getattr(args, "personas_file", None),
        role_value_mode,
    )
    if role_file is not None and getattr(args, "role_file", None) is None and getattr(args, "personas_file", None) is not None:
        args.personas_file = role_file
    elif role_file is not None:
        args.role_file = role_file
    if role_file and role_value_mode in {"legacy_hard", "soft_value", "expertise_only"}:
        from profiles import load_profiles

        resolved = load_profiles(role_file, role_value_mode)
        available = sorted(resolved)
        if len(available) < 2:
            raise ValueError(f"{role_value_mode} requires at least two profile entries")
        if random_persona:
            import random as _random

            chosen = _random.Random(random_seed if random_seed is not None else 0).sample(available, 2)
        else:
            alpha_key = getattr(args, "alpha_role_key", None) or ("agent_01" if "agent_01" in resolved else available[0])
            remaining = [key for key in available if key != alpha_key]
            beta_key = getattr(args, "beta_role_key", None) or ("agent_02" if "agent_02" in remaining else remaining[0])
            chosen = [alpha_key, beta_key]
        for agent, profile_key in zip(("alpha", "beta"), chosen):
            if profile_key not in resolved:
                raise KeyError(f"Profile key not found in role file: {profile_key}")
            profile = resolved[profile_key]
            body = profile.to_dict()
            role_keys[agent] = profile_key
            personas[agent] = profile.role.label
            persona_params[agent] = {
                "_resolved_profile": body,
                "role": profile.role.label,
                "communication_style": profile.persona.communication_style,
                "evidence_demand": profile.persona.evidence_demand,
                "concession_tendency": profile.persona.concession_tendency,
                "consensus_orientation": profile.persona.consensus_orientation,
                "dominance": profile.persona.dominance,
            }
        return personas, persona_params, role_keys
    if role_file:
        path = Path(role_file).expanduser()
        with path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("Role/persona file must contain a JSON object.")

        # random_persona モード: role_file から重複なしで2エージェントをランダム選択
        if random_persona:
            import random as _random
            agent_keys = [k for k in sorted(loaded) if isinstance(loaded[k], (dict, str))]
            if len(agent_keys) < 2:
                raise ValueError(f"random_persona requires >=2 agent entries in {role_file}, got {len(agent_keys)}")
            rng = _random.Random(random_seed if random_seed is not None else 0)
            chosen = rng.sample(agent_keys, 2)
            apply_role_entry("alpha", chosen[0], loaded[chosen[0]], personas, persona_params, role_keys)
            apply_role_entry("beta", chosen[1], loaded[chosen[1]], personas, persona_params, role_keys)
            return personas, persona_params, role_keys

        for key in ("alpha", "beta"):
            if key in loaded:
                if isinstance(loaded[key], dict):
                    persona_params[key] = loaded[key]
                    personas[key] = str(loaded[key].get("role", PERSONAS[key]))
                    role_keys[key] = key
                else:
                    personas[key] = str(loaded[key])
                    role_keys[key] = key

        missing_slots = [slot for slot in ("alpha", "beta") if role_keys[slot] is None]
        if missing_slots:
            used_keys: set[str] = {key for key in role_keys.values() if key is not None}
        if "alpha" in missing_slots:
            alpha_key, alpha_entry = get_role_entry(loaded, args.alpha_role_key, "agent_01", used_keys)
            used_keys.add(alpha_key)
            apply_role_entry("alpha", alpha_key, alpha_entry, personas, persona_params, role_keys)
        if "beta" in missing_slots:
            beta_key, beta_entry = get_role_entry(loaded, args.beta_role_key, "agent_02", used_keys)
            apply_role_entry("beta", beta_key, beta_entry, personas, persona_params, role_keys)

    if args.persona_params_file:
        path = Path(args.persona_params_file).expanduser()
        with path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        for key in ("alpha", "beta"):
            if key in loaded and isinstance(loaded[key], dict):
                persona_params[key] = loaded[key]
                personas[key] = str(loaded[key].get("role", personas[key]))
    if role_value_mode == "legacy_hard" and not role_file:
        persona_params = {agent: dict(values) for agent, values in DEFAULT_PERSONA_PARAMS.items()}
        role_keys = {"alpha": "alpha-default-legacy", "beta": "beta-default-legacy"}
    if args.alpha_persona:
        personas["alpha"] = args.alpha_persona
        persona_params["alpha"] = None
        role_keys["alpha"] = None
    if args.beta_persona:
        personas["beta"] = args.beta_persona
        persona_params["beta"] = None
        role_keys["beta"] = None
    return personas, persona_params, role_keys


def add_persona_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--role-value-mode",
        choices=("legacy_hard", "soft_value", "expertise_only"),
        default=None,
        help="Role/Value regime. Explicit selection enables independent V measurement.",
    )
    parser.add_argument("--alpha-persona", default=None)
    parser.add_argument("--beta-persona", default=None)
    parser.add_argument("--personas-file", default=None, help='JSON file such as {"alpha": "...", "beta": "..."}')
    parser.add_argument("--role-file", default=None, help="Role JSON file. Supports agent_01/agent_02 style keys.")
    parser.add_argument("--alpha-role-key", default=None, help="Role key to assign to alpha, e.g. agent_01.")
    parser.add_argument("--beta-role-key", default=None, help="Role key to assign to beta, e.g. agent_08.")
    parser.add_argument("--persona-params-file", default=None, help="Structured persona parameter JSON file.")
    parser.add_argument("--random-persona", action="store_true", default=False,
                        help="role_file から重複なしで2エージェントをランダム選択する。")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="random_persona の抽選シード（未指定時はゲーム seed を使用）。")


def _role_body(role: Any | None) -> dict[str, Any] | None:
    if role is None:
        return None
    if hasattr(role, "to_dict"):
        role = role.to_dict()
    return role if isinstance(role, dict) else None


def _resolved_role_from_params(persona_params: dict[str, object] | None) -> dict[str, Any] | None:
    if not persona_params:
        return None
    resolved = persona_params.get("_resolved_profile")
    if not isinstance(resolved, dict):
        return None
    return _role_body(resolved.get("role"))


def format_state(state, agent_name: str | None = None, role: Any | None = None) -> str:
    rescue = "未送信" if state.rescue_eta is None else f"救助到着まであと{state.rescue_eta}ターン"

    def _v(value, hidden: bool) -> str:
        return "不明（パートナーに問い合わせ）" if hidden else str(value)

    def hidden(field: str, *aliases: str) -> bool:
        return not _state_field_visible(agent_name, role, field, *aliases)

    return "\n".join(
        [
            f"turn: {_v(state.turn, hidden('turn'))}",
            f"scenario: {_v(state.scenario_id, hidden('scenario_id', 'scenario'))}",
            f"event: {_v(f'{EVENT_LABELS[state.current_event]} ({state.current_event.value})', hidden('current_event', 'event'))}",
            f"oxygen: {_v(state.oxygen, hidden('oxygen'))}",
            f"power: {_v(state.power, hidden('power'))}",
            f"hull_damage: {_v(state.hull_damage, hidden('hull_damage'))}",
            f"flooding: {_v(state.flooding, hidden('flooding'))}",
            f"communication: {_v(state.communication, hidden('communication'))}",
            f"pod_readiness: {_v(state.pod_readiness, hidden('pod_readiness'))}",
            f"pod_integrity: {_v(state.pod_integrity, hidden('pod_integrity'))}",
            f"rescue_eta: {_v(rescue, hidden('rescue_eta'))}",
            f"morale: {_v(state.morale, hidden('morale'))}",
        ]
    )


def action_list() -> str:
    # turn_game.step と共有するcanonical効果を提示し、記号と意味の取り違えを防ぐ。
    return "\n".join(
        f"{action.value}. {ACTION_LABELS[action]}（効果: {ACTION_EFFECT_DESCRIPTIONS[action]}）"
        for action in ALL_ACTIONS
    )


def _state_field_visible(
    agent_name: str | None,
    role: Any | None,
    field: str,
    *aliases: str,
) -> bool:
    role_mapping = _role_body(role)
    if role_mapping is not None:
        scope = {str(item) for item in role_mapping.get("observation_scope", [])}
        return any(name in scope for name in (field, *aliases))
    # Legacy fallback: historical alpha/beta visibility remains unchanged.
    hidden_fields = {
        "alpha": {"communication", "pod_integrity", "pod_readiness"},
        "beta": {"hull_damage", "flooding"},
    }.get(agent_name, set())
    return field not in hidden_fields


_PROJECTED_FIELDS = (
    "oxygen", "power", "hull_damage", "flooding", "communication",
    "pod_readiness", "pod_integrity", "rescue_eta", "morale",
)

_OUTCOME_CAUSE_FIELDS = {
    "loss_oxygen": ("oxygen",),
    "loss_power": ("power",),
    "loss_hull": ("hull_damage",),
    "loss_flooding": ("flooding",),
    "loss_escape_failed": (
        "oxygen", "power", "flooding", "pod_readiness", "pod_integrity"
    ),
}


def _outcome_visible(outcome: str, agent_name: str, role: Any | None) -> bool:
    fields = _OUTCOME_CAUSE_FIELDS.get(outcome, ())
    return not fields or all(
        _state_field_visible(agent_name, role, field) for field in fields
    )


def _compact_projected_state(state: Any, agent_name: str, role: Any | None) -> str:
    values: list[str] = []
    for field in _PROJECTED_FIELDS:
        if _state_field_visible(agent_name, role, field):
            values.append(f"{field}={getattr(state, field)}")
        else:
            values.append(f"{field}=hidden")
    outcome = state.outcome if _outcome_visible(state.outcome, agent_name, role) else "hidden"
    values.append(f"outcome={outcome}")
    return ", ".join(values)


def _visible_safety_label(
    safe: bool,
    reason: str,
    projected: Any,
    agent_name: str,
    role: Any | None,
) -> str:
    if safe:
        return "safe"
    if not _outcome_visible(projected.outcome, agent_name, role):
        return "unsafe:hidden_constraint"
    return f"unsafe:{reason}"


def decision_support_block(state: Any, agent_name: str, role: Any | None = None) -> str:
    """Roleの可視範囲を守りつつ、ターン遷移と全Actionの確定予測を示す。"""
    event_visible = _state_field_visible(agent_name, role, "current_event", "event")
    event_effect = (
        EVENT_EFFECT_DESCRIPTIONS[state.current_event]
        if event_visible
        else "現在イベントは非可視。必要なら観測可能な相手へ確認"
    )
    projections: list[str] = []
    for action in ALL_ACTIONS:
        safe, reason, projected = preview_action_safety(state, action)
        safety = _visible_safety_label(safe, reason, projected, agent_name, role)
        projections.append(
            f"{action.value}: [{safety}] {_compact_projected_state(projected, agent_name, role)}"
        )
    return (
        "【TURN_TRANSITION id=turn-transition】\n"
        f"Actionより先にターン開始消費が適用される: {TURN_START_BASE_EFFECT}\n"
        f"現在イベントの効果: {event_effect}\n"
        "その後に選択Actionの効果が適用され、敗北条件を判定する。\n\n"
        "【ACTION_CATALOG id=action-catalog】\n"
        f"{action_list()}\n\n"
        "【PROJECTED_STATE_AFTER id=projected-state-after】\n"
        "以下は既知の固定効果による予測。25%事故など未確定の確率分岐は含めない。\n"
        + "\n".join(projections)
    )


def final_vote_repair_feedback(
    state: Any,
    rejected_action: Action,
    agent_name: str,
    role: Any | None = None,
) -> str:
    """拒否理由、計算順序、予測結果、安全候補を同時に返す。"""
    safe, reason, projected = preview_action_safety(state, rejected_action)
    safe_candidates = [
        action.value for action in ALL_ACTIONS
        if preview_action_safety(state, action)[0]
    ]
    event_effect = (
        EVENT_EFFECT_DESCRIPTIONS[state.current_event]
        if _state_field_visible(agent_name, role, "current_event", "event")
        else "hidden_current_event"
    )
    visible_reason = (
        reason
        if _outcome_visible(projected.outcome, agent_name, role)
        else "unsafe_hidden_constraint"
    )
    return (
        f"{visible_reason or ('safe' if safe else 'unsafe')}; rejected_action={rejected_action.value}; "
        f"calculation=state_before -> turn_start({TURN_START_BASE_EFFECT}) -> "
        f"event({event_effect}) -> "
        f"action({ACTION_EFFECT_DESCRIPTIONS[rejected_action]}); "
        f"projected_state_after=({_compact_projected_state(projected, agent_name, role)}); "
        f"safe_candidates={','.join(safe_candidates) if safe_candidates else 'none'}"
    )


def decision_relevant_hidden_fields(
    state: Any,
    observer_scope: set[str],
    partner_scope: set[str],
) -> list[str]:
    """現在の判断に影響し、相手だけが観測できる危険・経路情報を返す。"""
    candidates: list[str] = []
    event = str(getattr(getattr(state, "current_event", None), "value", ""))
    if getattr(state, "hull_damage", 0) >= 3 or event in {"pressure_spike", "current_change", "hull_fracture"}:
        candidates.append("hull_damage")
    if getattr(state, "flooding", 0) >= 3 or event in {"leak_surge", "pod_flooding", "current_change"}:
        candidates.append("flooding")
    if getattr(state, "communication", 0) >= 2 or event in {"signal_window", "relay_short"}:
        candidates.append("communication")
    if (
        getattr(state, "pod_readiness", 0) >= 1
        or getattr(state, "pod_integrity", 0) >= 1
        or event == "pod_flooding"
    ):
        candidates.extend(["pod_readiness", "pod_integrity"])
    return [
        field for field in dict.fromkeys(candidates)
        if field not in observer_scope and field in partner_scope
    ]


def _fields_already_shared(transcript: list[dict[str, Any]], speaker: str, fields: list[str]) -> set[str]:
    """相手の有効発話に明示されたcanonical field名を抽出する。"""
    shared: set[str] = set()
    for item in transcript:
        if item.get("speaker") == speaker:
            continue
        text = " ".join(str(item.get(key, "")) for key in ("message", "reason"))
        for field in fields:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(field)}(?![A-Za-z0-9_])", text, re.IGNORECASE):
                shared.add(field)
    return shared


def schedule_decision_opportunities(seed: int, turn: int, schedule_seed: int = 0, max_opportunities: int = 3) -> int:
    """ゲームseed、ターン、固定スケジュールseedから決定論的に意思決定機会数を返す。"""
    return ((seed + turn * 31 + schedule_seed * 37) % max_opportunities) + 1


def allocate_discussion_budgets(
    opportunity_count: int,
    max_discussion_turns: int,
    discussion_token_budget: int,
    n_speakers: int = 2,
) -> tuple[list[int], list[int]]:
    """実際の opportunity_count で発言数とトークン予算を配分。端数は早い機会から。

    第1回目の意思決定機会には、各エージェントが1回ずつ発言できる最小数を確保する。
    合計は max_discussion_turns ・ discussion_token_budget を超えない。
    """
    if opportunity_count <= 0:
        return [], []

    # 発言数配分
    messages_per = max_discussion_turns // opportunity_count
    message_remainder = max_discussion_turns - messages_per * opportunity_count
    message_limits = [messages_per] * opportunity_count
    for i in range(opportunity_count):
        if i < message_remainder:
            message_limits[i] += 1
    # 第1回機会には少なくとも全エージェント1回ずつ
    message_limits[0] = max(n_speakers, message_limits[0])
    # 合計が max_discussion_turns を超えないよう後続機会から削減
    if sum(message_limits) > max_discussion_turns:
        excess = sum(message_limits) - max_discussion_turns
        for i in range(opportunity_count - 1, 0, -1):
            cut = min(excess, message_limits[i])
            message_limits[i] -= cut
            excess -= cut
            if excess <= 0:
                break
        # 後続機会だけでは補えない場合（max_discussion_turns < n_speakers 等）は第1機会を抑制
        if excess > 0:
            message_limits[0] = max(0, message_limits[0] - excess)

    # トークン配分は発言数配分に比例
    total_messages = sum(message_limits)
    if total_messages == 0:
        token_limits = [0] * opportunity_count
    else:
        token_limits = [
            (discussion_token_budget * message_limits[i]) // total_messages
            for i in range(opportunity_count)
        ]
        # 端数は早い機会から
        token_shortfall = discussion_token_budget - sum(token_limits)
        for i in range(opportunity_count):
            if token_shortfall <= 0:
                break
            add = min(token_shortfall, max(0, message_limits[i]))
            token_limits[i] += add
            token_shortfall -= add

    return message_limits, token_limits


def priority_agent(seed: int, turn: int) -> str:
    """フォールバック優先エージェントを返す。"""
    return "alpha" if (seed + turn) % 2 == 0 else "beta"


def format_transcript_text(transcript: list[dict[str, Any]]) -> str:
    """会話トランスクリプトをプロンプト用テキストに整形する。"""
    if not transcript:
        return "まだ議論はありません。"
    lines: list[str] = []
    for item in transcript:
        speaker = item.get("speaker", "unknown")
        parts: list[str] = [f"{speaker}:"]
        speech_act = item.get("speech_act")
        if speech_act:
            parts.append(f"[{speech_act}]")
        message = item.get("message", "")
        if message:
            parts.append(message)
        extras: list[str] = []
        if item.get("action"):
            extras.append(f"action={item['action']}")
        if item.get("reason"):
            extras.append(f"reason={item['reason']}")
        if "ready" in item:
            extras.append(f"ready={item['ready']}")
        if item.get("v_proposal"):
            extras.append(f"v_proposal={_canonical_json(item['v_proposal'])}")
        if item.get("v_star_response"):
            extras.append(f"v_star_response={_canonical_json(item['v_star_response'])}")
        if extras:
            parts.append(f"({' | '.join(extras)})")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _normalize_speech_act(value: Any) -> SpeechAct | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    for act in SpeechAct:
        if act.value == key:
            return act
    for act, label in SPEECH_ACT_LABELS.items():
        if label.lower() == key:
            return act
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object only if it is the entire (stripped) text.

    Any surrounding prose, markdown fences, or extra JSON fragments make the
    output a contract violation rather than a valid message.
    """
    text = text.strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            start = match.start()
            if text[:start].strip() == "" and text[start + end:].strip() == "":
                return payload
            # Found a valid JSON object but it is embedded in extra text.
            return None
    return None


def _is_unanswerable_response(response: dict[str, Any]) -> bool:
    """Detect an explicit unanswerable marker in a reply.

    Model may state it cannot observe the requested value using keywords such
    as '不明' or '観測できない'. It may also explicitly set 'unanswerable': true
    in the JSON payload.
    """
    raw_payload = response.get("raw_payload")
    if isinstance(raw_payload, dict) and raw_payload.get("unanswerable") is True:
        return True
    message = str(response.get("message", "")).strip().lower()
    reason = str(response.get("reason", "")).strip().lower()
    indicators = ["不明", "観測できない", "unanswerable", "cannot observe", "not observable", "観測不能"]
    return any(indicator in message or indicator in reason for indicator in indicators)


def extract_json_action(response: str) -> tuple[Action | None, str, str, bool]:
    text = response.strip()
    payload = _extract_json_object(text)
    if payload is not None:
        try:
            action_text = str(payload.get("action", "")).strip().upper()
            reason = str(payload.get("reason", "")).strip()
            message = str(payload.get("message", "")).strip()
            ready = bool(payload.get("ready", False))
            if action_text in {action.value for action in ALL_ACTIONS}:
                return Action(action_text), reason, message, ready
        except (TypeError, ValueError):
            pass
    letter_match = re.search(r"\b([ABCDEF])\b", text.upper())
    if letter_match:
        return Action(letter_match.group(1)), text[:160], text[:160], False
    return None, text[:160], text[:160], False


def _coerce_str_or_none(value: Any) -> str | None:
    """文字列または整数を正規化した文字列に変換。None/空文字は None にする。

    注意: 呼び出し側で事前に型検査を行うこと。この関数は正規化済みの値に対して
    文字列表記を統一するためだけに使い、未知の型を暗黙に受理しないこと。
    """
    if value is None:
        return None
    if isinstance(value, (str, int)):
        text = str(value).strip()
        return text or None
    return None


def _discussion_validation_reason(payload: Any) -> str:
    """自由議論JSONの契約違反理由を、監査可能な安定コードで返す。"""
    if not isinstance(payload, dict):
        return "not_json_object"
    required_keys = ("speech_act", "message", "action", "reason", "reply_to_message_id")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        return "missing_required_keys:" + ",".join(missing)

    speech_act = _normalize_speech_act(payload.get("speech_act"))
    if speech_act is None:
        return "invalid_speech_act"

    message = payload.get("message")
    reason = payload.get("reason")
    if not isinstance(message, str) or not message.strip():
        return "invalid_or_empty_message"
    if not isinstance(reason, str) or not reason.strip():
        return "invalid_or_empty_reason"

    action_val = payload.get("action")
    if speech_act in QUESTION_SPEECH_ACTS:
        if action_val is not None:
            if not isinstance(action_val, str):
                return "invalid_question_action"
            if action_val.strip().upper() not in {a.value for a in ALL_ACTIONS}:
                return "invalid_question_action"
    else:
        if not isinstance(action_val, str):
            return "non_question_action_required"
        if action_val.strip().upper() not in {a.value for a in ALL_ACTIONS}:
            return "invalid_non_question_action"

    # addressed_to は正規化前に厳密に型検査する。
    # 仕様: 質問の場合は alpha/beta/null/省略、質問以外は null/省略。
    # null/省略の質問は二者ゲームの呼び出し側で相手へ安全に補完する。
    # 整数や辞書などの暗黙の型変換は契約違反として拒否する。
    raw_addressed_to = payload.get("addressed_to")
    if raw_addressed_to is not None and not isinstance(raw_addressed_to, str):
        return "invalid_addressed_to_type"
    addressed_to = _coerce_str_or_none(raw_addressed_to)
    if addressed_to is not None and addressed_to.strip().lower() not in {"alpha", "beta"}:
        return "invalid_addressed_to_agent"
    if speech_act not in QUESTION_SPEECH_ACTS and addressed_to is not None:
        return "addressed_to_for_non_question"

    # reply_to_message_id は正規化前に厳密に型検査する。
    # 仕様: str | int | null のみ受理。辞書やリストは契約違反として拒否する。
    raw_reply_to = payload.get("reply_to_message_id")
    if raw_reply_to is not None and not isinstance(raw_reply_to, (str, int)):
        return "invalid_reply_to_message_id_type"
    # bool は int の派生型だが、JSON契約では受理しない
    if isinstance(raw_reply_to, bool):
        return "invalid_reply_to_message_id_type"

    return ""


def _is_valid_discussion_payload(payload: dict[str, Any]) -> bool:
    """自由議論 JSON が必須キー・型・値の契約を満たすか検証する。"""
    return not _discussion_validation_reason(payload)


def extract_json_discussion(response: str) -> tuple[SpeechAct | None, str, Action | None, str, str | None, str | None, bool]:
    """自由議論用JSONをパースする。契約違反は有効発話として扱わない。"""
    text = response.strip()
    payload = _extract_json_object(text)
    if payload is None or not _is_valid_discussion_payload(payload):
        return None, text[:160], None, "", None, None, False

    speech_act = _normalize_speech_act(payload["speech_act"])
    message = str(payload["message"]).strip()
    reason = str(payload["reason"]).strip()
    raw_action = payload["action"]
    action = None if raw_action is None else Action(str(raw_action).strip().upper())
    reply_to_message_id = _coerce_str_or_none(payload["reply_to_message_id"])
    addressed_to = _coerce_str_or_none(payload.get("addressed_to"))
    if addressed_to is not None:
        addressed_to = addressed_to.lower()
    requires_response = speech_act in QUESTION_SPEECH_ACTS
    return speech_act, message, action, reason, reply_to_message_id, addressed_to, requires_response


def format_persona(agent_name: str, persona: str, persona_params: dict[str, object] | None) -> str:
    if not persona_params:
        return persona
    resolved = persona_params.get("_resolved_profile")
    if isinstance(resolved, dict):
        role = resolved.get("role") or {}
        presentation = resolved.get("persona") or {}
        value = resolved.get("value")
        value_text = "明示的な初期優先度なし" if value is None else _canonical_json(_qualitative_value_profile(value))
        mode = str(resolved.get("role_value_mode", "soft_value"))
        value_guidance = (
            "priority_levels は固定された意思決定基準です。変更・再交渉しないでください。"
            if mode == "legacy_hard"
            else "初期Vは暫定基準であり、観測事実、相手の根拠、受諾済みV*により更新可能です。"
        )
        return "\n".join(
            [
                "【ROLE id=role-profile】",
                _canonical_json(role),
                "【PERSONA id=persona-profile】",
                _canonical_json(presentation),
                "【INITIAL_VALUE id=initial-v】",
                value_text,
                value_guidance,
            ]
        )
    priority_weights = persona_params.get("priority_weights", {})
    if isinstance(priority_weights, dict):
        priority_text = ", ".join([f"{key}={_qualitative_level(value)}" for key, value in priority_weights.items()])
    else:
        priority_text = str(priority_weights)
    return "\n".join(
        [
            f"name: {agent_name}",
            f"role: {persona_params.get('role', persona)}",
            f"priority_levels: {priority_text}",
            f"risk_tolerance: {_qualitative_scalar_level(persona_params.get('risk_tolerance', 'mid'))}",
            f"goal_focus: {persona_params.get('goal_focus', 'unspecified')}",
            f"communication_style: {persona_params.get('communication_style', 'unspecified')}",
            f"concession_tendency: {_qualitative_scalar_level(persona_params.get('concession_tendency', 'mid'))}",
            f"evidence_demand: {_qualitative_scalar_level(persona_params.get('evidence_demand', 'mid'))}",
            f"notes: {persona_params.get('notes', '')}",
        ]
    )


def _procedure_block(condition: str) -> str:
    text = CONDITION_PROCEDURES.get(condition, "")
    return f"{text}\n\n" if text else ""


def _speech_act_guide() -> str:
    return "\n".join([f"- {act.value}: {label}" for act, label in SPEECH_ACT_LABELS.items()])


def _win_loss_block() -> str:
    return (
        "勝利条件:\n"
        "- 通信救助: communication >= 3 となり、救助到着（rescue_eta=0）まで生存する\n"
        "- 自力脱出: 行動 F で pod_readiness >= 2, pod_integrity >= 2, oxygen >= 3, power >= 2, flooding <= 3\n\n"
        "敗北条件:\n"
        "- oxygen <= 0\n"
        "- power <= 0\n"
        "- hull_damage >= 5\n"
        "- flooding >= 5\n"
        "- 行動 F を未達条件で実行すると重大損傷または敗北"
    )


def _role_evidence(agent_name: str, state, role: Any | None = None) -> str:
    role_mapping = _role_body(role)
    if role_mapping is None:
        return role_specific_evidence(agent_name, state)
    scope = [str(item) for item in role_mapping.get("observation_scope", [])]
    state_values = state.as_dict() if hasattr(state, "as_dict") else vars(state)
    observations = {
        field: state_values.get(field)
        for field in scope
        if field in state_values
    }
    return "\n".join(
        [
            f"expertise_domains: {_canonical_json(role_mapping.get('expertise_domains', []))}",
            f"responsibility: {role_mapping.get('responsibility', '')}",
            f"observation_scope: {_canonical_json(scope)}",
            f"role_observations: {_canonical_json(observations)}",
        ]
    )


def _question_context(open_question: dict[str, Any] | None, can_ask_question: bool, remaining_messages: int, remaining_tokens: int) -> str:
    parts: list[str] = []
    if open_question is not None:
        parts.append(
            f"【未回答の質問への回答】\n"
            f"{open_question['speaker']} からの質問（ID: {open_question['message_id']}）:\n"
            f"{open_question['message']}\n"
            f"reply_to_message_id には {open_question['message_id']} を指定して回答してください。"
        )
    budget_note = f"残り発言枠: {remaining_messages}, 残りトークン予算: {remaining_tokens}"
    if not can_ask_question:
        parts.append(f"【注意】{budget_note}。質問を出すための余裕がないため、質問は避けてください。")
    else:
        parts.append(f"【注意】{budget_note}。質問を出す場合は reply_to_message_id への回答分の余裕を残してください。")
    return "\n\n".join(parts)


def _discussion_json_contract(agent_name: str, open_question: dict[str, Any] | None) -> str:
    """自由議論の状況に合った必須 JSON スキーマと例を返す。"""
    other_agent = "beta" if agent_name == "alpha" else "alpha"
    required_keys = (
        "必須キー: speech_act, message, action, reason, reply_to_message_id\n"
        "- action: 情報要求の質問では null 可。質問以外は必ず A-F\n"
        "- addressed_to: 任意キー。質問では alpha/beta/null、質問以外は null。"
        "二者ゲームで質問時に null/省略なら相手一名へ補完される。未知名や他の型は禁止\n"
        "- reply_to_message_id: 回答対象の質問ID。回答以外は null"
    )

    if open_question is not None:
        answer_example = {
            "speech_act": "evidence",
            "message": "質問への短い回答",
            "action": "A",
            "reason": "短い理由",
            # 回答先は reply_to_message_id で一意に決まる。非質問発話に
            # addressed_to を設定すると直上のJSON契約と矛盾する。
            "addressed_to": None,
            "reply_to_message_id": str(open_question["message_id"]),
        }
        return (
            f"{required_keys}\n"
            f"今は質問ID {open_question['message_id']} への回答が必須です。"
            "新しい質問や別の話題を出さず、reply_to_message_id を省略しないでください。\n"
            f"回答JSON例:\n{json.dumps(answer_example, ensure_ascii=False, separators=(',', ':'))}"
        )

    statement_example = {
        "speech_act": "evidence",
        "message": "相手への短い発言",
        "action": "A",
        "reason": "短い理由",
        "addressed_to": None,
        "reply_to_message_id": None,
    }
    question_example = {
        "speech_act": "information_request",
        "message": "相手への短い質問",
        "action": None,
        "reason": "確認したい理由",
        "addressed_to": other_agent,
        "reply_to_message_id": None,
        "requested_fields": ["oxygen"],
        "reask_reason": "",
    }
    return (
        f"{required_keys}\n"
        f"質問時の任意キー: requested_fields(質問するstate field名のリスト), reask_reason(既に閉じた質問を再質問する場合の新しい根拠)\n"
        f"通常発言JSON例:\n{json.dumps(statement_example, ensure_ascii=False, separators=(',', ':'))}\n"
        f"質問JSON例:\n{json.dumps(question_example, ensure_ascii=False, separators=(',', ':'))}"
    )


def v_measurement_prompt(
    agent_name: str,
    state,
    *,
    phase: str,
    current_v: dict[str, Any] | None = None,
    persona: str = "",
    persona_params: dict[str, object] | None = None,
    transcript: list[dict[str, Any]] | None = None,
    final_vote: dict[str, Any] | None = None,
    v_state: dict[str, Any] | None = None,
    role: Any | None = None,
) -> str:
    """Condition-symmetric V measurement. It intentionally contains no framework hint."""
    if phase not in {"before", "after"}:
        raise ValueError(f"Unknown V measurement phase: {phase}")
    current = "" if current_v is None else f"\n現在Vの参考記録:\n{_canonical_json(current_v)}\n"
    criteria = list(DEFAULT_VALUE_CRITERIA_SCHEMA.criteria)
    criteria_json = _canonical_json(criteria)
    levels_instruction = _canonical_json(
        {criterion: "<high|mid|low>" for criterion in criteria}
    )
    if phase == "before":
        contract = (
            f'{{"v_before":{{"ordered_criteria":{criteria_json},'
            f'"priority_levels":{levels_instruction},"confidence_level":"<high|mid|low>"}},'
            f'"action_before":"<現在の観測から選ぶA-Fのいずれか>","reason_before":"短い理由"}}'
        )
        instruction = "相手の発言を見る前の、あなた自身の暫定判断基準と行動案を記録してください。"
    else:
        contract = (
            f'{{"v_after":{{"ordered_criteria":{criteria_json},'
            f'"priority_levels":{levels_instruction},"confidence_level":"<high|mid|low>"}},'
            f'"reason_after":"短い理由"}}'
        )
        instruction = "最終投票確定後の、あなた自身の現在の判断基準を記録してください。"
    after_context = ""
    if phase == "after":
        after_context = (
            f"\n【FINAL_V_STATE id=final-v-state】\n{_v_state_block(v_state)}\n"
            f"【FINAL_VOTES id=final-votes】\n{_canonical_json(final_vote or {})}\n"
            f"【DISCUSSION_HISTORY id=history】\n{format_transcript_text(transcript or [])}\n"
        )
    return f"""【GAME_RULES_AND_JSON_CONTRACT id=v-measurement-{phase}】
あなたは深海研究施設トラブルの意思決定エージェント {agent_name} です。
{instruction}
これは測定であり、合意やV*形成を指示するものではありません。

【VALUE_CRITERIA_SCHEMA id={DEFAULT_VALUE_CRITERIA_SCHEMA.id}】
version: {DEFAULT_VALUE_CRITERIA_SCHEMA.version}
criteria: {criteria_json}
上記criteriaから1つでも欠けたり、未知の項目を追加したりしないでください。
priority_levels は ROLE_PERSONA_INITIAL_VALUE と CURRENT_OBSERVATION における、あなた自身の現在の優先順位から導出してください。
5項目すべてを high / mid / low のいずれかで示してください。小数、百分率、合計値による重み付けは禁止します。

【CURRENT_OBSERVATION id=state】
{format_state(state, agent_name, role)}

{decision_support_block(state, agent_name, role)}

【ROLE_PERSONA_INITIAL_VALUE id=agent-profile】
{format_persona(agent_name, persona, persona_params)}{current}
{after_context}
必ず次のJSONだけを返してください。説明文やMarkdownは不要です。
{contract}
"""


def _v_state_block(v_state: dict[str, Any] | None) -> str:
    if not v_state:
        return "【CURRENT_V_STATE id=current-v】\nv_star_status: not_recorded"
    status = str(v_state.get("v_star_status", "unresolved"))
    lines = ["【CURRENT_V_STATE id=current-v】", f"v_star_status: {status}"]
    current_v = v_state.get("current_v")
    if current_v is not None:
        lines.append(f"current_v: {_canonical_json(current_v)}")
    shared_before = v_state.get("shared_v_before")
    if shared_before:
        lines.append(f"shared_v_before_and_actions: {_canonical_json(shared_before)}")
    if v_state.get("pending_proposals"):
        lines.append(f"pending_v_proposals: {_canonical_json(v_state['pending_proposals'])}")
    if status == "accepted" and v_state.get("v_star"):
        lines.extend(
            [
                f"v_star_id: {v_state.get('v_star_id')}",
                f"accepted_v_star: {_canonical_json(v_state['v_star'])}",
                "この受諾済みV*を同一ターンの後続判断で優先してください。",
            ]
        )
    else:
        lines.append("受諾済みV*はありません。一致したものとして補完しないでください。")
    return "\n".join(lines)


def discussion_prompt(
    agent_name: str,
    persona: str,
    persona_params: dict[str, object] | None,
    state,
    transcript: list[dict[str, Any]],
    max_discussion_turns: int,
    condition: str = "control",
    open_question: dict[str, Any] | None = None,
    can_ask_question: bool = True,
    remaining_messages: int = 0,
    remaining_tokens: int = 0,
    v_state: dict[str, Any] | None = None,
    role: Any | None = None,
    required_question_fields: list[str] | None = None,
    counter_allowed: bool = True,
) -> str:
    context = _question_context(open_question, can_ask_question, remaining_messages, remaining_tokens)
    json_contract = _discussion_json_contract(agent_name, open_question)
    criteria_example = _canonical_json(list(DEFAULT_VALUE_CRITERIA_SCHEMA.criteria))
    counter_guide = (
        f"counterの場合は v_star_response.counter_proposal に proposal_id、ordered_criteria({criteria_example})、scope、任意のpriority_levels(high/mid/low)を含む完全な代替案を入れてください。"
        if counter_allowed
        else "counter上限に達しています。新しいcounterは出さず、最新提案をacceptまたはrejectしてください。"
    )
    v_sharing_guide = (
        f"HIVC-D条件では必要に応じ v_proposal と v_star_response を追加できます。\n"
        f"HIVC-D条件で自分の事前V測定を明示共有する場合だけ share_v_before=true を追加できます。\n"
        f'v_proposal={{"proposal_id":"一意ID","ordered_criteria":{criteria_example},"priority_levels":{{各criterion:"high|mid|low"}},"scope":"turn"}}\n'
        f"Vの強さを示す場合はpriority_levelsだけを使い、数値weights・小数・百分率を使わないでください。\n"
        f'v_star_response={{"response":"accept|reject|counter","proposal_id":"対象ID"}}。対象IDなしの応答は無効です。\n'
        f"自分が提示した v_proposal を受諾する場合、同じ JSON に v_star_response={{\"response\":\"accept\",\"proposal_id\":\"<v_proposal.proposal_id>\"}} を必ず含めてください。"
        f"{counter_guide}"
        if condition in {"hivc_d", "hivc_d_prescribed_v1"}
        else ""
    )
    required_question_guide = ""
    if required_question_fields:
        required_question_guide = (
            "【REQUIRED_INFORMATION_CHECK id=required-information-question】\n"
            "判断に必要な次の情報は相手だけが観測できます: "
            + ", ".join(required_question_fields)
            + "\n今回は結論を述べず、speech_act=information_request、action=null、"
            "requested_fieldsに上記fieldを入れて相手へ質問してください。\n"
        )
    return f"""【GAME_RULES_AND_JSON_CONTRACT id=discussion-contract】
あなたは深海研究施設トラブルの意思決定エージェントです。

{_win_loss_block()}

{_v_state_block(v_state)}

【FRAMEWORK id={condition}】
{_procedure_block(condition)}

【ROLE_PERSONA_VALUE id=agent-profile】
あなたのペルソナ設定:
{format_persona(agent_name, persona, persona_params)}

現在状態（あなたの担当分野のみ可視）:
{format_state(state, agent_name, role)}

{decision_support_block(state, agent_name, role)}

あなたの役割固有情報:
{_role_evidence(agent_name, state, role)}

【DISCUSSION_HISTORY id=history】
これまでの議論:
{format_transcript_text(transcript)}

{context}

{required_question_guide}

自由議論の発言目的は以下のいずれかを speech_act として選んでください:
{_speech_act_guide()}

この自由議論フェーズでは最大 {max_discussion_turns} 発言までです。
action は必須です。speech_act が information_request / question_objection / question の質問である場合だけ null を使えます。
evidence、proposal、tradeoff、concession_integration、objection、answer を含む質問以外の全speech_actでは、actionに必ずA-Fの一つを入れてください。
自分のactionとreason内で「採用・選択・実行する」と述べるactionを一致させてください。
ready は不要です。
{v_sharing_guide}
証拠・状態の情報要求は speech_act="information_request", action=null を使えます。
反論を伴う質問は speech_act="question_objection" を使ってください。addressed_to は相手名、null、または省略が可能です。
必ず次のJSONだけを返してください。説明文やMarkdownは不要です。
{json_contract}
"""


def decision_opportunity_prompt(
    agent_name: str,
    persona: str,
    persona_params: dict[str, object] | None,
    state,
    transcript: list[dict[str, Any]],
    condition: str,
    opportunity_index: int,
    opportunity_count: int,
    v_state: dict[str, Any] | None = None,
    role: Any | None = None,
) -> str:
    accepted = bool(v_state and v_state.get("v_star_status") == "accepted")
    action_reconciliation = ""
    if opportunity_index > 1 and not accepted:
        action_reconciliation = (
            "【ACTION_RECONCILIATION id=action-reconciliation】\n"
            "前回のAction投票は一致しませんでした。V*が未成立でも、Valueそのものを一致させる必要はありません。\n"
            "議論履歴にある相手のActionと理由を確認し、自分のValueを保持したまま同じ安全なActionへ合意できるか再検討してください。\n"
            "相手案へ変更する場合は、取り入れた根拠をreasonに示してください。\n\n"
        )
    v_contract = (
        ',"v_star_id":"' + str(v_state.get("v_star_id")) + '"'
        if accepted else ""
    )
    return f"""【GAME_RULES_AND_JSON_CONTRACT id=decision-contract】
あなたは深海研究施設トラブルの意思決定エージェントです。

{_win_loss_block()}

{_v_state_block(v_state)}

【FRAMEWORK id={condition}】
{_procedure_block(condition)}

【ROLE_PERSONA_VALUE id=agent-profile】
あなたのペルソナ設定:
{format_persona(agent_name, persona, persona_params)}

現在状態（あなたの担当分野のみ可視）:
{format_state(state, agent_name, role)}

{decision_support_block(state, agent_name, role)}

あなたの役割固有情報:
{_role_evidence(agent_name, state, role)}

【DISCUSSION_HISTORY id=history】
これまでの議論:
{format_transcript_text(transcript)}

{action_reconciliation}
これは第 {opportunity_index} / {opportunity_count} 回の意思決定機会です。
各エージェントは独立に最終案を一つだけ出してください。
出力には action（A-F）、短い reason、そして合意意思を表す ready（true/false）を含めてください。
全員が同じ action かつ ready=true なら合意成立です。
受諾済みV*が表示されている場合は、正確な v_star_id を返してください。
V*は数値スコアや唯一の正解Actionを定めるものではなく、複数の安全なActionを比較して理由を説明するための共通観点です。
システムはV*の参照・形式、reasonとactionの明示的矛盾、行動直後の安全性を検証します。整合性を自己申告する必要はありません。
必ず次のJSONだけを返してください。説明文やMarkdownは不要です。
{{"action":"A","reason":"短い理由","ready":true{v_contract}}}
"""


def v_proposal_required_prompt(
    agent_name: str,
    persona: str,
    persona_params: dict[str, object] | None,
    state,
    condition: str,
    transcript: list[dict[str, Any]],
    v_state: dict[str, Any] | None,
    role: Any | None = None,
) -> str:
    """自由議論で v_proposal が出なかった場合、HIVC-D条件で必須のV提案を求めるプロンプト。"""
    criteria = list(DEFAULT_VALUE_CRITERIA_SCHEMA.criteria)
    criteria_json = _canonical_json(criteria)
    levels_json = _canonical_json({c: "mid" for c in criteria})
    return f"""【GAME_RULES_AND_JSON_CONTRACT id=v-proposal-required】
あなたは深海研究施設トラブルの意思決定エージェント {agent_name} です。

自由議論では意見が分かれました。最終投票前に、グループで使う共通基準V*を一つ提案してください。
提案するVは、自分の役割・観測事実に基づいたもので構いません。

【VALUE_CRITERIA_SCHEMA id={DEFAULT_VALUE_CRITERIA_SCHEMA.id}】
version: {DEFAULT_VALUE_CRITERIA_SCHEMA.version}
criteria: {criteria_json}
上記criteriaから1つでも欠けたり、未知の項目を追加したりしないでください。
各criterionの強さは priority_levels の high / mid / low だけで表し、数値weights・小数・百分率を使わないでください。

【FRAMEWORK id={condition}】
{_procedure_block(condition)}

【ROLE_PERSONA_VALUE id=agent-profile】
{format_persona(agent_name, persona, persona_params)}

現在状態（あなたの担当分野のみ可視）:
{format_state(state, agent_name, role)}

{decision_support_block(state, agent_name, role)}

あなたの役割固有情報:
{_role_evidence(agent_name, state, role)}

【DISCUSSION_HISTORY id=history】
これまでの議論:
{format_transcript_text(transcript)}

{_v_state_block(v_state)}

必ず次のJSONだけを返してください。説明文やMarkdownは不要です。
{{"v_proposal":{{"proposal_id":"{agent_name}-turn{{state.turn}}-required","ordered_criteria":{criteria_json},"priority_levels":{levels_json},"scope":"turn"}},"v_star_response":{{"response":"accept","proposal_id":"{agent_name}-turn{{state.turn}}-required"}},"action":"A","reason":"短い理由"}}
"""


def v_proposal_response_prompt(
    agent_name: str,
    persona: str,
    persona_params: dict[str, object] | None,
    state,
    condition: str,
    transcript: list[dict[str, Any]],
    proposal: dict[str, Any],
    v_state: dict[str, Any] | None,
    role: Any | None = None,
    counter_allowed: bool = True,
) -> str:
    """v_proposal_required で出た提案に対し、相手エージェントに accept/reject/counter を求める。"""
    criteria = list(DEFAULT_VALUE_CRITERIA_SCHEMA.criteria)
    criteria_json = _canonical_json(criteria)
    response_choices = "accept / reject / counter" if counter_allowed else "accept / reject"
    counter_instruction = (
        "counter の場合は完全な代替案を v_star_response.counter_proposal に入れてください。\n"
        "counter を出す場合、自分のcounter提案にも明示的に同意するには "
        "v_star_response.self_accept を true に設定してください。これにより counter提案者自身の同意も記録されます。"
        if counter_allowed
        else "counter上限に達しています。新しいcounterは無効です。最新提案をacceptまたはrejectしてください。"
    )
    response_contract = (
        f'{{"v_star_response":{{"response":"accept|reject|counter","proposal_id":"{proposal.get("proposal_id", "")}",'
        f'"counter_proposal":{{"proposal_id":"{agent_name}-counter-turn{{state.turn}}","ordered_criteria":{criteria_json},'
        '"priority_levels":{"oxygen":"high","power":"mid","hull_damage":"mid","flooding":"mid","communication":"low"},"scope":"turn"},'
        '"self_accept":false}}}'
        if counter_allowed
        else f'{{"v_star_response":{{"response":"accept|reject","proposal_id":"{proposal.get("proposal_id", "")}"}}}}'
    )
    return f"""【GAME_RULES_AND_JSON_CONTRACT id=v-proposal-response】
あなたは深海研究施設トラブルの意思決定エージェント {agent_name} です。

相手から V* 提案が出ました。あなたの観測・役割から判断し、{response_choices} のいずれかを返してください。
{counter_instruction}
counterで優先度の強さを示す場合は priority_levels の high / mid / low だけを使い、数値weightsは使わないでください。

【VALUE_CRITERIA_SCHEMA id={DEFAULT_VALUE_CRITERIA_SCHEMA.id}】
version: {DEFAULT_VALUE_CRITERIA_SCHEMA.version}
criteria: {criteria_json}
上記criteriaから1つでも欠けたり、未知の項目を追加したりしないでください。

【FRAMEWORK id={condition}】
{_procedure_block(condition)}

【ROLE_PERSONA_VALUE id=agent-profile】
{format_persona(agent_name, persona, persona_params)}

現在状態（あなたの担当分野のみ可視）:
{format_state(state, agent_name, role)}

{decision_support_block(state, agent_name, role)}

あなたの役割固有情報:
{_role_evidence(agent_name, state, role)}

【DISCUSSION_HISTORY id=history】
これまでの議論:
{format_transcript_text(transcript)}

提案内容:
{_canonical_json(proposal)}

{_v_state_block(v_state)}

必ず次のJSONだけを返してください。説明文やMarkdownは不要です。
{response_contract}
"""


def run_prompt(model, tokenizer, prompt: str, max_new_tokens: int, enable_thinking: bool = False, thinking_budget: int | None = None) -> tuple[str, str]:
    """モデルにプロンプトを送り、(thinking_content, response_text) を返す。

    enable_thinking=True の場合、モデルは  <think>... </think> 内に思考を出力する。
    thinking_content はその内部テキスト、response_text は思考以降の最終応答。
    enable_thinking=False の場合、thinking_content は空文字。
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if thinking_budget is not None:
        template_kwargs["thinking_budget"] = thinking_budget
    text = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    full = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

    #  <think>... </think> ブロックを抽出（Qwen3 thinkingモード）
    think_match = re.search(r"<think>(.*?)</think>", full, flags=re.DOTALL)
    if think_match:
        thinking_content = think_match.group(1).strip()
        response_text = full[think_match.end():].strip()
    else:
        thinking_content = ""
        response_text = full.strip()

    return thinking_content, response_text


def _invoke_prompt(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    prompt_runner=None,
    agent: str | None = None,
) -> tuple[str, str]:
    """Invoke the local model or a narrowly injected, speaker-aware backend."""
    if prompt_runner is None:
        return run_prompt(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
    if agent not in {"alpha", "beta"}:
        raise ValueError("an injected prompt runner requires agent=alpha or agent=beta")
    return prompt_runner(
        agent,
        prompt,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )


def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quantization_config,
    )
    return model, tokenizer


def get_action(model, tokenizer, prompt: str, max_new_tokens: int, fallback: Action, enable_thinking: bool = False, thinking_budget: int | None = None, prompt_runner=None, agent: str | None = None) -> tuple[Action, str, str, bool, str, str]:
    """(action, reason, message, ready, raw_response, thinking) を返す。"""
    thinking, raw = _invoke_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=enable_thinking, thinking_budget=thinking_budget, prompt_runner=prompt_runner, agent=agent)
    action, reason, message, ready = extract_json_action(raw)
    if action is None:
        return fallback, f"invalid_response_fallback: {reason}", message, False, raw, thinking
    return action, reason, message, ready, raw, thinking


def get_discussion_message(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    fallback_action: Action | None = None,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    prompt_runner=None,
    agent: str | None = None,
) -> dict[str, Any]:
    """自由議論用の発言情報を dict で返す。JSON契約違反は有効発話として扱わない。"""
    thinking, raw = _invoke_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=enable_thinking, thinking_budget=thinking_budget, prompt_runner=prompt_runner, agent=agent)
    raw_payload: Any = _extract_json_object(raw.strip())
    if raw_payload is None:
        # JSON構文不正・壊れたJSON断片を有効発話にしない。監査用のrawは保持。
        return {
            "speech_act": None,
            "message": "",
            "action": None,
            "reason": "",
            "reply_to_message_id": None,
            "addressed_to": None,
            "requires_response": False,
            "raw": raw,
            "thinking": thinking,
            "raw_payload": None,
            "validation_reason": "not_json_object_or_extra_text",
            "invalid_discussion_output": True,
        }
    speech_act, message, action, reason, reply_to_message_id, addressed_to, requires_response = extract_json_discussion(raw)
    # 契約違反: 必須キー欠落、型不一致、無効なspeech_act/action、空のmessage/reason、
    # 質問でないのにaddressed_toが設定されている、質問でaddressed_toが欠落 等。
    if speech_act is None or not message:
        return {
            "speech_act": None,
            "message": "",
            "action": None,
            "reason": "",
            "reply_to_message_id": None,
            "addressed_to": None,
            "requires_response": False,
            "raw": raw,
            "thinking": thinking,
            "raw_payload": raw_payload,
            "validation_reason": _discussion_validation_reason(raw_payload) or "invalid_discussion_payload",
            "invalid_discussion_output": True,
        }
    # 質問の追加メタデータ: requested_fields と reask_reason
    requested_fields: list[str] = []
    reask_reason = ""
    if requires_response and isinstance(raw_payload, dict):
        requested_fields = _normalize_requested_fields(raw_payload.get("requested_fields"))
        raw_reask = raw_payload.get("reask_reason")
        if isinstance(raw_reask, str):
            reask_reason = raw_reask.strip()
    return {
        "speech_act": speech_act,
        "message": message,
        "action": action,
        "reason": reason,
        "reply_to_message_id": reply_to_message_id,
        "addressed_to": addressed_to,
        "requires_response": requires_response,
        "raw": raw,
        "thinking": thinking,
        "raw_payload": raw_payload,
        "validation_reason": "",
        "invalid_discussion_output": False,
        "requested_fields": requested_fields,
        "reask_reason": reask_reason,
    }


def extract_vote_v_fields(response: str) -> tuple[str, bool | None]:
    try:
        payload = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError):
        return "", None
    if not isinstance(payload, dict):
        return "", None
    v_star_id = str(payload.get("v_star_id", "")).strip()
    consistent = payload.get("v_star_consistent")
    return v_star_id, consistent if isinstance(consistent, bool) else None


def verify_vote_v_star_consistency(
    action: Action | None,
    reason: str,
    referenced_id: str,
    claimed_consistent: bool | None,
    v_star_id: str,
    v_star: dict[str, Any] | None,
    state: Any | None = None,
) -> bool | None:
    """V*の手続き的整合性を検証する。Vやリスクを数値採点してActionを順位付けしない。"""
    del claimed_consistent
    if action is None or referenced_id != v_star_id or not v_star:
        return False
    if state is not None:
        safe, _, _ = preview_action_safety(state, action)
        if not safe:
            return False
    ordered = v_star.get("ordered_criteria")
    if not isinstance(ordered, list) or not ordered:
        return False

    criteria = list(DEFAULT_VALUE_CRITERIA_SCHEMA.criteria)
    if set(map(str, ordered)) != set(criteria) or len(ordered) != len(criteria):
        return False

    # JSONのactionと、理由中で採用・選択・実行すると明言したactionが矛盾すれば拒否する。
    committed_actions: set[str] = set()
    for match in re.finditer(
        r"(?:^|[^A-Za-z0-9_])(?:action\s*)?([A-F])\s*(?:を)?\s*(?:採用|選択|実行|支持|推奨|choose|select|execute)",
        reason,
        flags=re.IGNORECASE,
    ):
        committed_actions.add(match.group(1).upper())
    for match in re.finditer(r"\b(?:choose|select|execute|adopt|recommend)\s+(?:action\s*)?([A-F])\b", reason, re.IGNORECASE):
        committed_actions.add(match.group(1).upper())
    if any(committed != action.value for committed in committed_actions):
        return False

    # V*は共通の熟議観点であり、重み付きutility、危険度倍率、順位差によって
    # 一つのActionをシステム側から強制しない。安全なAction間のtrade-offは
    # エージェントの観測・議論・合意に委ねる。
    return True


def run_one_game(
    model,
    tokenizer,
    condition: str,
    seed: int,
    personas: dict[str, str],
    persona_params: dict[str, dict[str, object] | None],
    role_keys: dict[str, str | None],
    max_new_tokens: int = 96,
    max_discussion_turns: int = 6,
    discussion_token_budget: int = 768,
    evaluator_rollouts: int = 24,
    live_jsonl_path: str | None = None,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    decision_schedule_seed: int = 0,
    max_decision_opportunities: int = 3,
    scenario_id: str | None = None,
    **kwargs: Any,
) -> list[dict[str, object]]:
    """1 ゲームを進行し、REQUIREMENTS §6 / §7.1 のターン別記録項目を含む行リストを返す。

    個人選択・個人理由・グループ理由・対立度を各行に記録する。
    質問と応答の閉包（§7.1.3）を実装する。
    live_jsonl_path を指定すると、各ターン終了時にその行を JSON 1 行として追記する。
    """
    role_value_mode = kwargs.get("role_value_mode")
    resolved_profiles = kwargs.get("resolved_profiles")
    prompt_runner = kwargs.get("prompt_runner")
    enable_v_flow = role_value_mode is not None
    max_v_counter_rounds = max(0, int(kwargs.get("max_v_counter_rounds", 1)))
    v_messages_per_counter_extension = max(
        0, int(kwargs.get("v_messages_per_counter_extension", 2))
    )
    state = initial_state(seed, scenario_id)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    speakers = ["alpha", "beta"]
    n_speakers = len(speakers)
    planned_route = "undecided"
    persistent_v_star: dict[str, Any] | None = None
    persistent_v_star_id = ""
    role_value_assignment_id = _role_value_assignment_id(personas, persona_params, role_keys, seed)

    def resolved_role_for(agent: str) -> Any | None:
        if isinstance(resolved_profiles, dict):
            resolved = resolved_profiles.get(agent)
            role = getattr(resolved, "role", None)
            if role is not None:
                return role
        return _resolved_role_from_params(persona_params.get(agent))

    def agent_observation_scope(agent: str) -> set[str]:
        """agentの observation_scope を小文字セットで返す。未定義の場合は空集合。"""
        role = resolved_role_for(agent)
        role_mapping = _role_body(role)
        if role_mapping is None:
            return set()
        return {str(item).strip().lower() for item in role_mapping.get("observation_scope", [])}

    while not state.done:
        q_values = estimate_q_values(state, n_rollouts=evaluator_rollouts, seed=seed + state.turn * 1000)
        optimal = best_action(q_values)
        allowed = acceptable_actions(q_values)
        safe_fallback_candidates = [
            candidate for candidate in ALL_ACTIONS
            if preview_action_safety(state, candidate)[0]
        ]
        fallback = (
            max(safe_fallback_candidates, key=lambda candidate: q_values[candidate])
            if safe_fallback_candidates
            else optimal
        )

        # §6.1: all framework conditions use this exact pre-discussion measurement.
        v_before: dict[str, dict[str, Any] | None] = {"alpha": None, "beta": None}
        action_before: dict[str, Action | None] = {"alpha": None, "beta": None}
        reason_before = {"alpha": "", "beta": ""}
        v_measurement_errors = {"alpha_before": "not_recorded", "beta_before": "not_recorded"}
        measurement_call_count = 0
        measurement_token_count = 0
        measurement_retry_count = 0
        max_v_measurement_retries = int(kwargs.get("max_v_measurement_retries", 2))
        if enable_v_flow:
            for agent in speakers:
                prompt = v_measurement_prompt(
                    agent,
                    state,
                    phase="before",
                    persona=personas[agent],
                    persona_params=persona_params[agent],
                    role=resolved_role_for(agent),
                )
                measured_v: dict[str, Any] | None = None
                measured_action: Action | None = None
                measured_reason = ""
                error = ""
                raw_measurement = ""
                for attempt in range(max_v_measurement_retries + 1):
                    _, raw_measurement = _invoke_prompt(
                        model, tokenizer, prompt, max_new_tokens,
                        enable_thinking=enable_thinking, thinking_budget=thinking_budget,
                        prompt_runner=prompt_runner, agent=agent,
                    )
                    measurement_call_count += 1
                    if tokenizer is not None:
                        measurement_token_count += len(tokenizer.encode(raw_measurement, add_special_tokens=False))
                    measured_v, measured_action, measured_reason, error = extract_json_v_measurement(raw_measurement)
                    if not error:
                        break
                    if attempt < max_v_measurement_retries:
                        measurement_retry_count += 1
                v_before[agent] = measured_v
                action_before[agent] = measured_action
                reason_before[agent] = measured_reason
                v_measurement_errors[f"{agent}_before"] = error or ("measurement_failed" if measured_v is None else "")

        # §6.2.1: v_alignment_required is determined from observed V and actions, not self-report.
        if enable_v_flow:
            turn_v_alignment_required, turn_v_alignment_requirement_reasons = v_alignment_required(
                action_before["alpha"], action_before["beta"], v_before["alpha"], v_before["beta"]
            )
        else:
            turn_v_alignment_required, turn_v_alignment_requirement_reasons = False, []

        # §6.2.2: reserve the base budget for mandatory V negotiation in hivc_d.
        # counter経路は自動acceptを廃止したため、明示的合意に到達する基礎4発話を確保:
        #   1. alpha: proposal + self-accept
        #   2. beta:  counter
        #   3. alpha: counter を accept
        #   4. beta:  自分の counter を accept
        # counter出力で counter_proposal と同時に self-accept を表現できるスキーマも許容する。
        # counterが実際に発生した場合は、下の必須ループで1ラウンドにつき既定2発話を
        # 動的に延長する。counter回数自体は max_v_counter_rounds で制限する。
        reserved_v_messages = 0
        reserved_v_tokens = 0
        if enable_v_flow and condition == "hivc_d" and turn_v_alignment_required:
            reserved_v_messages = 4
            reserved_v_tokens = reserved_v_messages * max_new_tokens

        # 少なくとも各エージェント1回ずつ発言できるよう実効値を確保
        base_max_discussion_turns = max(max_discussion_turns, n_speakers)
        if reserved_v_messages and base_max_discussion_turns > n_speakers:
            base_max_discussion_turns = max(n_speakers, base_max_discussion_turns - reserved_v_messages)
        effective_max_discussion_turns = base_max_discussion_turns
        if effective_max_discussion_turns != max_discussion_turns:
            print(f"[run_one_game] effective discussion budget {effective_max_discussion_turns} (reserved {reserved_v_messages} for V proposal)")

        opportunity_count = schedule_decision_opportunities(seed, state.turn, decision_schedule_seed, max_decision_opportunities)
        free_discussion_token_budget = max(0, discussion_token_budget - reserved_v_tokens)
        message_limits, token_limits = allocate_discussion_budgets(
            opportunity_count, effective_max_discussion_turns, free_discussion_token_budget, n_speakers=n_speakers
        )

        transcript: list[dict[str, Any]] = []
        token_budget_used = 0
        total_free_messages = 0
        decision_history: list[dict[str, Any]] = []
        group_action: Action | None = None
        decision_rule: str | None = None
        group_reason = ""
        fallback_used = False
        fallback_priority_agent: str | None = None
        final_attempt_index = 0

        alpha_vote: Action | None = None
        alpha_vote_reason = ""
        alpha_vote_message = ""
        alpha_vote_ready = False
        alpha_vote_raw = ""
        alpha_vote_thinking = ""
        beta_vote: Action | None = None
        beta_vote_reason = ""
        beta_vote_message = ""
        beta_vote_ready = False
        beta_vote_raw = ""
        beta_vote_thinking = ""
        final_vote_retry_count = 0
        rejected_final_votes: list[dict[str, Any]] = []
        max_final_vote_retries = max(0, int(kwargs.get("max_final_vote_retries", 1)))

        # 質問/応答閉包用の状態
        next_message_id = 1
        open_questions: list[dict[str, Any]] = []
        closed_questions: list[dict[str, Any]] = []
        forced_decision_with_open_question = False
        forced_decision_reason = ""
        question_response_latencies: list[int] = []
        question_count = 0
        answered_question_count = 0
        unanswerable_question_count = 0
        self_observable_question_count = 0
        duplicate_question_count = 0
        required_information_question_count = 0
        missing_required_information_question_count = 0
        invalid_discussion_output_count = 0
        invalid_attempt_count = 0
        repaired_invalid_output_count = 0
        invalid_discussion_outputs: list[dict[str, Any]] = []
        discussion_retry_count = 0
        max_discussion_retries = max(0, int(kwargs.get("max_discussion_retries", 1)))
        consecutive_duplicate_count = 0
        max_consecutive_duplicate_questions_recorded = 0
        last_duplicate_signature: tuple[str, str, str] | None = None
        next_speaker_override: str | None = None

        v_proposals: list[dict[str, Any]] = []
        v_responses: dict[str, list[dict[str, Any]]] = {"alpha": [], "beta": []}
        v_proposal_id_repairs: list[dict[str, str]] = []

        def register_proposal_id(
            proposal: dict[str, Any], proposal_speaker: str, message_id: str
        ) -> tuple[dict[str, Any], str, str]:
            enriched = {**proposal, "speaker": proposal_speaker, "message_id": message_id}
            repaired, original_id, repaired_id = ensure_unique_v_proposal_id(enriched, v_proposals)
            if repaired_id != original_id:
                v_proposal_id_repairs.append(
                    {
                        "original_id": original_id,
                        "repaired_id": repaired_id,
                        "speaker": proposal_speaker,
                        "message_id": message_id,
                    }
                )
            return repaired, original_id, repaired_id
        explicitly_shared_v_before: dict[str, dict[str, Any]] = {}
        inherited_game_v = persistent_v_star is not None
        v_star_status = "accepted" if inherited_game_v else ("unresolved" if enable_v_flow else "not_recorded")
        v_star_id = persistent_v_star_id if inherited_game_v else ""
        v_star: dict[str, Any] | None = persistent_v_star
        v_star_failure_reason = "" if inherited_game_v else ("missing_v_proposal" if enable_v_flow else "not_recorded")
        v_star_unresolved_reason = v_star_failure_reason
        v_proposal_required_prompt_issued = False
        missing_v_proposal_after_required_prompt = False
        v_negotiation_messages_used = 0
        v_counter_count = 0
        rejected_v_counter_count = 0
        v_negotiation_terminal_reason = ""

        # §6.2.3 V protocol state machine
        v_protocol_state = "I_SHARE"
        v_protocol_transition_history: list[dict[str, Any]] = [{"from": "init", "to": "I_SHARE", "reason": "turn_start"}]
        if enable_v_flow:
            v_protocol_state = "V_COMPARE"
            v_protocol_transition_history.append({"from": "I_SHARE", "to": "V_COMPARE", "reason": "v_before_measured"})
            if turn_v_alignment_required:
                if condition == "hivc_d":
                    v_protocol_state = "V_PROPOSE"
                    v_protocol_transition_history.append({"from": "V_COMPARE", "to": "V_PROPOSE", "reason": "v_alignment_required"})
                else:
                    v_protocol_state = "A_CHECK"
                    v_protocol_transition_history.append({"from": "V_COMPARE", "to": "A_CHECK", "reason": "alignment_required_but_not_hivc_d"})
            else:
                v_protocol_state = "V_NOT_REQUIRED"
                v_protocol_transition_history.append({"from": "V_COMPARE", "to": "V_NOT_REQUIRED", "reason": "no_alignment_required"})

        if enable_v_flow and condition == "hivc_d_prescribed_v1":
            v_star_id = f"seed{seed}-turn{state.turn}-prescribed-v1"
            criteria = list(DEFAULT_VALUE_CRITERIA_SCHEMA.criteria)
            priority_levels = {"oxygen": "high", "power": "high", "hull_damage": "mid", "flooding": "mid", "communication": "low"}
            v_star = {
                "proposal_id": v_star_id,
                "ordered_criteria": criteria,
                "priority_levels": priority_levels,
                "scope": "turn",
                "source": "external_prescription",
            }
            v_star_status = "accepted"
            v_star_failure_reason = ""
            v_star_unresolved_reason = ""

        def current_v_state(agent: str) -> dict[str, Any] | None:
            if not enable_v_flow:
                return None
            return {
                "current_v": v_before[agent],
                "shared_v_before": explicitly_shared_v_before,
                "pending_proposals": v_proposals,
                "v_star_status": v_star_status,
                "v_star_id": v_star_id,
                "v_star": v_star,
            }

        for opp_idx in range(1, opportunity_count + 1):
            opportunity_message_limit = message_limits[opp_idx - 1]
            opportunity_token_limit = token_limits[opp_idx - 1]
            opportunity_token_used = 0
            messages_this_opportunity = 0

            # 自由議論フェーズ
            while (
                messages_this_opportunity < opportunity_message_limit
                and total_free_messages < effective_max_discussion_turns
                and token_budget_used < free_discussion_token_budget
            ):
                if opportunity_token_used >= opportunity_token_limit:
                    break

                if next_speaker_override is not None:
                    speaker = next_speaker_override
                    next_speaker_override = None
                else:
                    speaker = speakers[total_free_messages % n_speakers]
                other_speaker = "beta" if speaker == "alpha" else "alpha"

                # この話者が回答すべき未回答質問（最古）
                open_for_speaker = [q for q in open_questions if q["addressed_to"] == speaker]
                question_to_answer = open_for_speaker[0] if open_for_speaker else None

                turn_remaining_messages = effective_max_discussion_turns - total_free_messages
                turn_remaining_tokens = discussion_token_budget - token_budget_used
                k = len(open_questions)
                # 新しい質問を出せるのは、自分宛の未回答質問がなく、
                # 残り発言・トークンですべての未回答質問に対する回答分を含められる場合
                can_ask_question = (
                    question_to_answer is None
                    and turn_remaining_messages >= k + 2
                    and turn_remaining_tokens >= (k + 2) * max_new_tokens
                )
                required_question_fields: list[str] = []
                if can_ask_question and question_count == 0:
                    candidate_fields = decision_relevant_hidden_fields(
                        state,
                        agent_observation_scope(speaker),
                        agent_observation_scope(other_speaker),
                    )
                    already_shared = _fields_already_shared(transcript, speaker, candidate_fields)
                    required_question_fields = [
                        field for field in candidate_fields if field not in already_shared
                    ]

                prompt = discussion_prompt(
                    speaker,
                    personas[speaker],
                    persona_params[speaker],
                    state,
                    transcript,
                    opportunity_message_limit,
                    condition,
                    open_question=question_to_answer,
                    can_ask_question=can_ask_question,
                    remaining_messages=turn_remaining_messages,
                    remaining_tokens=turn_remaining_tokens,
                    v_state=current_v_state(speaker),
                    role=resolved_role_for(speaker),
                    required_question_fields=required_question_fields,
                    counter_allowed=v_counter_count < max_v_counter_rounds,
                )

                # §6.2: invalid JSON出力に対する修復リトライ。
                # 同一agentへ修復プロンプトを再送し、retry上限後にinvalidとして確定する。
                # 各attemptの生成tokenをその場で予算へ加算し、実際の推論負荷と記録値を一致させる。
                response: dict[str, Any] = {}
                raw = ""
                token_count = 0
                attempt_audit: list[dict[str, Any]] = []
                max_attempts = max_discussion_retries + 1
                for attempt in range(max_discussion_retries + 1):
                    current_prompt = prompt
                    if attempt > 0:
                        repair_prefix = (
                            "【REPAIR_REQUEST id=discussion-repair】\n"
                            "直前の出力はJSON契約を満たしていません。"
                            "必須キー(speech_act, message, action, reason, reply_to_message_id)"
                            "をすべて含め、説明文やMarkdownなしでJSONのみ返してください。\n"
                            f"直前の出力(先頭): {raw[:200]}\n\n"
                        )
                        current_prompt = repair_prefix + prompt
                    response = get_discussion_message(
                        model,
                        tokenizer,
                        current_prompt,
                        max_new_tokens,
                        fallback_action=fallback,
                        enable_thinking=enable_thinking,
                        thinking_budget=thinking_budget,
                        prompt_runner=prompt_runner,
                        agent=speaker,
                    )
                    if required_question_fields and not response.get("invalid_discussion_output"):
                        returned_fields = set(response.get("requested_fields", []))
                        if (
                            not response.get("requires_response")
                            or not returned_fields.intersection(required_question_fields)
                        ):
                            response = {
                                **response,
                                "invalid_discussion_output": True,
                                "validation_reason": "required_information_question_missing",
                            }
                    raw = response["raw"]
                    attempt_token_count = 0
                    if tokenizer is not None:
                        attempt_token_count = len(tokenizer.encode(raw, add_special_tokens=False))
                    token_count += attempt_token_count
                    if not response.get("invalid_discussion_output"):
                        break
                    attempt_audit.append(
                        {
                            "message_id": str(next_message_id),
                            "speaker": speaker,
                            "agent": speaker,
                            "turn": state.turn,
                            "opportunity": opp_idx,
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "validation_reason": response.get("validation_reason", "invalid_discussion_payload"),
                            "raw": raw,
                            "raw_output": raw,
                            "raw_payload": response.get("raw_payload"),
                            "thinking": response["thinking"],
                            "token_count": attempt_token_count,
                            # ループ終了後、後続attemptの成否を見て確定する。
                            "recovered": False,
                            "final_exhausted": False,
                            # 後方互換: 旧監査consumer向け。
                            "retry_attempts": max_discussion_retries,
                        }
                    )
                    if attempt < max_discussion_retries:
                        discussion_retry_count += 1

                recovered = bool(attempt_audit) and not response.get("invalid_discussion_output")
                if attempt_audit:
                    invalid_attempt_count += len(attempt_audit)
                    if recovered:
                        repaired_invalid_output_count += 1
                    for audit_index, audit in enumerate(attempt_audit):
                        audit["recovered"] = recovered
                        audit["final_exhausted"] = bool(
                            not recovered and audit_index == len(attempt_audit) - 1
                        )
                    invalid_discussion_outputs.extend(attempt_audit)

                # JSON契約違反・壊れたJSON断片は有効発話として扱わない。
                # 監査用の別経路へ保存し、有効トランスクリプトには加えない。
                # リトライ上限後もinvalidの場合はここで確定する。
                if response.get("invalid_discussion_output"):
                    if required_question_fields:
                        missing_required_information_question_count += 1
                    invalid_discussion_output_count += 1
                    next_message_id += 1
                    total_free_messages += 1
                    messages_this_opportunity += 1
                    token_budget_used += token_count
                    opportunity_token_used += token_count
                    continue

                speech_act = response["speech_act"]
                # information_request / question_objection / question は内部表現 question として扱う
                is_question = response["requires_response"]
                addressed_to = response["addressed_to"]
                reply_to_message_id = response["reply_to_message_id"]
                requested_fields = response.get("requested_fields", [])
                reask_reason = response.get("reask_reason", "")

                # §6.6.3: 質問の重複検出をscope判定の前に行う。
                # 同一speakerが同一signatureのopen/closed questionを再送したら、
                # scope判定に入る前にduplicateとして処理する。
                # closed questionの再質問は reask_reason がある場合のみ許可する。
                # §6.6.4: question_count はJSON契約を満たした全質問試行を含む分母とする。
                # unanswerable/self_observable/duplicate いずれも question_count に含める。
                if is_question:
                    question_count += 1
                    if required_question_fields and set(requested_fields).intersection(required_question_fields):
                        required_information_question_count += 1
                    # addressed_to の正規化を先に行う（重複判定に必要）
                    if addressed_to != other_speaker and addressed_to != speaker:
                        addressed_to = other_speaker
                    if not addressed_to or addressed_to == speaker:
                        addressed_to = other_speaker
                    signature = _question_signature(
                        {
                            "speaker": speaker,
                            "addressed_to": addressed_to,
                            "action": response["action"].value if response["action"] else "",
                            "reason": response["reason"],
                            "message": response["message"],
                            "requested_fields": requested_fields,
                        }
                    )
                    # open question の重複検出
                    open_duplicate = any(
                        _question_signature(q) == signature and q["speaker"] == speaker
                        for q in open_questions
                    )
                    # closed question の再質問検出（reask_reason があれば許可）
                    closed_duplicate = any(
                        _question_signature(cq) == signature and cq["speaker"] == speaker
                        for cq in closed_questions
                    ) and not reask_reason
                    if open_duplicate or closed_duplicate:
                        duplicate_question_count += 1
                        consecutive_duplicate_count += 1
                        if consecutive_duplicate_count > max_consecutive_duplicate_questions_recorded:
                            max_consecutive_duplicate_questions_recorded = consecutive_duplicate_count
                        last_duplicate_signature = signature
                        transcript.append(
                            {
                                "speaker": speaker,
                                "speech_act": speech_act.value if speech_act else None,
                                "message": response["message"],
                                "action": response["action"].value if response["action"] else "",
                                "reason": response["reason"],
                                "message_id": str(next_message_id),
                                "addressed_to": addressed_to,
                                "requires_response": True,
                                "reply_to_message_id": None,
                                "duplicate_question": True,
                                "requested_fields": requested_fields,
                                "reask_reason": reask_reason,
                                "raw": raw,
                                "thinking": response["thinking"],
                            }
                        )
                        next_message_id += 1
                        next_speaker_override = addressed_to
                        continue
                    consecutive_duplicate_count = 0
                    last_duplicate_signature = None

                # §6.6.1: observation_scope に基づく宛先選択と回答不能判定。
                # requested_fields が明示されている場合、各fieldを観測できるagentを判定し:
                # - 質問者自身だけが観測可能なfieldのみなら self_observable_question として閉じる
                # - 全fieldを両者とも観測できない場合は unanswerable_question として閉じる
                # - 一部fieldだけ観測不能な場合は unanswerable_partial_fields として明示
                # - 相手が観測できるfieldがあれば、そのagentへルーティング
                unanswerable_by_scope = False
                self_observable_by_scope = False
                unanswerable_partial_fields: list[str] = []
                if is_question and requested_fields:
                    alpha_scope = agent_observation_scope("alpha")
                    beta_scope = agent_observation_scope("beta")
                    speaker_scope = agent_observation_scope(speaker)
                    other = "beta" if speaker == "alpha" else "alpha"
                    other_scope = agent_observation_scope(other)
                    neither_observes_fields: list[str] = []
                    only_speaker_fields: list[str] = []
                    only_other_fields: list[str] = []
                    both_observe_fields: list[str] = []
                    for field in requested_fields:
                        in_speaker = field in speaker_scope
                        in_other = field in other_scope
                        if not in_speaker and not in_other:
                            neither_observes_fields.append(field)
                        elif in_speaker and not in_other:
                            only_speaker_fields.append(field)
                        elif in_other and not in_speaker:
                            only_other_fields.append(field)
                        else:
                            both_observe_fields.append(field)
                    # 全fieldを両者とも観測できない場合は回答不能
                    # both_observe_fields がある場合は回答可能なので unanswerable にしない
                    if neither_observes_fields and not only_speaker_fields and not only_other_fields and not both_observe_fields:
                        unanswerable_by_scope = True
                    # 質問者自身だけが観測可能なfieldのみで、相手が観測できない場合は自己観測可能質問
                    # 相手に聞く必要がないので self_observable_question として閉じる
                    if not unanswerable_by_scope and only_speaker_fields and not only_other_fields and not both_observe_fields:
                        # 全fieldが質問者自身のみ観測可能、または質問者＋両者観測不能
                        # いずれにせよ相手は観測できないので自己観測可能として閉じる
                        self_observable_by_scope = True
                    # 一部fieldだけ観測不能な場合は明示的に記録
                    if neither_observes_fields and (only_speaker_fields or only_other_fields or both_observe_fields):
                        unanswerable_partial_fields = neither_observes_fields
                    # 宛先選択: 相手が観測できるfieldがあればそのagentへ優先ルーティング
                    # 現在のaddressed_toが観測できず、もう一方が観測できる場合は切り替える
                    if not unanswerable_by_scope and not self_observable_by_scope:
                        # 相手に観測させたいfield（質問者自身が観測できないfield）
                        fields_for_other = [f for f in requested_fields if f not in speaker_scope]
                        if fields_for_other:
                            current_can_observe = all(f in agent_observation_scope(addressed_to) for f in fields_for_other) if addressed_to else False
                            other_can_observe = all(f in other_scope for f in fields_for_other)
                            if not current_can_observe and other_can_observe:
                                addressed_to = other

                # 質問の宛先は従来通りもう一方のagentへ正規化する。
                # requested_fields によるscopeルーティングが有効な場合は上記で切り替え済み。
                # それ以外の場合（requested_fields なし、またはscope判定で切り替えなし）は
                # addressed_to が未設定・自分自身・無効な値の場合はもう一方へ正規化する。
                scope_routed = (
                    is_question
                    and bool(requested_fields)
                    and not unanswerable_by_scope
                    and not self_observable_by_scope
                    and addressed_to in ("alpha", "beta")
                    and addressed_to != speaker
                )
                if is_question and not scope_routed and addressed_to != other_speaker:
                    addressed_to = other_speaker

                if unanswerable_by_scope:
                    this_message_id = str(next_message_id)
                    transcript.append(
                        {
                            "speaker": speaker,
                            "speech_act": speech_act.value if speech_act else None,
                            "message": response["message"],
                            "action": response["action"].value if response["action"] else "",
                            "reason": response["reason"],
                            "message_id": this_message_id,
                            "addressed_to": addressed_to,
                            "requires_response": False,
                            "reply_to_message_id": None,
                            "raw": raw,
                            "thinking": response["thinking"],
                            "requested_fields": requested_fields,
                            "closed_as_unanswerable": True,
                            "unanswerable_reason": "neither_agent_observes_requested_fields",
                        }
                    )
                    next_message_id += 1
                    total_free_messages += 1
                    messages_this_opportunity += 1
                    token_budget_used += token_count
                    opportunity_token_used += token_count
                    unanswerable_question_count += 1
                    # §6.6.3: scope回答不能質問を closed_questions へ保存し、
                    # reask_reason なしの再送を抑止する
                    closed_questions.append({
                        "message_id": this_message_id,
                        "speaker": speaker,
                        "addressed_to": addressed_to,
                        "message": response["message"],
                        "reason": response["reason"],
                        "action": response["action"].value if response["action"] else "",
                        "requested_fields": requested_fields,
                    })
                    continue

                if self_observable_by_scope:
                    this_message_id = str(next_message_id)
                    transcript.append(
                        {
                            "speaker": speaker,
                            "speech_act": speech_act.value if speech_act else None,
                            "message": response["message"],
                            "action": response["action"].value if response["action"] else "",
                            "reason": response["reason"],
                            "message_id": this_message_id,
                            "addressed_to": addressed_to,
                            "requires_response": False,
                            "reply_to_message_id": None,
                            "raw": raw,
                            "thinking": response["thinking"],
                            "requested_fields": requested_fields,
                            "closed_as_self_observable": True,
                            "self_observable_reason": "only_speaker_observes_requested_fields",
                        }
                    )
                    next_message_id += 1
                    total_free_messages += 1
                    messages_this_opportunity += 1
                    token_budget_used += token_count
                    opportunity_token_used += token_count
                    self_observable_question_count += 1
                    # §6.6.3: 自己観測可能質問も closed_questions へ保存し、
                    # reask_reason なしの再送を抑止する
                    closed_questions.append({
                        "message_id": this_message_id,
                        "speaker": speaker,
                        "addressed_to": addressed_to,
                        "message": response["message"],
                        "reason": response["reason"],
                        "action": response["action"].value if response["action"] else "",
                        "requested_fields": requested_fields,
                    })
                    continue
                else:
                    consecutive_duplicate_count = 0
                    last_duplicate_signature = None

                # 有効な発話にのみ予算を加算
                token_budget_used += token_count
                opportunity_token_used += token_count

                # 回答すべき未回答質問があるのに質問を返した場合は無効（質問を閉じず同じagentに再試行）
                if is_question and question_to_answer is not None:
                    transcript.append(
                        {
                            "speaker": speaker,
                            "speech_act": speech_act.value if speech_act else None,
                            "message": response["message"],
                            "action": response["action"].value if response["action"] else "",
                            "reason": response["reason"],
                            "message_id": str(next_message_id),
                            "addressed_to": addressed_to,
                            "requires_response": False,
                            "reply_to_message_id": None,
                            "invalid_response_while_answer_required": True,
                            "invalid_response_while_answer_required_reason": (
                                f"{speaker} had question from {question_to_answer['speaker']} "
                                f"(id={question_to_answer['message_id']}) but returned a question"
                            ),
                            "raw": raw,
                            "thinking": response["thinking"],
                        }
                    )
                    next_message_id += 1
                    total_free_messages += 1
                    messages_this_opportunity += 1
                    next_speaker_override = speaker
                    continue

                # ターン全体の絶対上限で質問回答ができない場合は強制意思決定
                if is_question and not can_ask_question:
                    forced_decision_with_open_question = True
                    forced_decision_reason = (
                        f"turn_budget_exhausted_for_reply: {len(open_questions)} open questions, "
                        f"turn_remaining_messages={turn_remaining_messages}, turn_remaining_tokens={turn_remaining_tokens}"
                    )
                    this_message_id = str(next_message_id)
                    transcript.append(
                        {
                            "speaker": speaker,
                            "speech_act": speech_act.value if speech_act else None,
                            "message": response["message"],
                            "action": response["action"].value if response["action"] else "",
                            "reason": response["reason"],
                            "message_id": this_message_id,
                            "addressed_to": addressed_to,
                            "requires_response": True,
                            "reply_to_message_id": None,
                            "raw": raw,
                            "thinking": response["thinking"],
                        }
                    )
                    next_message_id += 1
                    open_questions.append(
                        {
                            "message_id": this_message_id,
                            "speaker": speaker,
                            "addressed_to": addressed_to,
                            "message": response["message"],
                            "timestamp": total_free_messages,
                        }
                    )
                    total_free_messages += 1
                    messages_this_opportunity += 1
                    break

                # 発言を記録
                this_message_id = str(next_message_id)
                proposal, v_response = parse_v_negotiation(
                    response.get("raw_payload"), speaker, this_message_id
                )
                if proposal is not None:
                    proposal, original_id, repaired_id = register_proposal_id(
                        proposal, speaker, this_message_id
                    )
                    if (
                        repaired_id != original_id
                        and v_response is not None
                        and v_response.get("proposal_id") == original_id
                    ):
                        v_response = {**v_response, "proposal_id": repaired_id}
                if (
                    v_response is not None
                    and v_response.get("response") == "counter"
                    and isinstance(v_response.get("counter_proposal"), dict)
                ):
                    if v_counter_count >= max_v_counter_rounds:
                        rejected_v_counter_count += 1
                        v_response = {
                            **v_response,
                            "counter_proposal": None,
                            "counter_rejected_reason": "counter_round_limit_reached",
                        }
                    else:
                        counter, original_counter_id, repaired_counter_id = register_proposal_id(
                            v_response["counter_proposal"], speaker, this_message_id
                        )
                        v_response = {**v_response, "counter_proposal": counter}
                        if v_response.get("self_accept_for_counter_id") == original_counter_id:
                            v_response["self_accept_for_counter_id"] = repaired_counter_id
                        v_counter_count += 1
                transcript_entry: dict[str, Any] = {
                    "speaker": speaker,
                    "speech_act": speech_act.value if speech_act else None,
                    "message": response["message"],
                    "action": response["action"].value if response["action"] else "",
                    "reason": response["reason"],
                    "message_id": this_message_id,
                    "addressed_to": addressed_to,
                    "requires_response": response["requires_response"],
                    "reply_to_message_id": reply_to_message_id,
                    "raw": raw,
                    "thinking": response["thinking"],
                    "v_proposal": proposal,
                    "v_star_response": v_response,
                }
                if is_question:
                    transcript_entry["requested_fields"] = requested_fields
                    if reask_reason:
                        transcript_entry["reask_reason"] = reask_reason
                    if unanswerable_partial_fields:
                        transcript_entry["unanswerable_partial_fields"] = unanswerable_partial_fields
                transcript.append(transcript_entry)
                raw_payload = response.get("raw_payload")
                if (
                    enable_v_flow
                    and condition in {"hivc_d", "hivc_d_prescribed_v1"}
                    and isinstance(raw_payload, dict)
                    and raw_payload.get("share_v_before") is True
                ):
                    explicitly_shared_v_before[speaker] = {
                        "v_before": v_before[speaker],
                        "action_before": action_before[speaker].value if action_before[speaker] else None,
                        "reason_before": reason_before[speaker],
                    }
                    transcript[-1]["shared_v_before"] = explicitly_shared_v_before[speaker]
                if proposal is not None:
                    v_proposals.append(proposal)
                    # 提案を出しただけでは受諾扱いにしない。自分の提案を受諾するには
                    # 同じ JSON に v_star_response: accept を含めるか、後続の V 応答で明示する。
                if v_response is not None:
                    v_responses[speaker].append(v_response)
                    # counter提案者は自分のcounterを自動受諾扱いにしない。
                    # 両agentが同一提案を明示的にacceptすることがV*受諾の要件であり、
                    # counter提案者も後続応答で明示的にacceptする必要がある。
                    # ただし counter出力で self_accept=true が同時表現された場合は、
                    # そのcounter提案への明示的acceptとして別途記録する。
                    if v_response.get("response") == "counter" and isinstance(v_response.get("counter_proposal"), dict):
                        counter = v_response["counter_proposal"]
                        v_proposals.append(counter)
                        self_accept_id = v_response.get("self_accept_for_counter_id")
                        if self_accept_id:
                            v_responses[speaker].append({
                                "response": "accept",
                                "proposal_id": self_accept_id,
                                "message_index": int(this_message_id) if str(this_message_id).isdigit() else 0,
                                "source": "counter_self_accept",
                            })
                if enable_v_flow and condition != "hivc_d_prescribed_v1" and (v_proposals or any(v_responses.values())):
                    v_star_status, v_star_id, v_star, reason = resolve_v_star(v_proposals, v_responses)
                    if v_star_status == "unresolved" and reason:
                        v_star_unresolved_reason = reason
                        v_star_failure_reason = reason
                    else:
                        v_star_failure_reason = ""
                        if v_star_status == "accepted":
                            v_star_unresolved_reason = ""
                    if v_star_status == "accepted" and v_star and v_star.get("scope") == "game":
                        persistent_v_star = v_star
                        persistent_v_star_id = v_star_id
                next_message_id += 1
                total_free_messages += 1
                messages_this_opportunity += 1

                if is_question:
                    # question_count は重複検出ブロックで既に加算済み
                    open_questions.append(
                        {
                            "message_id": this_message_id,
                            "speaker": speaker,
                            "addressed_to": addressed_to,
                            "message": response["message"],
                            "timestamp": total_free_messages - 1,
                            "reason": response["reason"],
                            "action": response["action"].value if response["action"] else "",
                            "requested_fields": requested_fields,
                        }
                    )
                else:
                    # 回答を処理
                    if question_to_answer is not None and reply_to_message_id is None:
                        # 回答すべき未回答質問があるのに reply_to_message_id を返さなかった
                        transcript[-1]["missing_reply_to_message_id_while_answer_required"] = True
                        transcript[-1]["missing_reply_to_message_id_while_answer_required_reason"] = (
                            f"{speaker} had question from {question_to_answer['speaker']} "
                            f"(id={question_to_answer['message_id']}) but reply_to_message_id was missing"
                        )
                        if not forced_decision_reason:
                            forced_decision_reason = "missing_reply_to_message_id_while_answer_required"
                    elif reply_to_message_id is not None:
                        answered_id = str(reply_to_message_id)
                        target_q = None
                        for q in open_questions:
                            if q["message_id"] == answered_id:
                                target_q = q
                                break
                        if target_q is not None and target_q["addressed_to"] == speaker:
                            latency = (total_free_messages - 1) - target_q["timestamp"]
                            question_response_latencies.append(latency)
                            if _is_unanswerable_response(response):
                                unanswerable_question_count += 1
                                transcript[-1]["closed_as_unanswerable"] = True
                            else:
                                answered_question_count += 1
                            # §6.6.3: 回答済み質問を closed_questions へ移動し、
                            # 同一signatureの再質問を reask_reason なしで抑止する
                            closed_questions.append(target_q)
                            open_questions = [q for q in open_questions if q["message_id"] != answered_id]
                        else:
                            # 質問の宛先ではないエージェントや存在しないIDを参照した無効な回答
                            transcript[-1]["reply_to_message_id_invalid"] = True
                            invalid_reason = f"replied_to_question_not_addressed_to_speaker: {answered_id}"
                            if target_q is not None:
                                invalid_reason += f" (addressed_to={target_q['addressed_to']})"
                            else:
                                invalid_reason += " (not_found)"
                            transcript[-1]["reply_to_message_id_invalid_reason"] = invalid_reason
                            if not forced_decision_reason:
                                forced_decision_reason = "invalid_reply_to_message_id"

            # 未回答質問が残っていれば、後続機会で回答を試行する
            if open_questions and opp_idx < opportunity_count and not forced_decision_with_open_question:
                continue

            # 未回答質問が残っていて、かつこれが最後の機会なら強制意思決定
            if open_questions and (opp_idx == opportunity_count or forced_decision_with_open_question):
                if not forced_decision_with_open_question:
                    forced_decision_with_open_question = True
                    if not forced_decision_reason:
                        forced_decision_reason = "absolute_budget_limit_reached"

            # §6.2.2/6.2.3: mandatory V negotiation loop for hivc_d.
            if enable_v_flow and condition == "hivc_d" and turn_v_alignment_required:
                while (
                    v_negotiation_messages_used
                    < reserved_v_messages
                    + v_counter_count * v_messages_per_counter_extension
                    and v_star_status != "accepted"
                    and not v_negotiation_terminal_reason
                ):
                    v_negotiation_messages_used += 1
                    v_proposal_required_prompt_issued = True
                    valid_terminal_reject = False

                    current_proposal = v_proposals[-1] if v_proposals else None
                    if current_proposal is None:
                        # 自由議論で有効提案が出ていない場合、優先agentに提案を要求
                        proposer = priority_agent(seed, state.turn)
                        if proposer not in speakers:
                            proposer = speakers[0]
                        agent = proposer
                        prompt = v_proposal_required_prompt(
                            agent,
                            personas[agent],
                            persona_params[agent],
                            state,
                            condition,
                            transcript,
                            v_state=current_v_state(agent),
                            role=resolved_role_for(agent),
                        )
                        kind = "proposal"
                    else:
                        # 最新提案に対してまだ受諾していないagentに応答を要求
                        pid = current_proposal["proposal_id"]
                        proposer = current_proposal.get("speaker")
                        if proposer not in speakers:
                            proposer = speakers[0]
                        missing = []
                        for a in speakers:
                            accepted = any(
                                r.get("proposal_id") == pid and r.get("response") == "accept"
                                for r in v_responses.get(a, [])
                            )
                            if not accepted:
                                missing.append(a)
                        if not missing:
                            # 全員受諾済みなら resolve_v_star が accepted を返すはず
                            break
                        # 提案者以外を優先してプロンプト
                        if proposer in missing and len(missing) == 1:
                            agent = proposer
                        else:
                            agent = [a for a in missing if a != proposer][0] if any(a != proposer for a in missing) else missing[0]
                        prompt = v_proposal_response_prompt(
                            agent,
                            personas[agent],
                            persona_params[agent],
                            state,
                            condition,
                            transcript,
                            current_proposal,
                            v_state=current_v_state(agent),
                            role=resolved_role_for(agent),
                            counter_allowed=v_counter_count < max_v_counter_rounds,
                        )
                        kind = "response"

                    _, raw = _invoke_prompt(
                        model, tokenizer, prompt, max_new_tokens,
                        enable_thinking=enable_thinking, thinking_budget=thinking_budget,
                        prompt_runner=prompt_runner, agent=agent,
                    )
                    if tokenizer is not None:
                        token_budget_used += len(tokenizer.encode(raw, add_special_tokens=False))
                    total_free_messages += 1
                    msg_id = str(next_message_id)
                    payload = _extract_json_object(raw)
                    if kind == "proposal":
                        proposal, self_response = parse_v_negotiation(payload, agent, msg_id)
                        if proposal is not None:
                            proposal, original_id, repaired_id = register_proposal_id(
                                proposal, agent, msg_id
                            )
                            if (
                                repaired_id != original_id
                                and self_response is not None
                                and self_response.get("proposal_id") == original_id
                            ):
                                self_response = {**self_response, "proposal_id": repaired_id}
                        transcript.append(
                            {
                                "speaker": agent,
                                "speech_act": None,
                                "message": "",
                                "action": "",
                                "reason": "",
                                "message_id": msg_id,
                                "addressed_to": None,
                                "requires_response": False,
                                "reply_to_message_id": None,
                                "raw": raw,
                                "thinking": "",
                                "v_proposal": proposal,
                                "v_star_response": self_response,
                            }
                        )
                        next_message_id += 1
                        if proposal is not None:
                            v_proposals.append(proposal)
                            # v_proposal_required プロンプトでの提案も、明示的な v_star_response: accept
                            # が同時に返されない限り、自動的には受諾扱いにしない。
                            # プロンプトが self-accept を同時要求するため、同じJSON内の
                            # v_star_response を v_responses へ記録する。
                            if self_response is not None:
                                v_responses[agent].append(self_response)
                            if v_protocol_state != "V_RESPOND":
                                v_protocol_transition_history.append({"from": v_protocol_state, "to": "V_RESPOND", "reason": "v_proposal_required_prompt_submitted"})
                                v_protocol_state = "V_RESPOND"
                        else:
                            if v_protocol_state not in {"V_PROPOSE", "V_RESPOND"}:
                                v_protocol_transition_history.append({"from": v_protocol_state, "to": "V_PROPOSE", "reason": "v_proposal_required_prompt_invalid"})
                                v_protocol_state = "V_PROPOSE"
                    else:
                        _, v_response = parse_v_negotiation(payload, agent, msg_id)
                        if (
                            v_response is not None
                            and v_response.get("response") == "counter"
                            and isinstance(v_response.get("counter_proposal"), dict)
                        ):
                            if v_counter_count >= max_v_counter_rounds:
                                rejected_v_counter_count += 1
                                v_response = {
                                    **v_response,
                                    "counter_proposal": None,
                                    "counter_rejected_reason": "counter_round_limit_reached",
                                }
                            else:
                                counter, original_counter_id, repaired_counter_id = register_proposal_id(
                                    v_response["counter_proposal"], agent, msg_id
                                )
                                v_response = {**v_response, "counter_proposal": counter}
                                if v_response.get("self_accept_for_counter_id") == original_counter_id:
                                    v_response["self_accept_for_counter_id"] = repaired_counter_id
                                v_counter_count += 1
                        transcript.append(
                            {
                                "speaker": agent,
                                "speech_act": None,
                                "message": "",
                                "action": "",
                                "reason": "",
                                "message_id": msg_id,
                                "addressed_to": None,
                                "requires_response": False,
                                "reply_to_message_id": None,
                                "raw": raw,
                                "thinking": "",
                                "v_star_response": v_response,
                            }
                        )
                        next_message_id += 1
                        if v_response is not None:
                            v_responses[agent].append(v_response)
                            valid_terminal_reject = (
                                v_response.get("response") == "reject"
                                and current_proposal is not None
                                and v_response.get("proposal_id") == current_proposal.get("proposal_id")
                            )
                            # counter提案者は自分のcounterを自動受諾扱いにしない。
                            # 両agentが同一提案を明示的にacceptすることがV*受諾の要件であり、
                            # counter提案者も後続応答で明示的にacceptする必要がある。
                            # ただし counter出力で self_accept=true が同時表現された場合は、
                            # そのcounter提案への明示的acceptとして別途記録する。
                            if (
                                v_response.get("response") == "counter"
                                and isinstance(v_response.get("counter_proposal"), dict)
                            ):
                                counter = v_response["counter_proposal"]
                                v_proposals.append(counter)
                                self_accept_id = v_response.get("self_accept_for_counter_id")
                                if self_accept_id:
                                    v_responses[agent].append({
                                        "response": "accept",
                                        "proposal_id": self_accept_id,
                                        "message_index": int(msg_id) if str(msg_id).isdigit() else 0,
                                        "source": "counter_self_accept",
                                    })

                    if enable_v_flow and condition != "hivc_d_prescribed_v1":
                        v_star_status, v_star_id, v_star, reason = resolve_v_star(v_proposals, v_responses)
                        if v_star_status == "unresolved" and reason:
                            v_star_unresolved_reason = reason
                            v_star_failure_reason = reason
                        else:
                            v_star_failure_reason = ""
                            if v_star_status == "accepted":
                                v_star_unresolved_reason = ""
                        if v_star_status == "accepted" and v_star and v_star.get("scope") == "game":
                            persistent_v_star = v_star
                            persistent_v_star_id = v_star_id

                    if v_star_status == "accepted":
                        v_star_failure_reason = ""
                        v_star_unresolved_reason = ""
                        v_protocol_transition_history.append({"from": v_protocol_state, "to": "A_CHECK", "reason": "v_star_accepted"})
                        v_protocol_state = "A_CHECK"
                        break
                    if valid_terminal_reject:
                        # 有効なrejectは当該提案への最終回答。同じagentへ同じproposalの
                        # accept/rejectを再要求せず、V*未成立のままAction調整へ進む。
                        v_negotiation_terminal_reason = "v_proposal_rejected"
                        break

                if v_star_status != "accepted":
                    missing_v_proposal_after_required_prompt = not v_proposals
                    if v_negotiation_terminal_reason:
                        v_star_failure_reason = v_negotiation_terminal_reason
                        v_star_unresolved_reason = v_negotiation_terminal_reason
                        transition_reason = v_negotiation_terminal_reason
                    else:
                        max_v_attempts = (
                            reserved_v_messages
                            + v_counter_count * v_messages_per_counter_extension
                        )
                        if v_negotiation_messages_used >= max_v_attempts:
                            v_star_unresolved_reason = v_star_failure_reason or v_star_unresolved_reason
                            v_star_failure_reason = "v_negotiation_budget_exhausted"
                        else:
                            v_star_failure_reason = v_star_unresolved_reason or v_star_failure_reason or "missing_matching_explicit_acceptance"
                        transition_reason = "v_negotiation_budget_exhausted"
                    if v_protocol_state != "A_CHECK":
                        v_protocol_transition_history.append({"from": v_protocol_state, "to": "A_CHECK", "reason": transition_reason})
                    v_protocol_state = "A_CHECK"
                    if not forced_decision_reason and not v_negotiation_terminal_reason:
                        forced_decision_reason = "v_negotiation_budget_exhausted"
            elif enable_v_flow and v_protocol_state not in {"A_CHECK", "FINAL_VOTE"}:
                v_protocol_transition_history.append({"from": v_protocol_state, "to": "A_CHECK", "reason": "pre_decision"})
                v_protocol_state = "A_CHECK"

            # 最終投票はシステム側で安全性とV*整合性を検証し、違反時は同じagentへ再試行する。
            # モデルの v_star_consistent 自己申告は判定に使用しない。
            safe_action_candidates = [
                candidate for candidate in ALL_ACTIONS
                if preview_action_safety(state, candidate)[0]
            ]
            validated_fallback = fallback
            if v_star_status == "accepted":
                v_consistent_fallbacks = [
                    candidate for candidate in safe_action_candidates
                    if verify_vote_v_star_consistency(
                        candidate, "", v_star_id, None, v_star_id, v_star, state
                    ) is True
                ]
                if v_consistent_fallbacks:
                    validated_fallback = max(
                        v_consistent_fallbacks, key=lambda candidate: q_values[candidate]
                    )

            def collect_validated_vote(agent: str) -> tuple[Action | None, str, str, bool, str, str, str, bool | None]:
                nonlocal final_vote_retry_count
                base_prompt = decision_opportunity_prompt(
                    agent,
                    personas[agent],
                    persona_params[agent],
                    state,
                    transcript,
                    condition,
                    opp_idx,
                    opportunity_count,
                    v_state=current_v_state(agent),
                    role=resolved_role_for(agent),
                )
                feedback = ""
                last: tuple[Action | None, str, str, bool, str, str] = (None, "", "", False, "", "")
                referenced_id = ""
                system_consistent: bool | None = None
                for attempt in range(max_final_vote_retries + 1):
                    prompt = base_prompt if not feedback else (
                        "【FINAL_VOTE_REPAIR id=final-vote-repair】\n"
                        f"直前の投票はシステム検証で拒否されました: {feedback}\n"
                        "別の安全かつ受諾V*に整合する行動をJSONだけで返してください。\n\n"
                        + base_prompt
                    )
                    last = get_action(
                        model,
                        tokenizer,
                        prompt,
                        max_new_tokens,
                        fallback,
                        enable_thinking=enable_thinking,
                        thinking_budget=thinking_budget,
                        prompt_runner=prompt_runner,
                        agent=agent,
                    )
                    action, reason, message, ready, raw, thinking = last
                    referenced_id, claimed_consistent = extract_vote_v_fields(raw)
                    rejection_reason = ""
                    if reason.startswith("invalid_response_fallback") or action is None:
                        rejection_reason = "invalid_final_vote"
                    elif safe_action_candidates:
                        safe, safety_reason, preview = preview_action_safety(state, action)
                        if not safe:
                            rejection_reason = "unsafe_action:" + final_vote_repair_feedback(
                                state, action, agent, resolved_role_for(agent)
                            )
                    if not rejection_reason and v_star_status == "accepted":
                        system_consistent = verify_vote_v_star_consistency(
                            action, reason, referenced_id, claimed_consistent, v_star_id, v_star, state
                        )
                        if system_consistent is False:
                            rejection_reason = "v_star_action_inconsistent"
                    if not rejection_reason:
                        return action, reason, message, ready, raw, thinking, referenced_id, system_consistent
                    rejected_final_votes.append(
                        {
                            "agent": agent,
                            "opportunity": opp_idx,
                            "attempt": attempt + 1,
                            "action": action.value if action else "",
                            "reason": reason,
                            "rejection_reason": rejection_reason,
                        }
                    )
                    feedback = rejection_reason
                    if attempt < max_final_vote_retries:
                        final_vote_retry_count += 1
                _, reason, message, ready, raw, thinking = last
                return None, reason, message, ready, raw, thinking, referenced_id, False

            (
                alpha_vote, alpha_vote_reason, alpha_vote_message, alpha_vote_ready,
                alpha_vote_raw, alpha_vote_thinking, alpha_vote_v_star_id, alpha_vote_system_consistent,
            ) = collect_validated_vote("alpha")
            (
                beta_vote, beta_vote_reason, beta_vote_message, beta_vote_ready,
                beta_vote_raw, beta_vote_thinking, beta_vote_v_star_id, beta_vote_system_consistent,
            ) = collect_validated_vote("beta")
            _, alpha_vote_v_star_claim = extract_vote_v_fields(alpha_vote_raw)
            _, beta_vote_v_star_claim = extract_vote_v_fields(beta_vote_raw)

            # 投票は全エージェントが出し終わってからトランスクリプトへ追加
            transcript.append(
                {
                    "speaker": "alpha",
                    "action": alpha_vote.value if alpha_vote else "",
                    "reason": alpha_vote_reason,
                    "message": alpha_vote_message,
                    "ready": str(alpha_vote_ready).lower(),
                    "raw": alpha_vote_raw,
                    "thinking": alpha_vote_thinking,
                }
            )
            transcript.append(
                {
                    "speaker": "beta",
                    "action": beta_vote.value if beta_vote else "",
                    "reason": beta_vote_reason,
                    "message": beta_vote_message,
                    "ready": str(beta_vote_ready).lower(),
                    "raw": beta_vote_raw,
                    "thinking": beta_vote_thinking,
                }
            )

            consensus = (
                alpha_vote is not None
                and beta_vote is not None
                and alpha_vote == beta_vote
                and alpha_vote_ready
                and beta_vote_ready
            )
            decision_history.append(
                {
                    "opportunity_index": opp_idx,
                    "opportunity_count": opportunity_count,
                    "alpha_vote": alpha_vote.value if alpha_vote else "",
                    "alpha_reason": alpha_vote_reason,
                    "alpha_ready": alpha_vote_ready,
                    "beta_vote": beta_vote.value if beta_vote else "",
                    "beta_reason": beta_vote_reason,
                    "beta_ready": beta_vote_ready,
                    "consensus": consensus,
                    "v_star_id": v_star_id,
                    "v_star_status": v_star_status,
                    "alpha_v_star_consistent": alpha_vote_system_consistent,
                    "beta_v_star_consistent": beta_vote_system_consistent,
                }
            )

            final_attempt_index = opp_idx
            if consensus:
                group_action = alpha_vote
                decision_rule = "consensus"
                group_reason = (
                    f"consensus on action {alpha_vote.value}: "
                    f"alpha reason={alpha_vote_reason}; beta reason={beta_vote_reason}"
                )
                break
            if forced_decision_with_open_question:
                break

        if group_action is None:
            priority = priority_agent(seed, state.turn)
            fallback_priority_agent = priority
            fallback_used = True
            if priority == "alpha":
                if alpha_vote is not None:
                    group_action = alpha_vote
                    group_reason = f"fallback priority agent alpha: {alpha_vote_reason}"
                    decision_rule = "fallback_priority"
                elif beta_vote is not None:
                    group_action = beta_vote
                    group_reason = f"fallback priority agent alpha invalid; using beta: {beta_vote_reason}"
                    decision_rule = "fallback_priority"
                else:
                    group_action = validated_fallback
                    group_reason = "both votes invalid; fallback to validated safe action"
                    decision_rule = "fallback_validated"

            else:
                if beta_vote is not None:
                    group_action = beta_vote
                    group_reason = f"fallback priority agent beta: {beta_vote_reason}"
                    decision_rule = "fallback_priority"
                elif alpha_vote is not None:
                    group_action = alpha_vote
                    group_reason = f"fallback priority agent beta invalid; using alpha: {alpha_vote_reason}"
                    decision_rule = "fallback_priority"
                else:
                    group_action = validated_fallback
                    group_reason = "both votes invalid; fallback to validated safe action"
                    decision_rule = "fallback_validated"

        # 最終防衛線: 将来の呼び出し側変更でも確実な即時敗北行動をstepへ渡さない。
        safety_override_used = False
        rejected_group_action = ""
        rejected_group_action_reason = ""
        group_safe, group_safety_reason, _ = preview_action_safety(state, group_action)
        if not group_safe and safe_fallback_candidates:
            rejected_group_action = group_action.value
            rejected_group_action_reason = group_safety_reason
            group_action = validated_fallback
            safety_override_used = True
            fallback_used = True
            decision_rule = "safety_fallback"
            group_reason = (
                f"system safety gate rejected {rejected_group_action}: {group_safety_reason}; "
                f"using safe action {group_action.value}"
            )

        if enable_v_flow and v_protocol_state != "FINAL_VOTE":
            v_protocol_transition_history.append({"from": v_protocol_state, "to": "FINAL_VOTE", "reason": "final_votes_collected"})
            v_protocol_state = "FINAL_VOTE"

        # §6.5: collect v_after only after final votes/group decision are fixed.
        v_after: dict[str, dict[str, Any] | None] = {"alpha": None, "beta": None}
        if enable_v_flow:
            for agent in speakers:
                prompt = v_measurement_prompt(
                    agent,
                    state,
                    phase="after",
                    current_v=v_before[agent],
                    persona=personas[agent],
                    persona_params=persona_params[agent],
                    transcript=transcript,
                    final_vote={
                        "alpha": {"action": alpha_vote.value if alpha_vote else None, "reason": alpha_vote_reason},
                        "beta": {"action": beta_vote.value if beta_vote else None, "reason": beta_vote_reason},
                        "group_action": group_action.value,
                    },
                    v_state=current_v_state(agent),
                    role=resolved_role_for(agent),
                )
                measured_v: dict[str, Any] | None = None
                error = ""
                raw_measurement = ""
                for attempt in range(max_v_measurement_retries + 1):
                    _, raw_measurement = _invoke_prompt(
                        model, tokenizer, prompt, max_new_tokens,
                        enable_thinking=enable_thinking, thinking_budget=thinking_budget,
                        prompt_runner=prompt_runner, agent=agent,
                    )
                    measurement_call_count += 1
                    if tokenizer is not None:
                        measurement_token_count += len(tokenizer.encode(raw_measurement, add_special_tokens=False))
                    measured_v, _, _, error = extract_json_v_measurement(raw_measurement)
                    if not error:
                        break
                    if attempt < max_v_measurement_retries:
                        measurement_retry_count += 1
                v_after[agent] = measured_v
                v_measurement_errors[f"{agent}_after"] = error or ("measurement_failed" if measured_v is None else "")

        alpha_v_star_consistent = v_star_status == "accepted" and verify_vote_v_star_consistency(
            alpha_vote, alpha_vote_reason, alpha_vote_v_star_id, alpha_vote_v_star_claim, v_star_id, v_star, state
        )
        beta_v_star_consistent = v_star_status == "accepted" and verify_vote_v_star_consistency(
            beta_vote, beta_vote_reason, beta_vote_v_star_id, beta_vote_v_star_claim, v_star_id, v_star, state
        )
        v_star_action_consistency: bool | None = (
            alpha_v_star_consistent and beta_v_star_consistent
            if v_star_status == "accepted"
            else None
        )
        result = step(state, group_action, rng)
        regret = q_values[optimal] - q_values[group_action]

        optimal_route_value = optimal_route(state, seed=seed + state.turn * 1000, n_rollouts=20)
        route = route_of_action(group_action)
        prev_route = planned_route
        if route in ("comms", "escape"):
            planned_route = route
        elif planned_route == "undecided":
            planned_route = optimal_route_value
        route_switch = (prev_route in ("comms", "escape") and planned_route in ("comms", "escape") and prev_route != planned_route)

        alpha_vote_value = alpha_vote.value if alpha_vote else ""
        beta_vote_value = beta_vote.value if beta_vote else ""
        individual_actions = f"{alpha_vote_value},{beta_vote_value}"
        individual_reasons = json.dumps(
            {"alpha": alpha_vote_reason, "beta": beta_vote_reason}, ensure_ascii=False
        )
        role_evidence = {
            "alpha": _role_evidence("alpha", state, resolved_role_for("alpha")),
            "beta": _role_evidence("beta", state, resolved_role_for("beta")),
        }

        unanswered_question_count = len(open_questions)
        silent_unanswered_question_count = (
            unanswered_question_count
            if unanswered_question_count > 0 and not forced_decision_with_open_question and not forced_decision_reason
            else 0
        )
        question_response_latency = float(np.mean(question_response_latencies)) if question_response_latencies else float("nan")

        resolved_by_agent = resolved_profiles if isinstance(resolved_profiles, dict) else {}

        def profile_metadata(agent: str) -> tuple[str, str, str, str]:
            resolved = resolved_by_agent.get(agent)
            role = getattr(resolved, "role", None)
            value = getattr(resolved, "value", None)
            if resolved is None:
                params = persona_params.get(agent) or {}
                separated = params.get("_resolved_profile")
                if isinstance(separated, dict):
                    role = separated.get("role")
                    value = separated.get("value")
            role_mapping_id = role.get("id") if isinstance(role, dict) else None
            value_mapping_id = value.get("id") if isinstance(value, dict) else None
            role_id = str(getattr(role, "id", None) or role_mapping_id or role_keys.get(agent) or "")
            value_id = str(getattr(value, "id", None) or value_mapping_id or "")
            value_body = value.to_dict() if hasattr(value, "to_dict") else value
            value_hash = _profile_sha256(value_body) if value_body is not None else ""
            role_body = role.to_dict() if hasattr(role, "to_dict") else role
            role_hash = _profile_sha256(role_body) if role_body is not None else ""
            return role_id, value_id, value_hash, role_hash

        alpha_role_id, alpha_value_id, alpha_value_hash, alpha_role_hash = profile_metadata("alpha")
        beta_role_id, beta_value_id, beta_value_hash, beta_role_hash = profile_metadata("beta")

        def compatibility_metadata(agent: str) -> tuple[str, str]:
            params = persona_params.get(agent) or {}
            separated = params.get("_resolved_profile")
            if not isinstance(separated, dict):
                return ("legacy-1" if role_value_mode == "legacy_hard" else "", "")
            role_body = separated.get("role") or {}
            warnings_body = separated.get("warnings") or []
            schema = str(role_body.get("schema_version", "")) if isinstance(role_body, dict) else ""
            return schema, _canonical_json(warnings_body)

        alpha_profile_schema, alpha_profile_warnings = compatibility_metadata("alpha")
        beta_profile_schema, beta_profile_warnings = compatibility_metadata("beta")
        distance_before = v_alignment_distance(v_before["alpha"], v_before["beta"])
        distance_after = v_alignment_distance(v_after["alpha"], v_after["beta"])

        row: dict[str, object] = {
            "game_id": seed,
            "seed": seed,
            "condition": condition,
            "scenario_id": state.scenario_id,
            "turn": state.turn,
            "event": state.current_event.value,
            "role_value_mode": role_value_mode or "legacy_unmeasured",
            "role_value_assignment_id": role_value_assignment_id,
            "value_criteria_schema_id": DEFAULT_VALUE_CRITERIA_SCHEMA.id if enable_v_flow else "",
            "value_criteria_schema_version": DEFAULT_VALUE_CRITERIA_SCHEMA.version if enable_v_flow else "",
            "alpha_role_id": alpha_role_id,
            "beta_role_id": beta_role_id,
            "alpha_role_sha256": alpha_role_hash,
            "beta_role_sha256": beta_role_hash,
            "alpha_profile_schema_version": alpha_profile_schema,
            "beta_profile_schema_version": beta_profile_schema,
            "alpha_profile_warnings": alpha_profile_warnings,
            "beta_profile_warnings": beta_profile_warnings,
            "alpha_value_profile_id": alpha_value_id,
            "beta_value_profile_id": beta_value_id,
            "alpha_value_profile_sha256": alpha_value_hash,
            "beta_value_profile_sha256": beta_value_hash,
            "alpha_role_key": role_keys["alpha"],
            "beta_role_key": role_keys["beta"],
            "alpha_persona": personas["alpha"],
            "beta_persona": personas["beta"],
            "alpha_persona_params": json.dumps(persona_params["alpha"], ensure_ascii=False, sort_keys=True),
            "beta_persona_params": json.dumps(persona_params["beta"], ensure_ascii=False, sort_keys=True),
            "state_before": json.dumps(state.as_dict(), ensure_ascii=False, sort_keys=True),
            "alpha_v_before": _canonical_json(v_before["alpha"]) if v_before["alpha"] is not None else "",
            "beta_v_before": _canonical_json(v_before["beta"]) if v_before["beta"] is not None else "",
            "alpha_action_before": action_before["alpha"].value if action_before["alpha"] else "",
            "beta_action_before": action_before["beta"].value if action_before["beta"] else "",
            "alpha_reason_before": reason_before["alpha"],
            "beta_reason_before": reason_before["beta"],
            "v_proposals": _canonical_json(v_proposals),
            "v_proposal_id_repair_count": len(v_proposal_id_repairs),
            "v_proposal_id_repairs": _canonical_json(v_proposal_id_repairs),
            "v_counter_count": v_counter_count,
            "rejected_v_counter_count": rejected_v_counter_count,
            "v_negotiation_message_budget": (
                reserved_v_messages + v_counter_count * v_messages_per_counter_extension
                if enable_v_flow and condition == "hivc_d" and turn_v_alignment_required
                else 0
            ),
            "v_negotiation_messages_used": v_negotiation_messages_used,
            "v_star_id": v_star_id,
            "v_star": _canonical_json(v_star) if v_star is not None else "",
            "v_star_scope": str(v_star.get("scope", "")) if v_star is not None else "",
            "v_star_status": v_star_status,
            "v_star_failure_reason": v_star_failure_reason,
            "v_star_unresolved_reason": v_star_unresolved_reason if enable_v_flow else "",
            "alpha_v_star_response": _canonical_json(v_responses["alpha"]),
            "beta_v_star_response": _canonical_json(v_responses["beta"]),
            "alpha_v_after": _canonical_json(v_after["alpha"]) if v_after["alpha"] is not None else "",
            "beta_v_after": _canonical_json(v_after["beta"]) if v_after["beta"] is not None else "",
            "alpha_vote_changed": (action_before["alpha"] != alpha_vote) if action_before["alpha"] is not None and alpha_vote is not None else "",
            "beta_vote_changed": (action_before["beta"] != beta_vote) if action_before["beta"] is not None and beta_vote is not None else "",
            "alpha_v_star_consistent": alpha_v_star_consistent if v_star_status == "accepted" else "",
            "beta_v_star_consistent": beta_v_star_consistent if v_star_status == "accepted" else "",
            "v_alignment_distance_before": distance_before,
            "v_alignment_distance_after": distance_after,
            "v_alignment_required": turn_v_alignment_required if enable_v_flow else "",
            "v_alignment_requirement_reasons": _canonical_json(turn_v_alignment_requirement_reasons) if enable_v_flow else "",
            "v_protocol_state": v_protocol_state if enable_v_flow else "",
            "v_protocol_transition_history": _canonical_json(v_protocol_transition_history) if enable_v_flow else "",
            "v_star_action_consistency": v_star_action_consistency if v_star_action_consistency is not None else "",
            "v_measurement_call_count": measurement_call_count,
            "v_measurement_token_count": measurement_token_count,
            "v_measurement_retry_count": measurement_retry_count,
            "v_measurement_errors": _canonical_json(v_measurement_errors),
            "individual_actions": individual_actions,
            "individual_reasons": individual_reasons,
            "discussion_turns": total_free_messages,
            "discussion_token_budget_used": token_budget_used,
            "discussion_transcript": json.dumps(transcript, ensure_ascii=False),
            "alpha_vote": alpha_vote_value,
            "alpha_vote_reason": alpha_vote_reason,
            "alpha_vote_message": alpha_vote_message,
            "alpha_vote_ready": str(alpha_vote_ready).lower(),
            "alpha_vote_raw": alpha_vote_raw,
            "alpha_vote_thinking": alpha_vote_thinking,
            "beta_vote": beta_vote_value,
            "beta_vote_reason": beta_vote_reason,
            "beta_vote_message": beta_vote_message,
            "beta_vote_ready": str(beta_vote_ready).lower(),
            "beta_vote_raw": beta_vote_raw,
            "beta_vote_thinking": beta_vote_thinking,
            "group_action": group_action.value,
            "group_action_label": ACTION_LABELS[group_action],
            "group_reason": group_reason,
            "decision_rule": decision_rule,
            "best_action": optimal.value,
            "acceptable_actions": ",".join(sorted(a.value for a in allowed)),
            "regret": round(float(regret), 3),
            "q_values": json.dumps({a.value: round(v, 3) for a, v in q_values.items()}, sort_keys=True),
            "state_after": json.dumps(result.state_after.as_dict(), ensure_ascii=False, sort_keys=True),
            "outcome": result.outcome,
            "terminal_score": terminal_score(result.state_after),
            "decision_opportunity_count": opportunity_count,
            "decision_attempts": final_attempt_index,
            "decision_attempt_index": final_attempt_index,
            "free_discussion_message_count": total_free_messages,
            "decision_history": json.dumps(decision_history, ensure_ascii=False),
            "final_vote_retry_count": final_vote_retry_count,
            "rejected_final_vote_count": len(rejected_final_votes),
            "rejected_final_votes": _canonical_json(rejected_final_votes),
            "fallback_used": fallback_used,
            "fallback_priority_agent": fallback_priority_agent,
            "safety_override_used": safety_override_used,
            "rejected_group_action": rejected_group_action,
            "rejected_group_action_reason": rejected_group_action_reason,
            "planned_route": planned_route,
            "optimal_route": optimal_route_value,
            "route_switch": route_switch,
            "premature": result.premature,
            "role_specific_evidence": json.dumps(role_evidence, ensure_ascii=False),
            "alpha_evidence": role_evidence["alpha"],
            "beta_evidence": role_evidence["beta"],
            "unanswered_question_count": unanswered_question_count,
            "silent_unanswered_question_count": silent_unanswered_question_count,
            "unanswerable_question_count": unanswerable_question_count,
            "self_observable_question_count": self_observable_question_count,
            "question_count": question_count,
            "answered_question_count": answered_question_count,
            "duplicate_question_count": duplicate_question_count,
            "required_information_question_count": required_information_question_count,
            "missing_required_information_question_count": missing_required_information_question_count,
            "max_consecutive_duplicate_questions": max_consecutive_duplicate_questions_recorded,
            "invalid_discussion_output_count": invalid_discussion_output_count,
            "invalid_attempt_count": invalid_attempt_count,
            "repaired_invalid_output_count": repaired_invalid_output_count,
            "invalid_discussion_outputs": _canonical_json(invalid_discussion_outputs),
            "discussion_retry_count": discussion_retry_count,
            "question_response_latency": question_response_latency,
            "forced_decision_with_open_question": forced_decision_with_open_question,
            "forced_decision_reason": forced_decision_reason,
            "v_proposal_required_prompt_issued": v_proposal_required_prompt_issued if enable_v_flow else "",
            "missing_v_proposal_after_required_prompt": missing_v_proposal_after_required_prompt if enable_v_flow else "",
        }
        rows.append(row)
        print(
            f"[{condition} seed={seed}] turn={row['turn']} event={row['event']} "
            f"disc={row['discussion_turns']} opp={row['decision_attempt_index']}/{row['decision_opportunity_count']} "
            f"alpha={alpha_vote_value} beta={beta_vote_value} "
            f"group={row['group_action']} rule={row['decision_rule']} "
            f"best={row['best_action']} regret={row['regret']} outcome={row['outcome']} "
            f"open_questions={unanswered_question_count} forced={forced_decision_with_open_question}"
        )
        if live_jsonl_path:
            with open(live_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        state = result.state_after

    return rows
