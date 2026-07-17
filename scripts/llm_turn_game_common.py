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
    ACTION_LABELS,
    ALL_ACTIONS,
    Action,
    EVENT_LABELS,
    acceptable_actions,
    best_action,
    estimate_q_values,
    initial_state,
    optimal_route,
    role_specific_evidence,
    route_of_action,
    step,
    terminal_score,
)


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
) -> None:
    """Record the exact per-game assignment as an authoritative snapshot.

    固定プロファイルでもseedごとに明示的な割当レコードを生成する。
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
    manifest.setdefault("game_profile_assignments", []).append({
        "role_value_assignment_id": assignment_id,
        "seed": seed,
        "agents": agents,
    })


def _normalize_v(
    value: Any,
    schema=DEFAULT_VALUE_CRITERIA_SCHEMA,
) -> dict[str, Any] | None:
    """Validate the model-facing V representation against the common ontology.

    V測定（v_before / v_after）は完全なcriteria集合とweightsを必要とする。
    """
    if not isinstance(value, dict):
        return None
    expected = set(schema.criteria)
    criteria = value.get("ordered_criteria")
    weights = value.get("weights")
    confidence = value.get("confidence")
    if not isinstance(criteria, list) or not criteria or not all(isinstance(v, str) and v.strip() for v in criteria):
        return None
    ordered = [v.strip() for v in criteria]
    if set(ordered) != expected or len(ordered) != len(expected):
        return None
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
    """Normalized L1 distance; NaN when comparable numeric V is unavailable."""
    if not first or not second or not isinstance(first.get("weights"), dict) or not isinstance(second.get("weights"), dict):
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
    """§6.2.1: ターン開始時にモデル自己申告に依存せず v_alignment_required を判定する。"""
    reasons: list[str] = []
    if alpha_action is not None and beta_action is not None and alpha_action != beta_action:
        reasons.append("action_before_mismatch")
    distance = v_alignment_distance(alpha_v, beta_v)
    if not math.isnan(distance) and distance >= threshold:
        reasons.append(f"l1_distance_{distance:.3f}_above_{threshold}")
    alpha_top = _top_criterion(alpha_v)
    beta_top = _top_criterion(beta_v)
    if alpha_top is not None and beta_top is not None and alpha_top != beta_top:
        reasons.append("top_criterion_mismatch")
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


def _question_signature(item: dict[str, Any]) -> tuple[str, str, str]:
    """質問の (speaker, addressed_to, normalized_fields) signature。

    requested_fields が明示されていない場合は action + reason + message を正規化して使用する。
    """
    speaker = str(item.get("speaker", "")).strip().lower()
    addressed_to = str(item.get("addressed_to", "")).strip().lower()
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
    weights が含まれる場合も同じcriteria集合を持つ必要がある。
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
    """Parse proposal and accept/reject/counter without ever inferring acceptance."""
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

    role_mapping = _role_body(role)
    if role_mapping is not None:
        scope = {str(item) for item in role_mapping.get("observation_scope", [])}

        def hidden(field: str, *aliases: str) -> bool:
            return not any(name in scope for name in (field, *aliases))
    else:
        # Legacy fallback: historical alpha/beta visibility remains unchanged.
        hide = {
            "alpha": {"communication": True, "pod_integrity": True, "pod_readiness": True},
            "beta": {"hull_damage": True, "flooding": True},
        }.get(agent_name, {})

        def hidden(field: str, *aliases: str) -> bool:
            return hide.get(field, False)

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
    return "\n".join([f"{action.value}. {ACTION_LABELS[action]}" for action in ALL_ACTIONS])


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
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


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


def extract_json_discussion(response: str) -> tuple[SpeechAct | None, str, Action | None, str, str | None, str | None, bool]:
    """自由議論用JSONをパースする。"""
    text = response.strip()
    payload = _extract_json_object(text)
    if payload is not None:
        try:
            speech_act = _normalize_speech_act(payload.get("speech_act"))
            message = str(payload.get("message", "")).strip()
            reason = str(payload.get("reason", "")).strip()
            reply_to_message_id = payload.get("reply_to_message_id")
            if reply_to_message_id is not None:
                reply_to_message_id = str(reply_to_message_id).strip() or None
            addressed_to = str(payload.get("addressed_to", "")).strip() or None
            # information_request / question_objection / question は内部表現 question として回答を要請
            requires_response = speech_act in QUESTION_SPEECH_ACTS
            action: Action | None = None
            if "action" in payload:
                action_text = str(payload["action"]).strip().upper()
                if action_text in {action.value for action in ALL_ACTIONS}:
                    action = Action(action_text)
            return speech_act, message, action, reason, reply_to_message_id, addressed_to, requires_response
        except (TypeError, ValueError):
            pass
    return None, text[:160], None, "", None, None, False


def format_persona(agent_name: str, persona: str, persona_params: dict[str, object] | None) -> str:
    if not persona_params:
        return persona
    resolved = persona_params.get("_resolved_profile")
    if isinstance(resolved, dict):
        role = resolved.get("role") or {}
        presentation = resolved.get("persona") or {}
        value = resolved.get("value")
        value_text = "明示的な初期重みなし" if value is None else _canonical_json(value)
        mode = str(resolved.get("role_value_mode", "soft_value"))
        value_guidance = (
            "priority_weights は固定された意思決定基準です。変更・再交渉しないでください。"
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
        priority_text = ", ".join([f"{key}={value}" for key, value in priority_weights.items()])
    else:
        priority_text = str(priority_weights)
    return "\n".join(
        [
            f"name: {agent_name}",
            f"role: {persona_params.get('role', persona)}",
            f"priority_weights: {priority_text}",
            f"risk_tolerance: {persona_params.get('risk_tolerance', 'unspecified')}  # 0.0=極めて慎重, 1.0=高リスク許容",
            f"goal_focus: {persona_params.get('goal_focus', 'unspecified')}",
            f"communication_style: {persona_params.get('communication_style', 'unspecified')}",
            f"concession_tendency: {persona_params.get('concession_tendency', 'unspecified')}  # 0.0=譲らない, 1.0=譲歩しやすい",
            f"evidence_demand: {persona_params.get('evidence_demand', 'unspecified')}  # 0.0=直感重視, 1.0=根拠要求が強い",
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
        "必須キー: speech_act, message, action, reason, addressed_to, reply_to_message_id\n"
        "- addressed_to: 質問の宛先。質問以外は null\n"
        "- reply_to_message_id: 回答対象の質問ID。回答以外は null"
    )

    if open_question is not None:
        answer_example = {
            "speech_act": "evidence",
            "message": "質問への短い回答",
            "action": "A",
            "reason": "短い理由",
            "addressed_to": open_question["speaker"],
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
        "speech_act": "question_objection",
        "message": "相手への短い質問",
        "action": "A",
        "reason": "確認したい理由",
        "addressed_to": other_agent,
        "reply_to_message_id": None,
    }
    return (
        f"{required_keys}\n"
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
    weight_example = {c: 0.2 for c in criteria}
    # 合計が1になるように最後の項目を微調整（浮動小数点誤差を避けるため）
    weight_example[criteria[-1]] = round(1.0 - sum(weight_example[c] for c in criteria[:-1]), 6)
    criteria_json = _canonical_json(criteria)
    weights_json = _canonical_json(weight_example)
    if phase == "before":
        contract = (
            f'{{"v_before":{{"ordered_criteria":{criteria_json},'
            f'"weights":{weights_json},"confidence":0.6}},'
            f'"action_before":"A","reason_before":"短い理由"}}'
        )
        instruction = "相手の発言を見る前の、あなた自身の暫定判断基準と行動案を記録してください。"
    else:
        contract = (
            f'{{"v_after":{{"ordered_criteria":{criteria_json},'
            f'"weights":{weights_json},"confidence":0.6}},'
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

【CURRENT_OBSERVATION id=state】
{format_state(state, agent_name, role)}

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
) -> str:
    context = _question_context(open_question, can_ask_question, remaining_messages, remaining_tokens)
    json_contract = _discussion_json_contract(agent_name, open_question)
    criteria_example = _canonical_json(list(DEFAULT_VALUE_CRITERIA_SCHEMA.criteria))
    v_sharing_guide = (
        f"HIVC-D条件では必要に応じ v_proposal と v_star_response を追加できます。\n"
        f"HIVC-D条件で自分の事前V測定を明示共有する場合だけ share_v_before=true を追加できます。\n"
        f'v_proposal={{"proposal_id":"一意ID","ordered_criteria":{criteria_example},"scope":"turn"}}\n'
        f'v_star_response={{"response":"accept|reject|counter","proposal_id":"対象ID"}}。対象IDなしの応答は無効です。\n'
        f"counterの場合は v_star_response.counter_proposal に proposal_id、ordered_criteria({criteria_example})、scope、任意のweightsを含む完全な代替案を必ず入れてください。"
        if condition in {"hivc_d", "hivc_d_prescribed_v1"}
        else ""
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

あなたの役割固有情報:
{_role_evidence(agent_name, state, role)}

選択可能な行動:
{action_list()}

【DISCUSSION_HISTORY id=history】
これまでの議論:
{format_transcript_text(transcript)}

{context}

自由議論の発言目的は以下のいずれかを speech_act として選んでください:
{_speech_act_guide()}

この自由議論フェーズでは最大 {max_discussion_turns} 発言までです。
行動案を述べたい場合は action（A-F）と reason を含めてください。
ready は不要です。
{v_sharing_guide}
質問をする場合は speech_act に "question_objection" を使い、addressed_to を指定してください。
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
    v_contract = (
        ',"v_star_id":"' + str(v_state.get("v_star_id")) + '","v_star_consistent":true'
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

あなたの役割固有情報:
{_role_evidence(agent_name, state, role)}

選択可能な行動:
{action_list()}

【DISCUSSION_HISTORY id=history】
これまでの議論:
{format_transcript_text(transcript)}

これは第 {opportunity_index} / {opportunity_count} 回の意思決定機会です。
各エージェントは独立に最終案を一つだけ出してください。
出力には action（A-F）、短い reason、そして合意意思を表す ready（true/false）を含めてください。
全員が同じ action かつ ready=true なら合意成立です。
受諾済みV*が表示されている場合は、正確な v_star_id と v_star_consistent を必ず返してください。
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
    weight_example = {c: 0.2 for c in criteria}
    weight_example[criteria[-1]] = round(1.0 - sum(weight_example[c] for c in criteria[:-1]), 6)
    weights_json = _canonical_json(weight_example)
    return f"""【GAME_RULES_AND_JSON_CONTRACT id=v-proposal-required】
あなたは深海研究施設トラブルの意思決定エージェント {agent_name} です。

自由議論では意見が分かれました。最終投票前に、グループで使う共通基準V*を一つ提案してください。
提案するVは、自分の役割・観測事実に基づいたもので構いません。

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

あなたの役割固有情報:
{_role_evidence(agent_name, state, role)}

選択可能な行動:
{action_list()}

【DISCUSSION_HISTORY id=history】
これまでの議論:
{format_transcript_text(transcript)}

{_v_state_block(v_state)}

必ず次のJSONだけを返してください。説明文やMarkdownは不要です。
{{"v_proposal":{{"proposal_id":"{agent_name}-turn{{state.turn}}-required","ordered_criteria":{criteria_json},"weights":{weights_json},"scope":"turn"}},"action":"A","reason":"短い理由"}}
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
) -> str:
    """v_proposal_required で出た提案に対し、相手エージェントに accept/reject/counter を求める。"""
    criteria = list(DEFAULT_VALUE_CRITERIA_SCHEMA.criteria)
    criteria_json = _canonical_json(criteria)
    return f"""【GAME_RULES_AND_JSON_CONTRACT id=v-proposal-response】
あなたは深海研究施設トラブルの意思決定エージェント {agent_name} です。

相手から V* 提案が出ました。あなたの観測・役割から判断し、accept / reject / counter のいずれかを返してください。
counter の場合は完全な代替案を v_star_response.counter_proposal に入れてください。

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

あなたの役割固有情報:
{_role_evidence(agent_name, state, role)}

選択可能な行動:
{action_list()}

【DISCUSSION_HISTORY id=history】
これまでの議論:
{format_transcript_text(transcript)}

提案内容:
{_canonical_json(proposal)}

{_v_state_block(v_state)}

必ず次のJSONだけを返してください。説明文やMarkdownは不要です。
{{"v_star_response":{{"response":"accept|reject|counter","proposal_id":"{proposal.get('proposal_id', '')}","counter_proposal":{{"proposal_id":"{agent_name}-counter-turn{{state.turn}}","ordered_criteria":{criteria_json},"scope":"turn"}}}}}}
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


def get_action(model, tokenizer, prompt: str, max_new_tokens: int, fallback: Action, enable_thinking: bool = False, thinking_budget: int | None = None) -> tuple[Action, str, str, bool, str, str]:
    """(action, reason, message, ready, raw_response, thinking) を返す。"""
    thinking, raw = run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=enable_thinking, thinking_budget=thinking_budget)
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
) -> dict[str, Any]:
    """自由議論用の発言情報を dict で返す。JSON契約違反は有効発話として扱わない。"""
    thinking, raw = run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=enable_thinking, thinking_budget=thinking_budget)
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
            "invalid_discussion_output": True,
        }
    speech_act, message, action, reason, reply_to_message_id, addressed_to, requires_response = extract_json_discussion(raw)
    if speech_act is None and action is None and not message:
        # JSON 解析ができなかった場合のみフォールバック行動を使用する
        if fallback_action is not None:
            action = fallback_action
            reason = "[fallback_action]"
        speech_act = SpeechAct.INFORMATION_REQUEST
    elif speech_act is None:
        speech_act = SpeechAct.INFORMATION_REQUEST
    if not message:
        message = raw[:160].strip()
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
        "invalid_discussion_output": False,
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
) -> bool:
    """Deterministic minimum check beyond the model's self-report."""
    if action is None or claimed_consistent is not True or referenced_id != v_star_id or not v_star:
        return False
    ordered = v_star.get("ordered_criteria")
    if not isinstance(ordered, list) or not ordered:
        return False
    top = str(ordered[0]).strip().casefold()
    normalized_reason = reason.casefold().replace("_", " ").replace("-", " ")
    candidates = {top, top.replace("_", " "), top.replace("-", " ")}
    return any(candidate and candidate in normalized_reason for candidate in candidates)


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
    enable_v_flow = role_value_mode is not None
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

    while not state.done:
        q_values = estimate_q_values(state, n_rollouts=evaluator_rollouts, seed=seed + state.turn * 1000)
        optimal = best_action(q_values)
        allowed = acceptable_actions(q_values)
        fallback = optimal

        # §6.1: all framework conditions use this exact pre-discussion measurement.
        v_before: dict[str, dict[str, Any] | None] = {"alpha": None, "beta": None}
        action_before: dict[str, Action | None] = {"alpha": None, "beta": None}
        reason_before = {"alpha": "", "beta": ""}
        v_measurement_errors = {"alpha_before": "not_recorded", "beta_before": "not_recorded"}
        measurement_call_count = 0
        measurement_token_count = 0
        measurement_retry_count = 0
        if enable_v_flow:
            for agent in speakers:
                _, raw_measurement = run_prompt(
                    model,
                    tokenizer,
                    v_measurement_prompt(
                        agent,
                        state,
                        phase="before",
                        persona=personas[agent],
                        persona_params=persona_params[agent],
                        role=resolved_role_for(agent),
                    ),
                    max_new_tokens,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                )
                measurement_call_count += 1
                if tokenizer is not None:
                    measurement_token_count += len(tokenizer.encode(raw_measurement, add_special_tokens=False))
                measured_v, measured_action, measured_reason, error = extract_json_v_measurement(raw_measurement)
                v_before[agent] = measured_v
                action_before[agent] = measured_action
                reason_before[agent] = measured_reason
                v_measurement_errors[f"{agent}_before"] = error

        # §6.2.1: v_alignment_required is determined from observed V and actions, not self-report.
        if enable_v_flow:
            turn_v_alignment_required, turn_v_alignment_requirement_reasons = v_alignment_required(
                action_before["alpha"], action_before["beta"], v_before["alpha"], v_before["beta"]
            )
        else:
            turn_v_alignment_required, turn_v_alignment_requirement_reasons = False, []

        # §6.2.2: reserve discussion budget for mandatory V proposal/response in hivc_d.
        reserved_v_messages = 0
        reserved_v_tokens = 0
        if enable_v_flow and condition == "hivc_d" and turn_v_alignment_required:
            reserved_v_messages = 2
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

        # 質問/応答閉包用の状態
        next_message_id = 1
        open_questions: list[dict[str, Any]] = []
        forced_decision_with_open_question = False
        forced_decision_reason = ""
        question_response_latencies: list[int] = []
        question_count = 0
        answered_question_count = 0
        duplicate_question_count = 0
        invalid_discussion_output_count = 0
        consecutive_duplicate_count = 0
        max_consecutive_duplicate_questions_recorded = 0
        last_duplicate_signature: tuple[str, str, str] | None = None
        next_speaker_override: str | None = None

        v_proposals: list[dict[str, Any]] = []
        v_responses: dict[str, list[dict[str, Any]]] = {"alpha": [], "beta": []}
        explicitly_shared_v_before: dict[str, dict[str, Any]] = {}
        inherited_game_v = persistent_v_star is not None
        v_star_status = "accepted" if inherited_game_v else ("unresolved" if enable_v_flow else "not_recorded")
        v_star_id = persistent_v_star_id if inherited_game_v else ""
        v_star: dict[str, Any] | None = persistent_v_star
        v_star_failure_reason = "" if inherited_game_v else ("missing_v_proposal" if enable_v_flow else "not_recorded")
        v_proposal_required_prompt_issued = False
        missing_v_proposal_after_required_prompt = False

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
            v_star = {
                "proposal_id": v_star_id,
                "ordered_criteria": ["avoid_immediate_loss", "advance_win_condition", "preserve_next_turn_options"],
                "scope": "turn",
                "source": "external_prescription",
            }
            v_star_status = "accepted"
            v_star_failure_reason = ""

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
                )

                response = get_discussion_message(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens,
                    fallback_action=fallback,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                )

                raw = response["raw"]
                token_count = 0
                if tokenizer is not None:
                    token_count = len(tokenizer.encode(raw, add_special_tokens=False))

                # JSON契約違反・壊れたJSON断片は有効発話として扱わない
                if response.get("invalid_discussion_output"):
                    invalid_discussion_output_count += 1
                    transcript.append(
                        {
                            "speaker": speaker,
                            "speech_act": None,
                            "message": "",
                            "action": "",
                            "reason": "",
                            "message_id": str(next_message_id),
                            "addressed_to": None,
                            "requires_response": False,
                            "reply_to_message_id": None,
                            "invalid_discussion_output": True,
                            "raw": raw,
                            "thinking": response["thinking"],
                        }
                    )
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
                if is_question and addressed_to != other_speaker:
                    addressed_to = other_speaker
                reply_to_message_id = response["reply_to_message_id"]

                # 質問の重複検出: 同一speakerが同一signatureのopen questionを再送したら、
                # 予算を消費せず宛先エージェントの回答ターンへ切り替える
                if is_question:
                    signature = _question_signature(
                        {
                            "speaker": speaker,
                            "addressed_to": addressed_to,
                            "action": response["action"].value if response["action"] else "",
                            "reason": response["reason"],
                            "message": response["message"],
                        }
                    )
                    if any(
                        _question_signature(q) == signature and q["speaker"] == speaker
                        for q in open_questions
                    ):
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
                                "raw": raw,
                                "thinking": response["thinking"],
                            }
                        )
                        next_message_id += 1
                        next_speaker_override = addressed_to
                        continue
                    consecutive_duplicate_count = 0
                    last_duplicate_signature = None
                else:
                    consecutive_duplicate_count = 0
                    last_duplicate_signature = None

                # 有効な発話にのみ予算を加算
                token_budget_used += token_count
                opportunity_token_used += token_count

                # 回答すべき未回答質問があるのに質問を返した場合は無効
                if is_question and question_to_answer is not None:
                    forced_decision_with_open_question = True
                    forced_decision_reason = (
                        f"invalid_response_while_answer_required: {speaker} had question "
                        f"from {question_to_answer['speaker']} (id={question_to_answer['message_id']}) "
                        "but returned a question"
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
                            "requires_response": False,
                            "reply_to_message_id": None,
                            "invalid_response_while_answer_required": True,
                            "raw": raw,
                            "thinking": response["thinking"],
                        }
                    )
                    next_message_id += 1
                    total_free_messages += 1
                    messages_this_opportunity += 1
                    break

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
                transcript.append(
                    {
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
                )
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
                    proposal = {**proposal, "speaker": speaker, "message_id": this_message_id}
                    v_proposals.append(proposal)
                if v_response is not None:
                    v_responses[speaker].append(v_response)
                    if v_response.get("response") == "counter" and isinstance(v_response.get("counter_proposal"), dict):
                        counter = {**v_response["counter_proposal"], "speaker": speaker, "message_id": this_message_id}
                        v_proposals.append(counter)
                if enable_v_flow and condition != "hivc_d_prescribed_v1" and (v_proposals or any(v_responses.values())):
                    v_star_status, v_star_id, v_star, v_star_failure_reason = resolve_v_star(v_proposals, v_responses)
                    if v_star_status == "accepted" and v_star and v_star.get("scope") == "game":
                        persistent_v_star = v_star
                        persistent_v_star_id = v_star_id
                next_message_id += 1
                total_free_messages += 1
                messages_this_opportunity += 1

                if is_question:
                    question_count += 1
                    open_questions.append(
                        {
                            "message_id": this_message_id,
                            "speaker": speaker,
                            "addressed_to": addressed_to,
                            "message": response["message"],
                            "timestamp": total_free_messages - 1,
                            "reason": response["reason"],
                            "action": response["action"].value if response["action"] else "",
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
                            answered_question_count += 1
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

            # §6.2.2: HIVC-Dで自由議論後も有効なV提案がない場合、v_proposal_requiredプロンプトを発行
            if (
                enable_v_flow
                and condition == "hivc_d"
                and turn_v_alignment_required
                and not v_proposal_required_prompt_issued
                and not v_proposals
            ):
                proposer = priority_agent(seed, state.turn)
                if proposer not in speakers:
                    proposer = speakers[0]
                responder = "beta" if proposer == "alpha" else "alpha"
                propose_prompt = v_proposal_required_prompt(
                    proposer,
                    personas[proposer],
                    persona_params[proposer],
                    state,
                    condition,
                    transcript,
                    v_state=current_v_state(proposer),
                    role=resolved_role_for(proposer),
                )
                _, raw_propose = run_prompt(
                    model, tokenizer, propose_prompt, max_new_tokens,
                    enable_thinking=enable_thinking, thinking_budget=thinking_budget,
                )
                if tokenizer is not None:
                    token_budget_used += len(tokenizer.encode(raw_propose, add_special_tokens=False))
                total_free_messages += 1
                propose_payload = _extract_json_object(raw_propose)
                propose_msg_id = str(next_message_id)
                proposal, _ = parse_v_negotiation(propose_payload, proposer, propose_msg_id)
                transcript.append(
                    {
                        "speaker": proposer,
                        "speech_act": None,
                        "message": "",
                        "action": "",
                        "reason": "",
                        "message_id": propose_msg_id,
                        "addressed_to": None,
                        "requires_response": False,
                        "reply_to_message_id": None,
                        "raw": raw_propose,
                        "thinking": "",
                        "v_proposal": proposal,
                    }
                )
                next_message_id += 1
                if proposal is not None:
                    proposal = {**proposal, "speaker": proposer, "message_id": propose_msg_id}
                    v_proposals.append(proposal)
                    v_protocol_transition_history.append({"from": v_protocol_state, "to": "V_RESPOND", "reason": "v_proposal_required_prompt_accepted"})
                    v_protocol_state = "V_RESPOND"
                    response_prompt = v_proposal_response_prompt(
                        responder,
                        personas[responder],
                        persona_params[responder],
                        state,
                        condition,
                        transcript,
                        proposal,
                        v_state=current_v_state(responder),
                        role=resolved_role_for(responder),
                    )
                    _, raw_response = run_prompt(
                        model, tokenizer, response_prompt, max_new_tokens,
                        enable_thinking=enable_thinking, thinking_budget=thinking_budget,
                    )
                    if tokenizer is not None:
                        token_budget_used += len(tokenizer.encode(raw_response, add_special_tokens=False))
                    total_free_messages += 1
                    response_payload = _extract_json_object(raw_response)
                    response_msg_id = str(next_message_id)
                    _, v_response = parse_v_negotiation(response_payload, responder, response_msg_id)
                    transcript.append(
                        {
                            "speaker": responder,
                            "speech_act": None,
                            "message": "",
                            "action": "",
                            "reason": "",
                            "message_id": response_msg_id,
                            "addressed_to": None,
                            "requires_response": False,
                            "reply_to_message_id": None,
                            "raw": raw_response,
                            "thinking": "",
                            "v_star_response": v_response,
                        }
                    )
                    next_message_id += 1
                    if v_response is not None:
                        v_responses[responder].append(v_response)
                        if (
                            v_response.get("response") == "counter"
                            and isinstance(v_response.get("counter_proposal"), dict)
                        ):
                            counter = {
                                **v_response["counter_proposal"],
                                "speaker": responder,
                                "message_id": response_msg_id,
                            }
                            v_proposals.append(counter)
                    if enable_v_flow and condition != "hivc_d_prescribed_v1":
                        v_star_status, v_star_id, v_star, v_star_failure_reason = resolve_v_star(v_proposals, v_responses)
                        if v_star_status == "accepted" and v_star and v_star.get("scope") == "game":
                            persistent_v_star = v_star
                            persistent_v_star_id = v_star_id
                    v_protocol_transition_history.append({"from": v_protocol_state, "to": "A_CHECK", "reason": "v_star_resolved"})
                    v_protocol_state = "A_CHECK"
                else:
                    missing_v_proposal_after_required_prompt = True
                    v_star_failure_reason = "missing_v_proposal_after_required_prompt"
                    v_protocol_transition_history.append({"from": v_protocol_state, "to": "A_CHECK", "reason": "v_proposal_required_prompt_rejected_or_invalid"})
                    v_protocol_state = "A_CHECK"
                v_proposal_required_prompt_issued = True

            if enable_v_flow and v_protocol_state not in {"A_CHECK", "FINAL_VOTE"}:
                v_protocol_transition_history.append({"from": v_protocol_state, "to": "A_CHECK", "reason": "pre_decision"})
                v_protocol_state = "A_CHECK"

            # 意思決定機会：例外時もエージェントの投票は実行する
            alpha_vote, alpha_vote_reason, alpha_vote_message, alpha_vote_ready, alpha_vote_raw, alpha_vote_thinking = get_action(
                model,
                tokenizer,
                decision_opportunity_prompt(
                    "alpha",
                    personas["alpha"],
                    persona_params["alpha"],
                    state,
                    transcript,
                    condition,
                    opp_idx,
                    opportunity_count,
                    v_state=current_v_state("alpha"),
                    role=resolved_role_for("alpha"),
                ),
                max_new_tokens,
                fallback,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
            )
            if alpha_vote_reason.startswith("invalid_response_fallback"):
                alpha_vote = None
            alpha_vote_v_star_id, alpha_vote_v_star_claim = extract_vote_v_fields(alpha_vote_raw)
            beta_vote, beta_vote_reason, beta_vote_message, beta_vote_ready, beta_vote_raw, beta_vote_thinking = get_action(
                model,
                tokenizer,
                decision_opportunity_prompt(
                    "beta",
                    personas["beta"],
                    persona_params["beta"],
                    state,
                    transcript,
                    condition,
                    opp_idx,
                    opportunity_count,
                    v_state=current_v_state("beta"),
                    role=resolved_role_for("beta"),
                ),
                max_new_tokens,
                fallback,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
            )
            if beta_vote_reason.startswith("invalid_response_fallback"):
                beta_vote = None
            beta_vote_v_star_id, beta_vote_v_star_claim = extract_vote_v_fields(beta_vote_raw)

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
                    group_action = fallback
                    group_reason = "both votes invalid; fallback to best action"
                    decision_rule = "fallback_best"
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
                    group_action = fallback
                    group_reason = "both votes invalid; fallback to best action"
                    decision_rule = "fallback_best"

        if enable_v_flow and v_protocol_state != "FINAL_VOTE":
            v_protocol_transition_history.append({"from": v_protocol_state, "to": "FINAL_VOTE", "reason": "final_votes_collected"})
            v_protocol_state = "FINAL_VOTE"

        # §6.5: collect v_after only after final votes/group decision are fixed.
        v_after: dict[str, dict[str, Any] | None] = {"alpha": None, "beta": None}
        if enable_v_flow:
            for agent in speakers:
                _, raw_measurement = run_prompt(
                    model,
                    tokenizer,
                    v_measurement_prompt(
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
                    ),
                    max_new_tokens,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                )
                measurement_call_count += 1
                if tokenizer is not None:
                    measurement_token_count += len(tokenizer.encode(raw_measurement, add_special_tokens=False))
                measured_v, _, _, error = extract_json_v_measurement(raw_measurement)
                v_after[agent] = measured_v
                v_measurement_errors[f"{agent}_after"] = error

        alpha_v_star_consistent = v_star_status == "accepted" and verify_vote_v_star_consistency(
            alpha_vote, alpha_vote_reason, alpha_vote_v_star_id, alpha_vote_v_star_claim, v_star_id, v_star
        )
        beta_v_star_consistent = v_star_status == "accepted" and verify_vote_v_star_consistency(
            beta_vote, beta_vote_reason, beta_vote_v_star_id, beta_vote_v_star_claim, v_star_id, v_star
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
            "v_star_id": v_star_id,
            "v_star": _canonical_json(v_star) if v_star is not None else "",
            "v_star_scope": str(v_star.get("scope", "")) if v_star is not None else "",
            "v_star_status": v_star_status,
            "v_star_failure_reason": v_star_failure_reason,
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
            "fallback_used": fallback_used,
            "fallback_priority_agent": fallback_priority_agent,
            "planned_route": planned_route,
            "optimal_route": optimal_route_value,
            "route_switch": route_switch,
            "premature": result.premature,
            "role_specific_evidence": json.dumps(role_evidence, ensure_ascii=False),
            "alpha_evidence": role_evidence["alpha"],
            "beta_evidence": role_evidence["beta"],
            "unanswered_question_count": unanswered_question_count,
            "question_count": question_count,
            "answered_question_count": answered_question_count,
            "duplicate_question_count": duplicate_question_count,
            "max_consecutive_duplicate_questions": max_consecutive_duplicate_questions_recorded,
            "invalid_discussion_output_count": invalid_discussion_output_count,
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
