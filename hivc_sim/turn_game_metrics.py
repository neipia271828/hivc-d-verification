"""REQUIREMENTS §6 評価指標とターン別記録項目。

本モジュールは LLM 実験ログの 1 ゲーム分（またはバッチ）のターン行リストから、
REQUIREMENTS §6 が定義する主要評価指標を計算する。torch に依存しない純粋 Python
実装であり、policy CLI の行（個人行動を持たない）と LLM 実験の行（alpha/beta 個人
票を持つ）の両方に対応する。

個人行動を持たない行（policy ロールアウト）では、minority_adoption_rate /
conflict_resolution_quality など個人票を必要とする指標は NaN となる。

conflict_level の定義:
    グループ内で個人選択がどれだけ分裂しているか。1 - max_share で定義し、
    2 エージェント同一行動なら 0.0、完全に割れれば 1.0 に近づく。

minority_adoption_rate の定義:
    少数派の個人案が基準行動（acceptable_actions）に含まれていた場合に、それが
    グループ行動として採用された割合。少数派が存在しないターンは除外。

plan_revision_quality の定義:
    イベント発生ターンにおいて、前ターンからの regret 変化が非正（改善または
    悪化なし）だった割合。regret が明示的に低下した割合も副指標として出す。

conflict_resolution_quality の定義:
    conflict_level が閾値（既定 0.5、2 体では不一致=1.0）以上のターンの regret
    平均。低いほど「対立が解決されて良い行動に落ちた」ことを示す。

agreement_rate_by_opportunity の定義:
    decision_history 中で全員合意に至った意思決定機会の割合。

fallback_rate の定義:
    fallback_used が true となるターンの割合。

discussion_diversity の定義:
    自由議論中の発言行為（speech_act）の種類数。
"""
from __future__ import annotations

import json
import math
import re
from typing import Iterable

import numpy as np

from profiles import DEFAULT_VALUE_CRITERIA_SCHEMA

CONFLICT_THRESHOLD = 0.5


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_action_set(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (set, list, tuple)):
        return {str(v).strip().upper() for v in value if str(v).strip()}
    return {str(v).strip().upper() for v in str(value).split(",") if v.strip()}


def _parse_json(value, default=None):
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def normalized_l1_distance(left: object, right: object) -> float:
    """Return a comparable V distance for qualitative levels or legacy weights.

    Both mappings must contain the same criteria.  Missing/invalid vectors are
    not measurement opportunities and therefore return NaN rather than zero.
    """
    left = _parse_json(left, left)
    right = _parse_json(right, right)
    if isinstance(left, dict) and isinstance(right, dict):
        left_levels = left.get("priority_levels")
        right_levels = right.get("priority_levels")
        if isinstance(left_levels, dict) and isinstance(right_levels, dict):
            if set(left_levels) != set(right_levels) or not left_levels:
                return float("nan")
            allowed = {"high", "mid", "low"}
            if any(str(value).lower() not in allowed for value in [*left_levels.values(), *right_levels.values()]):
                return float("nan")
            return sum(
                str(left_levels[key]).lower() != str(right_levels[key]).lower()
                for key in left_levels
            ) / len(left_levels)
    if isinstance(left, dict) and isinstance(left.get("weights"), dict):
        left = left["weights"]
    if isinstance(right, dict) and isinstance(right.get("weights"), dict):
        right = right["weights"]
    if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right) or not left:
        return float("nan")

    def normalize(values: dict) -> dict[str, float] | None:
        parsed: dict[str, float] = {}
        for key, value in values.items():
            number = _safe_float(value)
            if np.isnan(number) or number < 0:
                return None
            parsed[str(key)] = number
        total = sum(parsed.values())
        if total <= 0:
            return None
        return {key: value / total for key, value in parsed.items()}

    normalized_left = normalize(left)
    normalized_right = normalize(right)
    if normalized_left is None or normalized_right is None:
        return float("nan")
    return float(sum(abs(normalized_left[key] - normalized_right[key]) for key in normalized_left))


def _rate_with_counts(numerator: int, denominator: int) -> tuple[float, int, int]:
    rate = numerator / denominator if denominator else float("nan")
    return rate, numerator, denominator


def _v_proposal_count_and_ids(row: dict) -> tuple[int, set[str]]:
    """Return proposal count and IDs from a turn row.

    The current contract stores a list of proposals.  A single proposal object
    is accepted for transitional CSV compatibility.  Malformed entries still
    count as attempted proposals in the denominator, but cannot become the
    accepted numerator without a matching ID.
    """
    proposals = _parse_json(row.get("v_proposals"), row.get("v_proposals"))
    if isinstance(proposals, dict):
        proposals = [proposals] if proposals else []
    if not isinstance(proposals, list):
        return (0, set())
    ids: set[str] = set()
    count = 0
    for proposal in proposals:
        count += 1
        if isinstance(proposal, dict):
            proposal_id = str(proposal.get("proposal_id", proposal.get("id", ""))).strip()
        else:
            proposal_id = str(proposal).strip()
        if proposal_id:
            ids.add(proposal_id)
    return count, ids


def _has_v_proposal(row: dict) -> bool:
    count, _ = _v_proposal_count_and_ids(row)
    return count > 0


def _recorded_bool(value: object) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _v_alignment_required_for_row(row: dict) -> bool:
    """v_alignment_required 列があればそれを使い、なければ前ターンV距離・行動から推定する。"""
    required = row.get("v_alignment_required")
    if required is True or (isinstance(required, str) and required.strip().lower() == "true"):
        return True
    if required is False or (isinstance(required, str) and required.strip().lower() == "false"):
        return False
    before = normalized_l1_distance(row.get("alpha_v_before"), row.get("beta_v_before"))
    if not np.isnan(before) and before > 1e-12:
        return True
    alpha_action = str(row.get("alpha_action_before", "")).strip().upper()
    beta_action = str(row.get("beta_action_before", "")).strip().upper()
    if alpha_action and beta_action and alpha_action != beta_action:
        return True
    return False


def _v_schema_complete(row: dict) -> bool | None:
    """共通Vオントロジーに完全に従った V before/after レコードがあるかを判定する。

    検査対象:
    - alpha_v_before, beta_v_before
    - alpha_v_after, beta_v_after (存在する場合)
    - 新形式では priority_levels が全criteriaを持ち high/mid/low のいずれか
    - legacy形式では weights が有限・非負で合計1.0
    - ordered_criteria が期待集合に完全一致
    """
    errors = _parse_json(row.get("v_measurement_errors"), row.get("v_measurement_errors"))
    if isinstance(errors, dict) and any(
        isinstance(v, str) and v not in ("", "not_recorded") for v in errors.values()
    ):
        return False
    expected = set(DEFAULT_VALUE_CRITERIA_SCHEMA.criteria)

    def valid(v: object) -> bool:
        if not isinstance(v, dict):
            return False
        ordered = v.get("ordered_criteria")
        if not isinstance(ordered, (list, tuple)):
            return False
        if set(ordered) != expected or len(ordered) != len(expected):
            return False
        levels = v.get("priority_levels")
        if isinstance(levels, dict):
            return set(levels) == expected and all(
                isinstance(level, str) and level.lower() in {"high", "mid", "low"}
                for level in levels.values()
            )
        weights = v.get("weights")
        if not isinstance(weights, dict) or set(weights.keys()) != expected:
            return False
        # 全 weight 値が有限かつ非負
        for w in weights.values():
            if not isinstance(w, (int, float)) or not math.isfinite(float(w)) or float(w) < 0:
                return False
        # weight 合計が 1.0 に一致する（浮動小数点許容誤差内）
        total = sum(float(w) for w in weights.values())
        if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
            return False
        return True

    alpha_before = _parse_json(row.get("alpha_v_before"), row.get("alpha_v_before"))
    beta_before = _parse_json(row.get("beta_v_before"), row.get("beta_v_before"))
    alpha_after_raw = row.get("alpha_v_after")
    beta_after_raw = row.get("beta_v_after")
    alpha_after = _parse_json(alpha_after_raw, alpha_after_raw) if alpha_after_raw not in (None, "", "not_recorded") else None
    beta_after = _parse_json(beta_after_raw, beta_after_raw) if beta_after_raw not in (None, "", "not_recorded") else None

    # before は必須: 両方とも有効でなければ schema 不完全
    alpha_before_valid = valid(alpha_before)
    beta_before_valid = valid(beta_before)
    if not (alpha_before_valid and beta_before_valid):
        if alpha_before is not None or beta_before is not None:
            return False
        return None

    # after が存在する場合は同様に検査する
    for after_val in (alpha_after, beta_after):
        if after_val is not None and not valid(after_val):
            return False

    return True


def v_process_metrics(rows: list[dict]) -> dict[str, float | int]:
    """Compute §10.1 V-process metrics, including every rate denominator."""
    required_turns = 0
    proposal_on_required = 0
    proposal_count = 0
    accepted_proposals = 0
    unresolved_required = 0
    vote_revisions = 0
    vote_observations = 0
    consistent_actions = 0
    consistency_observations = 0
    distances_before: list[float] = []
    distances_after: list[float] = []
    gains: list[float] = []
    prompted_required = 0
    missing_after_prompt = 0
    schema_complete = 0
    schema_attempted = 0

    for row in rows:
        before = _safe_float(row.get("v_alignment_distance_before"))
        if np.isnan(before):
            before = normalized_l1_distance(row.get("alpha_v_before"), row.get("beta_v_before"))
        after = _safe_float(row.get("v_alignment_distance_after"))
        if np.isnan(after):
            after = normalized_l1_distance(row.get("alpha_v_after"), row.get("beta_v_after"))
        if not np.isnan(before):
            distances_before.append(before)
        if not np.isnan(after):
            distances_after.append(after)
        if not np.isnan(before) and not np.isnan(after):
            gains.append(before - after)

        row_proposal_count, proposal_ids = _v_proposal_count_and_ids(row)
        proposal = row_proposal_count > 0
        status = str(row.get("v_star_status", "")).strip().lower()
        v_star_id = str(row.get("v_star_id", "")).strip()
        required = _v_alignment_required_for_row(row)
        if required:
            required_turns += 1
            proposal_on_required += int(proposal)
            # Missing/malformed completion is not silently treated as alignment.
            unresolved_required += int(status != "accepted")
        if proposal:
            proposal_count += row_proposal_count
            accepted_proposals += int(
                status == "accepted" and bool(v_star_id) and v_star_id in proposal_ids
            )

        for agent in ("alpha", "beta"):
            changed = _recorded_bool(row.get(f"{agent}_vote_changed"))
            if changed is None:
                action_before = str(row.get(f"{agent}_action_before", "")).strip().upper()
                vote = str(row.get(f"{agent}_vote", "")).strip().upper()
                if action_before and vote:
                    changed = action_before != vote
            if changed is not None:
                vote_observations += 1
                vote_revisions += int(changed)

        row_consistency = _recorded_bool(row.get("v_star_action_consistency"))
        if status == "accepted" and row_consistency is not None:
            consistency_observations += 1
            consistent_actions += int(row_consistency)
        elif status == "accepted":
            # Older transitional rows may only have per-agent post-processed checks.
            agent_checks = [
                _recorded_bool(row.get(f"{agent}_v_star_consistent"))
                for agent in ("alpha", "beta")
            ]
            recorded = [value for value in agent_checks if value is not None]
            if recorded:
                consistency_observations += 1
                consistent_actions += int(len(recorded) == 2 and all(recorded))

        prompted = _recorded_bool(row.get("v_proposal_required_prompt_issued"))
        if prompted is True:
            prompted_required += 1
            missing = _recorded_bool(row.get("missing_v_proposal_after_required_prompt"))
            if missing is True:
                missing_after_prompt += 1

        schema_status = _v_schema_complete(row)
        if schema_status is not None:
            schema_attempted += 1
            schema_complete += int(schema_status)

    result: dict[str, float | int] = {}
    for name, numerator, denominator in (
        ("v_proposal_rate", proposal_on_required, required_turns),
        ("v_star_acceptance_rate", accepted_proposals, proposal_count),
        ("vote_revision_rate", vote_revisions, vote_observations),
        ("v_star_action_consistency", consistent_actions, consistency_observations),
        ("unresolved_v_rate", unresolved_required, required_turns),
        ("missing_v_proposal_after_required_prompt_rate", missing_after_prompt, prompted_required),
        ("v_schema_completeness_rate", schema_complete, schema_attempted),
    ):
        rate, numerator, denominator = _rate_with_counts(numerator, denominator)
        result[name] = rate
        result[f"{name}_numerator"] = numerator
        result[f"{name}_denominator"] = denominator
    result["v_alignment_distance_before"] = (
        float(np.mean(distances_before)) if distances_before else float("nan")
    )
    result["v_alignment_distance_before_n"] = len(distances_before)
    result["v_alignment_distance_after"] = (
        float(np.mean(distances_after)) if distances_after else float("nan")
    )
    result["v_alignment_distance_after_n"] = len(distances_after)
    result["v_alignment_gain"] = float(np.mean(gains)) if gains else float("nan")
    result["v_alignment_gain_n"] = len(gains)
    return result


def _game_key(row: dict) -> tuple[object, object]:
    """ゲーム識別子を (condition, seed_or_game_id) のタプルで返す。

    `condition` が存在し、かつ None でなければキーに含める。
    `seed` があればそれを優先し、なければ `game_id`、どちらもなければ
    後方互換の既定値を返す。
    """
    condition = row.get("condition")
    if condition is not None:
        if "seed" in row and row["seed"] is not None:
            return (condition, row["seed"])
        if "game_id" in row and row["game_id"] is not None:
            return (condition, row["game_id"])
        return (condition, "default")
    if "seed" in row and row["seed"] is not None:
        return (None, row["seed"])
    if "game_id" in row and row["game_id"] is not None:
        return (None, row["game_id"])
    return (None, "default")


def conflict_level(individual_actions: Iterable[str]) -> float:
    """個人選択の分裂度。1 - max_share。"""
    actions = [str(a).strip().upper() for a in individual_actions if str(a).strip()]
    if not actions:
        return float("nan")
    counts: dict[str, int] = {}
    for a in actions:
        counts[a] = counts.get(a, 0) + 1
    max_share = max(counts.values()) / len(actions)
    return 1.0 - max_share


def enrich_turn_row(row: dict) -> dict:
    """ターン行に conflict_level、group_reason、cross_role_evidence_used を補う。"""
    individual = _collect_individual_votes(row)
    if individual:
        row.setdefault("conflict_level", conflict_level(individual))
    else:
        row.setdefault("conflict_level", float("nan"))
    row.setdefault("group_reason", row.get("decision_rule", ""))
    row.setdefault("cross_role_evidence_used", _cross_role_evidence_used(row))
    return row


def expert_match_rate(rows: list[dict]) -> float:
    """group_action が acceptable_actions に含まれるターン割合。"""
    if not rows:
        return float("nan")
    matches = 0
    for row in rows:
        group = str(row.get("group_action", row.get("action", ""))).strip().upper()
        if group and group in _parse_action_set(row.get("acceptable_actions")):
            matches += 1
    return matches / len(rows)


def _collect_individual_votes(row: dict) -> list[str]:
    """個人票を集める。individual_actions があればそれを優先、なければ alpha/beta 票を使う。"""
    individual = row.get("individual_actions")
    if isinstance(individual, str) and individual.strip():
        return [v.strip().upper() for v in individual.split(",") if v.strip()]
    votes: list[str] = []
    for key in ("alpha_vote", "beta_vote"):
        value = str(row.get(key, "")).strip().upper()
        if value:
            votes.append(value)
    return votes


def minority_adoption_rate(rows: list[dict]) -> float:
    """少数派個人案が基準行動だった場合に採用された割合。

    個人票の多数派（plurality）を majority とし、それ以外を少数派案とする。
    同数の場合は group_action 側を majority とみなす（採用された側が勝者）。
    2 体で同数かつ group_action がいずれかと一致する場合、少数派は存在せず
    該当ターンは除外される（2 体では実質的に NaN になりうる）。
    """
    eligible = 0
    adopted = 0
    for row in rows:
        votes = _collect_individual_votes(row)
        if not votes:
            continue
        tally: dict[str, int] = {}
        for v in votes:
            tally[v] = tally.get(v, 0) + 1
        if len(tally) <= 1:
            continue  # 全員一致、少数派なし
        group = str(row.get("group_action", "")).strip().upper()
        max_count = max(tally.values())
        # 同数タイブレーク: group_action を majority に優先する
        majority = group if tally.get(group, 0) == max_count else max(tally, key=tally.get)
        minority_actions = {v for v in tally if v != majority}
        acceptable = _parse_action_set(row.get("acceptable_actions"))
        for minority in minority_actions:
            if minority in acceptable:
                eligible += 1
                if group == minority:
                    adopted += 1
    if eligible == 0:
        return float("nan")
    return adopted / eligible


def plan_revision_quality(rows: list[dict]) -> dict[str, float]:
    """イベント発生ターンで regret が改善した割合。

    rows はゲーム識別子ごとに turn 昇順で処理する。イベントは current_event / event 列。
    regret は "regret" 列（数値）。
    """
    groups: dict[tuple[object, object], list[dict]] = {}
    for row in rows:
        key = _game_key(row)
        groups.setdefault(key, []).append(row)
    eligible = 0
    not_worse = 0
    improved = 0
    for group in groups.values():
        ordered = sorted(group, key=lambda r: _safe_float(r.get("turn"), 0.0))
        prev_regret: float | None = None
        for row in ordered:
            event = str(row.get("event", row.get("current_event", "none")))
            regret = _safe_float(row.get("regret"))
            if event and event.lower() not in ("none", "", "nan"):
                if prev_regret is not None and not np.isnan(regret) and not np.isnan(prev_regret):
                    eligible += 1
                    delta = regret - prev_regret
                    if delta <= 0:
                        not_worse += 1
                    if delta < 0:
                        improved += 1
            if not np.isnan(regret):
                prev_regret = regret
    if eligible == 0:
        return {"plan_revision_quality": float("nan"), "plan_revision_improved_rate": float("nan")}
    return {
        "plan_revision_quality": not_worse / eligible,
        "plan_revision_improved_rate": improved / eligible,
    }


def conflict_resolution_quality(rows: list[dict], threshold: float = CONFLICT_THRESHOLD) -> float:
    """高 conflict ターンの regret 平均。"""
    regrets = []
    for row in rows:
        cl = _safe_float(row.get("conflict_level"))
        regret = _safe_float(row.get("regret"))
        if not np.isnan(cl) and cl >= threshold and not np.isnan(regret):
            regrets.append(regret)
    if not regrets:
        return float("nan")
    return float(np.mean(regrets))


def mean_regret(rows: list[dict]) -> float:
    regrets = [_safe_float(r.get("regret")) for r in rows]
    regrets = [r for r in regrets if not np.isnan(r)]
    if not regrets:
        return float("nan")
    return float(np.mean(regrets))


def terminal_metrics(rows: list[dict]) -> dict[str, float]:
    """ゲーム終端行から win/survival/return を集計。"""
    if not rows:
        return {"win_rate": float("nan"), "survival_rate": float("nan"), "mean_return": float("nan")}
    # ゲーム識別子ごとの最終ターン行を終端とみなす
    by_game: dict[tuple[object, object], dict] = {}
    for row in rows:
        key = _game_key(row)
        turn = _safe_float(row.get("turn"), 0.0)
        prev = by_game.get(key)
        if prev is None or turn >= _safe_float(prev.get("turn"), 0.0):
            by_game[key] = row
    terminal = list(by_game.values())
    win = float(np.mean([str(r.get("outcome", "")).lower() == "win" for r in terminal]))
    survival = float(
        np.mean([not str(r.get("outcome", "")).lower().startswith("loss_") for r in terminal])
    )
    ret = float(np.mean([_safe_float(r.get("terminal_score")) for r in terminal]))
    return {"win_rate": win, "survival_rate": survival, "mean_return": ret}


def agreement_rate_by_opportunity(rows: list[dict]) -> float:
    """decision_history 中で consensus=true となった意思決定機会の割合。"""
    total = 0
    agreed = 0
    for row in rows:
        history = _parse_json(row.get("decision_history"))
        if not isinstance(history, list) or not history:
            continue
        for attempt in history:
            if not isinstance(attempt, dict):
                continue
            total += 1
            if _to_bool(attempt.get("consensus")):
                agreed += 1
    if total == 0:
        return float("nan")
    return agreed / total


def fallback_rate(rows: list[dict]) -> float:
    """fallback_used が true となるターンの割合。"""
    total = 0
    fallback = 0
    for row in rows:
        value = row.get("fallback_used")
        if value is None or value == "":
            continue
        total += 1
        if _to_bool(value):
            fallback += 1
    if total == 0:
        return float("nan")
    return fallback / total


def discussion_diversity(rows: list[dict]) -> float:
    """自由議論中に使用された speech_act の種類数（最大5）。"""
    acts: set[str] = set()
    for row in rows:
        transcript = _parse_json(row.get("discussion_transcript"))
        if not isinstance(transcript, list):
            continue
        for item in transcript:
            if not isinstance(item, dict):
                continue
            if item.get("phase", "free") != "free":
                continue
            speech_act = item.get("speech_act")
            if speech_act:
                acts.add(str(speech_act).strip().lower())
    if not acts:
        return float("nan")
    return float(len(acts))


def route_choice_accuracy(rows: list[dict]) -> float:
    """planned_route が optimal_route と一致したターンの割合。"""
    total = 0
    correct = 0
    for row in rows:
        planned = str(row.get("planned_route", "")).strip().lower()
        optimal = str(row.get("optimal_route", "")).strip().lower()
        if planned in ("comms", "escape") and optimal in ("comms", "escape"):
            total += 1
            if planned == optimal:
                correct += 1
    if total == 0:
        return float("nan")
    return correct / total


def route_switch_quality(rows: list[dict]) -> float:
    """イベント発生ターンで、最適経路が変化した場合に適切に切り替えられた割合。

    rows はゲーム識別子ごとに turn 昇順で処理する。
    """
    groups: dict[tuple[object, object], list[dict]] = {}
    for row in rows:
        key = _game_key(row)
        groups.setdefault(key, []).append(row)
    eligible = 0
    good = 0
    for group in groups.values():
        ordered = sorted(group, key=lambda r: _safe_float(r.get("turn"), 0.0))
        prev_planned: str | None = None
        for row in ordered:
            event = str(row.get("event", row.get("current_event", "none")))
            planned = str(row.get("planned_route", "")).strip().lower()
            optimal = str(row.get("optimal_route", "")).strip().lower()
            if event and event.lower() not in ("none", "", "nan"):
                # 最適経路が前ターンの計画と異なるイベントターンを評価対象とする
                if (
                    prev_planned is not None
                    and optimal in ("comms", "escape")
                    and prev_planned != optimal
                ):
                    eligible += 1
                    if planned == optimal:
                        good += 1
            if planned in ("comms", "escape"):
                prev_planned = planned
    if eligible == 0:
        return float("nan")
    return good / eligible


def premature_launch_rate(rows: list[dict]) -> float:
    """行動 F を選んだターンのうち、発進条件未達だった割合。"""
    total = 0
    premature = 0
    for row in rows:
        action = str(row.get("group_action", row.get("action", ""))).strip().upper()
        if action == "F":
            total += 1
            if _to_bool(row.get("premature", False)):
                premature += 1
    if total == 0:
        return float("nan")
    return premature / total


def rescue_wait_failure_rate(rows: list[dict]) -> float:
    """通信救助要請後（rescue_eta 非 None）に敗北したゲームの割合。"""
    by_game: dict[tuple[object, object], dict] = {}
    for row in rows:
        key = _game_key(row)
        turn = _safe_float(row.get("turn"), 0.0)
        prev = by_game.get(key)
        if prev is None or turn >= _safe_float(prev.get("turn"), 0.0):
            by_game[key] = row
    rescue_sent = 0
    failed = 0
    for row in by_game.values():
        state_before = _parse_json(row.get("state_before"))
        state_after = _parse_json(row.get("state_after"))
        rescue_eta = None
        if isinstance(state_before, dict):
            rescue_eta = state_before.get("rescue_eta")
        if rescue_eta is None and isinstance(state_after, dict):
            rescue_eta = state_after.get("rescue_eta")
        if rescue_eta is not None:
            rescue_sent += 1
            outcome = str(row.get("outcome", "")).lower()
            if outcome.startswith("loss_"):
                failed += 1
    if rescue_sent == 0:
        return float("nan")
    return failed / rescue_sent


def _extract_terms(text: str) -> set[str]:
    """日本語2文字以上、英数字3文字以上のトークンを抽出する。"""
    lowered = text.lower()
    # 日本語（漢字・ひらがな・カタカナ）2文字以上
    ja = re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{2,}", lowered)
    # 英数字3文字以上
    en = re.findall(r"[a-z0-9_]{3,}", lowered)
    return set(ja) | set(en)


def _cross_role_evidence_used(row: dict) -> bool:
    """発言者ごとに相手役割の診断情報（自分の evidence には含まれない語）を参照したか。"""
    evidence = _parse_json(row.get("role_specific_evidence"))
    if not isinstance(evidence, dict):
        return False

    alpha_text = str(evidence.get("alpha", "")).lower()
    beta_text = str(evidence.get("beta", "")).lower()
    alpha_terms = _extract_terms(alpha_text)
    beta_terms = _extract_terms(beta_text)

    # alpha には beta 用語のうち alpha 用語に含まれないものが新規情報
    alpha_new_info = beta_terms - alpha_terms
    beta_new_info = alpha_terms - beta_terms

    speaker_texts = []
    transcript = _parse_json(row.get("discussion_transcript"))
    if isinstance(transcript, list):
        for item in transcript:
            if not isinstance(item, dict):
                continue
            speaker = str(item.get("speaker", "")).strip().lower()
            parts = []
            for key in ("message", "reason", "thinking"):
                value = item.get(key)
                if isinstance(value, str):
                    parts.append(value)
            if speaker and parts:
                speaker_texts.append((speaker, " ".join(parts).lower()))

    for key in ("alpha_vote_reason", "beta_vote_reason"):
        value = row.get(key)
        if isinstance(value, str):
            speaker = "alpha" if key == "alpha_vote_reason" else "beta"
            speaker_texts.append((speaker, value.lower()))

    for speaker, text in speaker_texts:
        if speaker == "alpha":
            terms = alpha_new_info
        elif speaker == "beta":
            terms = beta_new_info
        else:
            continue
        if any(term in text for term in terms):
            return True
    return False


def cross_role_evidence_use(rows: list[dict]) -> float:
    """group_reason または個人理由・発言に相手役割の診断情報が反映されたターン割合。"""
    total = 0
    used = 0
    for row in rows:
        if row.get("role_specific_evidence") is None:
            continue
        evidence = _parse_json(row.get("role_specific_evidence"))
        if not isinstance(evidence, dict):
            continue
        total += 1
        if _cross_role_evidence_used(row):
            used += 1
    if total == 0:
        return float("nan")
    return used / total


def unanswered_question_rate(rows: list[dict]) -> float:
    """意思決定開始時点で未回答の質問が残っていたターンの割合。"""
    total = 0
    unanswered = 0
    for row in rows:
        if "unanswered_question_count" not in row:
            continue
        total += 1
        if _safe_float(row.get("unanswered_question_count"), 0.0) > 0:
            unanswered += 1
    if total == 0:
        return float("nan")
    return unanswered / total


def question_response_latency_metric(rows: list[dict]) -> float:
    """質問から宛先エージェントの回答までに要した発言数の平均。"""
    latencies = []
    for row in rows:
        latency = _safe_float(row.get("question_response_latency"))
        if not np.isnan(latency):
            latencies.append(latency)
    if not latencies:
        return float("nan")
    return float(np.mean(latencies))


def forced_decision_with_open_question_rate(rows: list[dict]) -> float:
    """未回答質問を残したまま上限到達により意思決定へ進んだターンの割合。"""
    total = 0
    forced = 0
    for row in rows:
        if "forced_decision_with_open_question" not in row:
            continue
        total += 1
        if _to_bool(row.get("forced_decision_with_open_question")):
            forced += 1
    if total == 0:
        return float("nan")
    return forced / total


def question_answer_rate(rows: list[dict]) -> float:
    """発行された質問に対する回答率。"""
    total = 0
    answered = 0
    for row in rows:
        q = _safe_float(row.get("question_count"), 0.0)
        if q > 0:
            total += int(q)
            answered += _safe_float(row.get("answered_question_count"), 0.0)
    return answered / total if total else float("nan")


def duplicate_question_rate(rows: list[dict]) -> float:
    """発行された質問のうち重複と判定された割合。"""
    total = 0
    dup = 0
    for row in rows:
        q = _safe_float(row.get("question_count"), 0.0)
        if q > 0:
            total += int(q)
            dup += _safe_float(row.get("duplicate_question_count"), 0.0)
    return dup / total if total else float("nan")


def max_consecutive_duplicate_questions_metric(rows: list[dict]) -> int | float:
    """1ゲーム内で記録された最大連続重複質問数。"""
    if not rows:
        return 0
    return int(max(_safe_float(row.get("max_consecutive_duplicate_questions", 0.0), 0.0) for row in rows))


def invalid_discussion_output_rate(rows: list[dict]) -> float:
    """論理発話のうち、全retry後も修復できなかった割合（既存定義）。"""
    total = 0
    invalid = 0
    for row in rows:
        messages = _safe_float(row.get("discussion_turns", 0.0), 0.0)
        if messages > 0:
            total += int(messages)
            invalid += _safe_float(row.get("invalid_discussion_output_count"), 0.0)
    return invalid / total if total else float("nan")


def discussion_attempt_metrics(rows: list[dict]) -> dict[str, float | int]:
    """JSON契約違反をattempt単位で集計し、修復成否を論理発話単位で集計する。"""
    invalid_attempts = 0
    total_attempts = 0
    repaired_outputs = 0
    final_failures = 0
    for row in rows:
        discussion_turns = int(_safe_float(row.get("discussion_turns"), 0.0))
        retries = int(_safe_float(row.get("discussion_retry_count"), 0.0))
        total_attempts += discussion_turns + retries

        recorded_attempts = row.get("invalid_attempt_count")
        if recorded_attempts in (None, ""):
            audit = _parse_json(row.get("invalid_discussion_outputs"), [])
            if isinstance(audit, list):
                recorded_attempts = len(audit)
            else:
                # 旧CSVでは最終失敗だけが記録されていたため、安全な下限として扱う。
                recorded_attempts = row.get("invalid_discussion_output_count", 0)
        invalid_attempts += int(_safe_float(recorded_attempts, 0.0))
        repaired_outputs += int(_safe_float(row.get("repaired_invalid_output_count"), 0.0))
        final_failures += int(_safe_float(row.get("invalid_discussion_output_count"), 0.0))

    repair_opportunities = repaired_outputs + final_failures
    return {
        "invalid_discussion_attempt_rate": (
            invalid_attempts / total_attempts if total_attempts else float("nan")
        ),
        "invalid_discussion_attempt_count": invalid_attempts,
        "discussion_attempt_count": total_attempts,
        "discussion_repair_success_rate": (
            repaired_outputs / repair_opportunities if repair_opportunities else float("nan")
        ),
        "repaired_invalid_output_count": repaired_outputs,
        "discussion_repair_opportunity_count": repair_opportunities,
    }


def silent_unanswered_question_count(rows: list[dict]) -> int:
    """失敗理由なしで意思決定へ持ち越した未回答数の合計。"""
    return int(sum(_safe_float(row.get("silent_unanswered_question_count"), 0.0) for row in rows))


def compute_summary_metrics(rows: list[dict], threshold: float = CONFLICT_THRESHOLD) -> dict[str, float | int]:
    """REQUIREMENTS §6 の主要評価指標を全て計算して返す。"""
    enriched = [enrich_turn_row(dict(row)) for row in rows]
    summary = terminal_metrics(enriched)
    summary["mean_regret"] = mean_regret(enriched)
    summary["expert_match_rate"] = expert_match_rate(enriched)
    summary["minority_adoption_rate"] = minority_adoption_rate(enriched)
    summary["conflict_resolution_quality"] = conflict_resolution_quality(enriched, threshold)
    summary.update(plan_revision_quality(enriched))
    summary["agreement_rate_by_opportunity"] = agreement_rate_by_opportunity(enriched)
    summary["fallback_rate"] = fallback_rate(enriched)
    summary["discussion_diversity"] = discussion_diversity(enriched)
    summary["route_choice_accuracy"] = route_choice_accuracy(enriched)
    summary["route_switch_quality"] = route_switch_quality(enriched)
    summary["premature_launch_rate"] = premature_launch_rate(enriched)
    summary["rescue_wait_failure_rate"] = rescue_wait_failure_rate(enriched)
    summary["cross_role_evidence_use"] = cross_role_evidence_use(enriched)
    summary["unanswered_question_rate"] = unanswered_question_rate(enriched)
    summary["question_response_latency"] = question_response_latency_metric(enriched)
    summary["forced_decision_with_open_question_rate"] = forced_decision_with_open_question_rate(enriched)
    summary["question_answer_rate"] = question_answer_rate(enriched)
    summary["duplicate_question_rate"] = duplicate_question_rate(enriched)
    summary["max_consecutive_duplicate_questions"] = max_consecutive_duplicate_questions_metric(enriched)
    summary["invalid_discussion_output_rate"] = invalid_discussion_output_rate(enriched)
    summary.update(discussion_attempt_metrics(enriched))
    summary["silent_unanswered_question_count"] = silent_unanswered_question_count(enriched)
    summary.update(v_process_metrics(enriched))
    return summary
