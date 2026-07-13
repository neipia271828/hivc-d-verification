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
import re
from typing import Iterable

import numpy as np

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


def compute_summary_metrics(rows: list[dict], threshold: float = CONFLICT_THRESHOLD) -> dict[str, float]:
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
    return summary
