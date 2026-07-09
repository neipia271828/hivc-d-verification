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
"""
from __future__ import annotations

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
    """ターン行に conflict_level と group_reason（欠損時）を補う。"""
    individual = _collect_individual_votes(row)
    if individual:
        row.setdefault("conflict_level", conflict_level(individual))
    else:
        row.setdefault("conflict_level", float("nan"))
    row.setdefault("group_reason", row.get("decision_rule", ""))
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

    rows は turn 昇順を前提とする。イベントは current_event / event 列。
    regret は "regret" 列（数値）。
    """
    ordered = sorted(rows, key=lambda r: _safe_float(r.get("turn"), 0.0))
    prev_regret: float | None = None
    eligible = 0
    not_worse = 0
    improved = 0
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
    # seed ごとの最終ターン行を終端とみなす
    by_seed: dict[object, dict] = {}
    for row in rows:
        seed = row.get("seed", row.get("game_id", 0))
        turn = _safe_float(row.get("turn"), 0.0)
        prev = by_seed.get(seed)
        if prev is None or turn >= _safe_float(prev.get("turn"), 0.0):
            by_seed[seed] = row
    terminal = list(by_seed.values())
    win = float(np.mean([str(r.get("outcome", "")).lower() == "win" for r in terminal]))
    survival = float(
        np.mean([not str(r.get("outcome", "")).lower().startswith("loss_") for r in terminal])
    )
    ret = float(np.mean([_safe_float(r.get("terminal_score")) for r in terminal]))
    return {"win_rate": win, "survival_rate": survival, "mean_return": ret}


def compute_summary_metrics(rows: list[dict], threshold: float = CONFLICT_THRESHOLD) -> dict[str, float]:
    """REQUIREMENTS §6 の主要評価指標を全て計算して返す。"""
    enriched = [enrich_turn_row(dict(row)) for row in rows]
    summary = terminal_metrics(enriched)
    summary["mean_regret"] = mean_regret(enriched)
    summary["expert_match_rate"] = expert_match_rate(enriched)
    summary["minority_adoption_rate"] = minority_adoption_rate(enriched)
    summary["conflict_resolution_quality"] = conflict_resolution_quality(enriched, threshold)
    summary.update(plan_revision_quality(enriched))
    return summary
