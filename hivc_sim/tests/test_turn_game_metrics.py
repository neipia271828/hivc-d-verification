from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from turn_game_metrics import (  # noqa: E402
    agreement_rate_by_opportunity,
    conflict_level,
    compute_summary_metrics,
    discussion_diversity,
    expert_match_rate,
    fallback_rate,
    minority_adoption_rate,
    plan_revision_quality,
    conflict_resolution_quality,
)


def test_conflict_level_two_agents() -> None:
    # 1 - max_share: 2体同一=0.0, 2体分裂=0.5, 3体全異存=1.0-1/3
    assert conflict_level(["A", "A"]) == 0.0
    assert conflict_level(["A", "B"]) == 0.5
    assert abs(conflict_level(["A", "B", "C"]) - (1.0 - 1 / 3)) < 1e-9


def test_expert_match_rate() -> None:
    rows = [
        {"group_action": "A", "acceptable_actions": "A,B"},
        {"group_action": "C", "acceptable_actions": "A,B"},
    ]
    assert expert_match_rate(rows) == 0.5


def test_minority_adoption_rate_adopted() -> None:
    # 3 agents: votes A,B,B (B majority), group adopts A (minority) which is acceptable
    rows = [{"individual_actions": "A,B,B", "acceptable_actions": "A", "group_action": "A"}]
    assert minority_adoption_rate(rows) == 1.0


def test_minority_adoption_rate_rejected() -> None:
    # 3 agents: A,B,B (B majority), minority A is acceptable but group adopts B
    rows = [{"individual_actions": "A,B,B", "acceptable_actions": "A", "group_action": "B"}]
    assert minority_adoption_rate(rows) == 0.0


def test_minority_adoption_rate_no_minority() -> None:
    rows = [{"alpha_vote": "A", "beta_vote": "A", "acceptable_actions": "A", "group_action": "A"}]
    assert np.isnan(minority_adoption_rate(rows))


def test_minority_adoption_rate_two_agent_tie_is_nan() -> None:
    # 2体同数は少数派不在のため NaN
    rows = [{"alpha_vote": "A", "beta_vote": "B", "acceptable_actions": "A", "group_action": "A"}]
    assert np.isnan(minority_adoption_rate(rows))


def test_plan_revision_quality_improves() -> None:
    rows = [
        {"turn": 0, "event": "none", "regret": 100.0},
        {"turn": 1, "event": "leak_surge", "regret": 50.0},
        {"turn": 2, "event": "none", "regret": 50.0},
        {"turn": 3, "event": "pressure_spike", "regret": 80.0},
    ]
    result = plan_revision_quality(rows)
    # 2 event turns: turn1 (100->50 improved), turn3 (50->80 worse)
    assert result["plan_revision_quality"] == 0.5
    assert result["plan_revision_improved_rate"] == 0.5


def test_conflict_resolution_quality() -> None:
    # conflict_level を直接与える（enrich なしで呼ぶため）
    rows = [
        {"conflict_level": 0.5, "regret": 40.0},  # 2体分裂
        {"conflict_level": 0.0, "regret": 10.0},  # 一致
        {"conflict_level": 0.5, "regret": 60.0},  # 2体分裂
    ]
    # threshold=0.5: conflict>=0.5 の regret 40,60 -> mean 50
    assert conflict_resolution_quality(rows) == 50.0


def test_compute_summary_metrics_full() -> None:
    rows = [
        {
            "seed": 42, "turn": 0, "event": "none",
            "alpha_vote": "A", "beta_vote": "A",
            "group_action": "A", "acceptable_actions": "A,B",
            "regret": 0.0, "outcome": "running", "terminal_score": 500.0,
        },
        {
            "seed": 42, "turn": 1, "event": "leak_surge",
            "alpha_vote": "D", "beta_vote": "C",
            "group_action": "D", "acceptable_actions": "C,D",
            "regret": 5.0, "outcome": "win", "terminal_score": 1200.0,
        },
    ]
    summary = compute_summary_metrics(rows)
    assert summary["win_rate"] == 1.0
    assert summary["expert_match_rate"] == 1.0
    assert summary["mean_regret"] == 2.5
    # minority: turn1 minority is C (beta), acceptable has C,D; group=D so not adopted -> 0.0
    assert summary["minority_adoption_rate"] == 0.0
    assert "plan_revision_quality" in summary
    assert "conflict_resolution_quality" in summary
    assert "agreement_rate_by_opportunity" in summary
    assert "fallback_rate" in summary
    assert "discussion_diversity" in summary


def test_agreement_rate_by_opportunity() -> None:
    rows = [
        {
            "decision_history": [
                {"opportunity_index": 0, "consensus": True},
                {"opportunity_index": 1, "consensus": False},
            ]
        },
        {"decision_history": [{"opportunity_index": 0, "consensus": True}]},
    ]
    # 3 attempts, 2 consensus
    assert agreement_rate_by_opportunity(rows) == 2 / 3


def test_fallback_rate() -> None:
    rows = [
        {"fallback_used": "true"},
        {"fallback_used": "false"},
        {"fallback_used": "false"},
    ]
    assert fallback_rate(rows) == 1 / 3


def test_discussion_diversity() -> None:
    rows = [
        {
            "discussion_transcript": [
                {"phase": "free", "speech_act": "evidence"},
                {"phase": "free", "speech_act": "question_objection"},
                {"phase": "decision", "action": "A"},
            ]
        },
        {
            "discussion_transcript": [
                {"phase": "free", "speech_act": "evidence"},
            ]
        },
    ]
    # distinct free speech acts: evidence, question_objection -> 2
    assert discussion_diversity(rows) == 2.0
