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
    forced_decision_with_open_question_rate,
    minority_adoption_rate,
    plan_revision_quality,
    question_response_latency_metric,
    route_switch_quality,
    conflict_resolution_quality,
    unanswered_question_rate,
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


def test_plan_revision_quality_cross_game_boundary_is_nan() -> None:
    # 別ゲームの turn 0 同士を混在させても、前ターンを参照しない
    rows = [
        {"seed": 1, "turn": 0, "event": "none", "regret": 100.0},
        {"seed": 2, "turn": 0, "event": "leak_surge", "regret": 80.0},
    ]
    result = plan_revision_quality(rows)
    assert np.isnan(result["plan_revision_quality"])
    assert np.isnan(result["plan_revision_improved_rate"])


def test_plan_revision_quality_condition_boundary_is_nan() -> None:
    # 同一 seed でも condition が異なれば前ターン扱いしない
    rows = [
        {"seed": 1, "condition": "control", "turn": 0, "event": "none", "regret": 100.0},
        {"seed": 1, "condition": "hivc_d", "turn": 0, "event": "leak_surge", "regret": 80.0},
    ]
    result = plan_revision_quality(rows)
    assert np.isnan(result["plan_revision_quality"])
    assert np.isnan(result["plan_revision_improved_rate"])


def test_plan_revision_quality_interleaved_games() -> None:
    rows = [
        {"seed": 1, "turn": 0, "event": "none", "regret": 100.0},
        {"seed": 2, "turn": 0, "event": "none", "regret": 50.0},
        {"seed": 1, "turn": 1, "event": "leak_surge", "regret": 80.0},
        {"seed": 2, "turn": 1, "event": "pressure_spike", "regret": 40.0},
    ]
    result = plan_revision_quality(rows)
    # 各ゲーム内で turn1 が前ターンより改善している
    assert result["plan_revision_quality"] == 1.0
    assert result["plan_revision_improved_rate"] == 1.0


def test_plan_revision_quality_game_order_independent() -> None:
    rows_game_order = [
        {"seed": 1, "turn": 0, "event": "none", "regret": 100.0},
        {"seed": 1, "turn": 1, "event": "leak_surge", "regret": 80.0},
        {"seed": 2, "turn": 0, "event": "none", "regret": 50.0},
        {"seed": 2, "turn": 1, "event": "pressure_spike", "regret": 40.0},
    ]
    rows_interleaved = [
        {"seed": 1, "turn": 0, "event": "none", "regret": 100.0},
        {"seed": 2, "turn": 0, "event": "none", "regret": 50.0},
        {"seed": 1, "turn": 1, "event": "leak_surge", "regret": 80.0},
        {"seed": 2, "turn": 1, "event": "pressure_spike", "regret": 40.0},
    ]
    assert plan_revision_quality(rows_game_order) == plan_revision_quality(rows_interleaved)


def test_route_switch_quality_cross_game_boundary_turn0_is_nan() -> None:
    # 別 seed の turn 0 行を混在させても、ゲーム境界だけで分母が増えない
    rows = [
        {"seed": 1, "turn": 0, "event": "none", "planned_route": "escape", "optimal_route": "comms"},
        {"seed": 2, "turn": 0, "event": "leak_surge", "planned_route": "escape", "optimal_route": "comms"},
    ]
    assert np.isnan(route_switch_quality(rows))


def test_route_switch_quality_condition_boundary_turn0_is_nan() -> None:
    # 同一 seed でも condition が異なれば前ターン扱いしない
    rows = [
        {"seed": 1, "condition": "control", "turn": 0, "event": "none", "planned_route": "escape", "optimal_route": "comms"},
        {"seed": 1, "condition": "hivc_d", "turn": 0, "event": "leak_surge", "planned_route": "escape", "optimal_route": "comms"},
    ]
    assert np.isnan(route_switch_quality(rows))


def test_route_switch_quality_interleaved_games() -> None:
    rows = [
        {"seed": 1, "turn": 0, "event": "none", "planned_route": "escape", "optimal_route": "comms"},
        {"seed": 2, "turn": 0, "event": "none", "planned_route": "escape", "optimal_route": "comms"},
        {"seed": 1, "turn": 1, "event": "leak_surge", "planned_route": "comms", "optimal_route": "comms"},
        {"seed": 2, "turn": 1, "event": "pressure_spike", "planned_route": "comms", "optimal_route": "comms"},
    ]
    # 各ゲームで turn0 -> turn1 に optimal が escape から comms へ変化し、planned も切り替わっている
    assert route_switch_quality(rows) == 1.0


def test_route_switch_quality_game_order_independent() -> None:
    rows_game_order = [
        {"seed": 1, "turn": 0, "event": "none", "planned_route": "escape", "optimal_route": "comms"},
        {"seed": 1, "turn": 1, "event": "leak_surge", "planned_route": "comms", "optimal_route": "comms"},
        {"seed": 2, "turn": 0, "event": "none", "planned_route": "escape", "optimal_route": "comms"},
        {"seed": 2, "turn": 1, "event": "pressure_spike", "planned_route": "comms", "optimal_route": "comms"},
    ]
    rows_interleaved = [
        {"seed": 1, "turn": 0, "event": "none", "planned_route": "escape", "optimal_route": "comms"},
        {"seed": 2, "turn": 0, "event": "none", "planned_route": "escape", "optimal_route": "comms"},
        {"seed": 1, "turn": 1, "event": "leak_surge", "planned_route": "comms", "optimal_route": "comms"},
        {"seed": 2, "turn": 1, "event": "pressure_spike", "planned_route": "comms", "optimal_route": "comms"},
    ]
    assert route_switch_quality(rows_game_order) == route_switch_quality(rows_interleaved)


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


def test_unanswered_question_rate() -> None:
    rows = [
        {"unanswered_question_count": 0},
        {"unanswered_question_count": 1},
        {"unanswered_question_count": 2},
    ]
    assert unanswered_question_rate(rows) == 2 / 3


def test_question_response_latency_metric() -> None:
    rows = [
        {"question_response_latency": 1.0},
        {"question_response_latency": 3.0},
        {"question_response_latency": float("nan")},
    ]
    assert question_response_latency_metric(rows) == 2.0


def test_forced_decision_with_open_question_rate() -> None:
    rows = [
        {"forced_decision_with_open_question": False},
        {"forced_decision_with_open_question": True},
        {"forced_decision_with_open_question": "true"},
    ]
    assert forced_decision_with_open_question_rate(rows) == 2 / 3


def test_compute_summary_metrics_includes_question_metrics() -> None:
    rows = [
        {
            "seed": 42, "turn": 0, "event": "none",
            "alpha_vote": "A", "beta_vote": "A",
            "group_action": "A", "acceptable_actions": "A,B",
            "regret": 0.0, "outcome": "running", "terminal_score": 500.0,
            "unanswered_question_count": 0,
            "question_response_latency": 1.0,
            "forced_decision_with_open_question": False,
        },
    ]
    summary = compute_summary_metrics(rows)
    assert "unanswered_question_rate" in summary
    assert "question_response_latency" in summary
    assert "forced_decision_with_open_question_rate" in summary
    assert summary["unanswered_question_rate"] == 0.0
    assert summary["question_response_latency"] == 1.0
    assert summary["forced_decision_with_open_question_rate"] == 0.0
