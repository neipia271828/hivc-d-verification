from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from turn_game import (  # noqa: E402
    ACCEPTABLE_DELTA,
    Action,
    Event,
    GameState,
    acceptable_actions,
    best_action,
    estimate_q_values,
    heuristic_policy,
    initial_state,
    optimal_route,
    play_policy_game,
    pod_ready_status,
    random_policy,
    role_specific_evidence,
    step,
    summarize_games,
    terminal_score,
)
from scripts.llm_turn_game_common import format_state  # noqa: E402


def test_initial_state_is_reproducible() -> None:
    assert initial_state(123) == initial_state(123)
    assert initial_state(123).turn == 0


def test_communication_win_condition() -> None:
    rng = np.random.default_rng(0)
    state = GameState(
        turn=0,
        oxygen=8,
        power=6,
        hull_damage=2,
        flooding=1,
        communication=3,
        rescue_eta=1,
        morale=80,
        current_event=Event.SIGNAL_WINDOW,
    )
    result = step(state, Action.REPAIR_COMMUNICATION, rng)
    assert result.state_after.done
    assert result.state_after.outcome == "win"
    assert terminal_score(result.state_after) > 1000


def test_escape_win_condition() -> None:
    rng = np.random.default_rng(0)
    state = GameState(
        turn=0,
        oxygen=5,
        power=5,
        hull_damage=2,
        flooding=2,
        communication=0,
        pod_readiness=2,
        pod_integrity=2,
        morale=80,
        current_event=Event.NONE,
    )
    result = step(state, Action.EXECUTE_ESCAPE, rng)
    assert result.state_after.done
    assert result.state_after.outcome == "win"


def test_premature_escape_is_loss() -> None:
    rng = np.random.default_rng(0)
    state = GameState(
        turn=0,
        oxygen=5,
        power=5,
        hull_damage=2,
        flooding=2,
        communication=0,
        pod_readiness=1,
        pod_integrity=1,
        morale=80,
        current_event=Event.NONE,
    )
    result = step(state, Action.EXECUTE_ESCAPE, rng)
    assert result.premature
    assert result.state_after.done
    assert result.state_after.outcome.startswith("loss_")


def test_flooding_action_reduces_flooding() -> None:
    rng = np.random.default_rng(1)
    state = GameState(flooding=4, current_event=Event.NONE)
    result = step(state, Action.SEAL_FLOODING, rng)
    assert result.state_after.flooding < state.flooding


def test_q_values_include_all_actions() -> None:
    state = GameState(current_event=Event.NONE)
    q_values = estimate_q_values(state, n_rollouts=8, policy=random_policy, seed=5)
    assert set(q_values) == set(Action)
    assert all(isinstance(value, float) for value in q_values.values())


def test_acceptable_actions_uses_delta() -> None:
    q_values = {
        Action.STABILIZE_OXYGEN: 100.0,
        Action.REPAIR_POWER: 100.0 - ACCEPTABLE_DELTA + 0.1,
        Action.REPAIR_COMMUNICATION: 50.0,
        Action.SEAL_FLOODING: 10.0,
    }
    allowed = acceptable_actions(q_values)
    assert Action.STABILIZE_OXYGEN in allowed
    assert Action.REPAIR_POWER in allowed
    assert best_action(q_values) == Action.STABILIZE_OXYGEN


def test_play_policy_game_outputs_evaluation_rows() -> None:
    rows = play_policy_game(heuristic_policy, seed=42, evaluator_rollouts=6)
    assert rows
    first = rows[0]
    assert "q_values" in first
    assert "regret" in first
    assert "best_action" in first
    assert "acceptable_actions" in first

    summary = summarize_games(rows)
    assert summary["games"] == 1.0
    assert 0.0 <= summary["win_rate"] <= 1.0


def test_route_reversal_changes_optimal_route() -> None:
    """route_reversal では、POD_FLOODING → BACKUP_POWER_FOUND イベントで勝ち筋が変化する。"""
    rng = np.random.default_rng(0)
    state = initial_state(seed=0, scenario_id="route_reversal")
    assert state.scenario_id == "route_reversal"
    assert state.current_event == Event.POD_FLOODING

    route0 = optimal_route(state, seed=0, n_rollouts=40)

    # 1 手目を逃出路の準備にして次の状態を評価
    result = step(state, Action.PREP_POD, rng)
    s1 = result.state_after
    assert s1.current_event == Event.BACKUP_POWER_FOUND

    route1 = optimal_route(s1, seed=0, n_rollouts=40)

    assert route0 == "escape"
    assert route1 == "comms"


def test_beta_diagnosis_invariant_to_flooding() -> None:
    # oxygen、power、艇状態を固定し、flooding だけ変えても beta 診断文字列が一致する
    common = {
        "turn": 0,
        "oxygen": 5,
        "power": 5,
        "hull_damage": 1,
        "communication": 0,
        "pod_readiness": 2,
        "pod_integrity": 2,
        "current_event": Event.NONE,
    }
    s0 = GameState(flooding=0, **common)
    s4 = GameState(flooding=4, **common)
    assert role_specific_evidence("beta", s0) == role_specific_evidence("beta", s4)


def test_beta_pod_ready_diagnosis_lists_visible_shortfalls_only() -> None:
    state = GameState(
        turn=0,
        oxygen=1,
        power=5,
        hull_damage=1,
        flooding=4,
        communication=0,
        pod_readiness=1,
        pod_integrity=2,
        current_event=Event.NONE,
    )
    diag = pod_ready_status(state)
    assert "整備不足" in diag
    assert "酸素不足" in diag
    assert "艇損傷" not in diag
    assert "電力不足" not in diag
    assert "浸水" not in diag
    assert "船体" not in diag


def test_format_state_beta_hides_hull_and_flooding_values() -> None:
    state = GameState(
        turn=0,
        oxygen=5,
        power=5,
        hull_damage=2,
        flooding=4,
        communication=0,
        pod_readiness=1,
        pod_integrity=1,
        current_event=Event.NONE,
    )
    text = format_state(state, "beta")
    assert str(state.hull_damage) not in text
    assert str(state.flooding) not in text
    assert "不明（パートナーに問い合わせ）" in text


def test_escape_outcome_differs_by_flooding_only() -> None:
    rng = np.random.default_rng(0)
    common = {
        "turn": 0,
        "oxygen": 5,
        "power": 5,
        "hull_damage": 1,
        "communication": 0,
        "pod_readiness": 2,
        "pod_integrity": 2,
        "current_event": Event.NONE,
    }
    s0 = GameState(flooding=0, **common)
    s4 = GameState(flooding=4, **common)
    r0 = step(s0, Action.EXECUTE_ESCAPE, rng)
    r4 = step(s4, Action.EXECUTE_ESCAPE, rng)
    assert r0.state_after.outcome == "win"
    assert r4.state_after.outcome.startswith("loss_")


def test_beta_diagnosis_same_does_not_leak_flooding_failure() -> None:
    # flooding だけが未達条件でも、beta 診断にその差は出ない
    common = {
        "turn": 0,
        "oxygen": 5,
        "power": 5,
        "hull_damage": 1,
        "communication": 0,
        "pod_readiness": 2,
        "pod_integrity": 2,
        "current_event": Event.NONE,
    }
    s0 = GameState(flooding=0, **common)
    s4 = GameState(flooding=4, **common)
    assert role_specific_evidence("beta", s0) == role_specific_evidence("beta", s4)
    assert step(s0, Action.EXECUTE_ESCAPE, np.random.default_rng(0)).state_after.outcome == "win"
    assert step(s4, Action.EXECUTE_ESCAPE, np.random.default_rng(0)).state_after.outcome.startswith("loss_")
