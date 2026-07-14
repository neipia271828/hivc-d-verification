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
from scripts.llm_turn_game_common import (  # noqa: E402
    allocate_discussion_budgets,
    extract_json_discussion,
    format_state,
)


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


def test_allocate_discussion_budgets_uses_actual_opportunity_count() -> None:
    """実際の opportunity_count で予算を配分し、第1機会には全エージェント1回分以上を確保する。"""
    # max 24, 1 opportunity -> all 24 to first
    m, t = allocate_discussion_budgets(1, 24, 1024)
    assert m == [24]
    assert t == [1024]

    # max 24, 3 opportunities -> 8 each, first at least 2 (here 8)
    m, t = allocate_discussion_budgets(3, 24, 3072)
    assert m == [8, 8, 8]
    assert t == [1024, 1024, 1024]

    # max 24, 2 opportunities -> 12 each, first at least 2
    m, t = allocate_discussion_budgets(2, 24, 2000)
    assert m == [12, 12]
    assert t[0] >= 1000
    assert t[1] >= 1000

    # remainder distributed to early opportunities
    m, t = allocate_discussion_budgets(2, 25, 100)
    assert m[0] + m[1] == 25
    assert m[0] >= 2


def test_extract_json_discussion_parses_question_metadata() -> None:
    response = (
        '{"speech_act":"question_objection","message":"なぜ？","action":"C",'
        '"reason":"確認","addressed_to":"beta","requires_response":true}'
    )
    speech_act, message, action, reason, reply_id, addressed_to, requires = extract_json_discussion(response)
    assert speech_act.value == "question_objection"
    assert action == Action.REPAIR_COMMUNICATION
    assert addressed_to == "beta"
    assert requires is True
    assert reply_id is None

    response2 = (
        '{"speech_act":"evidence","message":"理由","reply_to_message_id":"1",'
        '"action":"C","reason":"回答"}'
    )
    speech_act2, message2, action2, reason2, reply_id2, addressed_to2, requires2 = extract_json_discussion(response2)
    assert reply_id2 == "1"
    assert requires2 is False


def test_run_one_game_question_response_closure(monkeypatch) -> None:
    """質問発言後に宛先エージェントが回答し、未回答を残さず意思決定に進む。"""
    import json
    from scripts.llm_turn_game_common import run_one_game

    call_count = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal call_count
        # 自由議論フェーズと意思決定機会を区別
        if "意思決定機会" in prompt:
            return "", '{"action":"C","reason":"vote C","message":"C","ready":true}'
        # 自由議論：1回目 alpha が質問、2回目 beta が回答
        if call_count == 0:
            call_count += 1
            return (
                "",
                '{"speech_act":"question_objection","message":"なぜ？","action":"C",'
                '"reason":"質問","addressed_to":"beta","requires_response":true}',
            )
        call_count += 1
        return (
            "",
            '{"speech_act":"evidence","message":"理由","action":"C",'
            '"reason":"回答","reply_to_message_id":"1"}',
        )

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)

    personas = {"alpha": "alpha", "beta": "beta"}
    persona_params = {"alpha": None, "beta": None}
    role_keys = {"alpha": "alpha", "beta": "beta"}
    rows = run_one_game(
        None,
        None,
        "control",
        seed=42,
        personas=personas,
        persona_params=persona_params,
        role_keys=role_keys,
        max_new_tokens=96,
        max_discussion_turns=2,
        discussion_token_budget=1024,
        evaluator_rollouts=4,
        scenario_id="comms_favored",
    )
    assert rows
    first = rows[0]
    transcript = json.loads(first["discussion_transcript"])
    assert transcript[0]["message_id"] == "1"
    assert transcript[0]["addressed_to"] == "beta"
    assert transcript[0]["requires_response"] is True
    assert transcript[1]["reply_to_message_id"] == "1"
    assert first["unanswered_question_count"] == 0
    assert first["forced_decision_with_open_question"] is False
    assert first["question_response_latency"] == 1.0
