from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

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
    sample_viable_event,
    step,
    summarize_games,
    terminal_score,
)
from scripts.llm_turn_game_common import (  # noqa: E402
    CONDITION_PROCEDURES,
    _extract_json_object,
    _normalize_requested_fields,
    _question_signature,
    allocate_discussion_budgets,
    decision_opportunity_prompt,
    decision_relevant_hidden_fields,
    discussion_prompt,
    extract_json_discussion,
    format_state,
    run_one_game,
)


def test_decision_relevant_hidden_fields_uses_scope_and_state() -> None:
    state = GameState(
        hull_damage=3,
        flooding=1,
        communication=2,
        pod_readiness=1,
        pod_integrity=0,
        current_event=Event.NONE,
    )
    alpha_scope = {"oxygen", "power", "hull_damage", "flooding"}
    beta_scope = {"oxygen", "power", "communication", "pod_readiness", "pod_integrity"}
    assert decision_relevant_hidden_fields(state, alpha_scope, beta_scope) == [
        "communication", "pod_readiness", "pod_integrity"
    ]
    assert decision_relevant_hidden_fields(state, beta_scope, alpha_scope) == ["hull_damage"]


def test_final_vote_retries_guaranteed_loss_and_uses_safe_action(monkeypatch) -> None:
    low_power = GameState(
        turn=4,
        oxygen=6,
        power=2,
        hull_damage=1,
        flooding=1,
        communication=0,
        current_event=Event.NONE,
        scenario_id="ambiguous",
    )
    monkeypatch.setattr("scripts.llm_turn_game_common.initial_state", lambda seed, scenario_id=None: low_power)

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        if "id=final-vote-repair" in prompt:
            return "", '{"action":"B","reason":"powerを回復するためBを選択する","ready":true}'
        if "id=decision-contract" in prompt:
            return "", '{"action":"A","reason":"Aを選択する","ready":true}'
        return "", '{"speech_act":"evidence","message":"状況共有","action":"B","reason":"powerが低い","addressed_to":null,"reply_to_message_id":null}'

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None, None, "control", 42,
        {"alpha": "alpha", "beta": "beta"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_discussion_turns=2,
        discussion_token_budget=512,
        evaluator_rollouts=1,
        max_decision_opportunities=1,
        max_final_vote_retries=1,
    )
    first = rows[0]
    assert first["alpha_vote"] == "B"
    assert first["beta_vote"] == "B"
    assert first["group_action"] == "B"
    assert first["final_vote_retry_count"] == 2
    assert first["rejected_final_vote_count"] == 2
    rejected = json.loads(first["rejected_final_votes"])
    assert all(item["rejection_reason"].startswith("unsafe_action:guaranteed_loss_power") for item in rejected)
    assert first["outcome"] != "loss_power"


def test_sample_viable_event_replaces_unavoidable_hull_fracture(monkeypatch) -> None:
    state = GameState(
        turn=3, oxygen=6, power=6, hull_damage=3, flooding=1,
        current_event=Event.NONE, scenario_id="ambiguous",
    )
    monkeypatch.setattr("turn_game.sample_event", lambda *args, **kwargs: Event.HULL_FRACTURE)
    event = sample_viable_event(np.random.default_rng(7), state, "ambiguous", 3)
    assert event != Event.HULL_FRACTURE
    candidate_state = GameState(**{**state.__dict__, "current_event": event})
    from turn_game import preview_action_safety
    assert any(preview_action_safety(candidate_state, action)[0] for action in Action)


def test_required_hidden_information_question_is_retried_and_answered(monkeypatch) -> None:
    from types import SimpleNamespace
    from profiles import Role

    state = GameState(
        turn=4, oxygen=8, power=8, hull_damage=1, flooding=1,
        communication=2, current_event=Event.SIGNAL_WINDOW, scenario_id="comms_favored",
    )
    monkeypatch.setattr("scripts.llm_turn_game_common.initial_state", lambda seed, scenario_id=None: state)
    alpha_role = Role(
        id="safety", label="安全", schema_version="2.0",
        expertise_domains=("safety",),
        observation_scope=("oxygen", "power", "hull_damage", "flooding"),
        responsibility="安全", feasibility_constraints=(),
    )
    beta_role = Role(
        id="comms", label="通信", schema_version="2.0",
        expertise_domains=("communication",),
        observation_scope=("oxygen", "power", "communication", "pod_readiness", "pod_integrity"),
        responsibility="通信", feasibility_constraints=(),
    )
    profiles = {
        "alpha": SimpleNamespace(role=alpha_role),
        "beta": SimpleNamespace(role=beta_role),
    }
    alpha_discussion_attempts = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal alpha_discussion_attempts
        if "id=decision-contract" in prompt:
            return "", '{"action":"B","reason":"Bを選択する","ready":true}'
        if "質問ID 1 への回答が必須" in prompt:
            return "", (
                '{"speech_act":"evidence","message":"communicationは2",'
                '"action":"B","reason":"回答","addressed_to":null,"reply_to_message_id":"1"}'
            )
        if "id=required-information-question" in prompt:
            alpha_discussion_attempts += 1
            if alpha_discussion_attempts == 1:
                return "", (
                    '{"speech_act":"evidence","message":"先に結論","action":"B",'
                    '"reason":"修理","addressed_to":null,"reply_to_message_id":null}'
                )
            return "", (
                '{"speech_act":"information_request","message":"communicationを確認したい",'
                '"action":null,"reason":"通信窓の判断に必要","addressed_to":"beta",'
                '"reply_to_message_id":null,"requested_fields":["communication"]}'
            )
        return "", (
            '{"speech_act":"evidence","message":"共有","action":"B",'
            '"reason":"共有","addressed_to":null,"reply_to_message_id":null}'
        )

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None, None, "control", 42,
        {"alpha": "alpha", "beta": "beta"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_discussion_turns=2,
        discussion_token_budget=512,
        evaluator_rollouts=1,
        max_decision_opportunities=1,
        max_discussion_retries=1,
        resolved_profiles=profiles,
    )
    first = rows[0]
    assert first["required_information_question_count"] == 1
    assert first["missing_required_information_question_count"] == 0
    assert first["question_count"] == 1
    assert first["answered_question_count"] == 1
    assert first["discussion_retry_count"] >= 1
    audit = json.loads(first["invalid_discussion_outputs"])
    assert audit[0]["validation_reason"] == "required_information_question_missing"
    assert audit[0]["recovered"] is True


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


def test_format_state_uses_role_observation_scope_instead_of_agent_name() -> None:
    state = initial_state(seed=42)
    role = {
        "expertise_domains": ["communications"],
        "observation_scope": ["communication", "pod_integrity"],
        "responsibility": "通信情報を共有する",
    }
    alpha_view = format_state(state, "alpha", role)
    assert f"communication: {state.communication}" in alpha_view
    assert f"pod_integrity: {state.pod_integrity}" in alpha_view
    assert "oxygen: 不明" in alpha_view
    assert "hull_damage: 不明" in alpha_view

    prompt = discussion_prompt("alpha", "persona", None, state, [], 2, role=role)
    assert 'expertise_domains: ["communications"]' in prompt
    assert "responsibility: 通信情報を共有する" in prompt


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
        '"reason":"確認","addressed_to":"beta","reply_to_message_id":null}'
    )
    speech_act, message, action, reason, reply_id, addressed_to, requires = extract_json_discussion(response)
    assert speech_act.value == "question_objection"
    assert action == Action.REPAIR_COMMUNICATION
    assert addressed_to == "beta"
    assert requires is True
    assert reply_id is None

    response2 = (
        '{"speech_act":"evidence","message":"理由","reply_to_message_id":"1",'
        '"action":"C","reason":"回答","addressed_to":null}'
    )
    speech_act2, message2, action2, reason2, reply_id2, addressed_to2, requires2 = extract_json_discussion(response2)
    assert reply_id2 == "1"
    assert requires2 is False

    # question_objection なら requires_response はモデル値によらず true
    response3 = (
        '{"speech_act":"question_objection","message":"なぜ？","action":"C",'
        '"reason":"確認","addressed_to":"beta","reply_to_message_id":null}'
    )
    speech_act3, _, _, _, _, _, requires3 = extract_json_discussion(response3)
    assert speech_act3.value == "question_objection"
    assert requires3 is True


def test_discussion_prompt_schema_includes_question_metadata_keys() -> None:
    prompt = discussion_prompt(
        "alpha",
        "alpha persona",
        None,
        GameState(current_event=Event.NONE),
        [],
        max_discussion_turns=6,
    )

    assert "必須キー: speech_act, message, action, reason, reply_to_message_id" in prompt
    assert "情報要求の質問では null 可" in prompt
    assert "質問以外の全speech_actでは、actionに必ずA-Fの一つ" in prompt
    assert "evidence、proposal、tradeoff" in prompt
    assert "null/省略なら相手一名へ補完" in prompt
    assert '"addressed_to":null,"reply_to_message_id":null' in prompt
    assert '"speech_act":"information_request"' in prompt
    assert '"action":null' in prompt
    assert '"addressed_to":"beta","reply_to_message_id":null' in prompt
    assert "A. 酸素供給を安定化（効果: oxygen +3, power -1" in prompt
    assert "F. 自力脱出を実行（効果: 脱出条件を全て満たす時のみ勝利" in prompt


def test_discussion_answer_example_obeys_non_question_address_contract() -> None:
    prompt = discussion_prompt(
        "beta",
        "beta",
        None,
        GameState(current_event=Event.NONE),
        [],
        max_discussion_turns=6,
        open_question={
            "speaker": "alpha",
            "message_id": "q-1",
            "message": "oxygenは?",
        },
    )

    assert '"speech_act":"evidence"' in prompt
    assert '"addressed_to":null,"reply_to_message_id":"q-1"' in prompt
    assert '"addressed_to":"alpha","reply_to_message_id":"q-1"' not in prompt


def test_discussion_prompt_schema_requires_exact_reply_id_for_open_question() -> None:
    prompt = discussion_prompt(
        "beta",
        "beta persona",
        None,
        GameState(current_event=Event.NONE),
        [],
        max_discussion_turns=6,
        open_question={
            "message_id": "7",
            "speaker": "alpha",
            "addressed_to": "beta",
            "message": "通信を優先する根拠は？",
        },
        can_ask_question=False,
        remaining_messages=3,
        remaining_tokens=288,
    )

    assert "今は質問ID 7 への回答が必須です" in prompt
    assert "reply_to_message_id を省略しないでください" in prompt
    assert '"addressed_to":null,"reply_to_message_id":"7"' in prompt
    assert "通常発言JSON例" not in prompt


def test_hivc_d_protocol_is_detailed_and_injected_into_both_prompt_types() -> None:
    state = GameState(current_event=Event.NONE)
    discussion = discussion_prompt(
        "alpha",
        "alpha persona",
        None,
        state,
        [],
        max_discussion_turns=6,
        condition="hivc_d",
    )
    decision = decision_opportunity_prompt(
        "alpha",
        "alpha persona",
        None,
        state,
        [],
        "hivc_d",
        opportunity_index=1,
        opportunity_count=2,
    )

    for prompt in (discussion, decision):
        assert "HIVC-D 合意形成プロトコル：I → V → A" in prompt
        assert "I（Information: 情報の共有）" in prompt
        assert "V（Value: 判断基準の整合）" in prompt
        assert "A（Ability: 実行可能性の確認）" in prompt
        assert "共通基準 V*" in prompt
        assert "見えていない状態を推測で事実扱いせず" in prompt
        assert "最終投票前チェック" in prompt


def test_consulting_guide_matches_hivc_d_detail_without_hivc_d_terms() -> None:
    consulting = CONDITION_PROCEDURES["consulting"]
    hivc_d = CONDITION_PROCEDURES["hivc_d"]

    assert len(consulting) >= len(hivc_d) * 0.85
    assert "状況を整理する" in consulting
    assert "選択肢のリスクと便益を比較する" in consulting
    assert "実行前に確認する" in consulting
    assert "見えていない状態を推測で事実扱いせず" in consulting
    assert "最終投票前チェック" in consulting
    assert "V*" not in consulting
    assert "I（Information" not in consulting


def test_allocate_discussion_budgets_respects_total_and_token_proportional() -> None:
    """配分合計が max_discussion_turns / token_budget を超えず、トークンは発言数に比例。"""
    # 第1機会を2に確保して後続を0に削減
    m, t = allocate_discussion_budgets(3, 2, 100, n_speakers=2)
    assert sum(m) <= 2
    assert m[0] >= 2
    assert sum(t) <= 100
    assert t[0] == 100

    # 3機会に3発言：2,1,0
    m, t = allocate_discussion_budgets(3, 3, 100, n_speakers=2)
    assert sum(m) <= 3
    assert m == [2, 1, 0]
    assert sum(t) <= 100
    assert t[0] > 0 and t[1] > 0

    # 多めの発言とトークン
    m, t = allocate_discussion_budgets(2, 25, 1000, n_speakers=2)
    assert sum(m) <= 25
    assert m[0] >= 2
    assert sum(t) <= 1000


def test_run_one_game_question_response_closure(monkeypatch) -> None:
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
                '"reason":"質問","addressed_to":"beta","reply_to_message_id":null}',
            )
        call_count += 1
        return (
            "",
            '{"speech_act":"evidence","message":"理由","action":"C",'
            '"reason":"回答","addressed_to":null,"reply_to_message_id":"1"}',
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


def test_run_one_game_forced_decision_still_collects_votes(monkeypatch) -> None:
    """予算末尾で未回答質問が残っても、投票を取らずにフォールバックしない。"""
    import json
    from scripts.llm_turn_game_common import run_one_game

    call_count = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal call_count
        # 意思決定機会では無効応答を返して、フォールバック票が best にならないことを確認
        if "意思決定機会" in prompt:
            return "", "this is not json"
        # 自由議論：1回目 alpha が質問、2回目 beta も質問（回答なし）
        if call_count == 0:
            call_count += 1
            return (
                "",
                '{"speech_act":"question_objection","message":"なぜ？","action":"C",'
                '"reason":"質問","addressed_to":"beta","reply_to_message_id":null}',
            )
        call_count += 1
        return (
            "",
            '{"speech_act":"question_objection","message":"さらに？","action":"C",'
            '"reason":"追加質問","addressed_to":"alpha","reply_to_message_id":null}',
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
    # 投票は実行されたので decision_history が存在する
    decision_history = json.loads(first["decision_history"])
    assert len(decision_history) > 0
    assert decision_history[0]["alpha_vote"] == ""
    assert decision_history[0]["beta_vote"] == ""
    # 無効な票を best にしないため、alpha_vote は空で best_action とは異なる
    assert first["alpha_vote"] == ""
    assert first["beta_vote"] == ""
    assert first["forced_decision_with_open_question"] is True
    assert first["unanswered_question_count"] > 0


@pytest.mark.parametrize("target_field", [',"addressed_to":null', ""])
def test_run_one_game_question_with_null_or_missing_target_is_routed_to_partner(monkeypatch, target_field: str) -> None:
    """二者ゲームの質問で null/省略宛先は一意な相手へ安全に補完される。"""
    import json
    from scripts.llm_turn_game_common import run_one_game

    call_count = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal call_count
        if "意思決定機会" in prompt:
            return "", '{"action":"C","reason":"vote C","message":"C","ready":true}'
        # 自由議論：1回目 alpha が行動案を伴わない情報要求を出す。
        if call_count == 0:
            call_count += 1
            return (
                "",
                '{"speech_act":"information_request","message":"why?","action":null,'
                f'"reason":"質問"{target_field},"reply_to_message_id":null}}',
            )
        # 2回目 beta が回答
        call_count += 1
        return (
            "",
            '{"speech_act":"evidence","message":"理由","action":"C",'
            '"reason":"回答","addressed_to":null,"reply_to_message_id":"1"}',
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
    assert transcript[0]["addressed_to"] == "beta"
    assert transcript[0]["requires_response"] is True
    assert transcript[1]["reply_to_message_id"] == "1"
    assert first["unanswered_question_count"] == 0
    assert first["forced_decision_with_open_question"] is False
    assert first["question_response_latency"] == 1.0


def test_run_one_game_unknown_question_target_is_rejected_not_normalized(monkeypatch) -> None:
    """未知名の宛先は相手へ補正せず、無効なdiscussion出力として監査する。"""
    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        if "意思決定機会" in prompt:
            return "", '{"action":"C","reason":"vote","message":"C","ready":true}'
        return "", (
            '{"speech_act":"information_request","message":"why?","action":null,'
            '"reason":"質問","addressed_to":"gamma","reply_to_message_id":null}'
        )

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None,
        None,
        "control",
        seed=42,
        personas={"alpha": "alpha", "beta": "beta"},
        persona_params={"alpha": None, "beta": None},
        role_keys={"alpha": "alpha", "beta": "beta"},
        max_discussion_turns=2,
        discussion_token_budget=1024,
        evaluator_rollouts=4,
        max_discussion_retries=0,
        scenario_id="comms_favored",
    )
    first = rows[0]
    assert first["invalid_discussion_output_count"] >= 1
    transcript = json.loads(first["discussion_transcript"])
    assert all(item.get("message") != "why?" for item in transcript)
    assert all(item.get("addressed_to") != "gamma" for item in transcript)
    assert first["question_count"] == 0


def test_run_one_game_fake_reply_from_non_addressee_keeps_question_open(monkeypatch) -> None:
    """質問の宛先以外、または質問者自身が reply_to_message_id を指定しても質問を閉じない。"""
    import json
    from scripts.llm_turn_game_common import run_one_game

    call_count = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal call_count
        if "意思決定機会" in prompt:
            return "", '{"action":"C","reason":"vote C","message":"C","ready":true}'
        # 1 alpha -> beta 質問
        if call_count == 0:
            call_count += 1
            return (
                "",
                '{"speech_act":"question_objection","message":"Q1","action":"C",'
                '"reason":"質問","addressed_to":"beta","reply_to_message_id":null}',
            )
        # 2 beta は存在しない返信IDを指定（not_found）
        if call_count == 1:
            call_count += 1
            return (
                "",
                '{"speech_act":"evidence","message":"無効","action":"C",'
                '"reason":"回答","addressed_to":null,"reply_to_message_id":"999"}',
            )
        # 3 alpha が自分の質問 Q1 に回答（addressed_to mismatch）
        call_count += 1
        return (
            "",
            '{"speech_act":"evidence","message":"自答","action":"C",'
                '"reason":"回答","addressed_to":null,"reply_to_message_id":"1"}',
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
        max_discussion_turns=3,
        discussion_token_budget=1024,
        evaluator_rollouts=4,
        scenario_id="comms_favored",
    )
    assert rows
    first = rows[0]
    transcript = json.loads(first["discussion_transcript"])
    # 2, 3 番目は無効な回答参照
    assert transcript[1].get("reply_to_message_id_invalid") is True
    assert transcript[2].get("reply_to_message_id_invalid") is True
    assert first["unanswered_question_count"] == 1
    assert first["forced_decision_with_open_question"] is True
    assert "invalid_reply_to_message_id" in first["forced_decision_reason"]
    assert first["question_response_latency"] != first["question_response_latency"]  # nan


def test_run_one_game_missing_reply_to_while_answer_required(monkeypatch) -> None:
    """回答すべき未回答質問があるのに reply_to_message_id を返さない発言は無効。"""
    import json
    from scripts.llm_turn_game_common import run_one_game

    call_count = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal call_count
        if "意思決定機会" in prompt:
            return "", '{"action":"C","reason":"vote C","message":"C","ready":true}'
        # 1 alpha -> beta 質問
        if call_count == 0:
            call_count += 1
            return (
                "",
                '{"speech_act":"question_objection","message":"Q1","action":"C",'
                '"reason":"質問","addressed_to":"beta","reply_to_message_id":null}',
            )
        # 2 beta は reply_to_message_id を返さない一般発言
        call_count += 1
        return (
            "",
            '{"speech_act":"evidence","message":"一般論","action":"C",'
            '"reason":"一般発言","addressed_to":null,"reply_to_message_id":null}',
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
    assert first["unanswered_question_count"] == 1
    assert transcript[1].get("missing_reply_to_message_id_while_answer_required") is True
    assert first["forced_decision_with_open_question"] is True
    assert "missing_reply_to_message_id_while_answer_required" in first["forced_decision_reason"]


def test_run_one_game_question_while_answer_required_retries(monkeypatch) -> None:
    """未回答質問の宛先エージェントが回答せず新しい質問を返した場合、同じagentに再試行させる。"""
    import json
    from scripts.llm_turn_game_common import run_one_game

    call_count = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal call_count
        if "意思決定機会" in prompt:
            return "", '{"action":"C","reason":"vote C","message":"C","ready":true}'
        # 1 alpha -> beta 質問
        if call_count == 0:
            call_count += 1
            return (
                "",
                '{"speech_act":"question_objection","message":"Q1","action":"C",'
                '"reason":"質問","addressed_to":"beta","reply_to_message_id":null}',
            )
        # 2 beta は Q1 に回答せず alpha 宛の質問を返す（無効）
        if call_count == 1:
            call_count += 1
            return (
                "",
                '{"speech_act":"question_objection","message":"Q2","action":"C",'
                '"reason":"返質問","addressed_to":"alpha","reply_to_message_id":null}',
            )
        # 3 beta が Q1 に回答（再試行後）
        call_count += 1
        return (
            "",
            '{"speech_act":"evidence","message":"A1","action":"C",'
            '"reason":"回答","addressed_to":null,"reply_to_message_id":"1"}',
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
        max_discussion_turns=3,
        discussion_token_budget=1024,
        evaluator_rollouts=4,
        scenario_id="comms_favored",
    )
    assert rows
    first = rows[0]
    transcript = json.loads(first["discussion_transcript"])
    assert transcript[1].get("invalid_response_while_answer_required") is True
    assert first["forced_decision_with_open_question"] is False
    assert "invalid_response_while_answer_required" not in first["forced_decision_reason"]
    # Q1 は回答により閉じられている
    assert first["unanswered_question_count"] == 0


def test_extract_json_object_rejects_surrounding_text_and_markdown() -> None:
    assert _extract_json_object('{"a":1}') == {"a": 1}
    assert _extract_json_object('Some prose {"a":1}') is None
    assert _extract_json_object('```json\n{"a":1}\n```') is None
    assert _extract_json_object('{"a":1} extra') is None
    assert _extract_json_object('{"a":1}{"b":2}') is None


def test_extract_json_discussion_rejects_missing_keys_and_type_mismatch() -> None:
    # 必須キー欠落
    missing_keys = '{"speech_act":"evidence","message":"ok","action":"A","reason":"ok"}'
    speech_act, _, _, _, _, _, _ = extract_json_discussion(missing_keys)
    assert speech_act is None

    # message が空文字
    empty_message = '{"speech_act":"evidence","message":"","action":"A","reason":"ok","addressed_to":null,"reply_to_message_id":null}'
    speech_act2, _, _, _, _, _, _ = extract_json_discussion(empty_message)
    assert speech_act2 is None

    # action が無効な文字列
    invalid_action = '{"speech_act":"evidence","message":"ok","action":"X","reason":"ok","addressed_to":null,"reply_to_message_id":null}'
    speech_act3, _, action, _, _, _, _ = extract_json_discussion(invalid_action)
    assert speech_act3 is None

    # 質問では action=null と addressed_to=null/省略を受理する
    missing_addressed = '{"speech_act":"question_objection","message":"Q","action":"A","reason":"Q","addressed_to":null,"reply_to_message_id":null}'
    speech_act4, _, action4, _, _, target4, requires4 = extract_json_discussion(missing_addressed)
    assert speech_act4 is not None and action4 == Action("A")
    assert target4 is None and requires4 is True

    missing_target = '{"speech_act":"information_request","message":"Q","action":null,"reason":"Q","reply_to_message_id":null}'
    speech_act4b, _, action4b, _, _, target4b, requires4b = extract_json_discussion(missing_target)
    assert speech_act4b is not None and action4b is None
    assert target4b is None and requires4b is True

    # 非質問で addressed_to が設定されている
    spurious_addressed = '{"speech_act":"evidence","message":"ok","action":"A","reason":"ok","addressed_to":"beta","reply_to_message_id":null}'
    speech_act5, _, _, _, _, _, _ = extract_json_discussion(spurious_addressed)
    assert speech_act5 is None

    # addressed_to に整数を設定（暗黙の型変換を拒否）
    int_addressed = '{"speech_act":"question_objection","message":"Q","action":"A","reason":"Q","addressed_to":123,"reply_to_message_id":null}'
    speech_act6, _, _, _, _, addressed_to6, _ = extract_json_discussion(int_addressed)
    assert speech_act6 is None

    # 未知名・辞書の宛先は二者ゲームでも補完せず拒否する
    for invalid_target in ('"gamma"', '{}'):
        raw = ('{"speech_act":"information_request","message":"Q","action":null,'
               f'"reason":"Q","addressed_to":{invalid_target},"reply_to_message_id":null}}')
        invalid_speech_act, *_ = extract_json_discussion(raw)
        assert invalid_speech_act is None

    # 非質問は action=null を受理しない
    null_statement_action = '{"speech_act":"evidence","message":"ok","action":null,"reason":"ok","addressed_to":null,"reply_to_message_id":null}'
    invalid_statement, _, invalid_action_value, _, _, _, _ = extract_json_discussion(null_statement_action)
    assert invalid_statement is None and invalid_action_value is None

    # reply_to_message_id に辞書を設定（正規化でNoneになるのを防ぐ）
    dict_reply = '{"speech_act":"evidence","message":"ok","action":"A","reason":"ok","addressed_to":null,"reply_to_message_id":{}}'
    speech_act7, _, _, _, reply_id7, _, _ = extract_json_discussion(dict_reply)
    assert speech_act7 is None

    # reply_to_message_id に bool を設定（int派生型だが拒否）
    bool_reply = '{"speech_act":"evidence","message":"ok","action":"A","reason":"ok","addressed_to":null,"reply_to_message_id":true}'
    speech_act8, _, _, _, reply_id8, _, _ = extract_json_discussion(bool_reply)
    assert speech_act8 is None


def test_normalize_requested_fields() -> None:
    assert _normalize_requested_fields(None) == []
    assert _normalize_requested_fields("oxygen") == ["oxygen"]
    assert _normalize_requested_fields(["Oxygen", "Power", "oxygen"]) == ["oxygen", "power"]
    assert _normalize_requested_fields(["  ", ""]) == []
    assert _normalize_requested_fields(123) == []
    assert _normalize_requested_fields({"a": 1}) == []


def test_question_signature_uses_requested_fields_when_present() -> None:
    """requested_fields があればそれを signature に使い、action/reason/message は無視する。"""
    sig_with_fields = _question_signature({
        "speaker": "Alpha",
        "addressed_to": "Beta",
        "requested_fields": ["Oxygen", "Power"],
        "action": "A",
        "reason": "r1",
        "message": "m1",
    })
    sig_same_fields_diff_text = _question_signature({
        "speaker": "alpha",
        "addressed_to": "beta",
        "requested_fields": ["power", "oxygen"],
        "action": "B",
        "reason": "r2",
        "message": "m2",
    })
    assert sig_with_fields == sig_same_fields_diff_text

    sig_diff_fields = _question_signature({
        "speaker": "alpha",
        "addressed_to": "beta",
        "requested_fields": ["hull_damage"],
        "action": "A",
        "reason": "r1",
        "message": "m1",
    })
    assert sig_with_fields != sig_diff_fields


def test_question_signature_falls_back_to_action_reason_message() -> None:
    """requested_fields がない場合は action+reason+message を signature に使う。"""
    sig1 = _question_signature({
        "speaker": "alpha",
        "addressed_to": "beta",
        "action": "A",
        "reason": "r1",
        "message": "m1",
    })
    sig2 = _question_signature({
        "speaker": "alpha",
        "addressed_to": "beta",
        "action": "A",
        "reason": "r1",
        "message": "m1",
    })
    assert sig1 == sig2
    sig3 = _question_signature({
        "speaker": "alpha",
        "addressed_to": "beta",
        "action": "B",
        "reason": "r1",
        "message": "m1",
    })
    assert sig1 != sig3


def test_invalid_discussion_output_triggers_retry_and_recovers(monkeypatch) -> None:
    """JSON 契約違反は同一agentへ修復リトライし、成功すれば監査経路へ保存しない。"""
    import json
    from scripts.llm_turn_game_common import run_one_game

    call_count = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal call_count
        if "意思決定機会" in prompt:
            return "", '{"action":"A","reason":"vote A","message":"A","ready":true}'
        call_count += 1
        if call_count == 1:
            return "", "this is not valid json"
        return "", '{"speech_act":"evidence","message":"ok","action":"A","reason":"ok","addressed_to":null,"reply_to_message_id":null}'

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None,
        None,
        "control",
        42,
        {"alpha": "alpha", "beta": "beta"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_new_tokens=96,
        max_discussion_turns=2,
        discussion_token_budget=1024,
        evaluator_rollouts=2,
        max_decision_opportunities=1,
        scenario_id="comms_favored",
    )
    assert rows
    first = rows[0]
    transcript = json.loads(first["discussion_transcript"])
    assert not any(item.get("raw") == "this is not valid json" for item in transcript)
    # リトライ成功時も、最初のinvalid attemptを回復済みとして監査保存する。
    audit = json.loads(first["invalid_discussion_outputs"])
    assert len(audit) == 1
    assert audit[0]["recovered"] is True
    assert audit[0]["final_exhausted"] is False
    assert audit[0]["attempt"] == 1
    assert audit[0]["max_attempts"] == 2
    assert audit[0]["turn"] == 0
    assert audit[0]["opportunity"] == 1
    assert audit[0]["agent"] == audit[0]["speaker"]
    assert "token_count" in audit[0]
    assert first["discussion_retry_count"] >= 1
    assert first["invalid_discussion_output_count"] == 0
    assert first["invalid_attempt_count"] == 1
    assert first["repaired_invalid_output_count"] == 1


def test_invalid_discussion_output_retry_exhaustion_records_audit(monkeypatch) -> None:
    """リトライ上限後もinvalidの場合は監査経路へ確定保存し、retry回数を記録する。"""
    import json
    from scripts.llm_turn_game_common import run_one_game

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        if "意思決定機会" in prompt:
            return "", '{"action":"A","reason":"vote A","message":"A","ready":true}'
        return "", "this is not valid json"

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None,
        None,
        "control",
        42,
        {"alpha": "alpha", "beta": "beta"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_new_tokens=96,
        max_discussion_turns=2,
        discussion_token_budget=1024,
        evaluator_rollouts=2,
        max_decision_opportunities=1,
        scenario_id="comms_favored",
    )
    assert rows
    first = rows[0]
    audit = json.loads(first["invalid_discussion_outputs"])
    assert len(audit) >= 1
    assert all(item["raw"] == "this is not valid json" for item in audit)
    assert all(item["raw_output"] == "this is not valid json" for item in audit)
    assert all(item["agent"] in {"alpha", "beta"} for item in audit)
    assert [item["attempt"] for item in audit[:2]] == [1, 2]
    assert all(item["max_attempts"] == 2 for item in audit)
    assert all(item["validation_reason"] == "not_json_object_or_extra_text" for item in audit)
    assert audit[0]["recovered"] is False
    assert audit[0]["final_exhausted"] is False
    assert audit[1]["final_exhausted"] is True
    assert audit[0]["retry_attempts"] >= 1
    assert first["discussion_retry_count"] >= 1
    assert first["invalid_discussion_output_count"] >= 1
    assert first["invalid_attempt_count"] >= 2
    assert first["repaired_invalid_output_count"] == 0


def test_scope_unanswerable_question_is_closed_and_not_resent(monkeypatch) -> None:
    """両agentとも観測できないfieldへの質問は unanswerable として閉じ、closed_questionsへ保存して再送を抑止する。

    発話上限を十分に確保し、同じspeakerが同じrequested_fieldsを再送した場合に
    duplicate_question として拒否されることを検証する。
    """
    import json
    from scripts.llm_turn_game_common import run_one_game

    # alpha: hull_damage/flooding を観測可能、communication は不可
    # beta:  communication を観測可能、hull_damage/flooding は不可
    # "morale" は両者とも観測不可 -> unanswerable
    from profiles import Role, Persona, Value, ResolvedProfile

    alpha_role = Role(
        id="safety", label="安全", schema_version="2.0",
        expertise_domains=("safety",), observation_scope=("oxygen", "power", "hull_damage", "flooding"),
        responsibility="安全", feasibility_constraints=(),
    )
    beta_role = Role(
        id="comms", label="通信", schema_version="2.0",
        expertise_domains=("comms",), observation_scope=("oxygen", "power", "communication"),
        responsibility="通信", feasibility_constraints=(),
    )
    alpha_persona = Persona(id="p1", version="1.0", communication_style="concise", evidence_demand=0.5, concession_tendency=0.5, consensus_orientation=0.5, dominance=0.5)
    beta_persona = Persona(id="p2", version="1.0", communication_style="concise", evidence_demand=0.5, concession_tendency=0.5, consensus_orientation=0.5, dominance=0.5)
    alpha_value = Value(id="v1", version="1.0", initial_priority_weights={"oxygen": 0.3, "power": 0.2, "hull_damage": 0.2, "flooding": 0.2, "communication": 0.1}, confidence=0.6, negotiable=True)
    beta_value = Value(id="v2", version="1.0", initial_priority_weights={"oxygen": 0.2, "power": 0.2, "hull_damage": 0.2, "flooding": 0.2, "communication": 0.2}, confidence=0.6, negotiable=True)
    resolved_profiles = {
        "alpha": ResolvedProfile(role=alpha_role, persona=alpha_persona, value=alpha_value, role_value_mode="soft_value"),
        "beta": ResolvedProfile(role=beta_role, persona=beta_persona, value=beta_value, role_value_mode="soft_value"),
    }

    question_count = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal question_count
        if "意思決定機会" in prompt:
            return "", '{"action":"A","reason":"vote","message":"A","ready":true}'
        if "id=v-measurement-before" in prompt:
            return "", '{"v_before":{"ordered_criteria":["oxygen","power","hull_damage","flooding","communication"],"weights":{"oxygen":0.2,"power":0.2,"hull_damage":0.2,"flooding":0.2,"communication":0.2},"confidence":0.6},"action_before":"A","reason_before":"ok"}'
        if "id=v-measurement-after" in prompt:
            return "", '{"v_after":{"ordered_criteria":["oxygen","power","hull_damage","flooding","communication"],"weights":{"oxygen":0.2,"power":0.2,"hull_damage":0.2,"flooding":0.2,"communication":0.2},"confidence":0.7},"reason_after":"ok"}'
        # alpha は常に morale (両者観測不可) への質問を出し続ける
        # beta は evidence を返す
        # observation_scope 行に hull_damage+flooding があれば alpha、communication があれば beta
        is_alpha_prompt = (
            "id=discussion-contract" in prompt
            and 'observation_scope: ["oxygen","power","hull_damage","flooding"]' in prompt
        )
        if is_alpha_prompt:
            question_count += 1
            return (
                "",
                '{"speech_act":"question_objection","message":"morale?","action":"A","reason":"確認",'
                '"addressed_to":"beta","reply_to_message_id":null,"requested_fields":["morale"]}',
            )
        return (
            "",
            '{"speech_act":"evidence","message":"ok","action":"A","reason":"ok",'
            '"addressed_to":null,"reply_to_message_id":null}',
        )

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None,
        None,
        "control",
        42,
        {"alpha": "alpha", "beta": "beta"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_new_tokens=96,
        max_discussion_turns=6,
        discussion_token_budget=2048,
        evaluator_rollouts=2,
        max_decision_opportunities=1,
        scenario_id="comms_favored",
        role_value_mode="soft_value",
        resolved_profiles=resolved_profiles,
    )
    assert rows
    first = rows[0]
    transcript = json.loads(first["discussion_transcript"])
    # 質問は unanswerable として閉じられる
    unanswerable_entries = [t for t in transcript if t.get("closed_as_unanswerable")]
    assert len(unanswerable_entries) >= 1
    assert unanswerable_entries[0].get("unanswerable_reason") == "neither_agent_observes_requested_fields"
    assert first["unanswerable_question_count"] >= 1
    # 同じspeakerが同じrequested_fieldsを再送した場合はduplicate_questionとして拒否される
    # 2回目以降のmorale質問はclosed_questionsにあるためduplicate扱い
    assert first["duplicate_question_count"] >= 1
    # §6.6.4: question_count は unanswerable + duplicate を含む全質問試行を分母とする
    assert first["question_count"] >= 2
    assert first["question_count"] >= first["unanswerable_question_count"] + first["duplicate_question_count"]
    # unanswerable質問の requires_response は False である（回答を求めない）
    assert unanswerable_entries[0].get("requires_response") is False
    # duplicate_question_rate が NaN にならないことを検証
    from hivc_sim.turn_game_metrics import duplicate_question_rate
    rate = duplicate_question_rate([first])
    assert rate == rate  # NaN check (NaN != NaN)
    assert rate >= 0.0


def test_scope_self_observable_question_is_closed(monkeypatch) -> None:
    """質問者自身だけが観測可能なfieldへの質問は self_observable_question として閉じる。

    alpha が hull_damage (alphaのみ観測可能、betaは観測不可) への質問を出した場合、
    相手に聞く必要がないので self_observable_question として閉じる。
    """
    import json
    from scripts.llm_turn_game_common import run_one_game
    from profiles import Role, Persona, Value, ResolvedProfile

    alpha_role = Role(
        id="safety", label="安全", schema_version="2.0",
        expertise_domains=("safety",), observation_scope=("oxygen", "power", "hull_damage", "flooding"),
        responsibility="安全", feasibility_constraints=(),
    )
    beta_role = Role(
        id="comms", label="通信", schema_version="2.0",
        expertise_domains=("comms",), observation_scope=("oxygen", "power", "communication"),
        responsibility="通信", feasibility_constraints=(),
    )
    alpha_persona = Persona(id="p1", version="1.0", communication_style="concise", evidence_demand=0.5, concession_tendency=0.5, consensus_orientation=0.5, dominance=0.5)
    beta_persona = Persona(id="p2", version="1.0", communication_style="concise", evidence_demand=0.5, concession_tendency=0.5, consensus_orientation=0.5, dominance=0.5)
    alpha_value = Value(id="v1", version="1.0", initial_priority_weights={"oxygen": 0.3, "power": 0.2, "hull_damage": 0.2, "flooding": 0.2, "communication": 0.1}, confidence=0.6, negotiable=True)
    beta_value = Value(id="v2", version="1.0", initial_priority_weights={"oxygen": 0.2, "power": 0.2, "hull_damage": 0.2, "flooding": 0.2, "communication": 0.2}, confidence=0.6, negotiable=True)
    resolved_profiles = {
        "alpha": ResolvedProfile(role=alpha_role, persona=alpha_persona, value=alpha_value, role_value_mode="soft_value"),
        "beta": ResolvedProfile(role=beta_role, persona=beta_persona, value=beta_value, role_value_mode="soft_value"),
    }

    question_count = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal question_count
        if "意思決定機会" in prompt:
            return "", '{"action":"A","reason":"vote","message":"A","ready":true}'
        if "id=v-measurement-before" in prompt:
            return "", '{"v_before":{"ordered_criteria":["oxygen","power","hull_damage","flooding","communication"],"weights":{"oxygen":0.2,"power":0.2,"hull_damage":0.2,"flooding":0.2,"communication":0.2},"confidence":0.6},"action_before":"A","reason_before":"ok"}'
        if "id=v-measurement-after" in prompt:
            return "", '{"v_after":{"ordered_criteria":["oxygen","power","hull_damage","flooding","communication"],"weights":{"oxygen":0.2,"power":0.2,"hull_damage":0.2,"flooding":0.2,"communication":0.2},"confidence":0.7},"reason_after":"ok"}'
        # alpha が hull_damage (alphaのみ観測可能) への質問を出す（1回だけ）
        is_alpha_prompt = (
            "id=discussion-contract" in prompt
            and 'observation_scope: ["oxygen","power","hull_damage","flooding"]' in prompt
        )
        if is_alpha_prompt and question_count == 0:
            question_count += 1
            return (
                "",
                '{"speech_act":"question_objection","message":"hull_damage?","action":"A","reason":"確認",'
                '"addressed_to":"beta","reply_to_message_id":null,"requested_fields":["hull_damage"]}',
            )
        return (
            "",
            '{"speech_act":"evidence","message":"ok","action":"A","reason":"ok",'
            '"addressed_to":null,"reply_to_message_id":null}',
        )

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None,
        None,
        "control",
        42,
        {"alpha": "alpha", "beta": "beta"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_new_tokens=96,
        max_discussion_turns=4,
        discussion_token_budget=2048,
        evaluator_rollouts=2,
        max_decision_opportunities=1,
        scenario_id="comms_favored",
        role_value_mode="soft_value",
        resolved_profiles=resolved_profiles,
    )
    assert rows
    first = rows[0]
    transcript = json.loads(first["discussion_transcript"])
    # 質問は self_observable として閉じられる
    self_observable_entries = [t for t in transcript if t.get("closed_as_self_observable")]
    assert len(self_observable_entries) >= 1
    assert self_observable_entries[0].get("self_observable_reason") == "only_speaker_observes_requested_fields"
    assert self_observable_entries[0]["requested_fields"] == ["hull_damage"]
    assert first["self_observable_question_count"] >= 1
    # §6.6.4: question_count は self_observable 質問も分母に含む
    assert first["question_count"] >= 1
    assert first["question_count"] >= first["self_observable_question_count"]


def test_scope_partial_unanswerable_with_both_observe_not_full_unanswerable(monkeypatch) -> None:
    """一部fieldが両者観測可能で一部が両者観測不能の場合、全体をunanswerableにせずpartial_fieldsに記録する。

    requested_fields=["oxygen", "morale"] で oxygen を両者が観測可能、morale を両者とも観測不可の場合、
    質問全体は unanswerable にならず、morale が unanswerable_partial_fields に記録される。
    """
    import json
    from scripts.llm_turn_game_common import run_one_game
    from profiles import Role, Persona, Value, ResolvedProfile

    # 両者とも oxygen を観測可能、morale は両者とも観測不可
    alpha_role = Role(
        id="safety", label="安全", schema_version="2.0",
        expertise_domains=("safety",), observation_scope=("oxygen", "power", "hull_damage"),
        responsibility="安全", feasibility_constraints=(),
    )
    beta_role = Role(
        id="comms", label="通信", schema_version="2.0",
        expertise_domains=("comms",), observation_scope=("oxygen", "power", "communication"),
        responsibility="通信", feasibility_constraints=(),
    )
    alpha_persona = Persona(id="p1", version="1.0", communication_style="concise", evidence_demand=0.5, concession_tendency=0.5, consensus_orientation=0.5, dominance=0.5)
    beta_persona = Persona(id="p2", version="1.0", communication_style="concise", evidence_demand=0.5, concession_tendency=0.5, consensus_orientation=0.5, dominance=0.5)
    alpha_value = Value(id="v1", version="1.0", initial_priority_weights={"oxygen": 0.3, "power": 0.2, "hull_damage": 0.2, "flooding": 0.2, "communication": 0.1}, confidence=0.6, negotiable=True)
    beta_value = Value(id="v2", version="1.0", initial_priority_weights={"oxygen": 0.2, "power": 0.2, "hull_damage": 0.2, "flooding": 0.2, "communication": 0.2}, confidence=0.6, negotiable=True)
    resolved_profiles = {
        "alpha": ResolvedProfile(role=alpha_role, persona=alpha_persona, value=alpha_value, role_value_mode="soft_value"),
        "beta": ResolvedProfile(role=beta_role, persona=beta_persona, value=beta_value, role_value_mode="soft_value"),
    }

    question_count = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal question_count
        if "意思決定機会" in prompt:
            return "", '{"action":"A","reason":"vote","message":"A","ready":true}'
        if "id=v-measurement-before" in prompt:
            return "", '{"v_before":{"ordered_criteria":["oxygen","power","hull_damage","flooding","communication"],"weights":{"oxygen":0.2,"power":0.2,"hull_damage":0.2,"flooding":0.2,"communication":0.2},"confidence":0.6},"action_before":"A","reason_before":"ok"}'
        if "id=v-measurement-after" in prompt:
            return "", '{"v_after":{"ordered_criteria":["oxygen","power","hull_damage","flooding","communication"],"weights":{"oxygen":0.2,"power":0.2,"hull_damage":0.2,"flooding":0.2,"communication":0.2},"confidence":0.7},"reason_after":"ok"}'
        # alpha が oxygen(両者観測可能) + morale(両者観測不可) への質問を出す（1回だけ）
        is_alpha_prompt = (
            "id=discussion-contract" in prompt
            and 'observation_scope: ["oxygen","power","hull_damage"]' in prompt
        )
        if is_alpha_prompt and question_count == 0:
            question_count += 1
            return (
                "",
                '{"speech_act":"question_objection","message":"oxygen and morale?","action":"A","reason":"確認",'
                '"addressed_to":"beta","reply_to_message_id":null,"requested_fields":["oxygen","morale"]}',
            )
        # beta が回答
        return (
            "",
            '{"speech_act":"evidence","message":"回答","action":"A","reason":"回答",'
            '"addressed_to":null,"reply_to_message_id":"1"}',
        )

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None,
        None,
        "control",
        42,
        {"alpha": "alpha", "beta": "beta"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_new_tokens=96,
        max_discussion_turns=4,
        discussion_token_budget=2048,
        evaluator_rollouts=2,
        max_decision_opportunities=1,
        scenario_id="comms_favored",
        role_value_mode="soft_value",
        resolved_profiles=resolved_profiles,
    )
    assert rows
    first = rows[0]
    transcript = json.loads(first["discussion_transcript"])
    # 質問は unanswerable として閉じられない（oxygen が両者観測可能だから）
    unanswerable_entries = [t for t in transcript if t.get("closed_as_unanswerable")]
    assert len(unanswerable_entries) == 0
    # 質問は通常の質問として記録され、morale が partial_fields に記録される
    questions = [t for t in transcript if t.get("requires_response") and not t.get("closed_as_self_observable")]
    assert len(questions) >= 1
    assert "morale" in questions[0].get("unanswerable_partial_fields", [])
    assert first["unanswerable_question_count"] == 0
