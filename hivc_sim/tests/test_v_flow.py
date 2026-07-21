import math

import numpy as np

from scripts.llm_turn_game_common import (
    DEFAULT_VALUE_CRITERIA_SCHEMA,
    _canonical_json,
    _profile_sha256,
    append_profile_assignment,
    build_value_manifest,
    decision_opportunity_prompt,
    decision_support_block,
    discussion_prompt,
    ensure_unique_v_proposal_id,
    extract_json_discussion,
    format_persona,
    format_transcript_text,
    final_vote_repair_feedback,
    parse_v_negotiation,
    resolve_v_star,
    v_alignment_required,
    v_alignment_distance,
    verify_vote_v_star_consistency,
    v_measurement_prompt,
    v_proposal_required_prompt,
    v_proposal_response_prompt,
    _normalize_v,
    _normalize_v_proposal,
)
from turn_game import Action, Event, GameState, preview_action_safety, step


# 共通Vオントロジーに合わせた完全criteria例
_O = "oxygen"
_P = "power"
_H = "hull_damage"
_F = "flooding"
_C = "communication"
FULL_CRITERIA = [_O, _P, _H, _F, _C]


def test_preview_action_safety_rejects_guaranteed_power_loss() -> None:
    state = GameState(power=2, oxygen=6, current_event=Event.NONE)
    safe_a, reason_a, preview_a = preview_action_safety(state, Action.STABILIZE_OXYGEN)
    safe_b, reason_b, preview_b = preview_action_safety(state, Action.REPAIR_POWER)
    assert safe_a is False
    assert reason_a == "guaranteed_loss_power"
    assert preview_a.outcome == "loss_power"
    assert safe_b is True
    assert reason_b == ""
    assert not preview_b.outcome.startswith("loss_")


def test_escape_preview_matches_fixed_failed_escape_effects() -> None:
    state = GameState(
        oxygen=4, power=4, hull_damage=1, flooding=1,
        pod_readiness=0, pod_integrity=1, current_event=Event.NONE,
    )
    safe, reason, preview = preview_action_safety(state, Action.EXECUTE_ESCAPE)
    actual = step(state, Action.EXECUTE_ESCAPE, np.random.default_rng(3)).state_after
    assert safe is False
    assert reason == "escape_conditions_not_met"
    for field in (
        "oxygen", "power", "hull_damage", "flooding", "communication",
        "pod_readiness", "pod_integrity", "morale", "severe_risk_count", "outcome",
    ):
        assert getattr(preview, field) == getattr(actual, field)


def test_vote_consistency_does_not_trust_model_self_report() -> None:
    v_star = {
        "ordered_criteria": [_P, _O, _H, _F, _C],
        "weights": {_P: 0.50, _O: 0.20, _H: 0.10, _F: 0.10, _C: 0.10},
    }
    # false申告でも、参照IDと実際の行動効果が整合すればシステム判定はTrue。
    assert verify_vote_v_star_consistency(
        Action.REPAIR_POWER, "Bを選択する", "p1", False, "p1", v_star
    )


def test_decision_prompt_omits_v_consistency_self_declaration() -> None:
    prompt = decision_opportunity_prompt(
        "alpha", "persona", None, GameState(), [], "hivc_d", 1, 1,
        v_state={"v_star_status": "accepted", "v_star_id": "p1", "v_star": {
            "ordered_criteria": FULL_CRITERIA,
            "weights": {_O: .2, _P: .2, _H: .2, _F: .2, _C: .2},
        }},
    )
    assert '"v_star_id":"p1"' in prompt
    assert '"v_star_consistent":true' not in prompt
    assert "整合性を自己申告する必要はありません" in prompt


def test_v_alignment_is_required_only_when_actions_conflict() -> None:
    alpha_v = {
        "ordered_criteria": FULL_CRITERIA,
        "priority_levels": {_O: "high", _P: "mid", _H: "mid", _F: "low", _C: "low"},
    }
    beta_v = {
        "ordered_criteria": list(reversed(FULL_CRITERIA)),
        "priority_levels": {_O: "low", _P: "low", _H: "mid", _F: "mid", _C: "high"},
    }
    required, reasons = v_alignment_required(
        Action.REPAIR_POWER, Action.REPAIR_POWER, alpha_v, beta_v
    )
    assert required is False
    assert reasons == []

    required, reasons = v_alignment_required(
        Action.REPAIR_POWER, Action.REPAIR_COMMUNICATION, alpha_v, beta_v
    )
    assert required is True
    assert reasons == ["action_before_mismatch"]


def test_second_unresolved_decision_prompt_requests_action_not_value_reconciliation() -> None:
    prompt = decision_opportunity_prompt(
        "alpha", "persona", None, GameState(), [], "hivc_d", 2, 3,
        v_state={"v_star_status": "unresolved", "v_star_id": "", "v_star": None},
    )
    assert "id=action-reconciliation" in prompt
    assert "V*が未成立でも、Valueそのものを一致させる必要はありません" in prompt
    assert "同じ安全なActionへ合意できるか" in prompt


def test_duplicate_proposal_id_is_repaired_deterministically() -> None:
    first = {"proposal_id": "p1", "ordered_criteria": FULL_CRITERIA, "scope": "turn", "speaker": "alpha", "message_index": 1}
    duplicate = {"proposal_id": "p1", "ordered_criteria": list(reversed(FULL_CRITERIA)), "scope": "turn", "speaker": "beta", "message_index": 2}
    repaired, original_id, repaired_id = ensure_unique_v_proposal_id(duplicate, [first])
    repaired_again, _, repaired_id_again = ensure_unique_v_proposal_id(duplicate, [first])
    assert original_id == "p1"
    assert repaired_id != original_id
    assert repaired["proposal_id"] == repaired_id
    assert repaired_id_again == repaired_id
    assert repaired_again == repaired


def test_safe_v_tradeoff_is_not_retried_toward_numeric_optimum(monkeypatch) -> None:
    from scripts.llm_turn_game_common import run_one_game

    state = GameState(
        turn=4, oxygen=8, power=8, hull_damage=1, flooding=1,
        communication=0, current_event=Event.NONE, scenario_id="comms_favored",
    )
    monkeypatch.setattr("scripts.llm_turn_game_common.initial_state", lambda seed, scenario_id=None: state)
    weights = {_P: .5, _O: .2, _H: .1, _F: .1, _C: .1}
    before = (
        f'{{"v_before":{{"ordered_criteria":{_canonical_json([_P, _O, _H, _F, _C])},'
        f'"weights":{_canonical_json(weights)},"confidence":0.8}},'
        '"action_before":"B","reason_before":"power"}'
    )
    after = (
        f'{{"v_after":{{"ordered_criteria":{_canonical_json([_P, _O, _H, _F, _C])},'
        f'"weights":{_canonical_json(weights)},"confidence":0.8}},"reason_after":"power"}}'
    )
    discussion_calls = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal discussion_calls
        if "id=v-measurement-before" in prompt:
            return "", before
        if "id=v-measurement-after" in prompt:
            return "", after
        if "id=final-vote-repair" in prompt:
            return "", '{"action":"B","reason":"powerを回復するBを選択する","ready":true,"v_star_id":"p1"}'
        if "id=decision-contract" in prompt:
            return "", '{"action":"C","reason":"通信を直すCを選択する","ready":true,"v_star_id":"p1"}'
        discussion_calls += 1
        if discussion_calls == 1:
            return "", (
                '{"speech_act":"tradeoff","message":"power優先を提案","action":"B","reason":"power",'
                '"addressed_to":null,"reply_to_message_id":null,'
                f'"v_proposal":{{"proposal_id":"p1","ordered_criteria":{_canonical_json([_P, _O, _H, _F, _C])},'
                f'"weights":{_canonical_json(weights)},"scope":"turn"}},'
                '"v_star_response":{"response":"accept","proposal_id":"p1"}}'
            )
        return "", (
            '{"speech_act":"concession_integration","message":"p1を受諾","action":"B","reason":"power",'
            '"addressed_to":null,"reply_to_message_id":null,'
            '"v_star_response":{"response":"accept","proposal_id":"p1"}}'
        )

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None, None, "hivc_d", 42,
        {"alpha": "alpha", "beta": "beta"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_discussion_turns=2,
        discussion_token_budget=1024,
        evaluator_rollouts=1,
        max_decision_opportunities=1,
        max_final_vote_retries=1,
        role_value_mode="soft_value",
    )
    first = rows[0]
    assert first["v_star_status"] == "accepted"
    assert first["alpha_vote"] == "C"
    assert first["beta_vote"] == "C"
    assert first["v_star_action_consistency"] is True
    assert first["final_vote_retry_count"] == 0
    assert first["rejected_final_vote_count"] == 0


def test_v_measurement_prompts_require_qualitative_priority_levels() -> None:
    prompts = {
        phase: v_measurement_prompt(
            "alpha",
            GameState(current_event=Event.NONE),
            phase=phase,
            persona="safety persona",
            persona_params={"priority_weights": {_O: 0.4, _P: 0.1, _H: 0.2, _F: 0.2, _C: 0.1}},
        )
        for phase in ("before", "after")
    }
    uniform = '"weights":{"communication":0.2,"flooding":0.2,"hull_damage":0.2,"oxygen":0.2,"power":0.2}'
    for phase, prompt in prompts.items():
        assert uniform not in prompt
        assert '"action_before":"A"' not in prompt
        assert _canonical_json(FULL_CRITERIA) in prompt
        assert '"priority_levels"' in prompt and '"confidence_level"' in prompt
        assert '"weights"' not in prompt and '"confidence"' not in prompt
        assert "high / mid / low" in prompt
        assert "小数、百分率、合計値による重み付けは禁止" in prompt
        assert all(f'"{criterion}":"<high|mid|low>"' in prompt for criterion in FULL_CRITERIA)
        assert "ROLE_PERSONA_INITIAL_VALUE" in prompt
        assert f"v-measurement-{phase}" in prompt
    assert '"v_before"' in prompts["before"]
    assert '"action_before"' in prompts["before"]
    assert '"reason_before"' in prompts["before"]
    assert '"v_after"' in prompts["after"]
    assert '"reason_after"' in prompts["after"]


def test_qualitative_v_is_canonical_and_rejects_numeric_levels() -> None:
    levels = {_O: "high", _P: "mid", _H: "mid", _F: "low", _C: "low"}
    normalized = _normalize_v({
        "ordered_criteria": FULL_CRITERIA,
        "priority_levels": levels,
        "confidence_level": "mid",
    })
    assert normalized == {
        "ordered_criteria": FULL_CRITERIA,
        "priority_levels": levels,
        "confidence_level": "mid",
    }
    assert _normalize_v({
        "ordered_criteria": FULL_CRITERIA,
        "priority_levels": {**levels, _O: "0.8"},
    }) is None


def test_soft_value_persona_hides_numeric_weights_from_model() -> None:
    text = format_persona("alpha", "safety", {
        "_resolved_profile": {
            "role_value_mode": "soft_value",
            "role": {"id": "safety"},
            "persona": {"id": "evidence"},
            "value": {
                "id": "v1",
                "initial_priority_weights": {_O: 0.4, _P: 0.2, _H: 0.2, _F: 0.1, _C: 0.1},
                "confidence": 0.65,
                "negotiable": True,
            },
        }
    })
    assert "initial_priority_levels" in text
    assert '"oxygen":"high"' in text
    assert '"confidence_level":"mid"' in text
    assert "initial_priority_weights" not in text
    assert "0.65" not in text and "0.4" not in text


def test_all_model_prompts_include_action_catalog_transition_and_projections() -> None:
    state = GameState(
        oxygen=6, power=3, hull_damage=1, flooding=1,
        current_event=Event.RELAY_SHORT,
    )
    proposal = {"proposal_id": "p1", "ordered_criteria": FULL_CRITERIA, "scope": "turn"}
    prompts = [
        v_measurement_prompt("alpha", state, phase="before"),
        v_measurement_prompt("alpha", state, phase="after"),
        discussion_prompt("alpha", "persona", None, state, [], 2),
        decision_opportunity_prompt("alpha", "persona", None, state, [], "control", 1, 1),
        v_proposal_required_prompt("alpha", "persona", None, state, "hivc_d", [], None),
        v_proposal_response_prompt("beta", "persona", None, state, "hivc_d", [], proposal, None),
    ]
    for prompt in prompts:
        assert "【ACTION_CATALOG id=action-catalog】" in prompt
        assert "A. 酸素供給を安定化" in prompt
        assert "F. 自力脱出を実行" in prompt
        assert "Actionより先にターン開始消費が適用される: oxygen -1, power -1" in prompt
        assert "中継器短絡" not in prompt or "power -1, communication -1" in prompt
        assert "【PROJECTED_STATE_AFTER id=projected-state-after】" in prompt
        assert all(f"{action.value}: [" in prompt for action in Action)


def test_final_vote_repair_feedback_contains_arithmetic_and_safe_candidates() -> None:
    state = GameState(oxygen=6, power=3, current_event=Event.RELAY_SHORT)
    feedback = final_vote_repair_feedback(state, Action.STABILIZE_OXYGEN, "alpha")
    assert feedback.startswith("guaranteed_loss_power;")
    assert "turn_start(oxygen -1, power -1)" in feedback
    assert "event(power -1, communication -1)" in feedback
    assert "action(oxygen +3, power -1, morale +2)" in feedback
    assert "projected_state_after=" in feedback
    assert "safe_candidates=B" in feedback


def test_projected_state_does_not_reveal_role_hidden_failure_field() -> None:
    state = GameState(
        oxygen=6, power=6, hull_damage=3, flooding=1,
        current_event=Event.HULL_FRACTURE,
    )
    role = {"observation_scope": ["oxygen", "power"]}
    block = decision_support_block(state, "beta", role)
    assert "hull_damage=hidden" in block
    assert "outcome=loss_hull" not in block
    assert "guaranteed_loss_hull" not in block
    assert "unsafe:hidden_constraint" in block


def test_v_response_prompt_disables_counter_after_round_limit() -> None:
    prompt = v_proposal_response_prompt(
        "beta", "persona", None, GameState(current_event=Event.NONE), "hivc_d", [],
        {"proposal_id": "p1", "ordered_criteria": FULL_CRITERIA, "scope": "turn"},
        None, counter_allowed=False,
    )
    assert "counter上限に達しています" in prompt
    assert '"response":"accept|reject"' in prompt
    assert '"counter_proposal"' not in prompt


def test_nested_v_proposal_and_response_parse() -> None:
    raw = (
        '{"speech_act":"tradeoff","message":"基準案","action":"B","reason":"比較",'
        '"addressed_to":null,"reply_to_message_id":null,'
        f'"v_proposal":{{"proposal_id":"p1","ordered_criteria":{_canonical_json(FULL_CRITERIA)},"scope":"turn"}},'
        '"v_star_response":{"response":"accept","proposal_id":"p1"}}'
    )
    speech_act, message, _, _, _, _, _ = extract_json_discussion(raw)
    assert speech_act is not None and speech_act.value == "tradeoff"
    assert message == "基準案"

    import json

    proposal, response = parse_v_negotiation(json.loads(raw), "alpha", "1")
    assert proposal == {
        "proposal_id": "p1",
        "ordered_criteria": FULL_CRITERIA,
        "scope": "turn",
        "message_index": 1,
    }
    assert response == {"response": "accept", "proposal_id": "p1", "message_index": 1}


def test_v_star_requires_matching_explicit_acceptance() -> None:
    proposal = {"proposal_id": "p1", "ordered_criteria": FULL_CRITERIA, "scope": "turn", "message_index": 1}
    status, _, _, reason = resolve_v_star([proposal], {"alpha": [{"response": "accept", "proposal_id": "p1", "message_index": 2}], "beta": []})
    assert status == "unresolved"
    assert reason == "missing_matching_explicit_acceptance"

    status, proposal_id, accepted, reason = resolve_v_star(
        [proposal],
        {
            "alpha": [{"response": "accept", "proposal_id": "p1", "message_index": 2}],
            "beta": [{"response": "accept", "proposal_id": "p1", "message_index": 3}],
        },
    )
    assert (status, proposal_id, accepted, reason) == ("accepted", "p1", proposal, "")

    conflicting = {**proposal, "ordered_criteria": [_P, _O, _H, _F, _C]}
    status, _, _, reason = resolve_v_star(
        [proposal, conflicting],
        {
            "alpha": [{"response": "accept", "proposal_id": "p1", "message_index": 2}],
            "beta": [{"response": "accept", "proposal_id": "p1", "message_index": 3}],
        },
    )
    assert status == "unresolved"
    assert reason == "proposal_id_content_mismatch"

    status, _, _, _ = resolve_v_star(
        [proposal],
        {
            "alpha": [{"response": "accept", "proposal_id": "p1", "message_index": 0}],
            "beta": [{"response": "accept", "proposal_id": "p1", "message_index": 3}],
        },
    )
    assert status == "unresolved"


def test_v_transcript_counter_and_finite_weights() -> None:
    item = {
        "speaker": "alpha",
        "message": "counter",
        "v_proposal": {"proposal_id": "p1", "ordered_criteria": FULL_CRITERIA},
        "v_star_response": {
            "response": "counter",
            "proposal_id": "p0",
            "counter_proposal": {"proposal_id": "p1", "ordered_criteria": FULL_CRITERIA},
        },
    }
    text = format_transcript_text([item])
    assert "v_proposal=" in text and "counter_proposal" in text
    weights = {_O: 0.2, _P: 0.2, _H: 0.2, _F: 0.2, _C: 0.2}
    weights[_P] = float("nan")
    assert _normalize_v_proposal(
        {"proposal_id": "bad", "ordered_criteria": FULL_CRITERIA, "weights": weights},
        "fallback",
    ) is None


def test_vote_consistency_allows_safe_tradeoffs_without_numeric_ranking() -> None:
    v_star = {
        "ordered_criteria": [_P, _O, _H, _F, _C],
        "weights": {_P: 0.50, _O: 0.20, _H: 0.10, _F: 0.10, _C: 0.10},
    }
    assert verify_vote_v_star_consistency(
        Action.REPAIR_POWER, "現在の最善策を採る", "p1", True, "p1", v_star
    )
    assert verify_vote_v_star_consistency(
        Action.REPAIR_COMMUNICATION, "powerを最優先する", "p1", True, "p1", v_star
    )


def test_vote_consistency_rejects_reason_with_conflicting_committed_action() -> None:
    v_star = {
        "ordered_criteria": [_P, _O, _H, _F, _C],
        "weights": {_P: 0.50, _O: 0.20, _H: 0.10, _F: 0.10, _C: 0.10},
    }
    assert not verify_vote_v_star_consistency(
        Action.REPAIR_POWER,
        "Bを採用すると述べたが、最終的にはAを選択する",
        "p1",
        True,
        "p1",
        v_star,
    )


def test_vote_consistency_allows_pod_action_without_ontology_bypass() -> None:
    v_star = {
        "ordered_criteria": [_P, _O, _H, _F, _C],
        "weights": {_P: 0.50, _O: 0.20, _H: 0.10, _F: 0.10, _C: 0.10},
    }
    assert verify_vote_v_star_consistency(
        Action.PREP_POD, "Eを選択する", "p1", True, "p1", v_star
    ) is True


def test_vote_consistency_excludes_unsafe_competitors_after_turn_start_effects() -> None:
    """実smoke turn 3: unsafeなBをutility最大候補に含めて安全なAを拒否しない。"""
    state = GameState(
        turn=3,
        oxygen=1,
        power=1,
        hull_damage=2,
        flooding=0,
        communication=2,
        pod_readiness=2,
        pod_integrity=0,
        rescue_eta=1,
        current_event=Event.BACKUP_POWER_FOUND,
        scenario_id="ambiguous",
    )
    v_star = {
        "ordered_criteria": [_O, _P, _H, _F, _C],
        "weights": {_O: 0.2, _P: 0.2, _H: 0.2, _F: 0.2, _C: 0.2},
    }
    assert verify_vote_v_star_consistency(
        Action.STABILIZE_OXYGEN, "Aを選択する", "p1", None, "p1", v_star, state
    )
    assert not verify_vote_v_star_consistency(
        Action.REPAIR_POWER, "Bを選択する", "p1", None, "p1", v_star, state
    )


def test_vote_consistency_uses_relay_short_post_start_baseline() -> None:
    """安全なBを許容し、unsafeなAは独立した安全性検証で拒否する。"""
    state = GameState(
        turn=2,
        oxygen=3,
        power=3,
        hull_damage=2,
        flooding=2,
        communication=3,
        pod_readiness=2,
        pod_integrity=0,
        rescue_eta=2,
        current_event=Event.RELAY_SHORT,
        scenario_id="ambiguous",
    )
    v_star = {
        "ordered_criteria": [_C, _O, _P, _H, _F],
        "weights": {_C: 0.45, _O: 0.2, _P: 0.15, _H: 0.1, _F: 0.1},
    }
    assert verify_vote_v_star_consistency(
        Action.REPAIR_POWER, "Bを選択する", "p1", None, "p1", v_star, state
    )
    assert not preview_action_safety(state, Action.STABILIZE_OXYGEN)[0]


def test_vote_consistency_allows_flashx_turn2_communication_tradeoff() -> None:
    """oxygenに余裕があるsignal windowでは、V*最大重みがoxygenでもCを許容する。"""
    state = GameState(
        turn=2,
        oxygen=6,
        power=3,
        hull_damage=2,
        flooding=3,
        communication=0,
        pod_readiness=2,
        pod_integrity=0,
        current_event=Event.SIGNAL_WINDOW,
        scenario_id="ambiguous",
    )
    v_star = {
        "ordered_criteria": [_O, _C, _P, _H, _F],
        "weights": {_O: 0.4, _C: 0.3, _P: 0.1, _H: 0.1, _F: 0.1},
    }
    assert preview_action_safety(state, Action.REPAIR_COMMUNICATION)[0]
    assert verify_vote_v_star_consistency(
        Action.REPAIR_COMMUNICATION,
        "oxygenには余裕があり、通信窓を使うためCを選択する",
        "p1", None, "p1", v_star, state,
    )


def test_vote_consistency_allows_flashx_turn4_flooding_tradeoff_without_weights() -> None:
    """weights省略時も順位を数値重みに変換せず、安全なDを許容する。"""
    state = GameState(
        turn=4,
        oxygen=7,
        power=3,
        hull_damage=3,
        flooding=4,
        communication=0,
        pod_readiness=2,
        pod_integrity=0,
        current_event=Event.NONE,
        scenario_id="ambiguous",
    )
    v_star = {"ordered_criteria": [_O, _P, _H, _F, _C]}
    assert preview_action_safety(state, Action.SEAL_FLOODING)[0]
    assert verify_vote_v_star_consistency(
        Action.SEAL_FLOODING,
        "浸水4への対処としてDを選択する",
        "p1", None, "p1", v_star, state,
    )


def test_unresolved_v_is_not_inserted_as_accepted() -> None:
    prompt = decision_opportunity_prompt(
        "alpha",
        "persona",
        None,
        GameState(current_event=Event.NONE),
        [],
        "hivc_d",
        1,
        2,
        v_state={"current_v": None, "v_star_status": "unresolved", "v_star_id": "", "v_star": None},
    )
    assert "受諾済みV*はありません" in prompt
    assert '"v_star_id"' not in prompt


def test_v_distance_and_manifest_hash_match_runtime_body() -> None:
    first = {"weights": {"power": 0.7, "oxygen": 0.3}}
    second = {"weights": {"power": 0.2, "oxygen": 0.8}}
    assert math.isclose(v_alignment_distance(first, second), 1.0)

    manifest = build_value_manifest(
        {"model_path": "model", "seed": 42, "games": 2},
        {"alpha": "safety", "beta": "comms"},
        {"alpha": None, "beta": None},
        {"alpha": "r1", "beta": "r2"},
        role_value_mode="expertise_only",
        framework_ids=["control", "hivc_d"],
    )
    alpha = manifest["role_profiles"]["alpha"]
    assert alpha["sha256"] == _profile_sha256(alpha["body"])
    assert manifest["experiment_config_sha256"] == _profile_sha256(manifest["experiment_config"])
    append_profile_assignment(
        manifest,
        43,
        {"alpha": "safety", "beta": "comms"},
        {"alpha": None, "beta": None},
        {"alpha": "r1", "beta": "r2"},
    )
    assignment = manifest["game_profile_assignments"][0]["agents"]["alpha"]
    assert assignment["sha256"] == _profile_sha256(assignment["body"])


def test_run_one_game_carries_v_state_and_measures_after_vote(monkeypatch) -> None:
    from scripts.llm_turn_game_common import run_one_game

    seen_discussion: list[str] = []
    seen_after: list[str] = []

    before_v = (
        f'{{"v_before":{{"ordered_criteria":{_canonical_json([_P, _O, _H, _F, _C])},'
        f'"weights":{_canonical_json({_P:0.3, _O:0.25, _H:0.2, _F:0.15, _C:0.1})},"confidence":0.6}},'
        f'"action_before":"B","reason_before":"power"}}'
    )
    after_v = (
        f'{{"v_after":{{"ordered_criteria":{_canonical_json([_P, _O, _H, _F, _C])},'
        f'"weights":{_canonical_json({_P:0.3, _O:0.25, _H:0.2, _F:0.15, _C:0.1})},"confidence":0.7}},'
        f'"reason_after":"power"}}'
    )
    proposal_json = f'"v_proposal":{{"proposal_id":"p1","ordered_criteria":{_canonical_json([_P, _O, _H, _F, _C])},"scope":"turn"}}'

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        if "id=v-measurement-before" in prompt:
            return "", before_v
        if "id=v-measurement-after" in prompt:
            seen_after.append(prompt)
            return "", after_v
        if "id=decision-contract" in prompt:
            return "", (
                '{"action":"B","reason":"power を最優先",'
                '"ready":true,"v_star_id":"p1","v_star_consistent":true}'
            )
        seen_discussion.append(prompt)
        if "name: alpha" in prompt or "alpha persona" in prompt:
            return "", (
                '{"speech_act":"tradeoff","message":"proposal","action":"B","reason":"power",'
                '"addressed_to":null,"reply_to_message_id":null,'
                f'{proposal_json},'
                '"v_star_response":{"response":"accept","proposal_id":"p1"}}'
            )
        return "", (
            '{"speech_act":"concession_integration","message":"accept","action":"B","reason":"power",'
            '"addressed_to":null,"reply_to_message_id":null,'
            '"v_star_response":{"response":"accept","proposal_id":"p1"}}'
        )

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None,
        None,
        "hivc_d",
        42,
        {"alpha": "alpha persona", "beta": "beta persona"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_discussion_turns=2,
        discussion_token_budget=1024,
        evaluator_rollouts=2,
        max_decision_opportunities=2,
        role_value_mode="soft_value",
        scenario_id="comms_favored",
    )
    assert rows and all(row["v_star_status"] == "accepted" for row in rows)
    # V*は数値最適Actionを強制しないため、安全で参照・形式が正しいBは
    # 状態ごとの別候補とのutility比較によって拒否されない。
    assert all(row["alpha_v_star_consistent"] is True for row in rows)
    assert not any("shared_v_before_and_actions" in prompt for prompt in seen_discussion)
    assert seen_after and all("FINAL_VOTES" in prompt and "DISCUSSION_HISTORY" in prompt for prompt in seen_after)


def test_control_prompts_never_disclose_opponent_private_v(monkeypatch) -> None:
    from scripts.llm_turn_game_common import run_one_game

    captured: list[str] = []
    personas = {"alpha": "alpha_private", "beta": "beta_private"}

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        if "id=v-measurement-before" in prompt:
            return "", (
                f'{{"v_before":{{"ordered_criteria":{_canonical_json(FULL_CRITERIA)},'
                f'"weights":{_canonical_json({_O:0.2, _P:0.2, _H:0.2, _F:0.2, _C:0.2})},"confidence":0.7}},'
                '"action_before":"A","reason_before":"ok"}'
            )
        if "id=v-measurement-after" in prompt:
            return "", (
                f'{{"v_after":{{"ordered_criteria":{_canonical_json(FULL_CRITERIA)},'
                f'"weights":{_canonical_json({_O:0.2, _P:0.2, _H:0.2, _F:0.2, _C:0.2})},"confidence":0.7}},'
                '"reason_after":"ok"}'
            )
        captured.append(prompt)
        if "id=decision-contract" in prompt:
            return "", '{"action":"A","reason":"ok","ready":true}'
        return "", '{"speech_act":"evidence","message":"ok","action":"A","reason":"ok","addressed_to":null,"reply_to_message_id":null}'

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    run_one_game(
        None, None, "control", 42,
        personas,
        {"alpha": None, "beta": None}, {"alpha": "a", "beta": "b"},
        max_discussion_turns=2, evaluator_rollouts=1,
        max_decision_opportunities=1, role_value_mode="soft_value",
        scenario_id="comms_favored",
    )
    alpha_prompts = [p for p in captured if "alpha_private" in p]
    beta_prompts = [p for p in captured if "beta_private" in p]
    assert alpha_prompts and all("beta_private" not in p for p in alpha_prompts)
    assert beta_prompts and all("alpha_private" not in p for p in beta_prompts)


def test_proposal_without_explicit_self_accept_is_unresolved() -> None:
    proposal = {"proposal_id": "p1", "ordered_criteria": FULL_CRITERIA, "scope": "turn", "message_index": 1}
    status, _, _, reason = resolve_v_star([proposal], {"alpha": [], "beta": []})
    assert status == "unresolved"
    assert reason == "missing_matching_explicit_acceptance"


def test_proposal_with_explicit_self_and_other_accept_is_accepted() -> None:
    proposal = {"proposal_id": "p1", "ordered_criteria": FULL_CRITERIA, "scope": "turn", "message_index": 1}
    responses = {
        "alpha": [{"response": "accept", "proposal_id": "p1", "message_index": 1}],
        "beta": [{"response": "accept", "proposal_id": "p1", "message_index": 2}],
    }
    status, proposal_id, _, reason = resolve_v_star([proposal], responses)
    assert status == "accepted"
    assert proposal_id == "p1"
    assert reason == ""


def test_counter_proposal_requires_explicit_self_accept() -> None:
    """counter提案者は自分のcounterを明示的にacceptしない限り受諾扱いにならない。"""
    original = {"proposal_id": "p0", "ordered_criteria": FULL_CRITERIA, "scope": "turn", "message_index": 1}
    counter = {"proposal_id": "p2", "ordered_criteria": FULL_CRITERIA, "scope": "turn", "message_index": 2}
    proposals = [original, counter]
    # alpha は counter を出したが自分の counter を明示的に accept していない
    responses_no_self_accept = {
        "alpha": [
            {"response": "counter", "proposal_id": "p0", "message_index": 2, "counter_proposal": counter},
        ],
        "beta": [{"response": "accept", "proposal_id": "p2", "message_index": 3}],
    }
    status, _, _, reason = resolve_v_star(proposals, responses_no_self_accept)
    assert status == "unresolved"
    assert reason == "missing_matching_explicit_acceptance"

    # alpha が自分の counter を明示的に accept すれば受諾扱いになる
    responses_with_self_accept = {
        "alpha": [
            {"response": "counter", "proposal_id": "p0", "message_index": 2, "counter_proposal": counter},
            {"response": "accept", "proposal_id": "p2", "message_index": 2},
        ],
        "beta": [{"response": "accept", "proposal_id": "p2", "message_index": 3}],
    }
    status, proposal_id, _, _ = resolve_v_star(proposals, responses_with_self_accept)
    assert status == "accepted"
    assert proposal_id == "p2"


def test_v_negotiation_budget_exhaustion_records_reason(monkeypatch) -> None:
    from scripts.llm_turn_game_common import run_one_game

    alpha_before = (
        f'{{"v_before":{{"ordered_criteria":{_canonical_json([_P, _O, _H, _F, _C])},'
        f'"weights":{_canonical_json({_P: 0.3, _O: 0.25, _H: 0.2, _F: 0.15, _C: 0.1})},"confidence":0.6}},'
        '"action_before":"B","reason_before":"power"}'
    )
    beta_before = (
        f'{{"v_before":{{"ordered_criteria":{_canonical_json([_O, _P, _H, _F, _C])},'
        f'"weights":{_canonical_json({_O: 0.3, _P: 0.25, _H: 0.2, _F: 0.15, _C: 0.1})},"confidence":0.6}},'
        '"action_before":"A","reason_before":"oxygen"}'
    )
    after_v = (
        f'{{"v_after":{{"ordered_criteria":{_canonical_json(FULL_CRITERIA)},'
        f'"weights":{_canonical_json({_O: 0.2, _P: 0.2, _H: 0.2, _F: 0.2, _C: 0.2})},"confidence":0.7}},'
        '"reason_after":"ok"}'
    )

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        if "id=v-measurement-before" in prompt:
            return "", alpha_before if "alpha です" in prompt else beta_before
        if "id=v-measurement-after" in prompt:
            return "", after_v
        if "id=decision-contract" in prompt:
            return "", '{"action":"A","reason":"ok","ready":true}'
        if "id=v-proposal-required" in prompt:
            return (
                "",
                f'{{"v_proposal":{{"proposal_id":"alpha-turn0-required","ordered_criteria":{_canonical_json(FULL_CRITERIA)},"scope":"turn"}},'
                '"v_star_response":{"response":"accept","proposal_id":"alpha-turn0-required"},"action":"A","reason":"proposal"}',
            )
        if "id=v-proposal-response" in prompt:
            # 応答者はいずれも明示的な accept を返さない
            return "", "not valid json"
        # 自由議論では V proposal を出さない
        return "", '{"speech_act":"evidence","message":"ok","action":"A","reason":"ok","addressed_to":null,"reply_to_message_id":null}'

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None,
        None,
        "hivc_d",
        42,
        {"alpha": "alpha", "beta": "beta"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_new_tokens=96,
        max_discussion_turns=2,
        discussion_token_budget=1024,
        evaluator_rollouts=2,
        max_decision_opportunities=1,
        role_value_mode="soft_value",
        scenario_id="comms_favored",
    )
    assert rows
    first = rows[0]
    assert first["v_star_status"] == "unresolved"
    assert first["v_star_failure_reason"] == "v_negotiation_budget_exhausted"
    assert first["v_star_unresolved_reason"] == "missing_matching_explicit_acceptance"
    assert first["v_proposal_required_prompt_issued"] is True
    assert first["missing_v_proposal_after_required_prompt"] is False


def test_valid_reject_terminates_proposal_without_repeated_prompt(monkeypatch) -> None:
    from scripts.llm_turn_game_common import run_one_game

    state = GameState(
        turn=4, oxygen=8, power=8, hull_damage=1, flooding=1,
        communication=0, current_event=Event.NONE, scenario_id="ambiguous",
    )
    monkeypatch.setattr("scripts.llm_turn_game_common.initial_state", lambda seed, scenario_id=None: state)
    levels = {_O: "high", _P: "mid", _H: "mid", _F: "low", _C: "low"}
    response_calls = 0
    reconciliation_prompts: list[str] = []

    def measurement(phase: str, action: str) -> str:
        reason_key = "reason_before" if phase == "before" else "reason_after"
        action_part = f',"action_before":"{action}"' if phase == "before" else ""
        return (
            f'{{"v_{phase}":{{"ordered_criteria":{_canonical_json(FULL_CRITERIA)},'
            f'"priority_levels":{_canonical_json(levels)},"confidence_level":"mid"}}'
            f'{action_part},"{reason_key}":"ok"}}'
        )

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal response_calls
        if "id=v-measurement-before" in prompt:
            return "", measurement("before", "A" if " alpha です" in prompt else "B")
        if "id=v-measurement-after" in prompt:
            return "", measurement("after", "")
        if "id=v-proposal-required" in prompt:
            return "", (
                f'{{"v_proposal":{{"proposal_id":"p1","ordered_criteria":{_canonical_json(FULL_CRITERIA)},'
                f'"priority_levels":{_canonical_json(levels)},"scope":"turn"}},'
                '"v_star_response":{"response":"accept","proposal_id":"p1"},'
                '"action":"A","reason":"proposal"}'
            )
        if "id=v-proposal-response" in prompt:
            response_calls += 1
            return "", '{"v_star_response":{"response":"reject","proposal_id":"p1"}}'
        if "id=decision-contract" in prompt:
            if "id=action-reconciliation" in prompt:
                reconciliation_prompts.append(prompt)
            action = "B" if "第 1 / 2" in prompt and "beta-persona" in prompt else "A"
            return "", f'{{"action":"{action}","reason":"安全な{action}を選択する","ready":true}}'
        return "", (
            '{"speech_act":"tradeoff","message":"actionを比較する","action":"A",'
            '"reason":"安全性を比較","addressed_to":null,"reply_to_message_id":null}'
        )

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None, None, "hivc_d", 43,
        {"alpha": "alpha-persona", "beta": "beta-persona"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_discussion_turns=2,
        discussion_token_budget=1024,
        evaluator_rollouts=1,
        max_decision_opportunities=2,
        role_value_mode="soft_value",
        scenario_id="ambiguous",
    )
    first = rows[0]
    assert response_calls == 1
    assert len(reconciliation_prompts) == 2
    assert first["v_negotiation_messages_used"] == 2
    assert first["v_star_status"] == "unresolved"
    assert first["v_star_failure_reason"] == "v_proposal_rejected"
    assert first["v_star_unresolved_reason"] == "v_proposal_rejected"
    assert '"reason":"v_proposal_rejected"' in first["v_protocol_transition_history"]
    assert first["decision_rule"] == "consensus"
    assert first["fallback_used"] is False


def test_v_measurement_retry_is_recorded(monkeypatch) -> None:
    from scripts.llm_turn_game_common import run_one_game

    attempts = 0
    valid_v = (
        f'{{"v_before":{{"ordered_criteria":{_canonical_json(FULL_CRITERIA)},'
        f'"weights":{_canonical_json({_O: 0.2, _P: 0.2, _H: 0.2, _F: 0.2, _C: 0.2})},"confidence":0.7}},'
        '"action_before":"A","reason_before":"ok"}'
    )

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal attempts
        if "id=v-measurement-before" in prompt:
            attempts += 1
            if attempts <= 2:
                return "", "invalid json"
            return "", valid_v
        if "id=v-measurement-after" in prompt:
            return "", '{"v_after":{"ordered_criteria":["oxygen"],"weights":{"oxygen":1.0},"confidence":0.7},"reason_after":"ok"}'
        if "id=decision-contract" in prompt:
            return "", '{"action":"A","reason":"ok","ready":true}'
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
        role_value_mode="soft_value",
        scenario_id="comms_favored",
    )
    assert rows
    first = rows[0]
    assert first["v_measurement_retry_count"] >= 1


def test_counter_happy_path_reaches_accepted_in_run_one_game(monkeypatch) -> None:
    """counter経路で4発話以内に明示的合意に到達することをrun_one_game統合テストで検証する。

    経路:
      1. alpha: proposal + self-accept
      2. beta:  counter (self_accept=true で自分のcounterにも同意)
      3. alpha: counter を accept
    これで v_star_status=accepted になる。
    """
    from scripts.llm_turn_game_common import run_one_game

    alpha_before = (
        f'{{"v_before":{{"ordered_criteria":{_canonical_json([_P, _O, _H, _F, _C])},'
        f'"weights":{_canonical_json({_P: 0.3, _O: 0.25, _H: 0.2, _F: 0.15, _C: 0.1})},"confidence":0.6}},'
        '"action_before":"B","reason_before":"power"}'
    )
    beta_before = (
        f'{{"v_before":{{"ordered_criteria":{_canonical_json([_O, _P, _H, _F, _C])},'
        f'"weights":{_canonical_json({_O: 0.3, _P: 0.25, _H: 0.2, _F: 0.15, _C: 0.1})},"confidence":0.6}},'
        '"action_before":"A","reason_before":"oxygen"}'
    )
    after_v = (
        f'{{"v_after":{{"ordered_criteria":{_canonical_json(FULL_CRITERIA)},'
        f'"weights":{_canonical_json({_O: 0.2, _P: 0.2, _H: 0.2, _F: 0.2, _C: 0.2})},"confidence":0.7}},'
        '"reason_after":"ok"}'
    )

    call_count = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal call_count
        if "id=v-measurement-before" in prompt:
            return "", alpha_before if "alpha です" in prompt else beta_before
        if "id=v-measurement-after" in prompt:
            return "", after_v
        if "id=decision-contract" in prompt:
            return "", '{"action":"A","reason":"ok","ready":true}'
        if "id=v-proposal-required" in prompt:
            # alpha が proposal + self-accept を返す
            return (
                "",
                f'{{"v_proposal":{{"proposal_id":"alpha-turn0-required","ordered_criteria":{_canonical_json(FULL_CRITERIA)},"scope":"turn"}},'
                '"v_star_response":{"response":"accept","proposal_id":"alpha-turn0-required"},"action":"A","reason":"proposal"}',
            )
        if "id=v-proposal-response" in prompt:
            call_count += 1
            if call_count == 1:
                # beta が counter を出し、self_accept=true で自分のcounterにも同意
                return (
                    "",
                    f'{{"v_star_response":{{"response":"counter","proposal_id":"alpha-turn0-required",'
                    f'"counter_proposal":{{"proposal_id":"beta-counter-turn0","ordered_criteria":{_canonical_json(FULL_CRITERIA)},"scope":"turn"}},'
                    '"self_accept":true}}',
                )
            # alpha が beta の counter を accept
            return "", '{"v_star_response":{"response":"accept","proposal_id":"beta-counter-turn0"}}'
        # 自由議論では V proposal を出さない
        return "", '{"speech_act":"evidence","message":"ok","action":"A","reason":"ok","addressed_to":null,"reply_to_message_id":null}'

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None,
        None,
        "hivc_d",
        42,
        {"alpha": "alpha", "beta": "beta"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_new_tokens=96,
        max_discussion_turns=6,
        discussion_token_budget=2048,
        evaluator_rollouts=2,
        max_decision_opportunities=1,
        role_value_mode="soft_value",
        scenario_id="comms_favored",
    )
    assert rows
    first = rows[0]
    assert first["v_star_status"] == "accepted"
    assert first["v_star_failure_reason"] == ""
    assert first["v_star_unresolved_reason"] == ""
    assert first["v_counter_count"] == 1
    assert first["v_negotiation_message_budget"] == 6
    assert first["v_negotiation_messages_used"] == 3


def test_counter_without_self_accept_needs_extra_message(monkeypatch) -> None:
    """counter出力でself_accept=trueがない場合、4発話目でcounter提案者が明示的にacceptする必要がある。"""
    from scripts.llm_turn_game_common import run_one_game

    alpha_before = (
        f'{{"v_before":{{"ordered_criteria":{_canonical_json([_P, _O, _H, _F, _C])},'
        f'"weights":{_canonical_json({_P: 0.3, _O: 0.25, _H: 0.2, _F: 0.15, _C: 0.1})},"confidence":0.6}},'
        '"action_before":"B","reason_before":"power"}'
    )
    beta_before = (
        f'{{"v_before":{{"ordered_criteria":{_canonical_json([_O, _P, _H, _F, _C])},'
        f'"weights":{_canonical_json({_O: 0.3, _P: 0.25, _H: 0.2, _F: 0.15, _C: 0.1})},"confidence":0.6}},'
        '"action_before":"A","reason_before":"oxygen"}'
    )
    after_v = (
        f'{{"v_after":{{"ordered_criteria":{_canonical_json(FULL_CRITERIA)},'
        f'"weights":{_canonical_json({_O: 0.2, _P: 0.2, _H: 0.2, _F: 0.2, _C: 0.2})},"confidence":0.7}},'
        '"reason_after":"ok"}'
    )

    response_call_count = 0

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        nonlocal response_call_count
        if "id=v-measurement-before" in prompt:
            return "", alpha_before if "alpha です" in prompt else beta_before
        if "id=v-measurement-after" in prompt:
            return "", after_v
        if "id=decision-contract" in prompt:
            return "", '{"action":"A","reason":"ok","ready":true}'
        if "id=v-proposal-required" in prompt:
            return (
                "",
                f'{{"v_proposal":{{"proposal_id":"alpha-turn0-required","ordered_criteria":{_canonical_json(FULL_CRITERIA)},"scope":"turn"}},'
                '"v_star_response":{"response":"accept","proposal_id":"alpha-turn0-required"},"action":"A","reason":"proposal"}',
            )
        if "id=v-proposal-response" in prompt:
            response_call_count += 1
            if response_call_count == 1:
                # beta が counter を出すが self_accept なし
                return (
                    "",
                    f'{{"v_star_response":{{"response":"counter","proposal_id":"alpha-turn0-required",'
                    f'"counter_proposal":{{"proposal_id":"beta-counter-turn0","ordered_criteria":{_canonical_json(FULL_CRITERIA)},"scope":"turn"}}}}}}',
                )
            if response_call_count == 2:
                # alpha が beta の counter を accept
                return "", '{"v_star_response":{"response":"accept","proposal_id":"beta-counter-turn0"}}'
            # beta が自分の counter を明示的に accept (4発話目)
            return "", '{"v_star_response":{"response":"accept","proposal_id":"beta-counter-turn0"}}'
        return "", '{"speech_act":"evidence","message":"ok","action":"A","reason":"ok","addressed_to":null,"reply_to_message_id":null}'

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    rows = run_one_game(
        None,
        None,
        "hivc_d",
        42,
        {"alpha": "alpha", "beta": "beta"},
        {"alpha": None, "beta": None},
        {"alpha": "a", "beta": "b"},
        max_new_tokens=96,
        max_discussion_turns=6,
        discussion_token_budget=2048,
        evaluator_rollouts=2,
        max_decision_opportunities=1,
        role_value_mode="soft_value",
        scenario_id="comms_favored",
    )
    assert rows
    first = rows[0]
    assert first["v_star_status"] == "accepted"
    assert first["v_star_failure_reason"] == ""
    assert first["v_star_unresolved_reason"] == ""
    assert first["v_counter_count"] == 1
    assert first["v_negotiation_message_budget"] == 6
    assert first["v_negotiation_messages_used"] == 4
