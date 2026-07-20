import math

from scripts.llm_turn_game_common import (
    DEFAULT_VALUE_CRITERIA_SCHEMA,
    _canonical_json,
    _profile_sha256,
    append_profile_assignment,
    build_value_manifest,
    decision_opportunity_prompt,
    extract_json_discussion,
    format_transcript_text,
    parse_v_negotiation,
    resolve_v_star,
    v_alignment_distance,
    verify_vote_v_star_consistency,
    v_measurement_prompt,
    _normalize_v_proposal,
)
from turn_game import Action, Event, GameState


# 共通Vオントロジーに合わせた完全criteria例
_O = "oxygen"
_P = "power"
_H = "hull_damage"
_F = "flooding"
_C = "communication"
FULL_CRITERIA = [_O, _P, _H, _F, _C]


def test_v_measurement_prompts_require_derived_weights_without_copyable_answers() -> None:
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
        assert '"weights"' in prompt and '"confidence"' in prompt
        assert all(f'"{criterion}":"<{criterion}の現在の優先度。0以上1以下>"' in prompt for criterion in FULL_CRITERIA)
        assert "プレースホルダーや例示値をコピーせず" in prompt
        assert "ROLE_PERSONA_INITIAL_VALUE" in prompt
        assert f"v-measurement-{phase}" in prompt
    assert '"v_before"' in prompts["before"]
    assert '"action_before"' in prompts["before"]
    assert '"reason_before"' in prompts["before"]
    assert '"v_after"' in prompts["after"]
    assert '"reason_after"' in prompts["after"]


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


def test_vote_consistency_requires_top_criterion_in_reason() -> None:
    v_star = {"ordered_criteria": [_P, _O, _H, _F, _C]}
    assert verify_vote_v_star_consistency(
        Action.REPAIR_POWER, "power を最優先する", "p1", True, "p1", v_star
    )
    assert not verify_vote_v_star_consistency(
        Action.REPAIR_POWER, "安全だから", "p1", True, "p1", v_star
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
