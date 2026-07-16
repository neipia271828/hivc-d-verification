import math

from scripts.llm_turn_game_common import (
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
    _normalize_v_proposal,
)
from turn_game import Action, Event, GameState


def test_nested_v_proposal_and_response_parse() -> None:
    raw = (
        '{"speech_act":"tradeoff","message":"基準案","action":"B","reason":"比較",'
        '"addressed_to":null,"reply_to_message_id":null,'
        '"v_proposal":{"proposal_id":"p1","ordered_criteria":["power","oxygen"],"scope":"turn"},'
        '"v_star_response":{"response":"accept","proposal_id":"p1"}}'
    )
    speech_act, message, _, _, _, _, _ = extract_json_discussion(raw)
    assert speech_act is not None and speech_act.value == "tradeoff"
    assert message == "基準案"

    import json

    proposal, response = parse_v_negotiation(json.loads(raw), "alpha", "1")
    assert proposal == {
        "proposal_id": "p1",
        "ordered_criteria": ["power", "oxygen"],
        "scope": "turn",
        "message_index": 1,
    }
    assert response == {"response": "accept", "proposal_id": "p1", "message_index": 1}


def test_v_star_requires_matching_explicit_acceptance() -> None:
    proposal = {"proposal_id": "p1", "ordered_criteria": ["power", "oxygen"], "scope": "turn", "message_index": 1}
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

    conflicting = {**proposal, "ordered_criteria": ["oxygen", "power"]}
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
        "v_proposal": {"proposal_id": "p1", "ordered_criteria": ["power"]},
        "v_star_response": {
            "response": "counter",
            "proposal_id": "p0",
            "counter_proposal": {"proposal_id": "p1", "ordered_criteria": ["power"]},
        },
    }
    text = format_transcript_text([item])
    assert "v_proposal=" in text and "counter_proposal" in text
    assert _normalize_v_proposal(
        {"proposal_id": "bad", "ordered_criteria": ["power"], "weights": {"power": float("nan")}},
        "fallback",
    ) is None


def test_vote_consistency_requires_top_criterion_in_reason() -> None:
    v_star = {"ordered_criteria": ["preserve_power", "oxygen"]}
    assert verify_vote_v_star_consistency(
        Action.REPAIR_POWER, "preserve power を最優先する", "p1", True, "p1", v_star
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

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        if "id=v-measurement-before" in prompt:
            return "", (
                '{"v_before":{"ordered_criteria":["preserve_power","oxygen"],'
                '"weights":{"preserve_power":0.7,"oxygen":0.3},"confidence":0.6},'
                '"action_before":"B","reason_before":"preserve power"}'
            )
        if "id=v-measurement-after" in prompt:
            seen_after.append(prompt)
            return "", (
                '{"v_after":{"ordered_criteria":["preserve_power","oxygen"],'
                '"weights":{"preserve_power":0.7,"oxygen":0.3},"confidence":0.7},'
                '"reason_after":"preserve power"}'
            )
        if "id=decision-contract" in prompt:
            return "", (
                '{"action":"B","reason":"preserve power を最優先",'
                '"ready":true,"v_star_id":"p1","v_star_consistent":true}'
            )
        seen_discussion.append(prompt)
        if "name: alpha" in prompt or "alpha persona" in prompt:
            return "", (
                '{"speech_act":"tradeoff","message":"proposal","action":"B","reason":"preserve power",'
                '"addressed_to":null,"reply_to_message_id":null,'
                '"v_proposal":{"proposal_id":"p1","ordered_criteria":["preserve_power","oxygen"],"scope":"turn"},'
                '"v_star_response":{"response":"accept","proposal_id":"p1"}}'
            )
        return "", (
            '{"speech_act":"concession_integration","message":"accept","action":"B","reason":"preserve power",'
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

    def fake_run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=False, thinking_budget=None):
        if "id=v-measurement-before" in prompt:
            marker = "alpha_private" if "エージェント alpha" in prompt else "beta_private"
            return "", (
                '{"v_before":{"ordered_criteria":["' + marker + '"],'
                '"weights":{"' + marker + '":1.0},"confidence":0.7},'
                '"action_before":"A","reason_before":"' + marker + '"}'
            )
        if "id=v-measurement-after" in prompt:
            return "", '{"v_after":{"ordered_criteria":["final"],"weights":{"final":1.0},"confidence":0.7},"reason_after":"final"}'
        captured.append(prompt)
        if "id=decision-contract" in prompt:
            return "", '{"action":"A","reason":"ok","ready":true}'
        return "", '{"speech_act":"evidence","message":"ok","action":"A","reason":"ok","addressed_to":null,"reply_to_message_id":null}'

    monkeypatch.setattr("scripts.llm_turn_game_common.run_prompt", fake_run_prompt)
    run_one_game(
        None, None, "control", 42,
        {"alpha": "alpha persona", "beta": "beta persona"},
        {"alpha": None, "beta": None}, {"alpha": "a", "beta": "b"},
        max_discussion_turns=2, evaluator_rollouts=1,
        max_decision_opportunities=1, role_value_mode="soft_value",
        scenario_id="comms_favored",
    )
    alpha_prompts = [p for p in captured if "alpha_private" in p]
    beta_prompts = [p for p in captured if "beta_private" in p]
    assert alpha_prompts and all("beta_private" not in p for p in alpha_prompts)
    assert beta_prompts and all("alpha_private" not in p for p in beta_prompts)
