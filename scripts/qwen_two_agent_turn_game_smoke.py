from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "hivc_sim"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import merge_config_and_cli  # noqa: E402
from turn_game import (  # noqa: E402
    ACTION_LABELS,
    ALL_ACTIONS,
    Action,
    EVENT_LABELS,
    acceptable_actions,
    best_action,
    estimate_q_values,
    initial_state,
    step,
    terminal_score,
)


PERSONAS = {
    "alpha": "安全管理担当。酸素、浸水、船体損傷などの破局リスクを最優先で見る。",
    "beta": "通信・電力担当。救助要請に必要な通信復旧と電力維持を最優先で見る。",
}

DEFAULT_PERSONA_PARAMS: dict[str, dict[str, object]] = {
    "alpha": {
        "role": "安全管理担当",
        "priority_weights": {
            "oxygen": 0.30,
            "power": 0.10,
            "hull_damage": 0.25,
            "flooding": 0.25,
            "communication": 0.10,
        },
        "risk_tolerance": 0.20,
        "goal_focus": "survival_first",
        "communication_style": "cautious",
        "concession_tendency": 0.35,
        "evidence_demand": 0.70,
        "notes": "短期的な勝利よりも施設崩壊の回避を重視する。",
    },
    "beta": {
        "role": "通信・電力担当",
        "priority_weights": {
            "oxygen": 0.10,
            "power": 0.25,
            "hull_damage": 0.10,
            "flooding": 0.10,
            "communication": 0.45,
        },
        "risk_tolerance": 0.55,
        "goal_focus": "mission_first",
        "communication_style": "goal_directed",
        "concession_tendency": 0.45,
        "evidence_demand": 0.50,
        "notes": "勝利条件の達成速度を重視する。",
    },
}


def format_state(state) -> str:
    return "\n".join(
        [
            f"turn: {state.turn}",
            f"event: {EVENT_LABELS[state.current_event]} ({state.current_event.value})",
            f"oxygen: {state.oxygen}",
            f"power: {state.power}",
            f"hull_damage: {state.hull_damage}",
            f"flooding: {state.flooding}",
            f"communication: {state.communication}",
            f"morale: {state.morale}",
        ]
    )


def action_list() -> str:
    return "\n".join([f"{action.value}. {ACTION_LABELS[action]}" for action in ALL_ACTIONS])


def extract_json_action(response: str) -> tuple[Action | None, str, str, bool]:
    text = response.strip()
    json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            action_text = str(payload.get("action", "")).strip().upper()
            reason = str(payload.get("reason", "")).strip()
            message = str(payload.get("message", "")).strip()
            ready = bool(payload.get("ready", False))
            if action_text in {action.value for action in ALL_ACTIONS}:
                return Action(action_text), reason, message, ready
        except json.JSONDecodeError:
            pass

    letter_match = re.search(r"\b([ABCD])\b", text.upper())
    if letter_match:
        return Action(letter_match.group(1)), text[:160], text[:160], False
    return None, text[:160], text[:160], False


def run_prompt(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


def format_persona(agent_name: str, persona: str, persona_params: dict[str, object] | None) -> str:
    if not persona_params:
        return persona

    priority_weights = persona_params.get("priority_weights", {})
    if isinstance(priority_weights, dict):
        priority_text = ", ".join([f"{key}={value}" for key, value in priority_weights.items()])
    else:
        priority_text = str(priority_weights)

    return "\n".join(
        [
            f"name: {agent_name}",
            f"role: {persona_params.get('role', persona)}",
            f"priority_weights: {priority_text}",
            f"risk_tolerance: {persona_params.get('risk_tolerance', 'unspecified')}  # 0.0=極めて慎重, 1.0=高リスク許容",
            f"goal_focus: {persona_params.get('goal_focus', 'unspecified')}",
            f"communication_style: {persona_params.get('communication_style', 'unspecified')}",
            f"concession_tendency: {persona_params.get('concession_tendency', 'unspecified')}  # 0.0=譲らない, 1.0=譲歩しやすい",
            f"evidence_demand: {persona_params.get('evidence_demand', 'unspecified')}  # 0.0=直感重視, 1.0=根拠要求が強い",
            f"notes: {persona_params.get('notes', '')}",
        ]
    )


def discussion_prompt(
    agent_name: str,
    persona: str,
    persona_params: dict[str, object] | None,
    state,
    transcript: list[dict[str, str]],
    max_discussion_turns: int,
) -> str:
    if transcript:
        transcript_text = "\n".join(
            [
                f"{item['speaker']}: action={item['action']} ready={item['ready']} "
                f"reason={item['reason']} message={item['message']}"
                for item in transcript
            ]
        )
    else:
        transcript_text = "まだ議論はありません。"

    return f"""あなたは深海研究施設トラブルの意思決定エージェントです。
あなたのペルソナ設定:
{format_persona(agent_name, persona, persona_params)}

勝利条件:
- communication >= 3

敗北条件:
- oxygen <= 0
- power <= 0
- hull_damage >= 5
- flooding >= 5

現在状態:
{format_state(state)}

選択可能な行動:
{action_list()}

これまでの議論:
{transcript_text}

議論は最大 {max_discussion_turns} 発言までです。
相手と合意できるなら ready を true にしてください。
まだ説得・修正が必要なら ready を false にしてください。
次の発言として、提案する行動を1つ出してください。
必ず次のJSONだけを返してください。説明文やMarkdownは不要です。
{{"action":"A","reason":"短い理由","message":"相手への短い発言","ready":false}}
"""


def forced_vote_prompt(
    agent_name: str,
    persona: str,
    persona_params: dict[str, object] | None,
    state,
    transcript: list[dict[str, str]],
) -> str:
    transcript_text = "\n".join(
        [
            f"{item['speaker']}: action={item['action']} ready={item['ready']} "
            f"reason={item['reason']} message={item['message']}"
            for item in transcript
        ]
    )
    return f"""あなたは深海研究施設トラブルの意思決定エージェントです。
あなたのペルソナ設定:
{format_persona(agent_name, persona, persona_params)}

現在状態:
{format_state(state)}

選択可能な行動:
{action_list()}

ここまでの議論:
{transcript_text}

議論予算を使い切りました。最終票を1つだけ出してください。
必ず次のJSONだけを返してください。説明文やMarkdownは不要です。
{{"action":"A","reason":"最終判断の短い理由","message":"最終票","ready":true}}
"""


def get_action(model, tokenizer, prompt: str, max_new_tokens: int, fallback: Action) -> tuple[Action, str, str, bool, str]:
    raw = run_prompt(model, tokenizer, prompt, max_new_tokens)
    action, reason, message, ready = extract_json_action(raw)
    if action is None:
        return fallback, f"invalid_response_fallback: {reason}", message, False, raw
    return action, reason, message, ready, raw


def find_discussion_consensus(transcript: list[dict[str, str]]) -> Action | None:
    ready_by_speaker = {item["speaker"]: item for item in transcript if item["ready"] == "true"}
    if "alpha" in ready_by_speaker and "beta" in ready_by_speaker:
        alpha_ready = Action(ready_by_speaker["alpha"]["action"])
        beta_ready = Action(ready_by_speaker["beta"]["action"])
        if alpha_ready == beta_ready:
            return alpha_ready
    return None


def decide_group_action(turn: int, alpha_vote: Action, beta_vote: Action) -> tuple[Action, str]:
    if alpha_vote == beta_vote:
        return alpha_vote, "forced_vote_agreement"
    if turn % 2 == 0:
        return alpha_vote, "split_vote_alpha_priority"
    return beta_vote, "split_vote_beta_priority"


def get_role_entry(
    loaded: dict[str, object],
    requested_key: str | None,
    fallback_key: str,
    used_keys: set[str],
) -> tuple[str, dict[str, object] | str]:
    if requested_key:
        if requested_key not in loaded:
            raise KeyError(f"Role key not found in role file: {requested_key}")
        return requested_key, loaded[requested_key]  # type: ignore[index]

    if fallback_key in loaded and fallback_key not in used_keys:
        return fallback_key, loaded[fallback_key]  # type: ignore[index]

    for key in sorted(loaded):
        if key not in used_keys and isinstance(loaded[key], (dict, str)):
            return key, loaded[key]  # type: ignore[index]
    raise ValueError("No usable role entries found in role file.")


def apply_role_entry(
    agent_slot: str,
    role_key: str,
    role_entry: dict[str, object] | str,
    personas: dict[str, str],
    persona_params: dict[str, dict[str, object] | None],
    role_keys: dict[str, str | None],
) -> None:
    role_keys[agent_slot] = role_key
    if isinstance(role_entry, dict):
        params = dict(role_entry)
        params.setdefault("role_key", role_key)
        persona_params[agent_slot] = params
        personas[agent_slot] = str(params.get("role", PERSONAS[agent_slot]))
    else:
        persona_params[agent_slot] = None
        personas[agent_slot] = str(role_entry)


def load_personas(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, dict[str, object] | None], dict[str, str | None]]:
    personas = dict(PERSONAS)
    persona_params: dict[str, dict[str, object] | None] = {"alpha": None, "beta": None}
    role_keys: dict[str, str | None] = {"alpha": None, "beta": None}

    random_persona = getattr(args, "random_persona", False)
    random_seed = getattr(args, "random_seed", None)

    role_file = args.role_file or args.personas_file
    if role_file:
        path = Path(role_file).expanduser()
        with path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("Role/persona file must contain a JSON object.")

        # random_persona モード: role_file から重複なしで2エージェントをランダム選択
        if random_persona:
            import random as _random
            agent_keys = [k for k in sorted(loaded) if isinstance(loaded[k], (dict, str))]
            if len(agent_keys) < 2:
                raise ValueError(f"random_persona requires >=2 agent entries in {role_file}, got {len(agent_keys)}")
            rng = _random.Random(random_seed if random_seed is not None else 0)
            chosen = rng.sample(agent_keys, 2)
            apply_role_entry("alpha", chosen[0], loaded[chosen[0]], personas, persona_params, role_keys)
            apply_role_entry("beta", chosen[1], loaded[chosen[1]], personas, persona_params, role_keys)
            return personas, persona_params, role_keys

        for key in ("alpha", "beta"):
            if key in loaded:
                if isinstance(loaded[key], dict):
                    persona_params[key] = loaded[key]
                    personas[key] = str(loaded[key].get("role", PERSONAS[key]))
                    role_keys[key] = key
                else:
                    personas[key] = str(loaded[key])
                    role_keys[key] = key

        missing_slots = [slot for slot in ("alpha", "beta") if role_keys[slot] is None]
        if missing_slots:
            used_keys: set[str] = {key for key in role_keys.values() if key is not None}
        if "alpha" in missing_slots:
            alpha_key, alpha_entry = get_role_entry(loaded, args.alpha_role_key, "agent_01", used_keys)
            used_keys.add(alpha_key)
            apply_role_entry("alpha", alpha_key, alpha_entry, personas, persona_params, role_keys)
        if "beta" in missing_slots:
            beta_key, beta_entry = get_role_entry(loaded, args.beta_role_key, "agent_02", used_keys)
            apply_role_entry("beta", beta_key, beta_entry, personas, persona_params, role_keys)

    if args.persona_params_file:
        path = Path(args.persona_params_file).expanduser()
        with path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        for key in ("alpha", "beta"):
            if key in loaded and isinstance(loaded[key], dict):
                persona_params[key] = loaded[key]
                personas[key] = str(loaded[key].get("role", personas[key]))
    if args.alpha_persona:
        personas["alpha"] = args.alpha_persona
        persona_params["alpha"] = None
        role_keys["alpha"] = None
    if args.beta_persona:
        personas["beta"] = args.beta_persona
        persona_params["beta"] = None
        role_keys["beta"] = None
    return personas, persona_params, role_keys


SMOKE_ARG_TYPES: dict[str, type] = {
    "model_path": str, "seed": int, "max_new_tokens": int,
    "max_discussion_turns": int, "discussion_token_budget": int,
    "evaluator_rollouts": int, "output": str, "live_jsonl": str,
    "alpha_persona": str, "beta_persona": str, "personas_file": str,
    "role_file": str, "alpha_role_key": str, "beta_role_key": str,
    "persona_params_file": str,
    "random_persona": bool, "random_seed": int,
}

SMOKE_DEFAULTS: dict[str, object] = {
    "model_path": "/home/student222/models/Qwen3-14B",
    "seed": 42, "max_new_tokens": 96, "max_discussion_turns": 6,
    "discussion_token_budget": 768, "evaluator_rollouts": 24,
    "output": "hivc_sim/results/turn_game/qwen_two_agent_smoke.csv",
    "live_jsonl": None,
    "alpha_persona": None, "beta_persona": None, "personas_file": None,
    "role_file": None, "alpha_role_key": None, "beta_role_key": None,
    "persona_params_file": None,
    "random_persona": False, "random_seed": None,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Let two Qwen agents discuss and play one game.")
    parser.add_argument("--config", default=None,
                        help="YAML設定ファイルパス（指定時はその値が既定値になる）")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-discussion-turns", type=int, default=None)
    parser.add_argument("--discussion-token-budget", type=int, default=None)
    parser.add_argument("--alpha-persona", default=None)
    parser.add_argument("--beta-persona", default=None)
    parser.add_argument("--personas-file", default=None, help='JSON file such as {"alpha": "...", "beta": "..."}')
    parser.add_argument("--role-file", default=None, help="Role JSON file. Supports agent_01/agent_02 style keys.")
    parser.add_argument("--alpha-role-key", default=None, help="Role key to assign to alpha, e.g. agent_01.")
    parser.add_argument("--beta-role-key", default=None, help="Role key to assign to beta, e.g. agent_08.")
    parser.add_argument("--persona-params-file", default=None, help="Structured persona parameter JSON file.")
    parser.add_argument("--random-persona", action="store_true", default=False,
                        help="role_file から重複なしで2エージェントをランダム選択する。")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="random_persona の抽選シード。")
    parser.add_argument("--evaluator-rollouts", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--live-jsonl", default=None,
                        help="各ターン終了時にJSONLを追記するパス（visualize_game.html ライブモード用）")
    args = parser.parse_args()

    cli_overrides: dict[str, object] = {}
    for key in SMOKE_ARG_TYPES:
        value = getattr(args, key, None)
        if value is not None:
            cli_overrides[key] = value
    cfg = merge_config_and_cli(args.config, cli_overrides, SMOKE_DEFAULTS, SMOKE_ARG_TYPES)

    for key in ("alpha_persona", "beta_persona", "personas_file",
                "role_file", "alpha_role_key", "beta_role_key", "persona_params_file",
                "random_persona", "random_seed"):
        setattr(args, key, cfg[key])
    personas, persona_params, role_keys = load_personas(args)

    live_jsonl_path = cfg["live_jsonl"]
    if live_jsonl_path:
        live_path = Path(live_jsonl_path)
        if not live_path.is_absolute():
            live_path = REPO_ROOT / live_path
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_path.write_text("", encoding="utf-8")
        live_jsonl_path = str(live_path)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        device_map="auto",
        quantization_config=quantization_config,
    )

    state = initial_state(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    rows: list[dict[str, object]] = []
    print("start_state", state.as_dict())

    while not state.done:
        q_values = estimate_q_values(state, n_rollouts=cfg["evaluator_rollouts"], seed=cfg["seed"] + state.turn * 1000)
        optimal = best_action(q_values)
        allowed = acceptable_actions(q_values)

        fallback = optimal
        transcript: list[dict[str, str]] = []
        token_budget_used = 0
        speakers = ["alpha", "beta"]

        for discussion_index in range(cfg["max_discussion_turns"]):
            speaker = speakers[discussion_index % len(speakers)]
            action, reason, message, ready, raw = get_action(
                model,
                tokenizer,
                discussion_prompt(
                    speaker,
                    personas[speaker],
                    persona_params[speaker],
                    state,
                    transcript,
                    cfg["max_discussion_turns"],
                ),
                cfg["max_new_tokens"],
                fallback,
            )
            token_budget_used += len(tokenizer.encode(raw, add_special_tokens=False))
            transcript.append(
                {
                    "speaker": speaker,
                    "action": action.value,
                    "reason": reason,
                    "message": message,
                    "ready": str(ready).lower(),
                    "raw": raw,
                }
            )

            if find_discussion_consensus(transcript) is not None:
                break
            if token_budget_used >= cfg["discussion_token_budget"]:
                break

        consensus_action = find_discussion_consensus(transcript)
        if consensus_action is not None:
            alpha_vote = consensus_action
            beta_vote = consensus_action
            alpha_vote_reason = "discussion_consensus"
            beta_vote_reason = "discussion_consensus"
            alpha_vote_msg = ""
            beta_vote_msg = ""
            alpha_vote_ready = True
            beta_vote_ready = True
            alpha_vote_raw = ""
            beta_vote_raw = ""
            group_action = consensus_action
            decision_rule = "free_discussion_consensus"
        else:
            alpha_vote, alpha_vote_reason, alpha_vote_msg, alpha_vote_ready, alpha_vote_raw = get_action(
                model,
                tokenizer,
                forced_vote_prompt("alpha", personas["alpha"], persona_params["alpha"], state, transcript),
                cfg["max_new_tokens"],
                fallback,
            )
            beta_vote, beta_vote_reason, beta_vote_msg, beta_vote_ready, beta_vote_raw = get_action(
                model,
                tokenizer,
                forced_vote_prompt("beta", personas["beta"], persona_params["beta"], state, transcript),
                cfg["max_new_tokens"],
                fallback,
            )
            group_action, decision_rule = decide_group_action(state.turn, alpha_vote, beta_vote)

        result = step(state, group_action, rng)
        regret = q_values[optimal] - q_values[group_action]
        row = {
            "game_id": cfg["seed"],
            "seed": cfg["seed"],
            "condition": "smoke",
            "turn": state.turn,
            "event": state.current_event.value,
            "alpha_role_key": role_keys["alpha"],
            "beta_role_key": role_keys["beta"],
            "alpha_persona": personas["alpha"],
            "beta_persona": personas["beta"],
            "alpha_persona_params": json.dumps(persona_params["alpha"], ensure_ascii=False, sort_keys=True),
            "beta_persona_params": json.dumps(persona_params["beta"], ensure_ascii=False, sort_keys=True),
            "state_before": json.dumps(state.as_dict(), ensure_ascii=False, sort_keys=True),
            "discussion_turns": len(transcript),
            "discussion_token_budget_used": token_budget_used,
            "discussion_transcript": json.dumps(transcript, ensure_ascii=False),
            "alpha_vote": alpha_vote.value,
            "alpha_vote_reason": alpha_vote_reason,
            "alpha_vote_message": alpha_vote_msg,
            "alpha_vote_ready": str(alpha_vote_ready).lower(),
            "alpha_vote_raw": alpha_vote_raw,
            "beta_vote": beta_vote.value,
            "beta_vote_reason": beta_vote_reason,
            "beta_vote_message": beta_vote_msg,
            "beta_vote_ready": str(beta_vote_ready).lower(),
            "beta_vote_raw": beta_vote_raw,
            "group_action": group_action.value,
            "group_action_label": ACTION_LABELS[group_action],
            "decision_rule": decision_rule,
            "best_action": optimal.value,
            "acceptable_actions": ",".join(sorted(a.value for a in allowed)),
            "regret": round(float(regret), 3),
            "q_values": json.dumps({a.value: round(v, 3) for a, v in q_values.items()}, sort_keys=True),
            "state_after": json.dumps(result.state_after.as_dict(), ensure_ascii=False, sort_keys=True),
            "outcome": result.outcome,
            "terminal_score": terminal_score(result.state_after),
        }
        rows.append(row)
        print(
            f"turn={row['turn']} event={row['event']} "
            f"discussion_turns={row['discussion_turns']} "
            f"alpha={row['alpha_vote']} beta={row['beta_vote']} group={row['group_action']} "
            f"rule={row['decision_rule']} best={row['best_action']} "
            f"regret={row['regret']} outcome={row['outcome']}"
        )
        if live_jsonl_path:
            with open(live_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        state = result.state_after

    output_path = Path(cfg["output"])
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("final_state", state.as_dict())
    print("output", output_path)


if __name__ == "__main__":
    main()
