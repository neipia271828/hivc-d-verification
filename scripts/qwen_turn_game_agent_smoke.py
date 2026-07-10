from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

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


def build_prompt(state) -> str:
    actions = "\n".join([f"{action.value}. {ACTION_LABELS[action]}" for action in ALL_ACTIONS])
    return f"""あなたは深海研究施設トラブルの意思決定エージェントです。
目的は、施設が壊滅する前に通信を復旧して救助要請することです。

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
{actions}

次の1手だけを選んでください。
必ず次のJSONだけを返してください。説明文やMarkdownは不要です。
{{"action":"A","reason":"短い理由"}}
"""


def extract_action(response: str) -> tuple[Action | None, str]:
    text = response.strip()
    json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            action_text = str(payload.get("action", "")).strip().upper()
            reason = str(payload.get("reason", "")).strip()
            if action_text in {action.value for action in ALL_ACTIONS}:
                return Action(action_text), reason
        except json.JSONDecodeError:
            pass

    letter_match = re.search(r"\b([ABCD])\b", text.upper())
    if letter_match:
        return Action(letter_match.group(1)), text
    return None, text


def generate_action(model, tokenizer, state, max_new_tokens: int, enable_thinking: bool = False) -> tuple[Action | None, str, str, str]:
    """(action, reason, response_text, thinking) を返す。"""
    prompt = build_prompt(state)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    full = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    think_match = re.search(r"⁠\*\*(.*?)\*\*", full, flags=re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        response = full[think_match.end():].strip()
    else:
        thinking = ""
        response = full.strip()
    action, reason = extract_action(response)
    return action, reason, response, thinking


AGENT_ARG_TYPES: dict[str, type] = {
    "model_path": str, "seed": int, "max_new_tokens": int,
    "evaluator_rollouts": int, "output": str,
    "enable_thinking": bool,
}

AGENT_DEFAULTS: dict[str, object] = {
    "model_path": "/home/student222/models/Qwen3-14B",
    "seed": 42, "max_new_tokens": 96, "evaluator_rollouts": 24,
    "output": "hivc_sim/results/turn_game/qwen_agent_smoke.csv",
    "enable_thinking": False,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Let Qwen play one turn-game episode.")
    parser.add_argument("--config", default=None,
                        help="YAML設定ファイルパス（指定時はその値が既定値になる）")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--evaluator-rollouts", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--enable-thinking", default=None, type=str,
                        help="Qwen3 thinkingモード (true/false)。config未指定時は false。")
    args = parser.parse_args()

    cli_overrides: dict[str, object] = {}
    for key in AGENT_ARG_TYPES:
        value = getattr(args, key, None)
        if value is not None:
            cli_overrides[key] = value
    cfg = merge_config_and_cli(args.config, cli_overrides, AGENT_DEFAULTS, AGENT_ARG_TYPES)

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
    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    rows: list[dict[str, object]] = []
    print("start_state", state.as_dict())

    while not state.done:
        q_values = estimate_q_values(state, n_rollouts=cfg["evaluator_rollouts"], seed=cfg["seed"] + state.turn * 1000)
        optimal = best_action(q_values)
        allowed = acceptable_actions(q_values)

        action, reason, raw_response, thinking = generate_action(model, tokenizer, state, cfg["max_new_tokens"], enable_thinking=cfg["enable_thinking"])
        if action is None:
            action = optimal
            reason = f"invalid_response_fallback: {reason[:120]}"

        step_seed = int(torch.randint(0, 1_000_000_000, (1,), generator=rng).item()) + cfg["seed"] + state.turn
        import numpy as np

        result = step(state, action, np.random.default_rng(step_seed))
        regret = q_values[optimal] - q_values[action]
        row = {
            "turn": state.turn,
            "event": state.current_event.value,
            "state_before": json.dumps(state.as_dict(), ensure_ascii=False, sort_keys=True),
            "raw_response": raw_response,
            "thinking": thinking,
            "action": action.value,
            "action_label": ACTION_LABELS[action],
            "reason": reason,
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
            f"turn={row['turn']} event={row['event']} action={row['action']} "
            f"best={row['best_action']} regret={row['regret']} outcome={row['outcome']}"
        )
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
