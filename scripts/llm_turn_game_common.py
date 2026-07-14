"""REQUIREMENTS §7 の3条件（control / consulting / hivc_d）バッチ実験用 共有モジュール。

REQUIREMENTS §7.1 のマルチエージェント議論・意思決定機会を実装する。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# hivc_sim/turn_game.py を import テストでも解決できるよう、リポジトリルートから hivc_sim を sys.path に追加
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HIVC_SIM_PATH = _REPO_ROOT / "hivc_sim"
if str(_HIVC_SIM_PATH) not in sys.path:
    sys.path.insert(0, str(_HIVC_SIM_PATH))

from turn_game import (
    ACTION_LABELS,
    ALL_ACTIONS,
    Action,
    EVENT_LABELS,
    acceptable_actions,
    best_action,
    estimate_q_values,
    initial_state,
    optimal_route,
    role_specific_evidence,
    route_of_action,
    step,
    terminal_score,
)


# REQUIREMENTS §7: 条件間で差をつけるのは合意形成手順の指示のみ。
CONDITION_PROCEDURES: dict[str, str] = {
    "control": "",
    "consulting": (
        "【合意形成の助言（一般論）】\n"
        "- リスクと便益を明示的に比較し、最悪の事態を避ける行動を優先する。\n"
        "- 相手の提案に対して根拠を一度問い、納得できれば譲歩する。\n"
        "- 残りターン数と資源残量から、今どの勝敗条件が最も差し迫っているかを確認する。"
    ),
    "hivc_d": (
        "【HIVC-D 合意形成手順】\n"
        "以下の順序で合意を診断・形成する。\n"
        "1. I（情報）の共有: 現在状態・イベント・資源残量を相手と同じ認識で確認する。\n"
        "2. V（判断基準）の整合: 各資源の優先順位が一致しているか確認し、不一致なら\n"
        "   双方が譲歩できる共通の優先順位 V* を一つ仮定して提案を作り直す。\n"
        "3. A（能力）の確認: 選んだ行動が現状の資源で実行可能か（敗北条件を踏ま倒さないか）\n"
        "   を最後に確認する。\n"
        "この順序を守り、まず情報をすり合わせ、次に基準を整合させ、最後に能力を確認する。"
    ),
}

CONDITIONS: tuple[str, ...] = ("control", "consulting", "hivc_d")


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


class SpeechAct(str, Enum):
    EVIDENCE = "evidence"
    QUESTION_OBJECTION = "question_objection"
    TRADEOFF = "tradeoff"
    CONCESSION_INTEGRATION = "concession_integration"
    INFORMATION_REQUEST = "information_request"


SPEECH_ACT_LABELS: dict[SpeechAct, str] = {
    SpeechAct.EVIDENCE: "根拠提示（状態・リスク・制約）",
    SpeechAct.QUESTION_OBJECTION: "質問・反論",
    SpeechAct.TRADEOFF: "トレードオフ比較",
    SpeechAct.CONCESSION_INTEGRATION: "譲歩案・統合案",
    SpeechAct.INFORMATION_REQUEST: "情報要請・要約",
}


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
    """ペルソナを読み込む。

    random_persona が真の場合、role_file から重複なしで2エージェントをランダム選択する。
    その場合 args.random_seed があればそれを使い、なければゲーム seed とは独立に
    決定論的に選ぶ（再現性のため）。
    """
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


def add_persona_args(parser: argparse.ArgumentParser) -> None:
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
                        help="random_persona の抽選シード（未指定時はゲーム seed を使用）。")


def format_state(state, agent_name: str | None = None) -> str:
    rescue = "未送信" if state.rescue_eta is None else f"救助到着まであと{state.rescue_eta}ターン"

    def _v(value, hidden: bool) -> str:
        return "不明（パートナーに問い合わせ）" if hidden else str(value)

    # alpha: 安全管理。船体・浸水は可視、通信・艇健全性は不可視。
    # beta: 通信・脱出艇。通信・艇状態は可視、船体・浸水は不可視。
    hide = {
        "alpha": {"communication": True, "pod_integrity": True, "pod_readiness": True},
        "beta": {"hull_damage": True, "flooding": True},
    }.get(agent_name, {})

    return "\n".join(
        [
            f"turn: {state.turn}",
            f"scenario: {state.scenario_id}",
            f"event: {EVENT_LABELS[state.current_event]} ({state.current_event.value})",
            f"oxygen: {state.oxygen}",
            f"power: {state.power}",
            f"hull_damage: {_v(state.hull_damage, hide.get('hull_damage', False))}",
            f"flooding: {_v(state.flooding, hide.get('flooding', False))}",
            f"communication: {_v(state.communication, hide.get('communication', False))}",
            f"pod_readiness: {_v(state.pod_readiness, hide.get('pod_readiness', False))}",
            f"pod_integrity: {_v(state.pod_integrity, hide.get('pod_integrity', False))}",
            f"rescue_eta: {rescue}",
            f"morale: {state.morale}",
        ]
    )


def action_list() -> str:
    return "\n".join([f"{action.value}. {ACTION_LABELS[action]}" for action in ALL_ACTIONS])


def schedule_decision_opportunities(seed: int, turn: int, schedule_seed: int = 0, max_opportunities: int = 3) -> int:
    """ゲームseed、ターン、固定スケジュールseedから決定論的に意思決定機会数を返す。"""
    return ((seed + turn * 31 + schedule_seed * 37) % max_opportunities) + 1


def allocate_discussion_budgets(
    opportunity_count: int,
    max_discussion_turns: int,
    discussion_token_budget: int,
    n_speakers: int = 2,
) -> tuple[list[int], list[int]]:
    """実際の opportunity_count で発言数とトークン予算を配分。端数は早い機会から。

    第1回目の意思決定機会には、各エージェントが1回ずつ発言できる最小数を確保する。
    合計は max_discussion_turns ・ discussion_token_budget を超えない。
    """
    if opportunity_count <= 0:
        return [], []

    # 発言数配分
    messages_per = max_discussion_turns // opportunity_count
    message_remainder = max_discussion_turns - messages_per * opportunity_count
    message_limits = [messages_per] * opportunity_count
    for i in range(opportunity_count):
        if i < message_remainder:
            message_limits[i] += 1
    # 第1回機会には少なくとも全エージェント1回ずつ
    message_limits[0] = max(n_speakers, message_limits[0])
    # 合計が max_discussion_turns を超えないよう後続機会から削減
    if sum(message_limits) > max_discussion_turns:
        excess = sum(message_limits) - max_discussion_turns
        for i in range(opportunity_count - 1, 0, -1):
            cut = min(excess, message_limits[i])
            message_limits[i] -= cut
            excess -= cut
            if excess <= 0:
                break
        # 後続機会だけでは補えない場合（max_discussion_turns < n_speakers 等）は第1機会を抑制
        if excess > 0:
            message_limits[0] = max(0, message_limits[0] - excess)

    # トークン配分は発言数配分に比例
    total_messages = sum(message_limits)
    if total_messages == 0:
        token_limits = [0] * opportunity_count
    else:
        token_limits = [
            (discussion_token_budget * message_limits[i]) // total_messages
            for i in range(opportunity_count)
        ]
        # 端数は早い機会から
        token_shortfall = discussion_token_budget - sum(token_limits)
        for i in range(opportunity_count):
            if token_shortfall <= 0:
                break
            add = min(token_shortfall, max(0, message_limits[i]))
            token_limits[i] += add
            token_shortfall -= add

    return message_limits, token_limits


def priority_agent(seed: int, turn: int) -> str:
    """フォールバック優先エージェントを返す。"""
    return "alpha" if (seed + turn) % 2 == 0 else "beta"


def format_transcript_text(transcript: list[dict[str, Any]]) -> str:
    """会話トランスクリプトをプロンプト用テキストに整形する。"""
    if not transcript:
        return "まだ議論はありません。"
    lines: list[str] = []
    for item in transcript:
        speaker = item.get("speaker", "unknown")
        parts: list[str] = [f"{speaker}:"]
        speech_act = item.get("speech_act")
        if speech_act:
            parts.append(f"[{speech_act}]")
        message = item.get("message", "")
        if message:
            parts.append(message)
        extras: list[str] = []
        if item.get("action"):
            extras.append(f"action={item['action']}")
        if item.get("reason"):
            extras.append(f"reason={item['reason']}")
        if "ready" in item:
            extras.append(f"ready={item['ready']}")
        if extras:
            parts.append(f"({' | '.join(extras)})")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _normalize_speech_act(value: Any) -> SpeechAct | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    for act in SpeechAct:
        if act.value == key:
            return act
    for act, label in SPEECH_ACT_LABELS.items():
        if label.lower() == key:
            return act
    return None


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
    letter_match = re.search(r"\b([ABCDEF])\b", text.upper())
    if letter_match:
        return Action(letter_match.group(1)), text[:160], text[:160], False
    return None, text[:160], text[:160], False


def extract_json_discussion(response: str) -> tuple[SpeechAct | None, str, Action | None, str, str | None, str | None, bool]:
    """自由議論用JSONをパースする。"""
    text = response.strip()
    json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            speech_act = _normalize_speech_act(payload.get("speech_act"))
            message = str(payload.get("message", "")).strip()
            reason = str(payload.get("reason", "")).strip()
            reply_to_message_id = payload.get("reply_to_message_id")
            if reply_to_message_id is not None:
                reply_to_message_id = str(reply_to_message_id).strip() or None
            addressed_to = str(payload.get("addressed_to", "")).strip() or None
            # speech_act が question_objection なら常に回答を要請
            requires_response = speech_act == SpeechAct.QUESTION_OBJECTION
            action: Action | None = None
            if "action" in payload:
                action_text = str(payload["action"]).strip().upper()
                if action_text in {action.value for action in ALL_ACTIONS}:
                    action = Action(action_text)
            return speech_act, message, action, reason, reply_to_message_id, addressed_to, requires_response
        except json.JSONDecodeError:
            pass
    return None, text[:160], None, "", None, None, False


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


def _procedure_block(condition: str) -> str:
    text = CONDITION_PROCEDURES.get(condition, "")
    return f"{text}\n\n" if text else ""


def _speech_act_guide() -> str:
    return "\n".join([f"- {act.value}: {label}" for act, label in SPEECH_ACT_LABELS.items()])


def _win_loss_block() -> str:
    return (
        "勝利条件:\n"
        "- 通信救助: communication >= 3 となり、救助到着（rescue_eta=0）まで生存する\n"
        "- 自力脱出: 行動 F で pod_readiness >= 2, pod_integrity >= 2, oxygen >= 3, power >= 2, flooding <= 3\n\n"
        "敗北条件:\n"
        "- oxygen <= 0\n"
        "- power <= 0\n"
        "- hull_damage >= 5\n"
        "- flooding >= 5\n"
        "- 行動 F を未達条件で実行すると重大損傷または敗北"
    )


def _role_evidence(agent_name: str, state) -> str:
    return role_specific_evidence(agent_name, state)


def _question_context(open_question: dict[str, Any] | None, can_ask_question: bool, remaining_messages: int, remaining_tokens: int) -> str:
    parts: list[str] = []
    if open_question is not None:
        parts.append(
            f"【未回答の質問への回答】\n"
            f"{open_question['speaker']} からの質問（ID: {open_question['message_id']}）:\n"
            f"{open_question['message']}\n"
            f"reply_to_message_id には {open_question['message_id']} を指定して回答してください。"
        )
    budget_note = f"残り発言枠: {remaining_messages}, 残りトークン予算: {remaining_tokens}"
    if not can_ask_question:
        parts.append(f"【注意】{budget_note}。質問を出すための余裕がないため、質問は避けてください。")
    else:
        parts.append(f"【注意】{budget_note}。質問を出す場合は reply_to_message_id への回答分の余裕を残してください。")
    return "\n\n".join(parts)


def discussion_prompt(
    agent_name: str,
    persona: str,
    persona_params: dict[str, object] | None,
    state,
    transcript: list[dict[str, Any]],
    max_discussion_turns: int,
    condition: str = "control",
    open_question: dict[str, Any] | None = None,
    can_ask_question: bool = True,
    remaining_messages: int = 0,
    remaining_tokens: int = 0,
) -> str:
    context = _question_context(open_question, can_ask_question, remaining_messages, remaining_tokens)
    return f"""あなたは深海研究施設トラブルの意思決定エージェントです。
あなたのペルソナ設定:
{format_persona(agent_name, persona, persona_params)}

{_win_loss_block()}

現在状態（あなたの担当分野のみ可視）:
{format_state(state, agent_name)}

あなたの役割固有情報:
{_role_evidence(agent_name, state)}

選択可能な行動:
{action_list()}

{_procedure_block(condition)}これまでの議論:
{format_transcript_text(transcript)}

{context}

自由議論の発言目的は以下のいずれかを speech_act として選んでください:
{_speech_act_guide()}

この自由議論フェーズでは最大 {max_discussion_turns} 発言までです。
行動案を述べたい場合は action（A-F）と reason を含めてください。
ready は不要です。
質問をする場合は speech_act に "question_objection" を使い、addressed_to を指定してください。
必ず次のJSONだけを返してください。説明文やMarkdownは不要です。
{{"speech_act":"evidence","message":"相手への短い発言","action":"A","reason":"短い理由"}}
"""


def decision_opportunity_prompt(
    agent_name: str,
    persona: str,
    persona_params: dict[str, object] | None,
    state,
    transcript: list[dict[str, Any]],
    condition: str,
    opportunity_index: int,
    opportunity_count: int,
) -> str:
    return f"""あなたは深海研究施設トラブルの意思決定エージェントです。
あなたのペルソナ設定:
{format_persona(agent_name, persona, persona_params)}

{_win_loss_block()}

現在状態（あなたの担当分野のみ可視）:
{format_state(state, agent_name)}

あなたの役割固有情報:
{_role_evidence(agent_name, state)}

選択可能な行動:
{action_list()}

{_procedure_block(condition)}これまでの議論:
{format_transcript_text(transcript)}

これは第 {opportunity_index} / {opportunity_count} 回の意思決定機会です。
各エージェントは独立に最終案を一つだけ出してください。
出力には action（A-F）、短い reason、そして合意意思を表す ready（true/false）を含めてください。
全員が同じ action かつ ready=true なら合意成立です。
必ず次のJSONだけを返してください。説明文やMarkdownは不要です。
{{"action":"A","reason":"短い理由","ready":true}}
"""


def run_prompt(model, tokenizer, prompt: str, max_new_tokens: int, enable_thinking: bool = False, thinking_budget: int | None = None) -> tuple[str, str]:
    """モデルにプロンプトを送り、(thinking_content, response_text) を返す。

    enable_thinking=True の場合、モデルは  <think>... </think> 内に思考を出力する。
    thinking_content はその内部テキスト、response_text は思考以降の最終応答。
    enable_thinking=False の場合、thinking_content は空文字。
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if thinking_budget is not None:
        template_kwargs["thinking_budget"] = thinking_budget
    text = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    full = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

    #  <think>... </think> ブロックを抽出（Qwen3 thinkingモード）
    think_match = re.search(r"<think>(.*?)</think>", full, flags=re.DOTALL)
    if think_match:
        thinking_content = think_match.group(1).strip()
        response_text = full[think_match.end():].strip()
    else:
        thinking_content = ""
        response_text = full.strip()

    return thinking_content, response_text


def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quantization_config,
    )
    return model, tokenizer


def get_action(model, tokenizer, prompt: str, max_new_tokens: int, fallback: Action, enable_thinking: bool = False, thinking_budget: int | None = None) -> tuple[Action, str, str, bool, str, str]:
    """(action, reason, message, ready, raw_response, thinking) を返す。"""
    thinking, raw = run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=enable_thinking, thinking_budget=thinking_budget)
    action, reason, message, ready = extract_json_action(raw)
    if action is None:
        return fallback, f"invalid_response_fallback: {reason}", message, False, raw, thinking
    return action, reason, message, ready, raw, thinking


def get_discussion_message(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    fallback_action: Action | None = None,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
) -> dict[str, Any]:
    """自由議論用の発言情報を dict で返す。"""
    thinking, raw = run_prompt(model, tokenizer, prompt, max_new_tokens, enable_thinking=enable_thinking, thinking_budget=thinking_budget)
    speech_act, message, action, reason, reply_to_message_id, addressed_to, requires_response = extract_json_discussion(raw)
    if speech_act is None and action is None and not message:
        # JSON 解析ができなかった場合のみフォールバック行動を使用する
        if fallback_action is not None:
            action = fallback_action
            reason = "[fallback_action]"
        speech_act = SpeechAct.INFORMATION_REQUEST
    elif speech_act is None:
        speech_act = SpeechAct.INFORMATION_REQUEST
    if not message:
        message = raw[:160].strip()
    return {
        "speech_act": speech_act,
        "message": message,
        "action": action,
        "reason": reason,
        "reply_to_message_id": reply_to_message_id,
        "addressed_to": addressed_to,
        "requires_response": requires_response,
        "raw": raw,
        "thinking": thinking,
    }


def run_one_game(
    model,
    tokenizer,
    condition: str,
    seed: int,
    personas: dict[str, str],
    persona_params: dict[str, dict[str, object] | None],
    role_keys: dict[str, str | None],
    max_new_tokens: int = 96,
    max_discussion_turns: int = 6,
    discussion_token_budget: int = 768,
    evaluator_rollouts: int = 24,
    live_jsonl_path: str | None = None,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    decision_schedule_seed: int = 0,
    max_decision_opportunities: int = 3,
    scenario_id: str | None = None,
    **kwargs: Any,
) -> list[dict[str, object]]:
    """1 ゲームを進行し、REQUIREMENTS §6 / §7.1 のターン別記録項目を含む行リストを返す。

    個人選択・個人理由・グループ理由・対立度を各行に記録する。
    質問と応答の閉包（§7.1.3）を実装する。
    live_jsonl_path を指定すると、各ターン終了時にその行を JSON 1 行として追記する。
    """
    _ = kwargs
    state = initial_state(seed, scenario_id)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    speakers = ["alpha", "beta"]
    n_speakers = len(speakers)
    planned_route = "undecided"

    # 少なくとも各エージェント1回ずつ発言できるよう実効値を確保
    effective_max_discussion_turns = max(max_discussion_turns, n_speakers)
    if effective_max_discussion_turns != max_discussion_turns:
        print(f"[run_one_game] max_discussion_turns {max_discussion_turns} is below {n_speakers}; bumping to {effective_max_discussion_turns}")

    while not state.done:
        q_values = estimate_q_values(state, n_rollouts=evaluator_rollouts, seed=seed + state.turn * 1000)
        optimal = best_action(q_values)
        allowed = acceptable_actions(q_values)
        fallback = optimal

        opportunity_count = schedule_decision_opportunities(seed, state.turn, decision_schedule_seed, max_decision_opportunities)
        message_limits, token_limits = allocate_discussion_budgets(
            opportunity_count, effective_max_discussion_turns, discussion_token_budget, n_speakers=n_speakers
        )

        transcript: list[dict[str, Any]] = []
        token_budget_used = 0
        total_free_messages = 0
        decision_history: list[dict[str, Any]] = []
        group_action: Action | None = None
        decision_rule: str | None = None
        group_reason = ""
        fallback_used = False
        fallback_priority_agent: str | None = None
        final_attempt_index = 0

        alpha_vote: Action | None = None
        alpha_vote_reason = ""
        alpha_vote_message = ""
        alpha_vote_ready = False
        alpha_vote_raw = ""
        alpha_vote_thinking = ""
        beta_vote: Action | None = None
        beta_vote_reason = ""
        beta_vote_message = ""
        beta_vote_ready = False
        beta_vote_raw = ""
        beta_vote_thinking = ""

        # 質問/応答閉包用の状態
        next_message_id = 1
        open_questions: list[dict[str, Any]] = []
        forced_decision_with_open_question = False
        forced_decision_reason = ""
        question_response_latencies: list[int] = []

        for opp_idx in range(1, opportunity_count + 1):
            opportunity_message_limit = message_limits[opp_idx - 1]
            opportunity_token_limit = token_limits[opp_idx - 1]
            opportunity_token_used = 0
            messages_this_opportunity = 0

            # 自由議論フェーズ
            while (
                messages_this_opportunity < opportunity_message_limit
                and total_free_messages < effective_max_discussion_turns
                and token_budget_used < discussion_token_budget
            ):
                if opportunity_token_used >= opportunity_token_limit:
                    break

                speaker = speakers[total_free_messages % n_speakers]
                other_speaker = "beta" if speaker == "alpha" else "alpha"

                # この話者が回答すべき未回答質問（最古）
                open_for_speaker = [q for q in open_questions if q["addressed_to"] == speaker]
                question_to_answer = open_for_speaker[0] if open_for_speaker else None

                turn_remaining_messages = effective_max_discussion_turns - total_free_messages
                turn_remaining_tokens = discussion_token_budget - token_budget_used
                k = len(open_questions)
                # 新しい質問を出せるのは、残り発言・トークンですべての未回答質問に対する回答分を含められる場合
                can_ask_question = (
                    turn_remaining_messages >= k + 2
                    and turn_remaining_tokens >= (k + 2) * max_new_tokens
                )

                prompt = discussion_prompt(
                    speaker,
                    personas[speaker],
                    persona_params[speaker],
                    state,
                    transcript,
                    opportunity_message_limit,
                    condition,
                    open_question=question_to_answer,
                    can_ask_question=can_ask_question,
                    remaining_messages=turn_remaining_messages,
                    remaining_tokens=turn_remaining_tokens,
                )

                response = get_discussion_message(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens,
                    fallback_action=fallback,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                )

                raw = response["raw"]
                token_count = 0
                if tokenizer is not None:
                    token_count = len(tokenizer.encode(raw, add_special_tokens=False))
                token_budget_used += token_count
                opportunity_token_used += token_count

                speech_act = response["speech_act"]
                is_question = speech_act == SpeechAct.QUESTION_OBJECTION and response["requires_response"]
                addressed_to = response["addressed_to"]
                if is_question and addressed_to != other_speaker:
                    addressed_to = other_speaker
                reply_to_message_id = response["reply_to_message_id"]

                # ターン全体の絶対上限で質問回答ができない場合は強制意思決定
                if is_question and not can_ask_question:
                    forced_decision_with_open_question = True
                    forced_decision_reason = (
                        f"turn_budget_exhausted_for_reply: {len(open_questions)} open questions, "
                        f"turn_remaining_messages={turn_remaining_messages}, turn_remaining_tokens={turn_remaining_tokens}"
                    )
                    this_message_id = str(next_message_id)
                    transcript.append(
                        {
                            "speaker": speaker,
                            "speech_act": speech_act.value if speech_act else None,
                            "message": response["message"],
                            "action": response["action"].value if response["action"] else "",
                            "reason": response["reason"],
                            "message_id": this_message_id,
                            "addressed_to": addressed_to,
                            "requires_response": True,
                            "reply_to_message_id": None,
                            "raw": raw,
                            "thinking": response["thinking"],
                        }
                    )
                    next_message_id += 1
                    open_questions.append(
                        {
                            "message_id": this_message_id,
                            "speaker": speaker,
                            "addressed_to": addressed_to,
                            "message": response["message"],
                            "timestamp": total_free_messages,
                        }
                    )
                    total_free_messages += 1
                    messages_this_opportunity += 1
                    break

                # 発言を記録
                this_message_id = str(next_message_id)
                transcript.append(
                    {
                        "speaker": speaker,
                        "speech_act": speech_act.value if speech_act else None,
                        "message": response["message"],
                        "action": response["action"].value if response["action"] else "",
                        "reason": response["reason"],
                        "message_id": this_message_id,
                        "addressed_to": addressed_to,
                        "requires_response": response["requires_response"],
                        "reply_to_message_id": reply_to_message_id,
                        "raw": raw,
                        "thinking": response["thinking"],
                    }
                )
                next_message_id += 1
                total_free_messages += 1
                messages_this_opportunity += 1

                if is_question:
                    open_questions.append(
                        {
                            "message_id": this_message_id,
                            "speaker": speaker,
                            "addressed_to": addressed_to,
                            "message": response["message"],
                            "timestamp": total_free_messages - 1,
                        }
                    )
                else:
                    # 回答を処理
                    if reply_to_message_id is None and open_for_speaker:
                        # 自動的に最古の未回答質問への回答とみなす
                        reply_to_message_id = open_for_speaker[0]["message_id"]
                        transcript[-1]["reply_to_message_id"] = reply_to_message_id

                    if reply_to_message_id is not None:
                        answered_id = str(reply_to_message_id)
                        target_q = None
                        for q in open_questions:
                            if q["message_id"] == answered_id:
                                target_q = q
                                break
                        if target_q is not None and target_q["addressed_to"] == speaker:
                            latency = (total_free_messages - 1) - target_q["timestamp"]
                            question_response_latencies.append(latency)
                            open_questions = [q for q in open_questions if q["message_id"] != answered_id]
                        else:
                            # 質問の宛先ではないエージェントや存在しないIDを参照した無効な回答
                            transcript[-1]["reply_to_message_id_invalid"] = True
                            invalid_reason = (
                                f"replied_to_question_not_addressed_to_speaker: {answered_id}"
                            )
                            if target_q is not None:
                                invalid_reason += f" (addressed_to={target_q['addressed_to']})"
                            else:
                                invalid_reason += " (not_found)"
                            transcript[-1]["reply_to_message_id_invalid_reason"] = invalid_reason

            # 未回答質問が残っていれば、後続機会で回答を試行する
            if open_questions and opp_idx < opportunity_count and not forced_decision_with_open_question:
                continue

            # 未回答質問が残っていて、かつこれが最後の機会なら強制意思決定
            if open_questions and (opp_idx == opportunity_count or forced_decision_with_open_question):
                if not forced_decision_with_open_question:
                    forced_decision_with_open_question = True
                    forced_decision_reason = "absolute_budget_limit_reached"

            # 意思決定機会：例外時もエージェントの投票は実行する
            alpha_vote, alpha_vote_reason, alpha_vote_message, alpha_vote_ready, alpha_vote_raw, alpha_vote_thinking = get_action(
                model,
                tokenizer,
                decision_opportunity_prompt(
                    "alpha",
                    personas["alpha"],
                    persona_params["alpha"],
                    state,
                    transcript,
                    condition,
                    opp_idx,
                    opportunity_count,
                ),
                max_new_tokens,
                fallback,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
            )
            if alpha_vote_reason.startswith("invalid_response_fallback"):
                alpha_vote = None
            beta_vote, beta_vote_reason, beta_vote_message, beta_vote_ready, beta_vote_raw, beta_vote_thinking = get_action(
                model,
                tokenizer,
                decision_opportunity_prompt(
                    "beta",
                    personas["beta"],
                    persona_params["beta"],
                    state,
                    transcript,
                    condition,
                    opp_idx,
                    opportunity_count,
                ),
                max_new_tokens,
                fallback,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
            )
            if beta_vote_reason.startswith("invalid_response_fallback"):
                beta_vote = None

            # 投票は全エージェントが出し終わってからトランスクリプトへ追加
            transcript.append(
                {
                    "speaker": "alpha",
                    "action": alpha_vote.value if alpha_vote else "",
                    "reason": alpha_vote_reason,
                    "message": alpha_vote_message,
                    "ready": str(alpha_vote_ready).lower(),
                    "raw": alpha_vote_raw,
                    "thinking": alpha_vote_thinking,
                }
            )
            transcript.append(
                {
                    "speaker": "beta",
                    "action": beta_vote.value if beta_vote else "",
                    "reason": beta_vote_reason,
                    "message": beta_vote_message,
                    "ready": str(beta_vote_ready).lower(),
                    "raw": beta_vote_raw,
                    "thinking": beta_vote_thinking,
                }
            )

            consensus = (
                alpha_vote is not None
                and beta_vote is not None
                and alpha_vote == beta_vote
                and alpha_vote_ready
                and beta_vote_ready
            )
            decision_history.append(
                {
                    "opportunity_index": opp_idx,
                    "opportunity_count": opportunity_count,
                    "alpha_vote": alpha_vote.value if alpha_vote else "",
                    "alpha_reason": alpha_vote_reason,
                    "alpha_ready": alpha_vote_ready,
                    "beta_vote": beta_vote.value if beta_vote else "",
                    "beta_reason": beta_vote_reason,
                    "beta_ready": beta_vote_ready,
                    "consensus": consensus,
                }
            )

            final_attempt_index = opp_idx
            if consensus:
                group_action = alpha_vote
                decision_rule = "consensus"
                group_reason = (
                    f"consensus on action {alpha_vote.value}: "
                    f"alpha reason={alpha_vote_reason}; beta reason={beta_vote_reason}"
                )
                break
            if forced_decision_with_open_question:
                break

        if group_action is None:
            priority = priority_agent(seed, state.turn)
            fallback_priority_agent = priority
            fallback_used = True
            if priority == "alpha":
                if alpha_vote is not None:
                    group_action = alpha_vote
                    group_reason = f"fallback priority agent alpha: {alpha_vote_reason}"
                    decision_rule = "fallback_priority"
                elif beta_vote is not None:
                    group_action = beta_vote
                    group_reason = f"fallback priority agent alpha invalid; using beta: {beta_vote_reason}"
                    decision_rule = "fallback_priority"
                else:
                    group_action = fallback
                    group_reason = "both votes invalid; fallback to best action"
                    decision_rule = "fallback_best"
            else:
                if beta_vote is not None:
                    group_action = beta_vote
                    group_reason = f"fallback priority agent beta: {beta_vote_reason}"
                    decision_rule = "fallback_priority"
                elif alpha_vote is not None:
                    group_action = alpha_vote
                    group_reason = f"fallback priority agent beta invalid; using alpha: {alpha_vote_reason}"
                    decision_rule = "fallback_priority"
                else:
                    group_action = fallback
                    group_reason = "both votes invalid; fallback to best action"
                    decision_rule = "fallback_best"

        result = step(state, group_action, rng)
        regret = q_values[optimal] - q_values[group_action]

        optimal_route_value = optimal_route(state, seed=seed + state.turn * 1000, n_rollouts=20)
        route = route_of_action(group_action)
        prev_route = planned_route
        if route in ("comms", "escape"):
            planned_route = route
        elif planned_route == "undecided":
            planned_route = optimal_route_value
        route_switch = (prev_route in ("comms", "escape") and planned_route in ("comms", "escape") and prev_route != planned_route)

        alpha_vote_value = alpha_vote.value if alpha_vote else ""
        beta_vote_value = beta_vote.value if beta_vote else ""
        individual_actions = f"{alpha_vote_value},{beta_vote_value}"
        individual_reasons = json.dumps(
            {"alpha": alpha_vote_reason, "beta": beta_vote_reason}, ensure_ascii=False
        )
        role_evidence = {
            "alpha": role_specific_evidence("alpha", state),
            "beta": role_specific_evidence("beta", state),
        }

        unanswered_question_count = len(open_questions)
        question_response_latency = float(np.mean(question_response_latencies)) if question_response_latencies else float("nan")

        row: dict[str, object] = {
            "game_id": seed,
            "seed": seed,
            "condition": condition,
            "scenario_id": state.scenario_id,
            "turn": state.turn,
            "event": state.current_event.value,
            "alpha_role_key": role_keys["alpha"],
            "beta_role_key": role_keys["beta"],
            "alpha_persona": personas["alpha"],
            "beta_persona": personas["beta"],
            "alpha_persona_params": json.dumps(persona_params["alpha"], ensure_ascii=False, sort_keys=True),
            "beta_persona_params": json.dumps(persona_params["beta"], ensure_ascii=False, sort_keys=True),
            "state_before": json.dumps(state.as_dict(), ensure_ascii=False, sort_keys=True),
            "individual_actions": individual_actions,
            "individual_reasons": individual_reasons,
            "discussion_turns": total_free_messages,
            "discussion_token_budget_used": token_budget_used,
            "discussion_transcript": json.dumps(transcript, ensure_ascii=False),
            "alpha_vote": alpha_vote_value,
            "alpha_vote_reason": alpha_vote_reason,
            "alpha_vote_message": alpha_vote_message,
            "alpha_vote_ready": str(alpha_vote_ready).lower(),
            "alpha_vote_raw": alpha_vote_raw,
            "alpha_vote_thinking": alpha_vote_thinking,
            "beta_vote": beta_vote_value,
            "beta_vote_reason": beta_vote_reason,
            "beta_vote_message": beta_vote_message,
            "beta_vote_ready": str(beta_vote_ready).lower(),
            "beta_vote_raw": beta_vote_raw,
            "beta_vote_thinking": beta_vote_thinking,
            "group_action": group_action.value,
            "group_action_label": ACTION_LABELS[group_action],
            "group_reason": group_reason,
            "decision_rule": decision_rule,
            "best_action": optimal.value,
            "acceptable_actions": ",".join(sorted(a.value for a in allowed)),
            "regret": round(float(regret), 3),
            "q_values": json.dumps({a.value: round(v, 3) for a, v in q_values.items()}, sort_keys=True),
            "state_after": json.dumps(result.state_after.as_dict(), ensure_ascii=False, sort_keys=True),
            "outcome": result.outcome,
            "terminal_score": terminal_score(result.state_after),
            "decision_opportunity_count": opportunity_count,
            "decision_attempts": final_attempt_index,
            "decision_attempt_index": final_attempt_index,
            "free_discussion_message_count": total_free_messages,
            "decision_history": json.dumps(decision_history, ensure_ascii=False),
            "fallback_used": fallback_used,
            "fallback_priority_agent": fallback_priority_agent,
            "planned_route": planned_route,
            "optimal_route": optimal_route_value,
            "route_switch": route_switch,
            "premature": result.premature,
            "role_specific_evidence": json.dumps(role_evidence, ensure_ascii=False),
            "alpha_evidence": role_evidence["alpha"],
            "beta_evidence": role_evidence["beta"],
            "unanswered_question_count": unanswered_question_count,
            "question_response_latency": question_response_latency,
            "forced_decision_with_open_question": forced_decision_with_open_question,
            "forced_decision_reason": forced_decision_reason,
        }
        rows.append(row)
        print(
            f"[{condition} seed={seed}] turn={row['turn']} event={row['event']} "
            f"disc={row['discussion_turns']} opp={row['decision_attempt_index']}/{row['decision_opportunity_count']} "
            f"alpha={alpha_vote_value} beta={beta_vote_value} "
            f"group={row['group_action']} rule={row['decision_rule']} "
            f"best={row['best_action']} regret={row['regret']} outcome={row['outcome']} "
            f"open_questions={unanswered_question_count} forced={forced_decision_with_open_question}"
        )
        if live_jsonl_path:
            with open(live_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        state = result.state_after

    return rows
