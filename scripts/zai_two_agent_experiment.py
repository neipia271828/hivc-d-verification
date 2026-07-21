"""Run the turn-game directly through Z.ai GLM chat completions."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "hivc_sim"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import merge_config_and_cli  # noqa: E402
from llm_turn_game_common import (  # noqa: E402
    CONDITIONS,
    CONDITION_PROCEDURES,
    _git_commit,
    add_persona_args,
    append_profile_assignment,
    build_value_manifest,
    condition_order_for_seed,
    load_personas,
    resolve_role_file_path,
    run_one_game,
    write_value_manifest,
)
from turn_game_metrics import compute_summary_metrics  # noqa: E402
from zai_api import ChatCompletion, RequestTrace, ZaiAPIError, ZaiClient  # noqa: E402


ARG_TYPES: dict[str, type] = {
    "conditions": list,
    "games": int,
    "seed": int,
    "model": str,
    "max_new_tokens": int,
    "max_discussion_turns": int,
    "discussion_token_budget": int,
    "evaluator_rollouts": int,
    "output_dir": str,
    "role_file": str,
    "alpha_role_key": str,
    "beta_role_key": str,
    "personas_file": str,
    "persona_params_file": str,
    "alpha_persona": str,
    "beta_persona": str,
    "random_persona": bool,
    "random_seed": int,
    "role_value_mode": str,
    "decision_schedule_seed": int,
    "max_decision_opportunities": int,
    "request_timeout_seconds": float,
    "http_max_retries": int,
    "http_backoff_seconds": float,
    "request_interval_seconds": float,
    "adaptive_rate_limit": bool,
    "adaptive_min_interval_seconds": float,
    "adaptive_initial_interval_seconds": float,
    "adaptive_max_interval_seconds": float,
    "adaptive_cooldown_seconds": float,
    "adaptive_max_retries": int,
    "compact_prompts": bool,
    "do_sample": bool,
    "temperature": float,
    "max_total_tokens": int,
    "parallel_conditions": bool,
    "api_concurrency": int,
    "api_concurrency_per_key": int,
    "api_key_envs": list,
    "thinking_enabled": bool,
}

DEFAULTS: dict[str, object] = {
    "conditions": list(CONDITIONS),
    "games": 1,
    "seed": 42,
    "model": "glm-4.7-flash",
    "max_new_tokens": 256,
    "max_discussion_turns": 6,
    "discussion_token_budget": 1536,
    "evaluator_rollouts": 24,
    "output_dir": "hivc_sim/results/turn_game/zai/runs",
    "role_file": "configs/profiles_soft_value.yaml",
    "alpha_role_key": None,
    "beta_role_key": None,
    "personas_file": None,
    "persona_params_file": None,
    "alpha_persona": None,
    "beta_persona": None,
    "random_persona": False,
    "random_seed": None,
    "role_value_mode": "soft_value",
    "decision_schedule_seed": 0,
    "max_decision_opportunities": 3,
    "request_timeout_seconds": 60.0,
    "http_max_retries": 5,
    "http_backoff_seconds": 1.0,
    "request_interval_seconds": 0.0,
    "adaptive_rate_limit": True,
    "adaptive_min_interval_seconds": 2.0,
    "adaptive_initial_interval_seconds": 2.5,
    "adaptive_max_interval_seconds": 12.0,
    "adaptive_cooldown_seconds": 60.0,
    "adaptive_max_retries": 4,
    "compact_prompts": True,
    "do_sample": False,
    "temperature": 0.2,
    "max_total_tokens": 20_000_000,
    "parallel_conditions": False,
    "api_concurrency": 1,
    "api_concurrency_per_key": 1,
    "api_key_envs": ["ZAI_API_KEY"],
    "thinking_enabled": False,
}
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
RATE_LIMIT_API_CODES = frozenset({1302, 1303, 1305})
SUPPORTED_MODELS = frozenset({"glm-4.7", "glm-4.7-flash", "glm-4.7-flashx"})


COMPACT_CONDITION_PROCEDURES = {
    "control": "",
    "consulting": (
        "【一般合意・リスク管理】\n"
        "可視情報だけを事実として共有し、不明点は相手へ質問する。質問には先に回答し、観測不能は不明とする。\n"
        "各案を直近の敗北リスク、勝利寄与、資源消費、次ターンの選択肢で比較する。対立時は採否を分ける事実を示し、"
        "強い根拠には譲歩する。実行可能性、直後の敗北悪化、未確認前提を確認してから投票する。"
    ),
    "hivc_d": (
        "【HIVC-D: I→V→A】\n"
        "I: 可視な状態・危険を根拠付きで共有。不明・矛盾は質問し、観測不能は不明とする。\n"
        "V: 優先基準を事実と分離して示す。相違時はproposal_id、全ordered_criteria、scope、任意weightsを持つV*を提案。"
        "同一内容・同一IDへの両者の明示acceptだけで成立し、reject/counter/欠落は未合意。\n"
        "A: V*で案を比較し、実行可能性、直後の敗北悪化、未確認前提を確認。reasonにIの事実、V*、Aの制約を簡潔に結ぶ。"
    ),
    "hivc_d_prescribed_v1": (
        "【HIVC-D prescribed V* v1】直近の破局回避→勝利寄与→次ターンの選択肢で比較する。外部V*への適応条件。"
    ),
}


def compact_prompt_text(prompt: str) -> str:
    """Shorten stable boilerplate without removing state, roles, history, or JSON contracts."""
    compact = prompt
    for condition, full in CONDITION_PROCEDURES.items():
        if full:
            compact = compact.replace(full, COMPACT_CONDITION_PROCEDURES[condition])
    replacements = {
        "必ず次のJSONだけを返してください。説明文やMarkdownは不要です。": "JSONのみ返す。説明・Markdown禁止。",
        "あなたは深海研究施設トラブルの意思決定エージェントです。": "深海事故の意思決定エージェント。",
        "あなたは深海研究施設トラブルの意思決定エージェント ": "深海事故の意思決定エージェント ",
        "現在状態（あなたの担当分野のみ可視）:": "担当範囲の現在状態:",
        "あなたの役割固有情報:": "役割固有情報:",
        "選択可能な行動:": "行動:",
        "これまでの議論:": "議論:",
    }
    for source, target in replacements.items():
        compact = compact.replace(source, target)
    compact = re.sub(r"[ \t]+\n", "\n", compact)
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    return compact.strip()


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _prompt_purpose(prompt: str) -> str:
    for purpose, marker in (
        ("v_before", "id=v-measurement-before"),
        ("v_after", "id=v-measurement-after"),
        ("decision", "id=decision-contract"),
        ("v_proposal_response", "id=v-proposal-response-contract"),
        ("discussion", "id=discussion-contract"),
    ):
        if marker in prompt:
            return purpose
    return "unknown"


class ProvenanceRecorder:
    """Store token/latency evidence without storing private prompts or API keys."""

    def __init__(self) -> None:
        self.http_requests: list[dict[str, Any]] = []
        self.completions: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def observe_request(self, trace: RequestTrace) -> None:
        with self._lock:
            self.http_requests.append({
                "request_id": trace.request_id,
                "path": trace.path,
                "status_code": trace.status_code,
                "api_code": trace.api_code,
                "latency_seconds": trace.latency_seconds,
                "retry_count": trace.retry_count,
            })

    def add_completion(
        self,
        *,
        completion: ChatCompletion,
        agent: str,
        seed: int,
        condition: str,
        purpose: str,
    ) -> None:
        with self._lock:
            self.completions.append({
                "request_id": completion.request_id,
                "agent": agent,
                "seed": seed,
                "condition": condition,
                "purpose": purpose,
                "model": completion.model,
                "finish_reason": completion.finish_reason,
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "cached_tokens": completion.usage.cached_tokens,
                "total_tokens": completion.usage.total_tokens,
            })

    def token_totals(self) -> dict[str, int]:
        with self._lock:
            return {
                key: sum(int(item[key]) for item in self.completions)
                for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "total_tokens")
            }

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            requests = list(self.http_requests)
            completions = list(self.completions)
        totals = {
            key: sum(int(item[key]) for item in completions)
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "total_tokens")
        }
        return {
            "schema_version": "zai-provenance-v1",
            "contains_private_prompts": False,
            "contains_credentials": False,
            "http_requests": requests,
            "completions": completions,
            "token_totals": totals,
        }


class UsageAwareTokenizer:
    """Expose API-reported completion counts to the common token-budget code."""

    def __init__(self) -> None:
        self._counts: dict[str, deque[int]] = defaultdict(deque)

    def register(self, text: str, token_count: int) -> None:
        self._counts[text].append(max(0, int(token_count)))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        counts = self._counts.get(text)
        count = counts.popleft() if counts else 0
        if counts is not None and not counts:
            self._counts.pop(text, None)
        return list(range(count))


def counterbalanced_condition_order(
    conditions: list[str], *, experiment_seed: int, game_seed: int
) -> list[str]:
    """Return a deterministic Latin-square row for this experiment seed.

    ``condition_order_for_seed`` supplies a reproducible base permutation while
    cyclic rotation makes every condition occupy every start/position equally
    over each complete block of ``len(conditions)`` seeds.
    """
    base_order = condition_order_for_seed(conditions, experiment_seed)
    if not base_order:
        return []
    offset = (game_seed - experiment_seed) % len(base_order)
    return base_order[offset:] + base_order[:offset]


def api_key_assignment_for_seed(
    conditions: list[str], api_key_envs: list[str], *, experiment_seed: int, game_seed: int
) -> dict[str, str]:
    """Assign conditions to credential slots without persisting credential values.

    The cyclic shift removes a condition/account confound over every complete
    block of credential slots. With three conditions, three accounts and 30
    seeds, each condition uses every account exactly ten times.
    """
    if not api_key_envs:
        raise ValueError("api_key_envs must contain at least one environment variable name")
    offset = (game_seed - experiment_seed) % len(api_key_envs)
    return {
        condition: api_key_envs[(index + offset) % len(api_key_envs)]
        for index, condition in enumerate(conditions)
    }


class ConditionRequestScheduler:
    """Fairly gate API calls made by condition workers.

    With a single API slot, a plain semaphore can repeatedly wake the same
    worker.  This scheduler instead gives active conditions one request each in
    deterministic round-robin order.  Finished workers unregister themselves so
    shorter games cannot block the remaining conditions.
    """

    def __init__(
        self, conditions: list[str], max_concurrency: int, *, fair_serial: bool = True
    ) -> None:
        self._active = deque(conditions)
        self._serial_round_robin = fair_serial and max_concurrency == 1 and len(conditions) > 1
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._condition = threading.Condition()
        self._busy = False

    @contextmanager
    def slot(self, condition: str):
        if not self._serial_round_robin:
            with self._semaphore:
                yield
            return

        with self._condition:
            self._condition.wait_for(
                lambda: condition not in self._active
                or (not self._busy and bool(self._active) and self._active[0] == condition)
            )
            if condition not in self._active:
                raise RuntimeError(f"condition worker is no longer active: {condition}")
            self._busy = True
        try:
            yield
        finally:
            with self._condition:
                self._busy = False
                if self._active and self._active[0] == condition:
                    self._active.rotate(-1)
                self._condition.notify_all()

    def unregister(self, condition: str) -> None:
        with self._condition:
            try:
                self._active.remove(condition)
            except ValueError:
                return
            self._condition.notify_all()


class AdaptiveRequestPacer:
    """Shared adaptive spacing and cooldown for all condition workers."""

    def __init__(
        self,
        *,
        min_interval: float,
        initial_interval: float,
        max_interval: float,
        cooldown_seconds: float,
        successes_before_decrease: int = 12,
        decrease_seconds: float = 0.25,
        increase_factor: float = 1.6,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        if min_interval < 0 or initial_interval < min_interval or max_interval < initial_interval:
            raise ValueError("adaptive intervals must satisfy 0 <= min <= initial <= max")
        self.min_interval = float(min_interval)
        self.current_interval = float(initial_interval)
        self.max_interval = float(max_interval)
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.successes_before_decrease = max(1, int(successes_before_decrease))
        self.decrease_seconds = max(0.0, float(decrease_seconds))
        self.increase_factor = max(1.0, float(increase_factor))
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last_started: float | None = None
        self._cooldown_until = 0.0
        self._success_streak = 0
        self.rate_limit_events = 0
        self.total_cooldown_seconds = 0.0

    def before_request(self) -> None:
        # The scheduler normally serializes calls, but the lock also protects
        # bounded-parallel configurations from reserving the same time slot.
        with self._lock:
            now = self._clock()
            spacing_until = now if self._last_started is None else self._last_started + self.current_interval
            wait_until = max(spacing_until, self._cooldown_until)
            delay = max(0.0, wait_until - now)
            if delay:
                self._sleep(delay)
                now = self._clock()
            self._last_started = now

    def note_success(self) -> None:
        with self._lock:
            self._success_streak += 1
            if self._success_streak >= self.successes_before_decrease:
                self.current_interval = max(
                    self.min_interval, self.current_interval - self.decrease_seconds
                )
                self._success_streak = 0

    def note_rate_limit(self) -> None:
        with self._lock:
            now = self._clock()
            self.rate_limit_events += 1
            self._success_streak = 0
            self.current_interval = min(
                self.max_interval,
                max(self.current_interval + 1.0, self.current_interval * self.increase_factor),
            )
            new_until = max(self._cooldown_until, now + self.cooldown_seconds)
            self.total_cooldown_seconds += max(0.0, new_until - max(now, self._cooldown_until))
            self._cooldown_until = new_until

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "current_interval_seconds": self.current_interval,
                "rate_limit_events": self.rate_limit_events,
                "total_cooldown_seconds": self.total_cooldown_seconds,
            }


class ZaiPromptRunner:
    def __init__(
        self,
        *,
        client: ZaiClient,
        recorder: ProvenanceRecorder,
        model: str,
        seed: int,
        condition: str,
        do_sample: bool,
        temperature: float,
        max_total_tokens: int,
        usage_tokenizer: UsageAwareTokenizer | None = None,
        request_semaphore: threading.Semaphore | None = None,
        request_scheduler: ConditionRequestScheduler | None = None,
        adaptive_pacer: AdaptiveRequestPacer | None = None,
        adaptive_max_retries: int = 0,
        compact_prompts: bool = False,
        thinking_enabled: bool = False,
        request_interval_seconds: float = 0.0,
        sleep=None,
    ) -> None:
        import time

        self.client = client
        self.recorder = recorder
        self.model = model
        self.seed = seed
        self.condition = condition
        self.do_sample = do_sample
        self.temperature = temperature
        self.max_total_tokens = max_total_tokens
        self.usage_tokenizer = usage_tokenizer
        self.request_semaphore = request_semaphore
        self.request_scheduler = request_scheduler
        self.adaptive_pacer = adaptive_pacer
        self.adaptive_max_retries = max(0, int(adaptive_max_retries))
        self.compact_prompts = bool(compact_prompts)
        self.thinking_enabled = thinking_enabled
        self.request_interval_seconds = max(0.0, request_interval_seconds)
        self.sleep = sleep or time.sleep

    def __call__(self, agent: str, prompt: str, **kwargs: Any) -> tuple[str, str]:
        effective_prompt = compact_prompt_text(prompt) if self.compact_prompts else prompt
        if self.request_scheduler is not None or self.request_semaphore is not None:
            with ExitStack() as stack:
                if self.request_scheduler is not None:
                    stack.enter_context(self.request_scheduler.slot(self.condition))
                if self.request_semaphore is not None:
                    stack.enter_context(self.request_semaphore)
                completion = self._complete_with_adaptive_retry(effective_prompt, kwargs)
                self._record_completion(completion, agent, effective_prompt)
                if self.adaptive_pacer is None and self.request_interval_seconds:
                    self.sleep(self.request_interval_seconds)
            return completion.reasoning_content, completion.content
        completion = self._complete_with_adaptive_retry(effective_prompt, kwargs)
        self._record_completion(completion, agent, effective_prompt)
        if self.adaptive_pacer is None and self.request_interval_seconds:
            self.sleep(self.request_interval_seconds)
        return completion.reasoning_content, completion.content

    def _complete_with_adaptive_retry(self, prompt: str, kwargs: dict[str, Any]) -> ChatCompletion:
        for attempt in range(self.adaptive_max_retries + 1):
            if self.adaptive_pacer is not None:
                self.adaptive_pacer.before_request()
            try:
                completion = self._complete(prompt, kwargs)
            except ZaiAPIError as exc:
                rate_limited = exc.status_code == 429 or exc.api_code in RATE_LIMIT_API_CODES
                retryable = (
                    rate_limited
                    or exc.status_code is None
                    or 500 <= exc.status_code <= 599
                )
                if self.adaptive_pacer is not None and rate_limited:
                    self.adaptive_pacer.note_rate_limit()
                if not retryable or attempt >= self.adaptive_max_retries:
                    raise
                continue
            if self.adaptive_pacer is not None:
                self.adaptive_pacer.note_success()
            return completion
        raise AssertionError("adaptive retry loop exhausted unexpectedly")

    def _record_completion(self, completion: ChatCompletion, agent: str, prompt: str) -> None:
        purpose = _prompt_purpose(prompt)
        self.recorder.add_completion(
            completion=completion,
            agent=agent,
            seed=self.seed,
            condition=self.condition,
            purpose=purpose,
        )
        if self.usage_tokenizer is not None and purpose != "decision":
            self.usage_tokenizer.register(completion.content, completion.usage.completion_tokens)
        if self.recorder.token_totals()["total_tokens"] > self.max_total_tokens:
            raise ZaiAPIError("Configured total token safety limit was exceeded")

    def _complete(self, prompt: str, kwargs: dict[str, Any]) -> ChatCompletion:
        return self.client.chat_completion(
            prompt=prompt,
            model=self.model,
            max_tokens=int(kwargs.get("max_new_tokens", 256)),
            do_sample=self.do_sample,
            temperature=self.temperature,
            response_format="json_object",
            thinking_enabled=self.thinking_enabled,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the two-agent experiment through Z.ai GLM-4.7-Flash")
    parser.add_argument("--config", default="configs/zai_experiment.yaml")
    parser.add_argument("--conditions", nargs="*", choices=list(CONDITIONS) + ["all"])
    parser.add_argument("--games", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--parallel-conditions", action="store_true", default=None)
    parser.add_argument("--api-concurrency", type=int)
    parser.add_argument("--dry-run", action="store_true")
    add_persona_args(parser)
    return parser


def _load_cfg(args: argparse.Namespace) -> dict[str, Any]:
    overrides = {key: getattr(args, key, None) for key in ARG_TYPES if getattr(args, key, None) is not None}
    cfg = merge_config_and_cli(args.config, overrides, DEFAULTS, ARG_TYPES)
    cfg["role_file"] = resolve_role_file_path(cfg.get("role_file"), cfg.get("role_value_mode"))
    conditions = cfg["conditions"]
    if isinstance(conditions, str):
        conditions = [item.strip() for item in conditions.split(",") if item.strip()]
    if "all" in conditions:
        conditions = list(CONDITIONS)
    if not conditions or any(item not in CONDITIONS for item in conditions):
        raise ValueError("conditions must be control, consulting, and/or hivc_d")
    cfg["conditions"] = list(conditions)
    if int(cfg["games"]) < 1:
        raise ValueError("games must be at least 1")
    if int(cfg["max_total_tokens"]) < 1:
        raise ValueError("max_total_tokens must be positive")
    if int(cfg["api_concurrency"]) < 1:
        raise ValueError("api_concurrency must be positive")
    if int(cfg["api_concurrency_per_key"]) < 1:
        raise ValueError("api_concurrency_per_key must be positive")
    api_key_envs = [str(item).strip() for item in cfg["api_key_envs"] if str(item).strip()]
    if not api_key_envs or len(api_key_envs) != len(set(api_key_envs)):
        raise ValueError("api_key_envs must contain unique, non-empty environment variable names")
    if any(not re.fullmatch(r"[A-Z_][A-Z0-9_]*", item) for item in api_key_envs):
        raise ValueError("api_key_envs may contain environment variable names only")
    cfg["api_key_envs"] = api_key_envs
    if int(cfg["adaptive_max_retries"]) < 0:
        raise ValueError("adaptive_max_retries must not be negative")
    min_interval = float(cfg["adaptive_min_interval_seconds"])
    initial_interval = float(cfg["adaptive_initial_interval_seconds"])
    max_interval = float(cfg["adaptive_max_interval_seconds"])
    if min_interval < 0 or not min_interval <= initial_interval <= max_interval:
        raise ValueError("adaptive intervals must satisfy 0 <= min <= initial <= max")
    if not 0 <= float(cfg["temperature"]) <= 1:
        raise ValueError("temperature must be between 0 and 1")
    cfg["model"] = str(cfg["model"]).lower()
    if cfg["model"] not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"model must be one of: {supported}")
    return cfg


def _persona_args(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    for key in (
        "role_file", "alpha_role_key", "beta_role_key", "personas_file", "persona_params_file",
        "alpha_persona", "beta_persona", "random_persona", "random_seed", "role_value_mode",
    ):
        setattr(args, key, cfg.get(key))


def validate_env() -> None:
    """Validate credentials and actual GLM-4.7-Flash JSON access with one tiny completion."""
    client = ZaiClient.from_env(max_retries=0)
    result = client.chat_completion(
        prompt='Return exactly this JSON object: {"ready":true}',
        model="glm-4.7-flash",
        max_tokens=128,
        do_sample=False,
        thinking_enabled=False,
    )
    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict) or payload.get("ready") is not True:
        raise ZaiAPIError("GLM-4.7-Flash validation returned an unexpected JSON response")
    print(json.dumps({
        "authenticated": True,
        "model": result.model,
        "json_mode": True,
        "ready": True,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        },
    }, ensure_ascii=False, sort_keys=True))


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_cfg(args)
    _persona_args(args, cfg)
    conditions = list(cfg["conditions"])
    api_key_envs = list(cfg["api_key_envs"])
    output_root = Path(str(cfg["output_dir"]))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    run_id = args.run_id or dt.datetime.now().strftime("zai-%Y%m%d-%H%M%S")
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run_id: {run_id!r}")
    run_dir = output_root / run_id
    experiment_seed = int(cfg["seed"])
    seeds = [experiment_seed + index for index in range(int(cfg["games"]))]
    condition_schedule = [
        {
            "seed": game_seed,
            "conditions": counterbalanced_condition_order(
                conditions, experiment_seed=experiment_seed, game_seed=game_seed
            ),
        }
        for game_seed in seeds
    ]
    key_assignment_schedule = [
        {
            "seed": game_seed,
            "condition_to_api_key_env": api_key_assignment_for_seed(
                conditions, api_key_envs, experiment_seed=experiment_seed, game_seed=game_seed
            ),
        }
        for game_seed in seeds
    ]
    effective_api_concurrency = (
        min(
            int(cfg["api_concurrency"]),
            len(conditions),
            len(api_key_envs) * int(cfg["api_concurrency_per_key"]),
        )
        if cfg["parallel_conditions"] else 1
    )
    if cfg["parallel_conditions"] and len(api_key_envs) > 1 and effective_api_concurrency > 1:
        scheduling_mode = "seed_counterbalanced_multi_account_parallel_api"
    elif cfg["parallel_conditions"] and effective_api_concurrency == 1 and len(conditions) > 1:
        scheduling_mode = "seed_counterbalanced_condition_workers_round_robin_serial_api"
    elif cfg["parallel_conditions"]:
        scheduling_mode = "seed_counterbalanced_condition_workers_bounded_parallel_api"
    else:
        scheduling_mode = "seed_counterbalanced_sequential_conditions"

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "network_calls": 0,
            "backend": "zai_chat_completions",
            "model": cfg["model"],
            "conditions": conditions,
            "seeds": seeds,
            "planned_games": int(cfg["games"]) * len(conditions),
            "parallel_conditions": bool(cfg["parallel_conditions"]),
            "condition_workers": len(conditions) if cfg["parallel_conditions"] else 1,
            "api_concurrency": int(cfg["api_concurrency"]),
            "api_concurrency_per_key": int(cfg["api_concurrency_per_key"]),
            "api_key_count": len(api_key_envs),
            "api_key_envs": api_key_envs,
            "api_key_assignment_by_seed": key_assignment_schedule,
            "effective_api_concurrency": effective_api_concurrency,
            "execution_scheduling_mode": scheduling_mode,
            "condition_order_strategy": "deterministic_cyclic_latin_square_v1",
            "condition_start_order_by_seed": condition_schedule,
            "api_requests_interleaved": scheduling_mode.endswith("round_robin_serial_api"),
            "adaptive_rate_limit": bool(cfg["adaptive_rate_limit"]),
            "adaptive_min_interval_seconds": float(cfg["adaptive_min_interval_seconds"]),
            "adaptive_initial_interval_seconds": float(cfg["adaptive_initial_interval_seconds"]),
            "adaptive_max_interval_seconds": float(cfg["adaptive_max_interval_seconds"]),
            "adaptive_cooldown_seconds": float(cfg["adaptive_cooldown_seconds"]),
            "prompt_profile": "zai-compact-v1" if cfg["compact_prompts"] else "full-v1",
            "max_new_tokens": int(cfg["max_new_tokens"]),
            "thinking_enabled": bool(cfg["thinking_enabled"]),
            "output": str(run_dir),
            "scientifically_comparable_to_qwen_gpu": False,
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse existing Z.ai run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    recorder = ProvenanceRecorder()
    client_kwargs = {
        "timeout": float(cfg["request_timeout_seconds"]),
        "max_retries": int(cfg["http_max_retries"]),
        "backoff_seconds": float(cfg["http_backoff_seconds"]),
        "trace_observer": recorder.observe_request,
    }
    clients: dict[str, ZaiClient] = {}
    for env_name in api_key_envs:
        api_key = os.environ.get(env_name, "").strip()
        if not api_key:
            raise ZaiAPIError(f"required credential environment variable is missing: {env_name}")
        if len(api_key_envs) == 1 and env_name == "ZAI_API_KEY":
            clients[env_name] = ZaiClient.from_env(**client_kwargs)
        else:
            clients[env_name] = ZaiClient(api_key=api_key, **client_kwargs)
    def new_adaptive_pacer() -> AdaptiveRequestPacer | None:
        if not cfg["adaptive_rate_limit"]:
            return None
        return AdaptiveRequestPacer(
            min_interval=float(cfg["adaptive_min_interval_seconds"]),
            initial_interval=float(cfg["adaptive_initial_interval_seconds"]),
            max_interval=float(cfg["adaptive_max_interval_seconds"]),
            cooldown_seconds=float(cfg["adaptive_cooldown_seconds"]),
        )

    # The provider can enforce limits above the credential level (account/IP/model).
    # When only one global request is allowed, all key slots must therefore share
    # one pacer so rotating credentials cannot accidentally exceed that limit.
    if effective_api_concurrency == 1:
        shared_pacer = new_adaptive_pacer()
        adaptive_pacers = {env_name: shared_pacer for env_name in api_key_envs}
    else:
        adaptive_pacers = {env_name: new_adaptive_pacer() for env_name in api_key_envs}
    personas, persona_params, role_keys = load_personas(args)
    value_manifest = build_value_manifest(
        cfg, personas, persona_params, role_keys,
        role_value_mode=str(cfg["role_value_mode"]),
        framework_ids=conditions,
        runner_version="zai_two_agent_experiment-v1",
    )
    value_manifest_path = run_dir / "value_manifest.json"
    write_value_manifest(value_manifest_path, value_manifest)
    manifest: dict[str, Any] = {
        "schema_version": "zai-experiment-manifest-v3",
        "run_id": run_id,
        "backend": "zai_chat_completions",
        "api_endpoint": "/api/paas/v4/chat/completions",
        "model": cfg["model"],
        "response_format": "json_object",
        "do_sample": bool(cfg["do_sample"]),
        "status": "running",
        "started_at": _now_iso(),
        "completed_at": None,
        "git_commit": _git_commit(),
        "conditions": conditions,
        "games_per_condition": int(cfg["games"]),
        "planned_games": int(cfg["games"]) * len(conditions),
        "parallel_conditions": bool(cfg["parallel_conditions"]),
        "condition_workers": len(conditions) if cfg["parallel_conditions"] else 1,
        "api_concurrency": int(cfg["api_concurrency"]),
        "api_concurrency_per_key": int(cfg["api_concurrency_per_key"]),
        "api_key_count": len(api_key_envs),
        "api_key_envs": api_key_envs,
        "api_key_assignment_by_seed": key_assignment_schedule,
        "effective_api_concurrency": effective_api_concurrency,
        "execution_scheduling_mode": scheduling_mode,
        "condition_order_strategy": "deterministic_cyclic_latin_square_v1",
        "condition_start_order_by_seed": condition_schedule,
        "api_requests_interleaved": scheduling_mode.endswith("round_robin_serial_api"),
        "adaptive_rate_limit": bool(cfg["adaptive_rate_limit"]),
        "adaptive_rate_state_by_api_key": {
            env_name: pacer.snapshot() if pacer else None
            for env_name, pacer in adaptive_pacers.items()
        },
        "adaptive_max_retries": int(cfg["adaptive_max_retries"]),
        "prompt_profile": "zai-compact-v1" if cfg["compact_prompts"] else "full-v1",
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "thinking_enabled": bool(cfg["thinking_enabled"]),
        "completed_game_conditions": 0,
        "token_totals": recorder.token_totals(),
        "scientific_comparability": {
            "comparable_to_qwen_gpu_runs": False,
            "comparable_to_previous_full_prompt_zai_runs": not bool(cfg["compact_prompts"]),
            "reason": "Different model/service from Qwen; compact-v1 prompt runs are also a new Z.ai experiment series.",
        },
        "provenance_artifact": "zai_provenance.json",
        "value_manifest": "value_manifest.json",
    }
    _write_json(run_dir / "manifest.json", manifest)
    all_rows: list[dict[str, object]] = []
    condition_rows: dict[str, list[dict[str, object]]] = {condition: [] for condition in conditions}
    global_request_semaphore = threading.BoundedSemaphore(effective_api_concurrency)

    def checkpoint() -> None:
        for condition in conditions:
            _write_csv(run_dir / f"{condition}_games.csv", condition_rows[condition])
        _write_csv(run_dir / "all_games.csv", all_rows)
        _write_json(run_dir / "zai_provenance.json", recorder.as_dict())
        manifest["token_totals"] = recorder.token_totals()
        manifest["adaptive_rate_state_by_api_key"] = {
            env_name: pacer.snapshot() if pacer else None
            for env_name, pacer in adaptive_pacers.items()
        }
        _write_json(run_dir / "manifest.json", manifest)
        write_value_manifest(value_manifest_path, value_manifest)

    try:
        for game_index in range(int(cfg["games"])):
            game_seed = int(cfg["seed"]) + game_index
            if cfg["random_persona"]:
                args.random_seed = cfg["random_seed"] if cfg["random_seed"] is not None else game_seed
                personas, persona_params, role_keys = load_personas(args)
            ordered_conditions = counterbalanced_condition_order(
                conditions, experiment_seed=experiment_seed, game_seed=game_seed
            )
            key_assignment = api_key_assignment_for_seed(
                conditions, api_key_envs, experiment_seed=experiment_seed, game_seed=game_seed
            )
            conditions_by_key: dict[str, list[str]] = {env_name: [] for env_name in api_key_envs}
            for condition in ordered_conditions:
                conditions_by_key[key_assignment[condition]].append(condition)
            if effective_api_concurrency == 1 and cfg["parallel_conditions"]:
                shared_scheduler = ConditionRequestScheduler(
                    ordered_conditions, 1, fair_serial=True
                )
                request_schedulers = {
                    env_name: shared_scheduler for env_name in api_key_envs
                }
            else:
                request_schedulers = {
                    env_name: ConditionRequestScheduler(
                        assigned_conditions,
                        int(cfg["api_concurrency_per_key"]),
                        fair_serial=bool(cfg["parallel_conditions"]),
                    )
                    for env_name, assigned_conditions in conditions_by_key.items()
                    if assigned_conditions
                }

            def run_condition(condition: str) -> tuple[str, list[dict[str, object]]]:
                usage_tokenizer = UsageAwareTokenizer()
                api_key_env = key_assignment[condition]
                request_scheduler = request_schedulers[api_key_env]
                try:
                    runner = ZaiPromptRunner(
                        client=clients[api_key_env],
                        recorder=recorder,
                        model=str(cfg["model"]),
                        seed=game_seed,
                        condition=condition,
                        do_sample=bool(cfg["do_sample"]),
                        temperature=float(cfg["temperature"]),
                        max_total_tokens=int(cfg["max_total_tokens"]),
                        usage_tokenizer=usage_tokenizer,
                        request_scheduler=request_scheduler,
                        request_semaphore=global_request_semaphore,
                        adaptive_pacer=adaptive_pacers[api_key_env],
                        adaptive_max_retries=int(cfg["adaptive_max_retries"]),
                        compact_prompts=bool(cfg["compact_prompts"]),
                        thinking_enabled=bool(cfg["thinking_enabled"]),
                        request_interval_seconds=float(cfg["request_interval_seconds"]),
                    )
                    rows = run_one_game(
                        None, usage_tokenizer, condition, game_seed, personas, persona_params, role_keys,
                        max_new_tokens=int(cfg["max_new_tokens"]),
                        max_discussion_turns=int(cfg["max_discussion_turns"]),
                        discussion_token_budget=int(cfg["discussion_token_budget"]),
                        evaluator_rollouts=int(cfg["evaluator_rollouts"]),
                        decision_schedule_seed=int(cfg["decision_schedule_seed"]),
                        max_decision_opportunities=int(cfg["max_decision_opportunities"]),
                        role_value_mode=cfg["role_value_mode"],
                        prompt_runner=runner,
                    )
                    return condition, rows
                finally:
                    request_scheduler.unregister(condition)

            if cfg["parallel_conditions"] and len(ordered_conditions) > 1:
                errors: list[BaseException] = []
                with ThreadPoolExecutor(
                    max_workers=len(ordered_conditions), thread_name_prefix=f"zai-seed-{game_seed}"
                ) as executor:
                    futures = [executor.submit(run_condition, condition) for condition in ordered_conditions]
                    for future in as_completed(futures):
                        try:
                            condition, rows = future.result()
                        except BaseException as exc:
                            errors.append(exc)
                            continue
                        append_profile_assignment(
                            value_manifest, game_seed, personas, persona_params, role_keys, condition=condition
                        )
                        condition_rows[condition].extend(rows)
                        all_rows.extend(rows)
                        manifest["completed_game_conditions"] = int(manifest["completed_game_conditions"]) + 1
                        checkpoint()
                if errors:
                    raise errors[0]
            else:
                for condition in ordered_conditions:
                    completed_condition, rows = run_condition(condition)
                    append_profile_assignment(
                        value_manifest, game_seed, personas, persona_params, role_keys,
                        condition=completed_condition,
                    )
                    condition_rows[completed_condition].extend(rows)
                    all_rows.extend(rows)
                    manifest["completed_game_conditions"] = int(manifest["completed_game_conditions"]) + 1
                    checkpoint()

        summaries: list[dict[str, object]] = []
        for condition in conditions:
            summaries.append({
                "condition": condition,
                "games": int(cfg["games"]),
                **compute_summary_metrics(condition_rows[condition]),
            })
        _write_csv(run_dir / "summary.csv", summaries)
        manifest.update({"status": "completed", "completed_at": _now_iso()})
    except BaseException:
        manifest.update({"status": "failed", "completed_at": _now_iso()})
        raise
    finally:
        checkpoint()


if __name__ == "__main__":
    main()
