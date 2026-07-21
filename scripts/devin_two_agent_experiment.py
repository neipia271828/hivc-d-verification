"""Run the turn-game through two isolated Devin API v3 organization sessions."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "hivc_sim"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import merge_config_and_cli  # noqa: E402
from devin_api import DevinAPIError, DevinClient, RequestTrace, Session, SessionStatus  # noqa: E402
from llm_turn_game_common import (  # noqa: E402
    CONDITIONS,
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


ARG_TYPES: dict[str, type] = {
    "conditions": list,
    "games": int,
    "seed": int,
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
    "poll_interval_seconds": float,
    "response_timeout_seconds": float,
    "response_max_retries": int,
    "max_acu_per_session": int,
    "max_total_acu": float,
}

DEFAULTS: dict[str, object] = {
    "conditions": list(CONDITIONS),
    "games": 1,
    "seed": 42,
    "max_new_tokens": 256,
    "max_discussion_turns": 6,
    "discussion_token_budget": 1536,
    "evaluator_rollouts": 24,
    "output_dir": "hivc_sim/results/turn_game/devin/runs",
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
    "request_timeout_seconds": 20.0,
    "http_max_retries": 3,
    "http_backoff_seconds": 1.0,
    "poll_interval_seconds": 5.0,
    "response_timeout_seconds": 120.0,
    "response_max_retries": 2,
    "max_acu_per_session": 1,
    "max_total_acu": 6.0,
}
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _strict_correlated_json(text: str, correlation_id: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("request_correlation_id") != correlation_id:
        return None
    return payload


class ProvenanceRecorder:
    """Records non-secret, non-prompt API/session evidence."""

    def __init__(self) -> None:
        self.http_requests: list[dict[str, Any]] = []
        self.sessions: list[dict[str, Any]] = []
        self.exchanges: list[dict[str, Any]] = []
        self._session_entries: dict[str, dict[str, Any]] = {}

    def observe_request(self, trace: RequestTrace) -> None:
        self.http_requests.append({
            "request_id": trace.request_id,
            "method": trace.method,
            "path": trace.path,
            "status_code": trace.status_code,
            "latency_seconds": trace.latency_seconds,
            "retry_count": trace.retry_count,
        })

    def add_session(self, *, session: Session, agent: str, seed: int, condition: str, initialization_correlation_id: str) -> None:
        entry = {
            "session_id": session.session_id,
            "agent": agent,
            "seed": seed,
            "condition": condition,
            "initialization_correlation_id": initialization_correlation_id,
            "status_lifecycle": [session.status.value],
            "reported_acu_consumption": session.acus_consumed,
            "archived": session.is_archived,
        }
        self.sessions.append(entry)
        self._session_entries[session.session_id] = entry

    def observe_session(self, session: Session) -> None:
        entry = self._session_entries.get(session.session_id)
        if entry is None:
            return
        lifecycle = entry["status_lifecycle"]
        if not lifecycle or lifecycle[-1] != session.status.value:
            lifecycle.append(session.status.value)
        entry["reported_acu_consumption"] = session.acus_consumed
        entry["archived"] = session.is_archived

    def total_reported_acu(self) -> float:
        return sum(float(item["reported_acu_consumption"]) for item in self.sessions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "devin-provenance-v1",
            "contains_private_prompts": False,
            "contains_credentials": False,
            "http_requests": self.http_requests,
            "sessions": self.sessions,
            "exchanges": self.exchanges,
            "total_reported_acu_consumption": self.total_reported_acu(),
        }


class DevinPromptChannel:
    """One game-local, agent-private Devin session."""

    def __init__(
        self,
        *,
        client: DevinClient,
        recorder: ProvenanceRecorder,
        session: Session,
        agent: str,
        poll_interval_seconds: float,
        response_timeout_seconds: float,
        response_max_retries: int,
        max_total_acu: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.recorder = recorder
        self.session = session
        self.agent = agent
        self.cursor: str | None = None
        self.seen_event_ids: set[str] = set()
        self.poll_interval_seconds = poll_interval_seconds
        self.response_timeout_seconds = response_timeout_seconds
        self.response_max_retries = response_max_retries
        self.max_total_acu = max_total_acu
        self.sleep = sleep

    def await_initialization(self, correlation_id: str) -> None:
        """Wait until the create-session prompt has completed before POSTing messages."""
        deadline = time.monotonic() + self.response_timeout_seconds
        while time.monotonic() < deadline:
            messages, cursor = self.client.list_messages(self.session.session_id, after=self.cursor)
            self.cursor = cursor or self.cursor
            for item in messages:
                self.seen_event_ids.add(item.event_id)
                if item.source != "devin":
                    continue
                payload = _strict_correlated_json(item.message, correlation_id)
                if payload is not None and payload.get("ready") is True:
                    self.session = self.client.get_session(self.session.session_id)
                    self.recorder.observe_session(self.session)
                    if self.recorder.total_reported_acu() > self.max_total_acu:
                        raise DevinAPIError("Configured total ACU safety limit was exceeded")
                    return
            self.session = self.client.get_session(self.session.session_id)
            self.recorder.observe_session(self.session)
            if self.session.status is SessionStatus.ERROR:
                raise DevinAPIError("Devin initialization session entered error status")
            if self.session.status is SessionStatus.SUSPENDED and self.session.status_detail not in {
                "inactivity", "user_request"
            }:
                raise DevinAPIError("Devin initialization session was suspended by a usage or account limit")
            self.sleep(self.poll_interval_seconds)
        raise DevinAPIError("Devin initialization response timed out")

    def _drain_current(self) -> None:
        messages, cursor = self.client.list_messages(self.session.session_id, after=self.cursor)
        self.cursor = cursor or self.cursor
        self.seen_event_ids.update(message.event_id for message in messages)

    def __call__(self, agent: str, prompt: str, **_kwargs: Any) -> tuple[str, str]:
        if agent != self.agent:
            raise RuntimeError(f"private prompt routing violation: {agent} cannot use {self.agent} session")
        self._drain_current()
        correlation_id = str(uuid.uuid4())
        event_ids: list[str] = []
        stale_count = 0
        malformed_count = 0
        started = time.monotonic()

        for response_attempt in range(self.response_max_retries + 1):
            if response_attempt == 0:
                message = (
                    f"REQUEST_CORRELATION_ID: {correlation_id}\n"
                    + prompt
                    + "\n\nFINAL RESPONSE CONTRACT: Return exactly one JSON object, with no Markdown or surrounding text. "
                    f"Add \"request_correlation_id\":\"{correlation_id}\" to the object required above."
                )
            else:
                message = (
                    f"REPAIR_REQUEST for REQUEST_CORRELATION_ID: {correlation_id}. "
                    "The previous response was missing, stale, or not one strict JSON object. "
                    f"Return only the corrected JSON object and include \"request_correlation_id\":\"{correlation_id}\"."
                )
            self.session = self.client.send_message(self.session.session_id, message)
            self.recorder.observe_session(self.session)
            deadline = time.monotonic() + self.response_timeout_seconds
            while time.monotonic() < deadline:
                messages, cursor = self.client.list_messages(self.session.session_id, after=self.cursor)
                self.cursor = cursor or self.cursor
                saw_devin_message = False
                for item in messages:
                    if item.event_id in self.seen_event_ids:
                        continue
                    self.seen_event_ids.add(item.event_id)
                    if item.source != "devin":
                        continue
                    saw_devin_message = True
                    event_ids.append(item.event_id)
                    payload = _strict_correlated_json(item.message, correlation_id)
                    if payload is not None:
                        self.session = self.client.get_session(self.session.session_id)
                        self.recorder.observe_session(self.session)
                        latency = round(time.monotonic() - started, 6)
                        self.recorder.exchanges.append({
                            "request_correlation_id": correlation_id,
                            "session_id": self.session.session_id,
                            "agent": self.agent,
                            "message_event_ids": event_ids,
                            "latency_seconds": latency,
                            "response_retry_count": response_attempt,
                            "stale_response_count": stale_count,
                            "malformed_response_count": malformed_count,
                            "reported_acu_consumption": self.session.acus_consumed,
                            "status": self.session.status.value,
                        })
                        if self.recorder.total_reported_acu() > self.max_total_acu:
                            raise DevinAPIError("Configured total ACU safety limit was exceeded")
                        return "", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    if correlation_id in item.message:
                        malformed_count += 1
                    else:
                        stale_count += 1
                self.session = self.client.get_session(self.session.session_id)
                self.recorder.observe_session(self.session)
                if self.session.status.terminal:
                    break
                if saw_devin_message and self.session.status_detail in {"waiting_for_user", "finished"}:
                    break
                self.sleep(self.poll_interval_seconds)
        self.recorder.exchanges.append({
            "request_correlation_id": correlation_id,
            "session_id": self.session.session_id,
            "agent": self.agent,
            "message_event_ids": event_ids,
            "latency_seconds": round(time.monotonic() - started, 6),
            "response_retry_count": self.response_max_retries,
            "stale_response_count": stale_count,
            "malformed_response_count": malformed_count,
            "reported_acu_consumption": self.session.acus_consumed,
            "status": self.session.status.value,
            "failed": True,
        })
        raise DevinAPIError("Devin response did not satisfy strict JSON/correlation requirements")


class SessionAllocator:
    """Allocates exactly two fresh sessions for each seed-condition game."""

    def __init__(self, client: DevinClient, recorder: ProvenanceRecorder, cfg: dict[str, Any]) -> None:
        self.client = client
        self.recorder = recorder
        self.cfg = cfg
        self.allocated_ids: set[str] = set()

    def allocate(self, seed: int, condition: str) -> dict[str, DevinPromptChannel]:
        channels: dict[str, DevinPromptChannel] = {}
        try:
            for agent in ("alpha", "beta"):
                correlation_id = str(uuid.uuid4())
                bootstrap = (
                    "This is a private, single-agent experimental session. Do not infer or simulate the other agent. "
                    "Wait for turn-game prompts. Return exactly one JSON object: "
                    f'{{"request_correlation_id":"{correlation_id}","ready":true}}'
                )
                session = self.client.create_session(
                    prompt=bootstrap,
                    title=f"HIVC-D Devin {condition} seed {seed} {agent}",
                    max_acu_limit=int(self.cfg["max_acu_per_session"]),
                    tags=["hivc-d-experiment", condition, agent],
                )
                if session.session_id in self.allocated_ids:
                    raise DevinAPIError("Devin returned a session ID that was already allocated")
                self.allocated_ids.add(session.session_id)
                self.recorder.add_session(
                    session=session,
                    agent=agent,
                    seed=seed,
                    condition=condition,
                    initialization_correlation_id=correlation_id,
                )
                channels[agent] = DevinPromptChannel(
                    client=self.client,
                    recorder=self.recorder,
                    session=session,
                    agent=agent,
                    poll_interval_seconds=float(self.cfg["poll_interval_seconds"]),
                    response_timeout_seconds=float(self.cfg["response_timeout_seconds"]),
                    response_max_retries=int(self.cfg["response_max_retries"]),
                    max_total_acu=float(self.cfg["max_total_acu"]),
                )
                channels[agent].await_initialization(correlation_id)
        except BaseException:
            for channel in channels.values():
                try:
                    ended = self.client.terminate_session(channel.session.session_id, archive=True)
                    self.recorder.observe_session(ended)
                except DevinAPIError:
                    try:
                        archived = self.client.archive_session(channel.session.session_id)
                        self.recorder.observe_session(archived)
                    except DevinAPIError:
                        pass
            raise
        return channels


class PromptRouter:
    def __init__(self, channels: dict[str, DevinPromptChannel]) -> None:
        if set(channels) != {"alpha", "beta"}:
            raise ValueError("prompt router requires independent alpha and beta channels")
        if channels["alpha"].session.session_id == channels["beta"].session.session_id:
            raise ValueError("alpha and beta must not share a Devin session")
        self.channels = channels

    def __call__(self, agent: str, prompt: str, **kwargs: Any) -> tuple[str, str]:
        try:
            channel = self.channels[agent]
        except KeyError:
            raise ValueError(f"unknown turn-game agent: {agent}") from None
        return channel(agent, prompt, **kwargs)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a two-session Devin API turn-game experiment")
    parser.add_argument("--config", default="configs/devin_experiment.yaml")
    parser.add_argument("--conditions", nargs="*", choices=list(CONDITIONS) + ["all"])
    parser.add_argument("--games", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
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
    if int(cfg["max_acu_per_session"]) < 1 or float(cfg["max_total_acu"]) <= 0:
        raise ValueError("ACU limits must be positive")
    return cfg


def _persona_args(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    for key in (
        "role_file", "alpha_role_key", "beta_role_key", "personas_file", "persona_params_file",
        "alpha_persona", "beta_persona", "random_persona", "random_seed", "role_value_mode",
    ):
        setattr(args, key, cfg.get(key))


def validate_env() -> None:
    """Make only GET /v3/self and print non-secret readiness metadata."""
    client = DevinClient.from_env(max_retries=0)
    identity = client.get_self()
    org_id = client.org_id or identity.get("org_id")
    if not isinstance(org_id, str) or not org_id:
        raise DevinAPIError("Authentication succeeded but GET /v3/self returned no org_id")
    print(json.dumps({
        "authenticated": True,
        "principal_type": identity.get("principal_type"),
        "service_user_id": identity.get("service_user_id"),
        "service_user_name": identity.get("service_user_name"),
        "org_id": org_id,
        "ready_for_organization_api": True,
    }, ensure_ascii=False, sort_keys=True))


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_cfg(args)
    _persona_args(args, cfg)
    conditions = list(cfg["conditions"])
    planned_sessions = int(cfg["games"]) * len(conditions) * 2
    output_root = Path(str(cfg["output_dir"]))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    run_id = args.run_id or dt.datetime.now().strftime("devin-%Y%m%d-%H%M%S")
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run_id: {run_id!r}")
    run_dir = output_root / run_id

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "network_calls": 0,
            "backend": "devin_api",
            "conditions": conditions,
            "seeds": [int(cfg["seed"]) + index for index in range(int(cfg["games"]))],
            "planned_session_count": planned_sessions,
            "sessions_per_seed_condition": 2,
            "output": str(run_dir),
            "scientifically_comparable_to_qwen_gpu": False,
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse existing Devin run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    recorder = ProvenanceRecorder()
    client = DevinClient.from_env(
        timeout=float(cfg["request_timeout_seconds"]),
        max_retries=int(cfg["http_max_retries"]),
        backoff_seconds=float(cfg["http_backoff_seconds"]),
        trace_observer=recorder.observe_request,
    )
    org_id = client.resolve_org_id()
    print(f"Devin organization ready: {org_id}; planned sessions: {planned_sessions}")

    personas, persona_params, role_keys = load_personas(args)
    value_manifest = build_value_manifest(
        cfg, personas, persona_params, role_keys,
        role_value_mode=str(cfg["role_value_mode"]),
        framework_ids=conditions,
        runner_version="devin_two_agent_experiment-v1",
    )
    value_manifest_path = run_dir / "value_manifest.json"
    write_value_manifest(value_manifest_path, value_manifest)
    manifest = {
        "schema_version": "devin-experiment-manifest-v1",
        "run_id": run_id,
        "backend": "devin_api",
        "api_version": "v3",
        "organization_scope": True,
        "status": "running",
        "started_at": _now_iso(),
        "completed_at": None,
        "git_commit": _git_commit(),
        "conditions": conditions,
        "games_per_condition": int(cfg["games"]),
        "planned_session_count": planned_sessions,
        "sessions_per_seed_condition": 2,
        "scientific_comparability": {
            "comparable_to_qwen_gpu_runs": False,
            "reason": "Different model, inference service, session lifecycle, and cost/runtime backend.",
        },
        "provenance_artifact": "devin_provenance.json",
        "value_manifest": "value_manifest.json",
    }
    _write_json(run_dir / "manifest.json", manifest)
    allocator = SessionAllocator(client, recorder, cfg)
    all_rows: list[dict[str, object]] = []
    condition_rows: dict[str, list[dict[str, object]]] = {condition: [] for condition in conditions}

    try:
        for game_index in range(int(cfg["games"])):
            game_seed = int(cfg["seed"]) + game_index
            if cfg["random_persona"]:
                args.random_seed = cfg["random_seed"] if cfg["random_seed"] is not None else game_seed
                personas, persona_params, role_keys = load_personas(args)
            for condition in condition_order_for_seed(conditions, game_seed):
                channels = allocator.allocate(game_seed, condition)
                try:
                    rows = run_one_game(
                        None, None, condition, game_seed, personas, persona_params, role_keys,
                        max_new_tokens=int(cfg["max_new_tokens"]),
                        max_discussion_turns=int(cfg["max_discussion_turns"]),
                        discussion_token_budget=int(cfg["discussion_token_budget"]),
                        evaluator_rollouts=int(cfg["evaluator_rollouts"]),
                        decision_schedule_seed=int(cfg["decision_schedule_seed"]),
                        max_decision_opportunities=int(cfg["max_decision_opportunities"]),
                        role_value_mode=cfg["role_value_mode"],
                        prompt_runner=PromptRouter(channels),
                    )
                    append_profile_assignment(
                        value_manifest, game_seed, personas, persona_params, role_keys, condition=condition
                    )
                    condition_rows[condition].extend(rows)
                    all_rows.extend(rows)
                finally:
                    for channel in channels.values():
                        try:
                            ended = client.terminate_session(channel.session.session_id, archive=True)
                            recorder.observe_session(ended)
                        except DevinAPIError:
                            try:
                                archived = client.archive_session(channel.session.session_id)
                                recorder.observe_session(archived)
                            except DevinAPIError:
                                # Preserve the primary error. Sanitized traces record both cleanup failures.
                                pass
                _write_json(run_dir / "devin_provenance.json", recorder.as_dict())
                write_value_manifest(value_manifest_path, value_manifest)

        summaries: list[dict[str, object]] = []
        for condition in conditions:
            _write_csv(run_dir / f"{condition}_games.csv", condition_rows[condition])
            summaries.append({
                "condition": condition,
                "games": int(cfg["games"]),
                **compute_summary_metrics(condition_rows[condition]),
            })
        _write_csv(run_dir / "all_games.csv", all_rows)
        _write_csv(run_dir / "summary.csv", summaries)
        manifest.update({"status": "completed", "completed_at": _now_iso()})
    except BaseException:
        manifest.update({"status": "failed", "completed_at": _now_iso()})
        raise
    finally:
        _write_json(run_dir / "devin_provenance.json", recorder.as_dict())
        _write_json(run_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
