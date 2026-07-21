from __future__ import annotations

import json
import pickle
import sys
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

import pytest

from hivc_sim.turn_game import Action
from scripts import devin_two_agent_experiment as experiment
from scripts.devin_api import (
    DevinAPIError,
    DevinClient,
    RequestTrace,
    Session,
    SessionMessage,
    SessionStatus,
)
from scripts.llm_turn_game_common import get_action
from scripts.local_preview import PreviewServer


def _json_response(data: object, status: int = 200, headers: dict[str, str] | None = None):
    return status, headers or {}, json.dumps(data).encode()


def _session(session_id: str, status: str = "running", acu: float = 0.0, archived: bool = False) -> dict:
    return {
        "session_id": session_id,
        "status": status,
        "status_detail": "working",
        "acus_consumed": acu,
        "is_archived": archived,
    }


def test_secret_only_enters_authorization_and_never_sanitized_error(monkeypatch) -> None:
    secret = "cog_super_secret_value"
    monkeypatch.setenv("DEVIN_API_KEY", secret)
    monkeypatch.setenv("DEVIN_ORG_ID", "org-test")
    seen = {}

    def transport(request, _timeout):
        seen["authorization"] = request.get_header("Authorization")
        seen["body"] = request.data
        return _json_response({"detail": secret}, status=401)

    client = DevinClient.from_env(transport=transport, max_retries=0)
    with pytest.raises(DevinAPIError) as caught:
        client.get_session("devin-one")
    assert secret not in str(caught.value)
    assert seen["authorization"] == f"Bearer {secret}"
    assert secret.encode() not in (seen["body"] or b"")
    assert secret not in repr(client)
    with pytest.raises(TypeError):
        pickle.dumps(client)


def test_client_session_endpoints_use_documented_v3_shapes(monkeypatch) -> None:
    monkeypatch.setenv("DEVIN_API_KEY", "cog_test")
    monkeypatch.setenv("DEVIN_ORG_ID", "org-test")
    calls = []

    def transport(request, _timeout):
        parsed = urllib.parse.urlparse(request.full_url)
        payload = json.loads(request.data) if request.data else None
        calls.append((request.method, parsed.path, urllib.parse.parse_qs(parsed.query), payload))
        return _json_response(_session("devin-one", status="exit", acu=0.5, archived="archive" in parsed.path))

    client = DevinClient(transport=transport)
    client.create_session(prompt="task", title="title", max_acu_limit=1, tags=["test"])
    client.get_session("devin-one")
    client.send_message("devin-one", "next")
    client.archive_session("devin-one")
    client.terminate_session("devin-one", archive=True)
    assert [(method, path) for method, path, _query, _payload in calls] == [
        ("POST", "/v3/organizations/org-test/sessions"),
        ("GET", "/v3/organizations/org-test/sessions/devin-one"),
        ("POST", "/v3/organizations/org-test/sessions/devin-one/messages"),
        ("POST", "/v3/organizations/org-test/sessions/devin-one/archive"),
        ("DELETE", "/v3/organizations/org-test/sessions/devin-one"),
    ]
    assert calls[0][3]["max_acu_limit"] == 1
    assert calls[-1][2] == {"archive": ["true"]}


def test_org_discovery_uses_self_and_fails_without_org(monkeypatch) -> None:
    monkeypatch.setenv("DEVIN_API_KEY", "cog_test")
    monkeypatch.delenv("DEVIN_ORG_ID", raising=False)
    paths = []

    def transport(request, _timeout):
        paths.append(urllib.parse.urlparse(request.full_url).path)
        return _json_response({"principal_type": "service_user", "org_id": "org-discovered"})

    client = DevinClient.from_env(transport=transport)
    assert client.resolve_org_id() == "org-discovered"
    assert paths == ["/v3/self"]

    client = DevinClient(transport=lambda *_: _json_response({"org_id": None}))
    with pytest.raises(DevinAPIError, match="no org_id"):
        client.resolve_org_id()


def test_http_retries_429_and_5xx_with_same_request_correlation_id(monkeypatch) -> None:
    monkeypatch.setenv("DEVIN_API_KEY", "cog_test")
    monkeypatch.setenv("DEVIN_ORG_ID", "org-test")
    statuses = [429, 503, 200]
    request_ids = []
    sleeps = []

    def transport(request, _timeout):
        request_ids.append(request.get_header("X-request-id"))
        status = statuses.pop(0)
        if status == 200:
            return _json_response(_session("devin-ok"))
        return _json_response({}, status=status, headers={"Retry-After": "0"})

    client = DevinClient(
        transport=transport,
        max_retries=3, backoff_seconds=0, sleep=sleeps.append,
    )
    assert client.get_session("devin-ok").session_id == "devin-ok"
    assert len(request_ids) == 3
    assert len(set(request_ids)) == 1
    assert len(sleeps) == 2


def test_paginated_messages_follow_cursor_and_preserve_order(monkeypatch) -> None:
    monkeypatch.setenv("DEVIN_API_KEY", "cog_test")
    monkeypatch.setenv("DEVIN_ORG_ID", "org-test")
    after_values = []

    def transport(request, _timeout):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
        after = query.get("after", [None])[0]
        after_values.append(after)
        if after is None:
            return _json_response({
                "items": [{"event_id": "e1", "source": "devin", "message": "one", "created_at": 1}],
                "end_cursor": "cursor-1", "has_next_page": True, "total": 2,
            })
        return _json_response({
            "items": [{"event_id": "e2", "source": "devin", "message": "two", "created_at": 2}],
            "end_cursor": "cursor-2", "has_next_page": False, "total": 2,
        })

    client = DevinClient(transport=transport)
    messages, cursor = client.list_messages("devin-one")
    assert [message.event_id for message in messages] == ["e1", "e2"]
    assert after_values == [None, "cursor-1"]
    assert cursor == "cursor-2"


class _AllocationClient:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.initialization_ids: dict[str, str] = {}

    def create_session(self, **kwargs):
        self.created.append(kwargs)
        session_id = f"devin-{len(self.created)}"
        marker = '"request_correlation_id":"'
        self.initialization_ids[session_id] = kwargs["prompt"].split(marker, 1)[1].split('"', 1)[0]
        return Session(session_id, SessionStatus.NEW, None, 0.0, False)

    def list_messages(self, session_id, *, after=None, first=100):
        if after is not None:
            return [], after
        correlation_id = self.initialization_ids[session_id]
        return [SessionMessage(
            f"ready-{session_id}", "devin",
            json.dumps({"request_correlation_id": correlation_id, "ready": True}), 1,
        )], f"cursor-{session_id}"

    def get_session(self, session_id):
        return Session(session_id, SessionStatus.RUNNING, "finished", 0.05, False)


def test_allocator_creates_independent_sessions_and_never_reuses_across_games() -> None:
    client = _AllocationClient()
    recorder = experiment.ProvenanceRecorder()
    cfg = dict(experiment.DEFAULTS)
    allocator = experiment.SessionAllocator(client, recorder, cfg)  # type: ignore[arg-type]
    first = allocator.allocate(42, "control")
    second = allocator.allocate(43, "control")
    ids = [channel.session.session_id for channels in (first, second) for channel in channels.values()]
    assert len(ids) == len(set(ids)) == 4
    assert first["alpha"].session.session_id != first["beta"].session.session_id
    assert [(item["seed"], item["agent"]) for item in recorder.sessions] == [
        (42, "alpha"), (42, "beta"), (43, "alpha"), (43, "beta")
    ]


class _RoutedChannel:
    def __init__(self, agent: str, session_id: str) -> None:
        self.agent = agent
        self.session = SimpleNamespace(session_id=session_id)
        self.prompts: list[str] = []

    def __call__(self, agent: str, prompt: str, **_kwargs):
        if agent != self.agent:
            raise RuntimeError(f"private prompt routing violation: {agent} cannot use {self.agent} session")
        self.prompts.append(prompt)
        return "", '{"action":"A","reason":"r","message":"m","ready":true}'


def test_prompt_router_preserves_private_agent_boundaries_and_common_injection() -> None:
    alpha = _RoutedChannel("alpha", "devin-alpha")
    beta = _RoutedChannel("beta", "devin-beta")
    router = experiment.PromptRouter({"alpha": alpha, "beta": beta})  # type: ignore[arg-type]
    action, *_ = get_action(
        None, None, "alpha-private", 32, Action.STABILIZE_OXYGEN,
        prompt_runner=router, agent="alpha",
    )
    assert action.value == Action.STABILIZE_OXYGEN.value
    assert alpha.prompts == ["alpha-private"]
    assert beta.prompts == []
    with pytest.raises(RuntimeError, match="routing violation"):
        alpha("beta", "must-not-cross")


class _ChannelClient:
    def __init__(self) -> None:
        self.correlation_id = ""
        self.list_calls = 0
        self.send_calls = 0

    def list_messages(self, _session_id, *, after=None, first=100):
        self.list_calls += 1
        if self.list_calls == 1:  # pre-send drain
            return [SessionMessage("old", "devin", "stale bootstrap", 0)], "baseline"
        if self.list_calls == 2:
            return [
                SessionMessage("stale", "devin", '{"request_correlation_id":"wrong"}', 1),
                SessionMessage("bad", "devin", f'prefix {{"request_correlation_id":"{self.correlation_id}"}}', 2),
                SessionMessage("good", "devin", json.dumps({
                    "request_correlation_id": self.correlation_id,
                    "action": "A", "reason": "ok", "message": "done", "ready": True,
                }), 3),
            ], "latest"
        return [], after

    def send_message(self, session_id, message):
        self.send_calls += 1
        marker = '"request_correlation_id":"'
        self.correlation_id = message.split(marker, 1)[1].split('"', 1)[0]
        return Session(session_id, SessionStatus.RUNNING, "working", 0.25, False)

    def get_session(self, session_id):
        return Session(session_id, SessionStatus.RUNNING, "working", 0.25, False)


def test_channel_requires_strict_json_and_matching_correlation_and_records_events() -> None:
    client = _ChannelClient()
    recorder = experiment.ProvenanceRecorder()
    session = Session("devin-alpha", SessionStatus.NEW, None, 0.0, False)
    recorder.add_session(session=session, agent="alpha", seed=42, condition="control", initialization_correlation_id="init")
    channel = experiment.DevinPromptChannel(
        client=client, recorder=recorder, session=session, agent="alpha",
        poll_interval_seconds=0, response_timeout_seconds=1, response_max_retries=1,
        max_total_acu=6, sleep=lambda _: None,
    )
    _, raw = channel("alpha", "private prompt")
    assert json.loads(raw)["action"] == "A"
    exchange = recorder.exchanges[0]
    assert exchange["message_event_ids"] == ["stale", "bad", "good"]
    assert exchange["stale_response_count"] == 1
    assert exchange["malformed_response_count"] == 1
    assert "private prompt" not in json.dumps(recorder.as_dict())


def test_channel_retries_stale_response_within_configured_bound() -> None:
    class RetryClient:
        def __init__(self):
            self.list_calls = 0
            self.send_calls = 0
            self.correlation_id = ""
            self.get_calls = 0

        def list_messages(self, _session_id, *, after=None, first=100):
            self.list_calls += 1
            if self.list_calls == 1:
                return [], "baseline"
            if self.list_calls == 2:
                return [SessionMessage("stale", "devin", '{"request_correlation_id":"old"}', 1)], "c1"
            return [SessionMessage("fixed", "devin", json.dumps({
                "request_correlation_id": self.correlation_id, "action": "A"
            }), 2)], "c2"

        def send_message(self, session_id, message):
            self.send_calls += 1
            marker = '"request_correlation_id":"'
            self.correlation_id = message.split(marker, 1)[1].split('"', 1)[0]
            return Session(session_id, SessionStatus.RUNNING, "working", 0.1, False)

        def get_session(self, session_id):
            self.get_calls += 1
            if self.get_calls == 1:
                return Session(session_id, SessionStatus.SUSPENDED, "inactivity", 0.1, False)
            return Session(session_id, SessionStatus.RUNNING, "waiting_for_user", 0.2, False)

    client = RetryClient()
    recorder = experiment.ProvenanceRecorder()
    session = Session("devin-alpha", SessionStatus.NEW, None, 0, False)
    recorder.add_session(session=session, agent="alpha", seed=42, condition="control", initialization_correlation_id="init")
    channel = experiment.DevinPromptChannel(
        client=client, recorder=recorder, session=session, agent="alpha",
        poll_interval_seconds=0, response_timeout_seconds=1, response_max_retries=1,
        max_total_acu=6, sleep=lambda _: None,
    )
    channel("alpha", "private")
    assert client.send_calls == 2
    assert recorder.exchanges[0]["response_retry_count"] == 1
    assert recorder.exchanges[0]["stale_response_count"] == 1


def test_dry_run_has_no_network_and_reports_six_sessions(monkeypatch, capsys) -> None:
    monkeypatch.setattr(experiment.DevinClient, "from_env", lambda **_: pytest.fail("network client created"))
    monkeypatch.setattr(sys, "argv", ["devin-experiment", "--dry-run", "--run-id", "planned"])
    experiment.main()
    output = capsys.readouterr().out
    plan = json.loads(output)
    assert plan["network_calls"] == 0
    assert plan["planned_session_count"] == 6
    assert plan["conditions"] == ["control", "consulting", "hivc_d"]


def test_validate_env_makes_only_self_request_and_prints_non_secret_readiness(monkeypatch, capsys) -> None:
    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.org_id = None

        def get_self(self):
            self.calls += 1
            return {"principal_type": "service_user", "service_user_id": "svc-1", "org_id": "org-1"}

    fake = FakeClient()
    monkeypatch.setattr(experiment.DevinClient, "from_env", lambda **_: fake)
    experiment.validate_env()
    output = json.loads(capsys.readouterr().out)
    assert fake.calls == 1
    assert output["ready_for_organization_api"] is True


def test_output_manifest_and_csv_are_preview_compatible(tmp_path: Path, monkeypatch) -> None:
    secret = "cog_must_not_reach_artifacts"
    monkeypatch.setenv("DEVIN_API_KEY", secret)
    class FakeClient:
        def resolve_org_id(self):
            return "org-test"

        def terminate_session(self, session_id, archive=True):
            return Session(session_id, SessionStatus.EXIT, "finished", 0.1, archive)

    class FakeAllocator:
        counter = 0

        def __init__(self, *_args):
            pass

        def allocate(self, _seed, _condition):
            FakeAllocator.counter += 2
            return {
                "alpha": SimpleNamespace(session=Session(f"devin-{FakeAllocator.counter - 1}", SessionStatus.NEW, None, 0, False)),
                "beta": SimpleNamespace(session=Session(f"devin-{FakeAllocator.counter}", SessionStatus.NEW, None, 0, False)),
            }

    monkeypatch.setattr(experiment.DevinClient, "from_env", lambda **_: FakeClient())
    monkeypatch.setattr(experiment, "SessionAllocator", FakeAllocator)
    monkeypatch.setattr(experiment, "run_one_game", lambda *_args, **kwargs: [{
        "condition": _args[2], "seed": _args[3], "turn": 1, "group_action": "A"
    }])
    monkeypatch.setattr(experiment, "compute_summary_metrics", lambda rows: {"turn_rows": len(rows)})
    monkeypatch.setattr(sys, "argv", [
        "devin-experiment", "--output-dir", str(tmp_path), "--run-id", "smoke",
    ])
    experiment.main()
    run_dir = tmp_path / "smoke"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["backend"] == "devin_api"
    assert manifest["planned_session_count"] == 6
    assert manifest["scientific_comparability"]["comparable_to_qwen_gpu_runs"] is False
    assert (run_dir / "all_games.csv").is_file()
    assert (run_dir / "summary.csv").is_file()
    runs = PreviewServer(tmp_path, 0, "127.0.0.1")._list_runs()
    assert runs[0]["run_id"] == "smoke"
    assert "all_games.csv" in runs[0]["files"]
    for artifact in run_dir.iterdir():
        if artifact.is_file():
            assert secret not in artifact.read_text(encoding="utf-8")
