from __future__ import annotations

import json
import pickle
import sys
import threading
import urllib.parse
from collections import Counter
from pathlib import Path

import pytest

from scripts import zai_two_agent_experiment as experiment
from scripts.local_preview import PreviewServer
from scripts.zai_api import ChatCompletion, TokenUsage, ZaiAPIError, ZaiClient


def _json_response(data: object, status: int = 200, headers: dict[str, str] | None = None):
    return status, headers or {}, json.dumps(data).encode()


def _completion(content: str = '{"ready":true}', *, prompt: int = 12, output: int = 4) -> dict:
    return {
        "request_id": "zai-request-1",
        "model": "glm-4.7-flash",
        "choices": [{
            "message": {"role": "assistant", "content": content, "reasoning_content": ""},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": output,
            "total_tokens": prompt + output,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    }


def test_secret_only_enters_authorization_and_client_is_not_serializable(monkeypatch) -> None:
    secret = "zai-secret-must-not-leak"
    monkeypatch.setenv("ZAI_API_KEY", secret)
    seen = {}

    def transport(request, _timeout):
        seen["authorization"] = request.get_header("Authorization")
        seen["payload"] = json.loads(request.data)
        return _json_response({"error": {"code": 1000, "message": secret}}, status=401)

    client = ZaiClient.from_env(transport=transport, max_retries=0)
    with pytest.raises(ZaiAPIError) as caught:
        client.chat_completion(prompt="private prompt")
    assert secret not in str(caught.value)
    assert seen["authorization"] == f"Bearer {secret}"
    assert secret not in json.dumps(seen["payload"])
    with pytest.raises(TypeError):
        pickle.dumps(client)


def test_chat_completion_uses_general_api_json_mode_and_records_usage(monkeypatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    seen = {}

    def transport(request, _timeout):
        seen["path"] = urllib.parse.urlparse(request.full_url).path
        seen["payload"] = json.loads(request.data)
        return _json_response(_completion('{"action":"A"}'))

    result = ZaiClient(transport=transport).chat_completion(
        prompt="return JSON", model="glm-4.7-flash", max_tokens=256, do_sample=False,
    )
    assert seen["path"] == "/api/paas/v4/chat/completions"
    assert seen["payload"]["model"] == "glm-4.7-flash"
    assert seen["payload"]["messages"] == [{"role": "user", "content": "return JSON"}]
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["payload"]["thinking"] == {"type": "disabled"}
    assert seen["payload"]["do_sample"] is False
    assert "temperature" not in seen["payload"]
    assert result.content == '{"action":"A"}'
    assert result.usage == TokenUsage(12, 4, 2, 16)


def test_retryable_zai_code_is_retried_with_same_request_id(monkeypatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    calls = []
    sleeps = []

    def transport(request, _timeout):
        payload = json.loads(request.data)
        calls.append(payload["request_id"])
        if len(calls) == 1:
            return _json_response({"error": {"code": 1303}}, status=429, headers={"Retry-After": "0"})
        return _json_response(_completion())

    client = ZaiClient(transport=transport, max_retries=2, backoff_seconds=0, sleep=sleeps.append)
    assert json.loads(client.chat_completion(prompt="x").content)["ready"] is True
    assert len(calls) == 2 and len(set(calls)) == 1
    assert len(sleeps) == 1


def test_prompt_runner_records_exact_token_usage_without_private_prompt() -> None:
    class FakeClient:
        def chat_completion(self, **_kwargs):
            return ChatCompletion(
                content='{"action":"A"}', reasoning_content="", model="glm-4.7-flash",
                request_id="r1", finish_reason="stop", usage=TokenUsage(100, 20, 10, 120),
            )

    recorder = experiment.ProvenanceRecorder()
    tokenizer = experiment.UsageAwareTokenizer()
    runner = experiment.ZaiPromptRunner(
        client=FakeClient(), recorder=recorder, model="glm-4.7-flash", seed=42,
        condition="control", do_sample=False, temperature=0.2, max_total_tokens=1000,
        usage_tokenizer=tokenizer,
    )
    _, raw = runner("alpha", "SENSITIVE_PROMPT_BODY id=decision-contract")
    assert json.loads(raw)["action"] == "A"
    assert recorder.token_totals() == {
        "prompt_tokens": 100, "completion_tokens": 20, "cached_tokens": 10, "total_tokens": 120,
    }
    artifact = json.dumps(recorder.as_dict())
    assert "SENSITIVE_PROMPT_BODY" not in artifact
    assert recorder.completions[0]["purpose"] == "decision"
    assert tokenizer.encode(raw) == []  # final-decision tokens are not charged to discussion budget


def test_compact_prompt_preserves_contract_and_shortens_framework() -> None:
    full = (
        "【GAME_RULES_AND_JSON_CONTRACT id=discussion-contract】\n"
        + experiment.CONDITION_PROCEDURES["hivc_d"]
        + '\n【CURRENT_OBSERVATION id=state】\noxygen: 5\n'
        + '必ず次のJSONだけを返してください。説明文やMarkdownは不要です。\n'
        + '{"speech_act":"evidence","action":"A"}'
    )
    compact = experiment.compact_prompt_text(full)
    assert len(compact) < len(full) * 0.6
    assert "I→V→A" in compact
    assert "両者の明示accept" in compact
    assert "oxygen: 5" in compact
    assert '{"speech_act":"evidence","action":"A"}' in compact
    assert "id=discussion-contract" in compact


def test_adaptive_pacer_decreases_on_success_and_cools_down_on_1302() -> None:
    now = [0.0]
    sleeps = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    pacer = experiment.AdaptiveRequestPacer(
        min_interval=2.0,
        initial_interval=3.0,
        max_interval=12.0,
        cooldown_seconds=60.0,
        successes_before_decrease=2,
        decrease_seconds=0.5,
        clock=lambda: now[0],
        sleep=sleep,
    )
    pacer.before_request()
    pacer.note_success()
    pacer.before_request()
    pacer.note_success()
    assert pacer.snapshot()["current_interval_seconds"] == 2.5
    pacer.note_rate_limit()
    assert pacer.snapshot()["current_interval_seconds"] == 4.0
    pacer.before_request()
    assert sleeps[-1] == 60.0
    assert pacer.snapshot()["rate_limit_events"] == 1


def test_prompt_runner_retries_rate_limit_through_shared_adaptive_pacer() -> None:
    now = [0.0]
    calls = []

    def sleep(seconds: float) -> None:
        now[0] += seconds

    class RateLimitedClient:
        def chat_completion(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise experiment.ZaiAPIError("limited", status_code=429, api_code=1302)
            return ChatCompletion(
                content='{"ready":true}', reasoning_content="", model="glm-4.7-flash",
                request_id="r2", finish_reason="stop", usage=TokenUsage(10, 3, 0, 13),
            )

    pacer = experiment.AdaptiveRequestPacer(
        min_interval=2.0, initial_interval=2.5, max_interval=12.0,
        cooldown_seconds=60.0, clock=lambda: now[0], sleep=sleep,
    )
    runner = experiment.ZaiPromptRunner(
        client=RateLimitedClient(), recorder=experiment.ProvenanceRecorder(),
        model="glm-4.7-flash", seed=42, condition="control", do_sample=False,
        temperature=0.2, max_total_tokens=1000, adaptive_pacer=pacer,
        adaptive_max_retries=2, compact_prompts=True,
    )
    _, raw = runner("alpha", "あなたは深海研究施設トラブルの意思決定エージェントです。 id=decision-contract")
    assert json.loads(raw)["ready"] is True
    assert len(calls) == 2
    assert pacer.snapshot()["rate_limit_events"] == 1
    assert now[0] >= 60.0


def test_api_completion_tokens_feed_common_discussion_budget_tokenizer() -> None:
    class FakeClient:
        def chat_completion(self, **_kwargs):
            return ChatCompletion(
                content='{"speech_act":"evidence"}', reasoning_content="", model="glm-4.7-flash",
                request_id="r1", finish_reason="stop", usage=TokenUsage(80, 17, 0, 97),
            )

    tokenizer = experiment.UsageAwareTokenizer()
    runner = experiment.ZaiPromptRunner(
        client=FakeClient(), recorder=experiment.ProvenanceRecorder(), model="glm-4.7-flash",
        seed=42, condition="control", do_sample=False, temperature=0.2,
        max_total_tokens=1000, usage_tokenizer=tokenizer,
    )
    _, raw = runner("alpha", "id=discussion-contract")
    assert len(tokenizer.encode(raw, add_special_tokens=False)) == 17
    assert tokenizer.encode(raw) == []


def test_serial_api_scheduler_interleaves_condition_requests_round_robin() -> None:
    conditions = ["control", "consulting", "hivc_d"]
    scheduler = experiment.ConditionRequestScheduler(conditions, max_concurrency=1)
    recorder = experiment.ProvenanceRecorder()
    barrier = threading.Barrier(len(conditions))

    class FakeClient:
        def chat_completion(self, **_kwargs):
            return ChatCompletion(
                content='{"ready":true}', reasoning_content="", model="glm-4.7-flash",
                request_id="r", finish_reason="stop", usage=TokenUsage(1, 1, 0, 2),
            )

    def worker(condition: str) -> None:
        runner = experiment.ZaiPromptRunner(
            client=FakeClient(), recorder=recorder, model="glm-4.7-flash", seed=42,
            condition=condition, do_sample=False, temperature=0.2, max_total_tokens=1000,
            request_scheduler=scheduler,
        )
        barrier.wait(timeout=2)
        try:
            for _ in range(5):
                runner("alpha", "id=discussion-contract")
        finally:
            scheduler.unregister(condition)

    threads = [threading.Thread(target=worker, args=(condition,)) for condition in conditions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()

    assert [item["condition"] for item in recorder.completions] == conditions * 5


def test_sequential_condition_mode_does_not_wait_for_inactive_workers() -> None:
    scheduler = experiment.ConditionRequestScheduler(
        ["control", "consulting", "hivc_d"], max_concurrency=1, fair_serial=False
    )
    for condition in ["control", "consulting", "hivc_d"]:
        with scheduler.slot(condition):
            pass
        with scheduler.slot(condition):
            pass
        scheduler.unregister(condition)


def test_ninety_game_condition_schedule_is_position_counterbalanced() -> None:
    """Three conditions x 30 seeds produces 90 condition-games without lead bias."""
    conditions = ["control", "consulting", "hivc_d"]
    orders = [
        experiment.counterbalanced_condition_order(
            conditions, experiment_seed=42, game_seed=game_seed
        )
        for game_seed in range(42, 72)
    ]
    assert orders == [
        experiment.counterbalanced_condition_order(
            conditions, experiment_seed=42, game_seed=game_seed
        )
        for game_seed in range(42, 72)
    ]
    for position in range(len(conditions)):
        assert Counter(order[position] for order in orders) == Counter({
            "control": 10, "consulting": 10, "hivc_d": 10,
        })


def test_ninety_game_api_key_assignment_counterbalances_accounts() -> None:
    conditions = ["control", "consulting", "hivc_d"]
    key_envs = ["ZAI_API_KEY", "ZAI_API_KEY_2", "ZAI_API_KEY_3"]
    assignments = [
        experiment.api_key_assignment_for_seed(
            conditions, key_envs, experiment_seed=42, game_seed=game_seed
        )
        for game_seed in range(42, 72)
    ]
    for condition in conditions:
        assert Counter(item[condition] for item in assignments) == Counter({
            "ZAI_API_KEY": 10, "ZAI_API_KEY_2": 10, "ZAI_API_KEY_3": 10,
        })


def test_dry_run_has_no_network_and_is_pinned_to_flash(monkeypatch, capsys) -> None:
    monkeypatch.setattr(experiment.ZaiClient, "from_env", lambda **_: pytest.fail("network client created"))
    monkeypatch.setattr(sys, "argv", ["zai-experiment", "--dry-run", "--run-id", "planned"])
    experiment.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["network_calls"] == 0
    assert plan["model"] == "glm-4.7-flash"
    assert plan["planned_games"] == 3


def test_dry_run_accepts_paid_flashx_model(monkeypatch, capsys) -> None:
    monkeypatch.setattr(experiment.ZaiClient, "from_env", lambda **_: pytest.fail("network client created"))
    monkeypatch.setattr(sys, "argv", [
        "zai-experiment", "--dry-run", "--run-id", "flashx-planned",
        "--model", "glm-4.7-flashx", "--conditions", "hivc_d", "--games", "1",
    ])
    experiment.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["network_calls"] == 0
    assert plan["model"] == "glm-4.7-flashx"
    assert plan["planned_games"] == 1


def test_dry_run_accepts_glm47_flagship_model(monkeypatch, capsys) -> None:
    monkeypatch.setattr(experiment.ZaiClient, "from_env", lambda **_: pytest.fail("network client created"))
    monkeypatch.setattr(sys, "argv", [
        "zai-experiment", "--dry-run", "--run-id", "glm47-planned",
        "--model", "glm-4.7", "--conditions", "hivc_d", "--games", "1",
    ])
    experiment.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["network_calls"] == 0
    assert plan["model"] == "glm-4.7"
    assert plan["planned_games"] == 1


def test_validate_env_checks_actual_flash_json_access(monkeypatch, capsys) -> None:
    class FakeClient:
        def chat_completion(self, **kwargs):
            assert kwargs["model"] == "glm-4.7-flash"
            return ChatCompletion(
                content='{"ready":true}', reasoning_content="", model="glm-4.7-flash",
                request_id="r1", finish_reason="stop", usage=TokenUsage(8, 4, 0, 12),
            )

    monkeypatch.setattr(experiment.ZaiClient, "from_env", lambda **_: FakeClient())
    experiment.validate_env()
    output = json.loads(capsys.readouterr().out)
    assert output["ready"] is True
    assert output["model"] == "glm-4.7-flash"
    assert output["usage"]["total_tokens"] == 12


def test_output_manifest_checkpoint_and_preview_are_compatible(tmp_path: Path, monkeypatch) -> None:
    secrets = {
        "ZAI_API_KEY": "zai-secret-1-must-not-reach-artifacts",
        "ZAI_API_KEY_2": "zai-secret-2-must-not-reach-artifacts",
        "ZAI_API_KEY_3": "zai-secret-3-must-not-reach-artifacts",
    }
    for env_name, secret in secrets.items():
        monkeypatch.setenv(env_name, secret)

    class FakeClient:
        pass

    monkeypatch.setattr(experiment.ZaiClient, "__init__", lambda self, **_: None)
    monkeypatch.setattr(experiment.ZaiClient, "from_env", lambda **_: FakeClient())
    barrier = threading.Barrier(3)
    worker_names: set[str] = set()

    def fake_run_one_game(*args, **_kwargs):
        worker_names.add(threading.current_thread().name)
        barrier.wait(timeout=2)
        return [{"condition": args[2], "seed": args[3], "turn": 1, "group_action": "A"}]

    monkeypatch.setattr(experiment, "run_one_game", fake_run_one_game)
    monkeypatch.setattr(experiment, "compute_summary_metrics", lambda rows: {"turn_rows": len(rows)})
    monkeypatch.setattr(sys, "argv", [
        "zai-experiment", "--output-dir", str(tmp_path), "--run-id", "smoke",
        "--parallel-conditions",
    ])
    experiment.main()
    run_dir = tmp_path / "smoke"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["backend"] == "zai_chat_completions"
    assert manifest["model"] == "glm-4.7-flash"
    assert manifest["completed_game_conditions"] == 3
    assert manifest["status"] == "completed"
    assert manifest["parallel_conditions"] is True
    assert manifest["condition_workers"] == 3
    assert manifest["api_concurrency"] == 3
    assert manifest["api_concurrency_per_key"] == 1
    assert manifest["api_key_count"] == 3
    assert manifest["effective_api_concurrency"] == 3
    assert manifest["api_requests_interleaved"] is False
    assert manifest["execution_scheduling_mode"] == (
        "seed_counterbalanced_multi_account_parallel_api"
    )
    assert manifest["api_key_assignment_by_seed"][0]["condition_to_api_key_env"] == {
        "control": "ZAI_API_KEY",
        "consulting": "ZAI_API_KEY_2",
        "hivc_d": "ZAI_API_KEY_3",
    }
    assert manifest["condition_order_strategy"] == "deterministic_cyclic_latin_square_v1"
    assert manifest["condition_start_order_by_seed"] == [{
        "seed": 42,
        "conditions": experiment.counterbalanced_condition_order(
            ["control", "consulting", "hivc_d"], experiment_seed=42, game_seed=42
        ),
    }]
    assert manifest["thinking_enabled"] is False
    assert len(worker_names) == 3
    provenance = json.loads((run_dir / "zai_provenance.json").read_text())
    assert provenance["contains_private_prompts"] is False
    assert (run_dir / "summary.csv").is_file()
    runs = PreviewServer(tmp_path, 0, "127.0.0.1")._list_runs()
    assert runs[0]["run_id"] == "smoke"
    assert "all_games.csv" in runs[0]["files"]
    for artifact in run_dir.iterdir():
        if artifact.is_file():
            content = artifact.read_text(encoding="utf-8")
            assert all(secret not in content for secret in secrets.values())
