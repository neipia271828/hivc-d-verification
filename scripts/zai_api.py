"""Secret-safe client for Z.ai's OpenAI-compatible chat completion API."""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
RETRYABLE_API_CODES = {1302, 1303, 1305, 1308, 1312}


class ZaiAPIError(RuntimeError):
    """A sanitized Z.ai API failure that never includes prompts or credentials."""

    def __init__(self, message: str, *, status_code: int | None = None, api_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.api_code = api_code


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    reasoning_content: str
    model: str
    request_id: str
    finish_reason: str
    usage: TokenUsage


@dataclass(frozen=True)
class RequestTrace:
    request_id: str
    path: str
    status_code: int | None
    api_code: int | None
    latency_seconds: float
    retry_count: int


Transport = Callable[[urllib.request.Request, float], tuple[int, Mapping[str, str], bytes]]
TraceObserver = Callable[[RequestTrace], None]


def _default_transport(request: urllib.request.Request, timeout: float) -> tuple[int, Mapping[str, str], bytes]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.headers or {}, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ZaiAPIError("Z.ai API transport failed; request details were suppressed") from None


def _as_nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _api_error_code(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    raw = error.get("code") if isinstance(error, dict) else payload.get("code")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


class ZaiClient:
    """Small standard-library client for non-streaming JSON chat completions."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 5,
        backoff_seconds: float = 1.0,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        trace_observer: TraceObserver | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("ZAI_API_KEY", "")
        if not key:
            raise ZaiAPIError("ZAI_API_KEY is required")
        self.__api_key = key
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)
        self._transport = transport or _default_transport
        self._sleep = sleep
        self._trace_observer = trace_observer

    @classmethod
    def from_env(cls, **kwargs: Any) -> "ZaiClient":
        return cls(**kwargs)

    def __getstate__(self) -> dict[str, Any]:
        raise TypeError("ZaiClient cannot be serialized because it contains credentials")

    def chat_completion(
        self,
        *,
        prompt: str,
        model: str = "glm-4.7-flash",
        max_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.2,
        response_format: str = "json_object",
        thinking_enabled: bool = False,
    ) -> ChatCompletion:
        path = "/chat/completions"
        client_request_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "request_id": client_request_id,
            "do_sample": bool(do_sample),
            "max_tokens": int(max_tokens),
            "stream": False,
            "response_format": {"type": response_format},
            "thinking": {"type": "enabled" if thinking_enabled else "disabled"},
        }
        if do_sample:
            payload["temperature"] = float(temperature)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": "Bearer " + self.__api_key,
            "Accept": "application/json",
            "Accept-Language": "en-US,en",
            "Content-Type": "application/json",
        }
        final_status: int | None = None
        final_api_code: int | None = None
        retry_count = 0
        started = time.monotonic()
        try:
            for attempt in range(self.max_retries + 1):
                request = urllib.request.Request(self.base_url + path, data=body, headers=headers, method="POST")
                try:
                    status, response_headers, response_body = self._transport(request, self.timeout)
                except ZaiAPIError:
                    if attempt >= self.max_retries:
                        raise
                    retry_count += 1
                    self._sleep(self.backoff_seconds * (2**attempt))
                    continue
                final_status = int(status)
                try:
                    decoded = json.loads(response_body.decode("utf-8")) if response_body else {}
                except (UnicodeDecodeError, json.JSONDecodeError):
                    decoded = None
                final_api_code = _api_error_code(decoded)
                retryable = (
                    final_status == 429
                    or 500 <= final_status <= 599
                    or final_api_code in RETRYABLE_API_CODES
                )
                if retryable and attempt < self.max_retries:
                    retry_count += 1
                    try:
                        retry_after = float(response_headers.get("Retry-After", ""))
                    except (TypeError, ValueError):
                        retry_after = self.backoff_seconds * (2**attempt)
                    self._sleep(max(0.0, retry_after) + random.uniform(0.0, min(0.25, self.backoff_seconds)))
                    continue
                if not 200 <= final_status <= 299:
                    suffix = f" (code={final_api_code})" if final_api_code is not None else ""
                    raise ZaiAPIError(
                        f"Z.ai API request failed: POST {path} returned HTTP {final_status}{suffix}",
                        status_code=final_status,
                        api_code=final_api_code,
                    )
                if not isinstance(decoded, dict):
                    raise ZaiAPIError("Z.ai API returned invalid JSON", status_code=final_status)
                choices = decoded.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise ZaiAPIError("Z.ai API response omitted choices", status_code=final_status)
                choice = choices[0]
                message = choice.get("message")
                if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                    raise ZaiAPIError("Z.ai API response omitted message content", status_code=final_status)
                usage_raw = decoded.get("usage") if isinstance(decoded.get("usage"), dict) else {}
                details = usage_raw.get("prompt_tokens_details")
                cached = details.get("cached_tokens") if isinstance(details, dict) else 0
                prompt_tokens = _as_nonnegative_int(usage_raw.get("prompt_tokens"))
                completion_tokens = _as_nonnegative_int(usage_raw.get("completion_tokens"))
                total_tokens = _as_nonnegative_int(usage_raw.get("total_tokens"))
                if total_tokens == 0 and (prompt_tokens or completion_tokens):
                    total_tokens = prompt_tokens + completion_tokens
                usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=_as_nonnegative_int(cached),
                    total_tokens=total_tokens,
                )
                return ChatCompletion(
                    content=message["content"],
                    reasoning_content=str(message.get("reasoning_content") or ""),
                    model=str(decoded.get("model") or model),
                    request_id=str(decoded.get("request_id") or client_request_id),
                    finish_reason=str(choice.get("finish_reason") or ""),
                    usage=usage,
                )
            raise ZaiAPIError("Z.ai API retries exhausted", status_code=final_status, api_code=final_api_code)
        finally:
            if self._trace_observer is not None:
                self._trace_observer(RequestTrace(
                    request_id=client_request_id,
                    path=path,
                    status_code=final_status,
                    api_code=final_api_code,
                    latency_seconds=round(time.monotonic() - started, 6),
                    retry_count=retry_count,
                ))
