"""Minimal, secret-safe client for Devin API v3 organization sessions."""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

DEFAULT_BASE_URL = "https://api.devin.ai/v3"


class DevinAPIError(RuntimeError):
    """An intentionally sanitized Devin API failure."""


class SessionStatus(str, Enum):
    NEW = "new"
    CLAIMED = "claimed"
    RUNNING = "running"
    EXIT = "exit"
    ERROR = "error"
    SUSPENDED = "suspended"
    RESUMING = "resuming"

    @property
    def terminal(self) -> bool:
        return self in {self.EXIT, self.ERROR, self.SUSPENDED}


@dataclass(frozen=True)
class Session:
    session_id: str
    status: SessionStatus
    status_detail: str | None
    acus_consumed: float
    is_archived: bool


@dataclass(frozen=True)
class SessionMessage:
    event_id: str
    source: str
    message: str
    created_at: float | int | None


@dataclass(frozen=True)
class MessagePage:
    items: tuple[SessionMessage, ...]
    end_cursor: str | None
    has_next_page: bool
    total: int | None


@dataclass(frozen=True)
class RequestTrace:
    request_id: str
    method: str
    path: str
    status_code: int | None
    latency_seconds: float
    retry_count: int


Transport = Callable[[urllib.request.Request, float], tuple[int, Mapping[str, str], bytes]]
TraceObserver = Callable[[RequestTrace], None]


def _default_transport(request: urllib.request.Request, timeout: float) -> tuple[int, Mapping[str, str], bytes]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.headers, response.read()
    except urllib.error.HTTPError as exc:
        # The response body may contain echoed request data. Never propagate it.
        return int(exc.code), exc.headers or {}, b""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DevinAPIError("Devin API transport failed; request details were suppressed") from None


class DevinClient:
    """Standard-library HTTP client for documented v3 organization endpoints."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 20.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        trace_observer: TraceObserver | None = None,
    ) -> None:
        api_key = os.environ.get("DEVIN_API_KEY", "")
        org_id = os.environ.get("DEVIN_ORG_ID")
        if not api_key:
            raise DevinAPIError("DEVIN_API_KEY is required")
        self.__api_key = api_key
        self.org_id = org_id.strip() if org_id and org_id.strip() else None
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)
        self._transport = transport or _default_transport
        self._sleep = sleep
        self._trace_observer = trace_observer

    @classmethod
    def from_env(cls, **kwargs: Any) -> "DevinClient":
        """Read authentication exclusively from the supported environment variables."""
        return cls(**kwargs)

    def __getstate__(self) -> dict[str, Any]:
        raise TypeError("DevinClient cannot be serialized because it contains credentials")

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        safe_path = path if path.startswith("/") else "/" + path
        url = self.base_url + safe_path
        if query:
            encoded = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
            if encoded:
                url += "?" + encoded
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        headers = {
            "Authorization": "Bearer " + self.__api_key,
            "Accept": "application/json",
            "X-Request-ID": request_id,
        }
        if body is not None:
            headers["Content-Type"] = "application/json"

        started = time.monotonic()
        final_status: int | None = None
        retry_count = 0
        try:
            for attempt in range(self.max_retries + 1):
                request = urllib.request.Request(url, data=body, headers=headers, method=method)
                try:
                    status, response_headers, response_body = self._transport(request, self.timeout)
                except Exception:
                    if attempt >= self.max_retries:
                        raise DevinAPIError("Devin API transport failed; request details were suppressed") from None
                    retry_count += 1
                    self._sleep(self.backoff_seconds * (2**attempt))
                    continue
                final_status = int(status)
                if final_status == 429 or 500 <= final_status <= 599:
                    if attempt < self.max_retries:
                        retry_count += 1
                        retry_after = None
                        try:
                            retry_after = float(response_headers.get("Retry-After", ""))
                        except (TypeError, ValueError):
                            pass
                        delay = retry_after if retry_after is not None else self.backoff_seconds * (2**attempt)
                        self._sleep(max(0.0, delay) + random.uniform(0.0, min(0.25, self.backoff_seconds)))
                        continue
                if not 200 <= final_status <= 299:
                    raise DevinAPIError(f"Devin API request failed: {method} {safe_path} returned HTTP {final_status}")
                try:
                    decoded = json.loads(response_body.decode("utf-8")) if response_body else {}
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise DevinAPIError(f"Devin API returned invalid JSON for {method} {safe_path}") from None
                if not isinstance(decoded, dict):
                    raise DevinAPIError(f"Devin API returned an unexpected shape for {method} {safe_path}")
                return decoded
            raise DevinAPIError(f"Devin API request retries exhausted: {method} {safe_path}")
        finally:
            if self._trace_observer is not None:
                self._trace_observer(RequestTrace(
                    request_id=request_id,
                    method=method,
                    path=safe_path,
                    status_code=final_status,
                    latency_seconds=round(time.monotonic() - started, 6),
                    retry_count=retry_count,
                ))

    def get_self(self) -> dict[str, Any]:
        return self._request("GET", "/self")

    def resolve_org_id(self) -> str:
        if self.org_id:
            return self.org_id
        identity = self.get_self()
        discovered = identity.get("org_id")
        if not isinstance(discovered, str) or not discovered.strip():
            raise DevinAPIError("DEVIN_ORG_ID is unset and GET /v3/self returned no org_id")
        self.org_id = discovered.strip()
        return self.org_id

    def _org_path(self, suffix: str) -> str:
        org_id = urllib.parse.quote(self.resolve_org_id(), safe="")
        return f"/organizations/{org_id}{suffix}"

    @staticmethod
    def _session(data: Mapping[str, Any]) -> Session:
        session_id = data.get("session_id") or data.get("devin_id")
        if not isinstance(session_id, str) or not session_id:
            raise DevinAPIError("Devin API session response omitted session_id")
        try:
            status = SessionStatus(str(data.get("status", "new")))
        except ValueError:
            raise DevinAPIError("Devin API session response contained an unknown status") from None
        acus = data.get("acus_consumed", 0)
        return Session(
            session_id=session_id,
            status=status,
            status_detail=str(data["status_detail"]) if data.get("status_detail") is not None else None,
            acus_consumed=float(acus or 0),
            is_archived=bool(data.get("is_archived", False)),
        )

    def create_session(self, *, prompt: str, title: str, max_acu_limit: int, tags: list[str] | None = None) -> Session:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "title": title,
            "max_acu_limit": int(max_acu_limit),
            "tags": list(tags or []),
            "devin_mode": "normal",
        }
        return self._session(self._request("POST", self._org_path("/sessions"), payload=payload))

    def get_session(self, session_id: str) -> Session:
        sid = urllib.parse.quote(session_id, safe="")
        return self._session(self._request("GET", self._org_path(f"/sessions/{sid}")))

    def list_message_page(self, session_id: str, *, after: str | None = None, first: int = 100) -> MessagePage:
        sid = urllib.parse.quote(session_id, safe="")
        data = self._request(
            "GET",
            self._org_path(f"/sessions/{sid}/messages"),
            query={"after": after, "first": max(1, min(int(first), 200))},
        )
        raw_items = data.get("items")
        if not isinstance(raw_items, list):
            raise DevinAPIError("Devin API messages response omitted items")
        items: list[SessionMessage] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise DevinAPIError("Devin API messages response contained an invalid item")
            event_id = raw.get("event_id")
            message = raw.get("message")
            source = raw.get("source")
            if not all(isinstance(value, str) for value in (event_id, message, source)):
                raise DevinAPIError("Devin API message omitted required fields")
            items.append(SessionMessage(event_id, source, message, raw.get("created_at")))
        cursor = data.get("end_cursor")
        return MessagePage(
            items=tuple(items),
            end_cursor=str(cursor) if cursor is not None else None,
            has_next_page=bool(data.get("has_next_page", False)),
            total=int(data["total"]) if data.get("total") is not None else None,
        )

    def list_messages(self, session_id: str, *, after: str | None = None, first: int = 100) -> tuple[list[SessionMessage], str | None]:
        items: list[SessionMessage] = []
        cursor = after
        while True:
            page = self.list_message_page(session_id, after=cursor, first=first)
            items.extend(page.items)
            next_cursor = page.end_cursor
            if not page.has_next_page:
                return items, next_cursor or cursor
            if not next_cursor or next_cursor == cursor:
                raise DevinAPIError("Devin API message pagination returned a repeated cursor")
            cursor = next_cursor

    def send_message(self, session_id: str, message: str) -> Session:
        sid = urllib.parse.quote(session_id, safe="")
        return self._session(self._request(
            "POST", self._org_path(f"/sessions/{sid}/messages"), payload={"message": message}
        ))

    def archive_session(self, session_id: str) -> Session:
        sid = urllib.parse.quote(session_id, safe="")
        return self._session(self._request("POST", self._org_path(f"/sessions/{sid}/archive")))

    def terminate_session(self, session_id: str, *, archive: bool = True) -> Session:
        sid = urllib.parse.quote(session_id, safe="")
        return self._session(self._request(
            "DELETE", self._org_path(f"/sessions/{sid}"), query={"archive": str(bool(archive)).lower()}
        ))
