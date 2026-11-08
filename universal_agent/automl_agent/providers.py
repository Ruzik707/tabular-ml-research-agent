"""Provider adapters exposing a canonical OpenAI-style tool-call response."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
import uuid
from concurrent.futures import Future
from contextlib import contextmanager
from typing import Any

import httpx

from .contracts import ModelResponse


class ProviderError(RuntimeError):
    """A provider failure classified for bounded retry and caller handling."""

    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ProviderCancelToken:
    """Request-scoped, thread-safe cancellation signal for one provider call."""

    def __init__(self) -> None:
        self.event = threading.Event()

    def cancel(self) -> None:
        self.event.set()


class _HTTPProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        retries: int,
        extra_body: dict | None,
        *,
        ollama: bool,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a string")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if not isinstance(retries, int) or retries < 0:
            raise ValueError("retries must be a non-negative integer")
        if extra_body is not None and not isinstance(extra_body, dict):
            raise TypeError("extra_body must be a dict or None")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.retries = retries
        self.extra_body = dict(extra_body or {})
        self.ollama = ollama
        self._legacy_cancel_event = threading.Event()
        self._request_scope = threading.local()
        self._active_lock = threading.Lock()
        self._active_cancel_events: set[threading.Event] = set()
        if ollama:
            if self.base_url.endswith("/v1"):
                self.base_url = self.base_url[:-3]
            self.endpoint = self.base_url + "/api/chat"
        else:
            self.endpoint = self.base_url + "/chat/completions"

    def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        timeout: float,
        max_tokens: int,
    ) -> ModelResponse:
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise TypeError("messages and tools must be lists")
        scoped_token = getattr(self._request_scope, "token", None)
        cancellation_event = scoped_token.event if scoped_token is not None else self._legacy_cancel_event
        if scoped_token is None:
            # Preserve the original direct-call contract: cancel_pending() is
            # armed for an active call and reset when a new direct call begins.
            cancellation_event.clear()
        with self._active_lock:
            self._active_cancel_events.add(cancellation_event)
        try:
            coroutine = self._complete_async(messages, tools, float(timeout), max_tokens, cancellation_event)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coroutine)

            # The public interface is synchronous. If called from an async host,
            # run the request in a private event loop so we do not nest asyncio.run.
            result: Future[ModelResponse] = Future()

            def run_in_thread() -> None:
                try:
                    result.set_result(asyncio.run(coroutine))
                except BaseException as exc:
                    result.set_exception(exc)

            thread = threading.Thread(target=run_in_thread, name="automl-provider", daemon=True)
            thread.start()
            try:
                return result.result(timeout=float(timeout) + 0.5)
            except TimeoutError:
                raise ProviderError("provider request timed out", code="timeout", retryable=True) from None
        finally:
            with self._active_lock:
                self._active_cancel_events.discard(cancellation_event)

    def new_cancel_token(self) -> ProviderCancelToken:
        """Create a cancellation token before starting a request worker."""
        return ProviderCancelToken()

    @contextmanager
    def request_scope(self, token: ProviderCancelToken):
        """Bind a token to the existing complete() call in the current thread."""
        if not isinstance(token, ProviderCancelToken):
            raise TypeError("token must be created by new_cancel_token()")
        previous = getattr(self._request_scope, "token", None)
        self._request_scope.token = token
        try:
            yield token
        finally:
            if previous is None:
                del self._request_scope.token
            else:
                self._request_scope.token = previous

    def cancel_pending(self) -> None:
        """Cancel all active calls, retaining compatibility with direct callers."""
        self._legacy_cancel_event.set()
        with self._active_lock:
            active = tuple(self._active_cancel_events)
        for event in active:
            event.set()

    async def _complete_async(
        self,
        messages: list[dict],
        tools: list[dict],
        timeout: float,
        max_tokens: int,
        cancellation_event: threading.Event | None = None,
    ) -> ModelResponse:
        if cancellation_event is None:
            cancellation_event = self._legacy_cancel_event
        if cancellation_event.is_set():
            raise ProviderError("provider request cancelled", code="cancelled", retryable=False)
        task = asyncio.create_task(self._complete_request(messages, tools, timeout, max_tokens))
        try:
            while True:
                if cancellation_event.is_set():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise ProviderError("provider request cancelled", code="cancelled", retryable=False)
                done, _ = await asyncio.wait({task}, timeout=0.05)
                if done:
                    return task.result()
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if cancellation_event.is_set():
                raise ProviderError("provider request cancelled", code="cancelled", retryable=False) from None
            raise

    async def _complete_request(
        self,
        messages: list[dict],
        tools: list[dict],
        timeout: float,
        max_tokens: int,
    ) -> ModelResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = self._payload(messages, tools, max_tokens)
        try:
            async with asyncio.timeout(timeout):
                async with httpx.AsyncClient(timeout=None) as client:
                    for attempt in range(self.retries + 1):
                        try:
                            response = await client.post(self.endpoint, headers=headers, json=payload)
                        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                                httpx.WriteError, httpx.RemoteProtocolError) as exc:
                            if attempt >= self.retries:
                                raise ProviderError(
                                    "provider transport failed", code="transport_error", retryable=True
                                ) from None
                            await asyncio.sleep(min(0.1 * (2**attempt), 0.8))
                            continue
                        if response.is_success:
                            try:
                                body = response.json()
                            except (ValueError, json.JSONDecodeError):
                                raise ProviderError(
                                    "provider returned invalid JSON", code="invalid_response", retryable=False
                                ) from None
                            return self._normalize(body)
                        error = self._http_error(response.status_code, response.content)
                        if error.retryable and attempt < self.retries:
                            await asyncio.sleep(min(0.1 * (2**attempt), 0.8))
                            continue
                        raise error
        except TimeoutError:
            raise ProviderError("provider request timed out", code="timeout", retryable=True) from None
        except httpx.InvalidURL:
            raise ProviderError("provider URL is invalid", code="invalid_url", retryable=False) from None
        except httpx.RequestError as exc:
            transient = isinstance(
                exc,
                (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError),
            )
            code = "transport_error" if transient else "request_error"
            raise ProviderError("provider transport failed", code=code, retryable=transient) from None
        raise ProviderError("provider request failed", code="unknown", retryable=False)

    def _payload(self, messages: list[dict], tools: list[dict], max_tokens: int) -> dict[str, Any]:
        raise NotImplementedError

    def _normalize(self, body: Any) -> ModelResponse:
        raise NotImplementedError

    @staticmethod
    def _http_error(status: int, body: bytes) -> ProviderError:
        text = body[:8192].decode("utf-8", errors="ignore").lower()
        if status == 401:
            code, message, retryable = "unauthorized", "provider rejected credentials", False
        elif status == 403:
            code, message, retryable = "forbidden", "provider denied the request", False
        elif status == 400 and (
            "context_length_exceeded" in text
            or "maximum context length" in text
            or "context length is" in text
            or "too many tokens" in text
        ):
            code, message, retryable = "context_length_exceeded", "provider context limit exceeded", False
        elif status == 429:
            code, message, retryable = "rate_limit", "provider rate limit", True
        elif status in (408, 425) or status >= 500:
            code, message, retryable = "server_error", "temporary provider error", True
        else:
            code, message, retryable = f"http_{status}", f"provider returned HTTP {status}", False
        return ProviderError(message, code=code, retryable=retryable)


class OpenAICompatibleProvider(_HTTPProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        retries: int = 2,
        extra_body: dict | None = None,
    ) -> None:
        super().__init__(base_url, api_key, model, retries, extra_body, ollama=False)

    def _payload(self, messages: list[dict], tools: list[dict], max_tokens: int) -> dict[str, Any]:
        payload = dict(self.extra_body)
        payload.update({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "parallel_tool_calls": False,
            "max_tokens": max_tokens,
        })
        return payload

    def _normalize(self, body: Any) -> ModelResponse:
        try:
            choice = body["choices"][0]
            message = dict(choice["message"])
            message.setdefault("role", "assistant")
            message["tool_calls"] = _canonical_tool_calls(message.get("tool_calls", []))
            usage = body.get("usage") or {}
            prompt_tokens = _int_or_zero(usage.get("prompt_tokens"))
            completion_tokens = _int_or_zero(usage.get("completion_tokens"))
            total_tokens = _int_or_zero(usage.get("total_tokens"), prompt_tokens + completion_tokens)
            finish_reason = choice.get("finish_reason") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError("provider response shape is invalid", code="invalid_response", retryable=False) from None
        return ModelResponse(
            message=message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=str(finish_reason),
        )


class OllamaProvider(_HTTPProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        retries: int = 2,
        extra_body: dict | None = None,
    ) -> None:
        super().__init__(base_url, api_key, model, retries, extra_body, ollama=True)

    def _payload(self, messages: list[dict], tools: list[dict], max_tokens: int) -> dict[str, Any]:
        extra = dict(self.extra_body)
        extra_options = extra.pop("options", {})
        if not isinstance(extra_options, dict):
            raise ProviderError("Ollama options must be an object", code="invalid_request", retryable=False)
        options = {"num_ctx": 32768, "temperature": 0.2, **extra_options}
        options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            **extra,
            "model": self.model,
            "messages": _ollama_request_messages(messages),
            "tools": tools,
            "stream": False,
            "think": False,
            "options": options,
        }
        return payload

    def _normalize(self, body: Any) -> ModelResponse:
        try:
            message = dict(body["message"])
            message.setdefault("role", "assistant")
            message["tool_calls"] = _canonical_tool_calls(message.get("tool_calls", []))
            prompt_tokens = _int_or_zero(body.get("prompt_eval_count"))
            completion_tokens = _int_or_zero(body.get("eval_count"))
            total_tokens = prompt_tokens + completion_tokens
            finish_reason = body.get("done_reason") or ("stop" if body.get("done") else "")
        except (KeyError, TypeError, AttributeError):
            raise ProviderError("provider response shape is invalid", code="invalid_response", retryable=False) from None
        return ModelResponse(
            message=message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=str(finish_reason),
        )


def _ollama_request_messages(messages: list[dict]) -> list[dict[str, Any]]:
    """Convert canonical assistant tool arguments into Ollama's object form."""
    try:
        prepared = copy.deepcopy(messages)
    except (TypeError, ValueError):
        raise ProviderError("provider messages are invalid", code="invalid_request", retryable=False) from None
    for message in prepared:
        if not isinstance(message, dict):
            raise ProviderError("provider message is invalid", code="invalid_request", retryable=False)
        if message.get("role") != "assistant" or "tool_calls" not in message:
            continue
        calls = message["tool_calls"]
        if not isinstance(calls, list):
            raise ProviderError("assistant tool calls are invalid", code="invalid_request", retryable=False)
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise ProviderError("assistant tool call is invalid", code="invalid_request", retryable=False)
            function = call["function"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, json.JSONDecodeError):
                    raise ProviderError(
                        "assistant tool arguments are malformed JSON",
                        code="invalid_tool_arguments",
                        retryable=False,
                    ) from None
            if not isinstance(arguments, dict):
                raise ProviderError(
                    "assistant tool arguments must be a JSON object",
                    code="invalid_tool_arguments",
                    retryable=False,
                )
            function["arguments"] = arguments
    return prepared


def _int_or_zero(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback


def _canonical_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if tool_calls is None:
        return []
    if not isinstance(tool_calls, list):
        raise ProviderError("provider tool calls are invalid", code="invalid_response", retryable=False)
    normalized = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise ProviderError("provider tool call is invalid", code="invalid_response", retryable=False)
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ProviderError("provider tool call is invalid", code="invalid_response", retryable=False)
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError):
                raise ProviderError("provider tool arguments are invalid", code="invalid_response", retryable=False) from None
        normalized.append({
            "id": call.get("id") or f"call_{uuid.uuid4().hex}",
            "type": "function",
            "function": {"name": function["name"], "arguments": arguments},
        })
    return normalized
