import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from automl_agent.providers import (
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderError,
)


@contextmanager
def fake_server(responder):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, dict(self.headers), request))
            status, payload, delay = responder(request, len(seen))
            if delay:
                time.sleep(delay)
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_openai_compatible_provider_normalizes_tools_usage_and_request_flags():
    response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "run_experiment", "arguments": {"code": "x=1"}}}],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    with fake_server(lambda *_: (200, response, 0)) as (base_url, seen):
        provider = OpenAICompatibleProvider(base_url, "test-key", "local-model")
        result = provider.complete(
            [{"role": "user", "content": "try"}],
            [{"type": "function", "function": {"name": "run_experiment", "parameters": {"type": "object"}}}],
            timeout=2,
            max_tokens=99,
        )
    path, headers, request = seen[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert request["parallel_tool_calls"] is False
    assert request["max_tokens"] == 99
    assert result.message["role"] == "assistant"
    assert result.message["tool_calls"][0]["id"]
    assert json.loads(result.message["tool_calls"][0]["function"]["arguments"]) == {"code": "x=1"}
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (11, 7, 18)
    assert result.finish_reason == "tool_calls"


def test_ollama_uses_native_api_and_normalizes_usage_and_tools():
    response = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "finish_research", "arguments": {"reason": "done"}}}],
        },
        "prompt_eval_count": 13,
        "eval_count": 5,
        "done_reason": "stop",
    }
    with fake_server(lambda *_: (200, response, 0)) as (base_url, seen):
        provider = OllamaProvider(base_url, "", "qwen-local")
        result = provider.complete([], [{"type": "function", "function": {"name": "finish_research"}}], timeout=2, max_tokens=77)
    path, _headers, request = seen[0]
    assert path == "/api/chat"
    assert request["stream"] is False
    assert request["think"] is False
    assert request["options"]["num_predict"] == 77
    assert len(request["tools"]) == 1
    assert result.prompt_tokens == 13
    assert result.completion_tokens == 5
    assert result.total_tokens == 18
    assert json.loads(result.message["tool_calls"][0]["function"]["arguments"]) == {"reason": "done"}


def test_only_transient_provider_failures_are_retried():
    response = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    def responder(_request, call_number):
        return (503, {"error": "temporary"}, 0) if call_number == 1 else (200, response, 0)

    with fake_server(responder) as (base_url, seen):
        result = OpenAICompatibleProvider(base_url, "k", "m", retries=2).complete([], [], timeout=2, max_tokens=3)
    assert result.message["content"] == "ok"
    assert len(seen) == 2


def test_permanent_error_is_not_retried_or_leaked_into_error():
    with fake_server(lambda *_: (401, {"error": "bad credential test-secret-42"}, 0)) as (base_url, seen):
        provider = OpenAICompatibleProvider(base_url, "test-secret-42", "m", retries=4)
        with pytest.raises(ProviderError) as caught:
            provider.complete([], [], timeout=2, max_tokens=3)
    assert caught.value.code == "unauthorized"
    assert not caught.value.retryable
    assert len(seen) == 1
    assert "test-secret-42" not in str(caught.value)


def test_context_overflow_has_a_distinct_nonretryable_error():
    payload = {"error": {"message": "maximum context length exceeded", "code": "context_length_exceeded"}}
    with fake_server(lambda *_: (400, payload, 0)) as (base_url, seen):
        provider = OpenAICompatibleProvider(base_url, "k", "m", retries=3)
        with pytest.raises(ProviderError) as caught:
            provider.complete([], [], timeout=2, max_tokens=3)
    assert caught.value.code == "context_length_exceeded"
    assert not caught.value.retryable
    assert len(seen) == 1


def test_timeout_covers_all_attempts_and_backoff():
    with fake_server(lambda *_: (200, {}, 0.2)) as (base_url, seen):
        provider = OpenAICompatibleProvider(base_url, "k", "m", retries=10)
        started = time.monotonic()
        with pytest.raises(ProviderError) as caught:
            provider.complete([], [], timeout=0.08, max_tokens=3)
        duration = time.monotonic() - started
    assert caught.value.code == "timeout"
    assert duration < 0.3
    assert len(seen) == 1


def test_ollama_roundtrips_canonical_assistant_tool_call_and_tool_result():
    response = {"message": {"role": "assistant", "content": "done"}}
    assistant_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_lookup",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"city":"Moscow"}'},
        }],
    }
    tool_result = {"role": "tool", "tool_call_id": "call_lookup", "content": "cold"}
    messages = [assistant_call, tool_result]
    with fake_server(lambda *_: (200, response, 0)) as (base_url, seen):
        result = OllamaProvider(base_url, "", "qwen-local").complete(
            messages, [], timeout=2, max_tokens=55
        )
    request_messages = seen[0][2]["messages"]
    assert request_messages[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Moscow"}
    assert request_messages[1] == tool_result
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"city":"Moscow"}'
    assert result.message["content"] == "done"


def test_ollama_uses_native_option_defaults_with_explicit_overrides():
    response = {"message": {"role": "assistant", "content": "ok"}}
    with fake_server(lambda *_: (200, response, 0)) as (base_url, seen):
        OllamaProvider(base_url, "", "qwen-local").complete([], [], timeout=2, max_tokens=41)
        OllamaProvider(
            base_url,
            "",
            "qwen-local",
            extra_body={"options": {"num_ctx": 8192, "temperature": 0.7, "top_p": 0.9}},
        ).complete([], [], timeout=2, max_tokens=42)
    assert seen[0][2]["options"] == {"num_ctx": 32768, "temperature": 0.2, "num_predict": 41}
    assert seen[1][2]["options"] == {"num_ctx": 8192, "temperature": 0.7, "top_p": 0.9, "num_predict": 42}


def test_ollama_rejects_malformed_canonical_tool_arguments_before_request():
    with fake_server(lambda *_: (200, {"message": {"role": "assistant", "content": "ok"}}, 0)) as (base_url, seen):
        provider = OllamaProvider(base_url, "", "qwen-local")
        with pytest.raises(ProviderError) as caught:
            provider.complete(
                [{
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "lookup", "arguments": '{"city":'}}],
                }],
                [],
                timeout=2,
                max_tokens=10,
            )
    assert caught.value.code == "invalid_tool_arguments"
    assert not caught.value.retryable
    assert seen == []
