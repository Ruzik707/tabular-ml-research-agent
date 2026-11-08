"""End-to-end protocol and measurement contract tests for benchmark_proxy."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_proxy.py"


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    requests = []
    lock = threading.Lock()

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.lock:
            self.requests.append((self.path, payload))
        if payload.get("messages", [{}])[-1].get("content") == "cause upstream error":
            body = json.dumps({"error": "fake failure"}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps(
            {
                "model": "qwen3.5:9b",
                "message": {
                    "role": "assistant",
                    "content": "private output sentinel",
                    "tool_calls": [
                        {"function": {"name": "lookup", "arguments": {"answer": 42}}}
                    ],
                },
                "done": True,
                "done_reason": "tool_calls",
                "prompt_eval_count": 11,
                "eval_count": 7,
                "total_duration": 1_500_000,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def proxy(tmp_path):
    _FakeOllamaHandler.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    ready = tmp_path / "ready.json"
    log = tmp_path / "calls.jsonl"
    command = [
        sys.executable,
        str(SCRIPT),
        "--port", "0",
        "--ready-file", str(ready),
        "--log", str(log),
        "--upstream", f"http://127.0.0.1:{upstream.server_port}",
        "--model", "qwen3.5:9b",
        "--num-ctx", "16384",
        "--num-predict", "2000",
        "--threads", "2",
        "--seed", "42",
    ]
    env = os.environ.copy()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "proxy did not publish its ready port"
        port = json.loads(ready.read_text())["port"]
        yield port, log, _FakeOllamaHandler.requests
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


def _post(port, path, value):
    body = json.dumps(value).encode()
    request = Request(f"http://127.0.0.1:{port}{path}", data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_native_protocol_is_normalized_and_upstream_response_is_passthrough(proxy):
    port, _, captured = proxy
    payload = {
        "model": "caller-model",
        "think": True,
        "stream": True,
        "messages": [{"role": "user", "content": "private prompt sentinel"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "format": {"type": "object", "properties": {"answer": {"type": "integer"}}},
        "options": {"num_ctx": 4096, "num_predict": 5000, "temperature": 0.9, "seed": 9, "num_thread": 8},
    }

    status, response = _post(port, "/api/chat", payload)

    assert status == 200
    assert response["message"]["content"] == "private output sentinel"
    assert len(captured) == 1 and captured[0][0] == "/api/chat"
    sent = captured[0][1]
    assert sent["model"] == "qwen3.5:9b"
    assert sent["messages"] == payload["messages"]
    assert sent["tools"] == payload["tools"] and sent["format"] == payload["format"]
    assert sent["think"] is False and sent["stream"] is False
    assert sent["options"] == {
        "num_ctx": 16384, "num_predict": 2000, "temperature": 0.2, "seed": 42, "num_thread": 2
    }


def test_openai_protocol_preserves_schema_tools_and_normalizes_tokens(proxy):
    port, _, captured = proxy
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": schema}}]
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "old", "type": "function",
         "function": {"name": "lookup", "arguments": "{\"key\":\"x\"}"}}]},
        {"role": "tool", "tool_call_id": "old", "content": "done"},
    ]
    payload = {
        "model": "ignored-model", "messages": messages, "tools": tools,
        "tool_choice": "auto", "response_format": {"type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": schema}},
        "max_tokens": 1500, "temperature": 0.9, "stream": True,
    }

    status, response = _post(port, "/v1/chat/completions", payload)

    assert status == 200
    sent = captured[0][1]
    assert captured[0][0] == "/api/chat"
    assert sent["model"] == "qwen3.5:9b" and sent["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"key": "x"}
    assert sent["messages"][1] == messages[1]
    assert sent["tools"] == tools and sent["tool_choice"] == "auto" and sent["format"] == schema
    assert sent["stream"] is False and sent["think"] is False
    assert sent["options"] == {
        "num_ctx": 16384, "num_predict": 1500, "temperature": 0.2, "seed": 42, "num_thread": 2
    }
    choice = response["choices"][0]
    assert choice["message"]["content"] == "private output sentinel"
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["id"].startswith("call_")
    assert json.loads(call["function"]["arguments"]) == {"answer": 42}
    assert response["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def test_measurement_log_has_started_completed_counts_and_no_prompt_or_content(proxy):
    port, log, _ = proxy
    _post(port, "/api/chat", {"messages": [{"role": "user", "content": "private prompt sentinel"}]})
    records = [json.loads(line) for line in log.read_text().splitlines()]

    assert [record["event"] for record in records] == ["started", "completed"]
    assert records[0]["request_id"] == records[1]["request_id"]
    assert records[0]["path"] == records[1]["path"] == "/api/chat"
    assert records[0]["started_at"] <= records[1]["completed_at"]
    assert records[1]["duration_seconds"] >= 0
    assert records[1]["prompt_eval_count"] == 11
    assert records[1]["eval_count"] == 7
    assert records[1]["total_duration"] == 1_500_000
    contents = log.read_text()
    assert "private prompt sentinel" not in contents and "private output sentinel" not in contents


def test_upstream_errors_are_logged_and_unknown_routes_fail(proxy):
    port, log, captured = proxy
    status, body = _post(port, "/api/chat", {"messages": [{"role": "user", "content": "cause upstream error"}]})
    assert status == 503 and body["error"] == "fake failure"
    before = len(captured)
    assert _post(port, "/wrong", {})[0] == 404
    assert len(captured) == before
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [record["event"] for record in records] == ["started", "error"]
    assert records[1]["status"] == 503


def test_proxy_refuses_nonlocal_upstream(tmp_path):
    ready = tmp_path / "ready.json"
    command = [
        sys.executable, str(SCRIPT), "--port", "0", "--ready-file", str(ready),
        "--log", str(tmp_path / "calls.jsonl"), "--upstream", "http://example.com",
        "--model", "qwen3.5:9b",
    ]

    result = subprocess.run(command, capture_output=True, text=True, timeout=5)

    assert result.returncode == 2
    assert "local HTTP Ollama origin" in result.stderr
    assert not ready.exists()
