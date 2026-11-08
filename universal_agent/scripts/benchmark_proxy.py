#!/usr/bin/env python3
"""Normalize benchmark LLM requests to one local Ollama chat configuration.

Both Ollama's native chat API and OpenAI-compatible chat completions are
forwarded to the same native ``/api/chat`` endpoint. The JSONL log contains
call metadata and timing only; prompts, responses, and headers are never logged.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _local_upstream(value: str) -> str:
    parsed = urlparse(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("--upstream must be a local HTTP Ollama origin")
    return value.rstrip("/")


def _predict_limit(requested, maximum: int) -> int:
    try:
        requested = int(requested)
    except (TypeError, ValueError, OverflowError):
        requested = maximum
    return max(1, min(requested, maximum))


def _iso_json_arguments(messages):
    """Copy OAI messages while decoding prior assistant tool argument strings."""
    output = []
    for message in messages:
        copied = dict(message)
        if copied.get("role") == "assistant" and isinstance(copied.get("tool_calls"), list):
            calls = []
            for call in copied["tool_calls"]:
                call_copy = dict(call)
                function = dict(call_copy.get("function") or {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                call_copy["function"] = function
                calls.append(call_copy)
            copied["tool_calls"] = calls
        output.append(copied)
    return output


def _response_format(value):
    if not isinstance(value, dict):
        return None
    kind = value.get("type")
    if kind == "json_object":
        return "json"
    if kind == "json_schema":
        schema = value.get("json_schema") or {}
        return schema.get("schema", schema)
    return None


def _native_request(path: str, payload: dict, config: dict) -> dict:
    if path == "/api/chat":
        messages = payload.get("messages")
        passthrough = {
            key: payload[key]
            for key in ("tools", "format", "tool_choice", "keep_alive")
            if key in payload
        }
        requested = (payload.get("options") or {}).get("num_predict")
    else:
        messages = payload.get("messages")
        passthrough = {}
        if "tools" in payload:
            passthrough["tools"] = payload["tools"]
        if "tool_choice" in payload:
            passthrough["tool_choice"] = payload["tool_choice"]
        response_format = _response_format(payload.get("response_format"))
        if response_format is not None:
            passthrough["format"] = response_format
        requested = payload.get("max_completion_tokens", payload.get("max_tokens"))
        if isinstance(messages, list):
            messages = _iso_json_arguments(messages)

    if not isinstance(messages, list):
        raise ValueError("messages must be an array")
    return {
        "model": config["model"],
        "messages": messages,
        **passthrough,
        "think": False,
        "stream": False,
        "options": {
            "num_ctx": config["num_ctx"],
            "num_predict": _predict_limit(requested, config["num_predict"]),
            "temperature": 0.2,
            "seed": config["seed"],
            "num_thread": config["threads"],
        },
    }


def _openai_response(native: dict, model: str) -> dict:
    message = native.get("message") or {}
    tool_calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        tool_calls.append({
            "id": call.get("id") or "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": function.get("name", ""), "arguments": arguments},
        })
    done_reason = native.get("done_reason")
    if tool_calls:
        finish_reason = "tool_calls"
    elif done_reason in {"length", "max_tokens"}:
        finish_reason = "length"
    else:
        finish_reason = "stop"
    prompt_tokens = int(native.get("prompt_eval_count") or 0)
    completion_tokens = int(native.get("eval_count") or 0)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": message.get("role", "assistant"),
                "content": message.get("content") or "",
                **({"tool_calls": tool_calls} if tool_calls else {}),
            },
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _handler(config: dict, log_path: str):
    log_lock = threading.Lock()

    def log_event(record):
        with log_lock:
            with open(log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            return

        def _reply(self, status: int, body: bytes, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str):
            body = json.dumps({"error": message}).encode("utf-8")
            self._reply(status, body)

        def do_POST(self):
            if self.path not in {"/api/chat", "/v1/chat/completions"}:
                self._error(404, "unsupported route")
                return

            started = time.monotonic()
            request_id = uuid.uuid4().hex
            path = self.path
            log_event({"event": "started", "request_id": request_id, "path": path,
                       "started_at": time.time()})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 32 * 1024 * 1024:
                    raise ValueError("invalid request body length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                outgoing = _native_request(path, payload, config)
                request = Request(
                    config["upstream"] + "/api/chat",
                    data=json.dumps(outgoing, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=config["upstream_timeout"]) as response:
                    status = response.status
                    native_body = response.read()
                native = json.loads(native_body)
                if path == "/api/chat":
                    body = native_body
                    content_type = "application/json"
                else:
                    body = json.dumps(_openai_response(native, config["model"]),
                                      ensure_ascii=False).encode("utf-8")
                    content_type = "application/json"
                log_event({
                    "event": "completed", "request_id": request_id, "path": path,
                    "status": status, "completed_at": time.time(),
                    "duration_seconds": round(time.monotonic() - started, 6),
                    "prompt_eval_count": native.get("prompt_eval_count"),
                    "eval_count": native.get("eval_count"),
                    "total_duration": native.get("total_duration"),
                    "load_duration": native.get("load_duration"),
                })
                self._reply(status, body, content_type)
            except HTTPError as exc:
                upstream_body = exc.read()
                status = exc.code
                log_event({"event": "error", "request_id": request_id, "path": path,
                           "status": status, "completed_at": time.time(),
                           "error_type": "upstream_http_error",
                           "duration_seconds": round(time.monotonic() - started, 6)})
                if path == "/api/chat":
                    self._reply(status, upstream_body)
                else:
                    self._error(status, "upstream returned an HTTP error")
            except (URLError, TimeoutError, OSError) as exc:
                log_event({"event": "error", "request_id": request_id, "path": path,
                           "status": 502, "completed_at": time.time(),
                           "error_type": type(exc).__name__,
                           "duration_seconds": round(time.monotonic() - started, 6)})
                self._error(502, "upstream request failed")
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                log_event({"event": "error", "request_id": request_id, "path": path,
                           "status": 400, "completed_at": time.time(),
                           "error_type": type(exc).__name__,
                           "duration_seconds": round(time.monotonic() - started, 6)})
                self._error(400, "invalid chat request or upstream JSON response")
            except Exception as exc:
                log_event({"event": "error", "request_id": request_id, "path": path,
                           "status": 502, "completed_at": time.time(),
                           "error_type": type(exc).__name__,
                           "duration_seconds": round(time.monotonic() - started, 6)})
                self._error(502, "proxy request failed")

    return Handler


def serve(args):
    config = {
        "upstream": _local_upstream(args.upstream),
        "model": args.model,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "threads": args.threads,
        "seed": args.seed,
        "upstream_timeout": args.upstream_timeout,
    }
    if min(config["num_ctx"], config["num_predict"], config["threads"]) < 1:
        raise ValueError("context, token cap, and thread count must be positive")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _handler(config, args.log))
    server.daemon_threads = True
    ready_path = os.path.abspath(args.ready_file)
    os.makedirs(os.path.dirname(ready_path), exist_ok=True)
    temporary = ready_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump({"port": server.server_port}, stream)
    os.replace(temporary, ready_path)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Normalize local benchmark chat requests for Ollama.")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--num-predict", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upstream-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    try:
        serve(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
