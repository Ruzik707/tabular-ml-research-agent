"""Durable, append-only JSONL events and atomic JSON snapshots."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - the MVP runs on macOS/Linux
    fcntl = None


_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
_SENSITIVE_FIELDS = {
    "api_key",
    "access_token",
    "refresh_token",
    "client_secret",
    "password",
    "authorization",
    "secret",
    "credentials",
    "credential",
}


class EventJournal:
    """A small append-only journal with tail-frame recovery and redaction."""

    def __init__(self, path: Path, secrets: list[str] | tuple[str, ...] = ()) -> None:
        self.path = Path(path)
        self.secrets = tuple(sorted({secret for secret in secrets if isinstance(secret, str) and secret}, key=len, reverse=True))
        lock_key = str(self.path.absolute())
        with _PATH_LOCKS_GUARD:
            self._thread_lock = _PATH_LOCKS.setdefault(lock_key, threading.RLock())

    def append(self, event: str, **payload: Any) -> dict:
        if not isinstance(event, str) or not event:
            raise ValueError("event must be a non-empty string")
        reserved = {"seq", "timestamp", "event"}.intersection(payload)
        if reserved:
            raise ValueError(f"payload uses reserved journal fields: {', '.join(sorted(reserved))}")
        redacted_payload = self._redact(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_APPEND, 0o600)
        try:
            with self._locked(fd):
                events = self._read_and_repair(fd)
                record = {
                    "seq": len(events) + 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "event": self._redact_text(event),
                    "payload": redacted_payload,
                }
                frame = (json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
                _write_all(fd, frame)
                os.fsync(fd)
                _fsync_directory(self.path.parent)
                return record
        finally:
            os.close(fd)

    def read_events(self) -> list[dict]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            with self._locked(fd):
                return self._read_and_repair(fd)
        finally:
            os.close(fd)

    @contextmanager
    def _locked(self, fd: int) -> Iterator[None]:
        with self._thread_lock:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)

    def _read_and_repair(self, fd: int) -> list[dict]:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            return []

        pieces = data.split(b"\n")
        has_final_newline = data.endswith(b"\n")
        complete = pieces[:-1] if has_final_newline else pieces[:-1]
        tail = b"" if has_final_newline else pieces[-1]
        events: list[dict] = []
        offset = 0
        for frame in complete:
            if not frame:
                raise ValueError(f"corrupt journal frame after sequence {len(events)}")
            events.append(_decode_frame(frame, expected_seq=len(events) + 1))
            offset += len(frame) + 1

        if tail:
            try:
                record_value = json.loads(tail.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Only an invalid final frame without a newline is a torn append.
                os.ftruncate(fd, offset)
                os.fsync(fd)
                return events
            events.append(_validate_record(record_value, expected_seq=len(events) + 1))
            _write_all(fd, b"\n")
            os.fsync(fd)
        return events

    def redact(self, value: Any) -> Any:
        """Use the canonical secret policy for public snapshots and reports too."""
        return self._redact(value)

    def _redact_text(self, value: str) -> str:
        for secret in self.secrets:
            value = value.replace(secret, "[REDACTED]")
        return value

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact_text(value)
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                safe_key = self._redact_text(str(key))
                normalized = safe_key.lower().replace("-", "_")
                if normalized in _SENSITIVE_FIELDS or normalized.endswith("_api_key") or normalized.endswith("_secret"):
                    result[safe_key] = "[REDACTED]"
                else:
                    result[safe_key] = self._redact(child)
            return result
        if isinstance(value, (list, tuple)):
            return [self._redact(item) for item in value]
        return value


def atomic_write_json(path: Path, value: dict) -> None:
    """Write a JSON snapshot via fsynced temp file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        temp_path = None
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _decode_frame(frame: bytes, *, expected_seq: int) -> dict:
    try:
        record = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"corrupt journal frame at sequence {expected_seq}") from None
    return _validate_record(record, expected_seq=expected_seq)


def _validate_record(record: Any, *, expected_seq: int) -> dict:
    if (
        not isinstance(record, dict)
        or isinstance(record.get("seq"), bool)
        or record.get("seq") != expected_seq
        or not isinstance(record.get("timestamp"), str)
        or not isinstance(record.get("event"), str)
        or not record.get("event")
        or not isinstance(record.get("payload"), dict)
    ):
        raise ValueError(f"corrupt journal sequence or shape at sequence {expected_seq}")
    return record


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
