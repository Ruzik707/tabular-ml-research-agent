import json
from pathlib import Path

import pytest

from automl_agent.journal import EventJournal, atomic_write_json


def test_journal_appends_durable_sequenced_redacted_events(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = EventJournal(path, secrets=["super-secret-key"])
    first = journal.append("request", text="header=super-secret-key", prompt_tokens=12)
    second = journal.append("result", api_key="super-secret-key", content="fine")
    assert first["seq"] == 1 and second["seq"] == 2
    assert first["timestamp"]
    assert first["event"] == "request"
    loaded = EventJournal(path, secrets=["super-secret-key"]).read_events()
    assert loaded == [first, second]
    assert "super-secret-key" not in path.read_text()
    assert loaded[0]["payload"]["prompt_tokens"] == 12
    assert loaded[1]["payload"]["api_key"] == "[REDACTED]"


def test_journal_recovers_only_an_incomplete_last_frame(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = EventJournal(path)
    journal.append("started", value=1)
    with path.open("ab") as stream:
        stream.write(b'{"seq":2,"event":"torn"')
        stream.flush()
    assert journal.read_events()[0]["event"] == "started"
    assert path.read_bytes().endswith(b"\n")
    next_event = journal.append("continued", value=2)
    assert next_event["seq"] == 2
    assert [event["event"] for event in journal.read_events()] == ["started", "continued"]


def test_journal_rejects_corruption_before_the_last_frame(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"seq":1,"timestamp":"x","event":"ok","payload":{}}\nnot-json\n{"seq":3}\n')
    with pytest.raises(ValueError, match="corrupt"):
        EventJournal(path).read_events()


def test_journal_rejects_invalid_complete_final_frame(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"seq":1,"timestamp":"x","event":"ok","payload":{}}\nnot-json\n')
    with pytest.raises(ValueError, match="corrupt"):
        EventJournal(path).read_events()


def test_atomic_write_json_replaces_file_with_complete_json(tmp_path):
    path = tmp_path / "state" / "snapshot.json"
    path.parent.mkdir()
    path.write_text('{"old": true}')
    atomic_write_json(path, {"new": [1, "ё"]})
    assert json.loads(path.read_text()) == {"new": [1, "ё"]}
    assert sorted(item.name for item in path.parent.iterdir()) == ["snapshot.json"]
