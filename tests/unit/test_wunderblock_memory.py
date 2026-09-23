"""Application backend selection and WorkItem memory contract."""

import json
from types import SimpleNamespace

import pytest

from dd_agents.chat.history import ChatMessage, MessageRole
from dd_agents.chat.memory import ChatMemory, ChatMemoryStore, MemoryType, SessionMetadata, create_memory_store


@pytest.fixture
def authority(monkeypatch):
    items = {}

    def write(namespace, key, value):
        items[(tuple(namespace), key)] = json.dumps(value)

    def records(namespace):
        return [{"value_json": value} for (ns, _), value in items.items() if ns == tuple(namespace)]

    api = SimpleNamespace(write=write, records=records)
    monkeypatch.setitem(__import__("sys").modules, "wb_framework_memory", api)
    monkeypatch.setenv("DD_CHAT_MEMORY_BACKEND", "wunderblock")
    return api


def test_default_still_uses_files(monkeypatch, tmp_path):
    monkeypatch.delenv("DD_CHAT_MEMORY_BACKEND", raising=False)
    assert type(create_memory_store(tmp_path)) is ChatMemoryStore


def test_unknown_backend_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("DD_CHAT_MEMORY_BACKEND", "typo")
    with pytest.raises(ValueError, match="DD_CHAT_MEMORY_BACKEND"):
        create_memory_store(tmp_path)


def test_new_handle_reads_memory_without_local_files(authority, tmp_path):
    path = tmp_path / "never-created"
    first = create_memory_store(path)
    first.ensure_dirs()
    memory = ChatMemory(
        id="m1",
        timestamp="2026-09-23T12:00:00Z",
        session_id="one",
        content="Acme evidence",
        topics=["Acme"],
        memory_type=MemoryType.INSIGHT,
    )
    first.save_memory(memory)
    resumed = create_memory_store(path)
    assert resumed.memory_count == 1
    assert resumed.search_memories("Acme evidence") == [memory]
    assert resumed.load_recent_memories() == [memory]
    assert not path.exists()


def test_backend_failure_never_falls_back_to_files(authority, tmp_path):
    def refuse(_):
        raise RuntimeError("memory denied")

    authority.records = refuse
    with pytest.raises(RuntimeError, match="memory denied"):
        create_memory_store(tmp_path).load_recent_memories()


def test_transcript_and_session_index_use_host_records(authority, tmp_path):
    store = create_memory_store(tmp_path)
    message = ChatMessage(role=MessageRole.USER, content="Acme evidence")
    store.save_session_transcript("s1", [message])
    record = authority.records(["due-diligence", "chat", "transcripts"])[0]
    assert json.loads(record["value_json"]) == [message.model_dump(mode="json")]
    meta = SessionMetadata(session_id="s1", start_time="2026-09-23T00:00:00Z")
    store.update_session_index(meta)
    meta.turn_count = 2
    store.update_session_index(meta)
    assert create_memory_store(tmp_path).load_session_index() == [meta]


def test_missing_host_module_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("DD_CHAT_MEMORY_BACKEND", "wunderblock")
    monkeypatch.setitem(__import__("sys").modules, "wb_framework_memory", None)
    with pytest.raises(ModuleNotFoundError):
        create_memory_store(tmp_path)
