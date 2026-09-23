"""Chat memory in the active Wunderblock WorkItem, selected by the application."""

from __future__ import annotations

import json
from importlib import import_module
from typing import TYPE_CHECKING

from dd_agents.chat.memory import ChatMemory, ChatMemoryStore, SessionMetadata

if TYPE_CHECKING:
    from pathlib import Path

    from dd_agents.chat.history import ChatMessage

ROOT = ["due-diligence", "chat"]


class WunderblockChatMemoryStore(ChatMemoryStore):
    """Reuse chat search and models; keep durable state in host memory tools.

    The admitted image supplies ``wb_framework_memory``. The host chooses the
    WorkItem scope. ``chat_dir`` does not select or widen that scope.
    """

    def __init__(self, chat_dir: Path) -> None:
        super().__init__(chat_dir)
        self._backend = import_module("wb_framework_memory")

    def ensure_dirs(self) -> None:
        """This backend needs no local memory directory."""

    def save_memory(self, memory: ChatMemory) -> None:
        self._backend.write([*ROOT, "memories"], memory.id, memory.model_dump(mode="json"))

    def _refresh_cache(self) -> None:
        memories = [
            ChatMemory.model_validate_json(item["value_json"]) for item in self._backend.records([*ROOT, "memories"])
        ]
        self._memories_cache = sorted(memories, key=lambda memory: (memory.timestamp, memory.id))

    def save_session_transcript(self, session_id: str, messages: list[ChatMessage]) -> None:
        self._backend.write(
            [*ROOT, "transcripts"], session_id, [message.model_dump(mode="json") for message in messages]
        )

    def update_session_index(self, meta: SessionMetadata) -> None:
        self._backend.write([*ROOT, "sessions"], meta.session_id, meta.model_dump(mode="json"))

    def load_session_index(self) -> list[SessionMetadata]:
        entries = [
            SessionMetadata.model_validate(json.loads(item["value_json"]))
            for item in self._backend.records([*ROOT, "sessions"])
        ]
        return sorted(entries, key=lambda entry: (entry.start_time, entry.session_id))
