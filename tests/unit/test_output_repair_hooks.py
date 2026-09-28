"""Specialist hooks reject bad writes and keep incomplete sessions working."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from dd_agents.hooks.factory import _build_pre_tool_hook, _build_stop_hook

if TYPE_CHECKING:
    from pathlib import Path


def subject_output(name: str) -> str:
    return json.dumps({"subject": name, "subject_safe_name": name, "findings": [], "file_headers": []})


@pytest.mark.parametrize("content", ["{}{}", '{"subject":"line\nbreak"}', "", "[]", '{"findings":false}'])
async def test_sdk_tool_id_does_not_bypass_json_guard(tmp_path: Path, content: str) -> None:
    target = tmp_path / "_dd" / "findings" / "legal" / "subject_a.json"
    hook = _build_pre_tool_hook("legal", tmp_path / "_dd", tmp_path)

    result = await hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": str(target), "content": content},
        },
        "toolu_123",
        {},
    )

    assert result["decision"] == "block"
    assert "outside" not in result["reason"]
    assert "continue_" not in result


async def test_sdk_tool_id_does_not_bypass_path_guard(tmp_path: Path) -> None:
    hook = _build_pre_tool_hook("legal", tmp_path / "_dd", tmp_path)
    result = await hook(
        {"tool_name": "Write", "tool_input": {"file_path": "/outside/project.json", "content": "{}"}},
        "toolu_123",
        {},
    )
    assert result["decision"] == "block"


async def test_valid_write_with_tool_id_is_allowed(tmp_path: Path) -> None:
    hook = _build_pre_tool_hook("legal", tmp_path / "_dd", tmp_path)
    result = await hook(
        {
            "tool_name": "Write",
            "tool_input": {
                "file_path": str(tmp_path / "_dd" / "findings" / "legal" / "subject_a.json"),
                "content": subject_output("subject_a"),
            },
        },
        "toolu_123",
        {},
    )
    assert result == {}


@pytest.mark.parametrize("bad_content", ["{}{}", "{}", "[]"])
async def test_stop_requests_repair_of_existing_invalid_file(tmp_path: Path, bad_content: str) -> None:
    output = tmp_path / "findings" / "legal"
    output.mkdir(parents=True)
    target = output / "subject_a.json"
    target.write_text(bad_content)
    hook = _build_stop_hook("legal", tmp_path, 1)

    result = await hook({"stop_hook_active": True}, None, {})

    assert result["decision"] == "block"
    assert "subject_a.json" in result["reason"]
    assert "continue_" not in result
    target.write_text(subject_output("subject_a"))
    assert await hook({"stop_hook_active": True}, None, {}) == {}


async def test_wrong_filename_cannot_satisfy_assigned_subject(tmp_path: Path) -> None:
    output = tmp_path / "findings" / "finance"
    output.mkdir(parents=True)
    (output / "_reference.json").write_text(subject_output("_reference"))
    hook = _build_stop_hook("finance", tmp_path, 1, subject_names=["reference"])

    result = await hook({}, None, {})

    assert result["decision"] == "block"
    assert "reference.json" in result["reason"]
    (output / "reference.json").write_text(subject_output("reference"))
    assert await hook({}, None, {}) == {}


async def test_stop_checks_subject_identity_not_only_filename(tmp_path: Path) -> None:
    output = tmp_path / "findings" / "finance"
    output.mkdir(parents=True)
    (output / "reference.json").write_text(subject_output("_reference"))
    hook = _build_stop_hook("finance", tmp_path, 1, subject_names=["reference"])
    result = await hook({}, None, {})
    assert result["decision"] == "block"
    assert "subject_safe_name" in result["reason"]


async def test_runner_passes_assigned_names_to_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dd_agents.agents.specialists import LegalAgent

    agent = LegalAgent(tmp_path, tmp_path / "_dd", "test_run")
    spawn = AsyncMock(return_value="Completed")
    monkeypatch.setattr(agent, "_spawn_agent", spawn)
    await agent.run({"subjects": ["subject_a", "reference"], "prompt": "Analyze assigned subjects"})
    assert spawn.call_args.kwargs["subject_names"] == ["subject_a", "reference"]


async def test_stop_blocks_unreadable_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    output = tmp_path / "findings" / "legal"
    output.mkdir(parents=True)
    (output / "subject_a.json").write_text(subject_output("subject_a"))

    def unreadable(*args: object, **kwargs: object) -> str:
        raise OSError("unreadable fixture")

    monkeypatch.setattr(Path, "read_text", unreadable)
    result = await _build_stop_hook("legal", tmp_path, 1)({}, None, {})
    assert result["decision"] == "block"
    assert "unreadable fixture" in result["reason"]
