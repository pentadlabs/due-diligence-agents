"""The SDK Stop hook must request continuation when artifacts are missing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dd_agents.hooks.factory import _build_stop_hook, build_hooks_for_agent

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("existing_subjects", [None, 0, 1])
async def test_missing_findings_block_stop_with_model_feedback(tmp_path: Path, existing_subjects: int | None) -> None:
    output = tmp_path / "findings" / "legal"
    if existing_subjects is not None:
        output.mkdir(parents=True)
        for index in range(existing_subjects):
            (output / f"subject_{index}.json").write_text("{}")
    hook = _build_stop_hook("legal", tmp_path, expected_subjects=2)

    result = await hook({"hook_event_name": "Stop", "stop_hook_active": False}, None, {})

    assert result["decision"] == "block"
    assert result["reason"]
    # continue=false terminates the SDK; it does not block an early stop.
    assert "continue_" not in result
    assert "stopReason" not in result


async def test_missing_manifest_blocks_stop_with_model_feedback(tmp_path: Path) -> None:
    (tmp_path / "findings" / "legal").mkdir(parents=True)
    hook = _build_stop_hook("legal", tmp_path, expected_subjects=0)

    result = await hook({"hook_event_name": "Stop", "stop_hook_active": False}, None, {})

    assert result["decision"] == "block"
    assert "coverage_manifest.json" in result["reason"]
    assert "continue_" not in result


async def test_completed_findings_allow_stop_without_a_manifest(tmp_path: Path) -> None:
    output = tmp_path / "findings" / "legal"
    output.mkdir(parents=True)
    (output / "subject_a.json").write_text("{}")
    (output / "subject_b.json").write_text("{}")
    hook = _build_stop_hook("legal", tmp_path, expected_subjects=2)

    result = await hook({"hook_event_name": "Stop", "stop_hook_active": True}, None, {})

    assert result == {}


def test_read_only_sessions_do_not_require_findings_files(tmp_path: Path) -> None:
    hooks = build_hooks_for_agent("executive_synthesis", tmp_path, tmp_path, expected_subjects=0)

    assert hooks is not None
    assert "PreToolUse" in hooks
    assert "Stop" not in hooks


def test_file_sessions_register_the_completion_hook(tmp_path: Path) -> None:
    hooks = build_hooks_for_agent("legal", tmp_path, tmp_path, expected_subjects=2)

    assert hooks is not None
    assert len(hooks["Stop"]) == 1
