"""Hook factory — builds SDK-compatible hook configurations for agents.

Wraps the individual hook functions from ``pre_tool``, ``post_tool``, and
``stop`` into async closures matching the ``claude_agent_sdk`` callback
signatures, then packages them as ``HookMatcher`` objects keyed by event type.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dd_agents.hooks.post_tool import validate_subject_json
from dd_agents.hooks.pre_tool import (
    BATCH_FILE_PATTERN,
    BLOCKED_FILENAMES,
    bash_guard,
    file_size_guard,
    finding_schema_guard,
    path_guard,
)
from dd_agents.hooks.stop import check_coverage, check_manifest
from dd_agents.utils.constants import COVERAGE_MANIFEST_JSON

if TYPE_CHECKING:
    from claude_agent_sdk import HookMatcher  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PreToolUse hook builder
# ---------------------------------------------------------------------------


def _build_pre_tool_hook(
    agent_name: str,
    run_dir: Path,
    project_dir: Path,
) -> Any:
    """Return an async PreToolUse callback compatible with the SDK.

    The callback receives ``(hook_input, tool_use_id, context)`` and returns
    a ``SyncHookJSONOutput``-compatible dict.
    """

    async def pre_tool_hook(hook_input: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        try:
            tn = hook_input.get("tool_name", "")
            ti = hook_input.get("tool_input", {})

            # 1. Bash guard
            result = bash_guard(tn, ti)
            if result["decision"] == "block":
                return {"decision": "block", "reason": result["reason"]}

            # 2. Path guard — only for Write/Edit
            result = path_guard(tn, ti, project_dir)
            if result["decision"] == "block":
                return {"decision": "block", "reason": result["reason"]}

            # 3. File size guard (warning only)
            result = file_size_guard(tn, ti)
            if result["reason"]:
                logger.warning("[%s] %s", agent_name, result["reason"])

            # 4. Aggregate file guard — block writes to known bad filenames
            if tn in ("Write", "Edit"):
                file_path = ti.get("file_path", "")
                filename = Path(file_path).name if file_path else ""
                if filename in BLOCKED_FILENAMES or BATCH_FILE_PATTERN.search(filename):
                    return {
                        "decision": "block",
                        "reason": (
                            f"Blocked write to aggregate filename '{filename}'. "
                            f"Findings must be per-subject, not aggregated."
                        ),
                    }

            # 5. Finding schema guard — validate JSON structure before write
            result = finding_schema_guard(tn, ti, run_dir)
            if result["decision"] == "block":
                return {"decision": "block", "reason": result["reason"]}

            return {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("PreToolUse hook error — blocking tool call for safety: %s", exc)
            return {"decision": "block", "reason": f"Hook guard error (fail-closed): {exc}"}

    return pre_tool_hook


# ---------------------------------------------------------------------------
# Stop hook builder
# ---------------------------------------------------------------------------


def _output_errors(output_dir: Path, subject_names: list[str] | None) -> list[str]:
    """Validate assigned artifacts, including files changed through Edit."""
    paths = (
        [output_dir / f"{name}.json" for name in subject_names]
        if subject_names is not None
        else sorted(path for path in output_dir.glob("*.json") if path.name != COVERAGE_MANIFEST_JSON)
    )
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            errors.append(f"Missing {path}. Write this assigned subject file.")
            continue
        content = path.read_text(encoding="utf-8")
        problems = validate_subject_json(str(path), content)
        if not problems and json.loads(content)["subject_safe_name"] != path.stem:
            problems.append(f"subject_safe_name must be '{path.stem}' to match the assigned filename")
        errors.extend(f"{path.name}: {problem}" for problem in problems)
    return errors


def _build_stop_hook(
    agent_name: str,
    run_dir: Path,
    expected_subjects: int,
    *,
    subject_names: list[str] | None = None,
) -> Any:
    """Return an async Stop callback compatible with the SDK.

    The callback receives ``(hook_input, tool_use_id, context)`` and returns
    a ``SyncHookJSONOutput``-compatible dict.
    """
    output_dir = run_dir / "findings" / agent_name

    async def stop_hook(hook_input: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        try:
            errors = _output_errors(output_dir, subject_names)
            if errors:
                return {"decision": "block", "reason": "Repair findings before stopping:\n" + "\n".join(errors)}

            # 1. Coverage check — must have produced all subject JSONs
            result = check_coverage(output_dir, expected_subjects)
            if result["decision"] == "block":
                # Block stopping and feed the missing work back to the model.
                # continue_=False would terminate the session instead.
                return {"decision": "block", "reason": result["reason"]}

            # 2. Manifest check — coverage_manifest.json must exist
            result = check_manifest(output_dir, expected_subjects)
            if result["decision"] == "block":
                return {"decision": "block", "reason": result["reason"]}

            # Note: audit log is written by the orchestrator AFTER the agent
            # session completes (_write_audit_log in engine.py), so checking
            # for it here would always warn.  QA audit (step 28, DoD #11)
            # validates audit logs exist post-pipeline.

            return {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stop hook could not validate findings: %s", exc)
            return {"decision": "block", "reason": f"Could not validate findings before stopping: {exc}"}

    return stop_hook


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_hooks_for_agent(
    agent_name: str,
    run_dir: Path,
    project_dir: Path,
    expected_subjects: int,
    *,
    subject_names: list[str] | None = None,
) -> dict[str, list[HookMatcher]] | None:
    """Build the complete hook configuration for a specialist agent.

    Returns a dict suitable for passing to ``ClaudeAgentOptions(hooks=...)``,
    or ``None`` if ``claude_agent_sdk`` is not installed.

    Parameters
    ----------
    agent_name:
        Agent identifier (e.g., ``"legal"``).
    run_dir:
        Path to the current run directory.
    project_dir:
        Path to the project root (for path guard scope).
    expected_subjects:
        Number of subject JSONs the agent is expected to produce.
    subject_names:
        Exact safe names assigned to this session, when available.
    """
    try:
        from claude_agent_sdk import HookMatcher as _HookMatcher  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("claude_agent_sdk not installed — hooks unavailable")
        return None

    hooks = {
        "PreToolUse": [
            _HookMatcher(
                hooks=[_build_pre_tool_hook(agent_name, run_dir, project_dir)],
                timeout=5.0,
            ),
        ],
    }
    # Read-only synthesis sessions return JSON in the SDK text stream.
    if expected_subjects > 0:
        hooks["Stop"] = [
            _HookMatcher(
                hooks=[_build_stop_hook(agent_name, run_dir, expected_subjects, subject_names=subject_names)],
                timeout=10.0,
            ),
        ]
    return hooks
