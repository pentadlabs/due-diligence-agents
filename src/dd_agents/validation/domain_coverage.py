"""Review coverage shared by merge, QA, and Definition of Done."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from dd_agents.hooks.post_tool import validate_subject_json

if TYPE_CHECKING:
    from pathlib import Path


def _merged_agents(data: dict[str, Any]) -> set[str]:
    agents: set[str] = set()
    for item in data.get("findings", []) + data.get("gaps", []):
        agent = item.get("agent")
        if isinstance(agent, str):
            agents.add(agent)
        metadata = item.get("metadata") or {}
        contributors = metadata.get("contributing_agents", [])
        if isinstance(contributors, list):
            agents.update(name for name in contributors if isinstance(name, str))
    return agents


def _review_output(path: Path, subject: str, agent: str) -> dict[str, Any] | None:
    try:
        content = path.read_text(encoding="utf-8")
        if validate_subject_json(str(path), content):
            return None
        data: dict[str, Any] = json.loads(content)
    except (OSError, ValueError):
        return None
    if data.get("subject_safe_name") != subject or data.get("agent", agent) != agent:
        return None
    if data.get("auto_generated") or data.get("source") == "coverage_gate":
        return None
    if not all(isinstance(data.get(key, []), list) for key in ("findings", "gaps", "file_headers")):
        return None
    return data


def covered_domains(
    merged: dict[str, Any],
    subject: str,
    expected: set[str],
    findings_dir: Path | None = None,
) -> set[str]:
    """Count surviving contributions and valid empty reviews, not file count.

    When source artifacts are available, every claimed review must have a
    valid matching output. Nonempty outputs still need a surviving finding,
    gap, or deduplication contribution in the merged result.
    """
    if merged.get("subject_safe_name") != subject or not all(
        isinstance(merged.get(key), list) for key in ("findings", "gaps")
    ):
        return set()
    try:
        represented = _merged_agents(merged)
    except (AttributeError, TypeError):
        return set()
    if findings_dir is None:
        return represented & expected
    covered: set[str] = set()
    for agent in expected:
        output = _review_output(findings_dir / agent / f"{subject}.json", subject, agent)
        if output is None:
            continue
        if agent in represented or (not output["findings"] and not output.get("gaps")):
            covered.add(agent)
    return covered
