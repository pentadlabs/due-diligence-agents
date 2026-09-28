"""Coverage survives deduplication and valid reviews with no issues."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from dd_agents.reporting.merge import FindingMerger
from dd_agents.validation.dod import DefinitionOfDoneChecker
from dd_agents.validation.qa_audit import QAAuditor

if TYPE_CHECKING:
    from pathlib import Path

AGENTS = ["commercial", "hr", "legal", "finance"]


def _make_run(root: Path) -> FindingMerger:
    finding = {
        "severity": "P2",
        "category": "retention",
        "title": "Retention issue",
        "description": "Source supports a retention issue.",
        "confidence": "high",
        "citations": [
            {
                "source_type": "file",
                "source_path": "file_1.pdf",
                "location": "Section 1",
                "exact_quote": "retention clause",
            }
        ],
    }
    for agent in AGENTS:
        path = root / "findings" / agent / "subject_a.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "subject": "Subject A",
                    "subject_safe_name": "subject_a",
                    "agent": agent,
                    "findings": [finding] if agent in ("commercial", "hr") else [],
                    "gaps": [],
                    "file_headers": [],
                }
            )
        )
    merger = FindingMerger(run_id="test")
    merged = merger.merge_all(root / "findings", expected_subjects=["subject_a"], active_agents=AGENTS)
    merger.write_merged(merged, root / "findings" / "merged")
    return merger


def _qa(root: Path) -> QAAuditor:
    return QAAuditor(root, root / "inventory", ["subject_a"], active_agents=AGENTS)


def test_merge_qa_and_dod_preserve_reviewed_domains(tmp_path: Path) -> None:
    merger = _make_run(tmp_path)
    merged = merger.merge_all(tmp_path / "findings", expected_subjects=["subject_a"], active_agents=AGENTS)
    assert len(merged["subject_a"].findings) == 1
    assert FindingMerger.check_agent_coverage(merged, tmp_path / "findings", AGENTS) == []
    assert _qa(tmp_path).check_domain_coverage()[1].passed
    checker = DefinitionOfDoneChecker(tmp_path, tmp_path / "inventory", ["subject_a"], active_agents=AGENTS)
    assert checker.check_12b_agent_coverage_in_merged().passed


@pytest.mark.parametrize("damage", ["missing", "corrupt", "identity", "schema", "fallback", "null", "gaps"])
def test_bad_source_output_cannot_claim_coverage(tmp_path: Path, damage: str) -> None:
    merger = _make_run(tmp_path)
    merged = merger.merge_all(tmp_path / "findings", expected_subjects=["subject_a"], active_agents=AGENTS)
    path = tmp_path / "findings" / "hr" / "subject_a.json"
    data = json.loads(path.read_text())
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("{}{}")
    else:
        changes = {
            "identity": {"subject_safe_name": "wrong"},
            "schema": {"findings": "wrong"},
            "fallback": {"auto_generated": True, "source": "coverage_gate"},
            "null": {"findings": None},
            "gaps": {"gaps": "wrong"},
        }
        path.write_text(json.dumps(data | changes[damage]))
    gaps = FindingMerger.check_agent_coverage(merged, tmp_path / "findings", AGENTS)
    assert len(gaps) == 1
    checker = DefinitionOfDoneChecker(tmp_path, tmp_path / "inventory", ["subject_a"], active_agents=AGENTS)
    assert not checker.check_12b_agent_coverage_in_merged().passed
    check = _qa(tmp_path).check_domain_coverage()[1]
    assert not check.passed
    assert check.details["subjects_with_missing_domains"] == [{"subject": "subject_a", "missing_domains": ["hr"]}]


@pytest.mark.parametrize("merged_content", [None, "{}{}", "[]", '{"subject_safe_name":"wrong"}'])
def test_missing_or_invalid_merged_output_fails(tmp_path: Path, merged_content: str | None) -> None:
    _make_run(tmp_path)
    path = tmp_path / "findings" / "merged" / "subject_a.json"
    if merged_content is None:
        path.unlink()
        path.parent.rmdir()
    else:
        path.write_text(merged_content)
    assert not _qa(tmp_path).check_domain_coverage()[1].passed

    checker = DefinitionOfDoneChecker(tmp_path, tmp_path / "inventory", ["subject_a"], active_agents=AGENTS)
    assert not checker.check_12b_agent_coverage_in_merged().passed


def test_nonempty_review_needs_surviving_contribution(tmp_path: Path) -> None:
    _make_run(tmp_path)
    path = tmp_path / "findings" / "merged" / "subject_a.json"
    merged = json.loads(path.read_text())
    merged["findings"][0]["metadata"].pop("contributing_agents")
    path.write_text(json.dumps(merged))
    check = _qa(tmp_path).check_domain_coverage()[1]
    assert not check.passed
    assert check.details["subjects_with_missing_domains"] == [{"subject": "subject_a", "missing_domains": ["hr"]}]
