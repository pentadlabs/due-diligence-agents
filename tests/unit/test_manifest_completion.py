"""Completion must include retry manifests and preserve critical DoD failures."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from dd_agents.cli import main
from dd_agents.orchestrator.engine import PipelineEngine
from dd_agents.orchestrator.state import PipelineState
from dd_agents.validation.dod import DefinitionOfDoneChecker

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_retry_outputs_receive_manifests(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}")
    engine = PipelineEngine(tmp_path, config)
    engine._active_agents = ["legal"]
    state = PipelineState(run_id="retry", project_dir=tmp_path, run_dir=tmp_path / "run", subject_safe_names=["acme"])
    (state.run_dir / "findings").mkdir(parents=True)

    async def finish_retry(**kwargs: object) -> None:
        output = state.run_dir / "findings" / "legal"
        output.mkdir(exist_ok=True)
        (output / "acme.json").write_text(json.dumps({"subject": "acme", "findings": [], "gaps": []}))

    with patch.object(engine, "_respawn_for_missing_subjects", side_effect=finish_retry):
        await engine._step_17_coverage_gate(state)

    check = DefinitionOfDoneChecker(state.run_dir, tmp_path, ["acme"], active_agents=["legal"])
    assert check.check_3_agent_manifests_valid().passed


@pytest.mark.parametrize("exit_code", [0, 1])
def test_cli_preserves_pipeline_exit_code(tmp_path: Path, exit_code: int) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "config_version": "1.0.0",
                "buyer": {"name": "Buyer"},
                "target": {"name": "Target"},
                "deal": {"type": "acquisition", "focus_areas": ["ip_ownership"]},
                "data_room": {"path": str(tmp_path)},
            }
        )
    )
    state = PipelineState(
        run_id="completion",
        project_dir=tmp_path,
        run_dir=tmp_path,
        exit_code=exit_code,
        audit_passed=True,
        validation_results={"dod": exit_code == 0},
    )

    def exit_process(code: int) -> None:
        raise SystemExit(code)

    with (
        patch.object(PipelineEngine, "run", new=AsyncMock(return_value=state)),
        patch("dd_agents.cli._terminate_child_processes"),
        patch("os._exit", side_effect=exit_process) as terminate,
    ):
        result = CliRunner().invoke(main, ["run", str(config)])
    terminate.assert_called_once_with(exit_code)
    assert result.exit_code == exit_code
    if exit_code:
        assert "critical DoD failures" in result.output
