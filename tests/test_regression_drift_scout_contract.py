import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = ROOT / ".agents" / "skills" / "automem-regression-drift-scout" / "SKILL.md"


def read_skill() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_automem_regression_drift_scout_skill_exists_with_required_contract():
    source = read_skill()

    assert "name: automem-regression-drift-scout" in source
    assert "description:" in source
    assert "agents: [codex]" in source
    assert "capabilities:" in source
    assert "filesystem: readonly" in source


def test_automem_regression_drift_scout_is_read_only_by_default():
    source = read_skill()

    assert "read-only" in source
    assert "Do not deploy" in source
    assert "Do not edit files" in source
    assert "Do not create issues or pull requests" in source
    assert not re.search(r"`make deploy`|\bmake deploy(?:\s|$)", source)


def test_automem_regression_drift_scout_uses_project_truth_sources():
    source = read_skill()

    assert "benchmarks/EXPERIMENT_LOG.md" in source
    assert "make bench-health" in source
    assert "make deploy-check" in source
    assert "gh run list" in source
    assert "railway status" in source


def test_automem_regression_drift_scout_output_contract_is_tiered():
    source = read_skill()

    assert "healthy" in source
    assert "needs_issue" in source
    assert "needs_pr_plan" in source
    assert "evidence bundle" in source
    assert "confidence score" in source
    assert "what to measure next" in source


def test_automem_regression_drift_scout_handles_partial_command_results():
    source = read_skill()

    assert "Do not trust the final `RECALL HEALTH: HEALTHY` banner by itself" in source
    assert "Connection refused" in source
    assert "railway CLI not found" in source
    assert "skipped surface" in source
