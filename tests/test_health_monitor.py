"""CLI precedence and non-destructive health-monitor recovery regression tests."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "health_monitor.py"


def load_monitor_module():
    spec = importlib.util.spec_from_file_location("health_monitor", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "env_values,cli_values,expected",
    [
        (None, [], (5, 50)),
        ((10, 30), [], (10, 30)),
        ((10, 30), ["--drift-threshold", "2", "--critical-threshold", "20"], (2, 20)),
        ((10, 30), ["--drift-threshold", "0", "--critical-threshold", "0"], (0, 0)),
        ((10, 30), ["--drift-threshold", "3"], (3, 30)),
    ],
)
def test_threshold_precedence(monkeypatch, env_values, cli_values, expected):
    monitor_module = load_monitor_module()
    for name in ["HEALTH_MONITOR_DRIFT_THRESHOLD", "HEALTH_MONITOR_CRITICAL_THRESHOLD"]:
        monkeypatch.delenv(name, raising=False)
    if env_values:
        monkeypatch.setenv("HEALTH_MONITOR_DRIFT_THRESHOLD", str(env_values[0]))
        monkeypatch.setenv("HEALTH_MONITOR_CRITICAL_THRESHOLD", str(env_values[1]))
    observed = []

    def run_check(self):
        observed.append((self.drift_threshold_percent, self.critical_threshold_percent))
        return {}

    monkeypatch.setattr(monitor_module.HealthMonitor, "run_check", run_check)
    monkeypatch.setattr("sys.argv", [str(SCRIPT_PATH), "--once", *cli_values])
    monitor_module.main()
    assert observed == [expected]


@pytest.mark.parametrize("graph_count,vector_count,should_recover", [(1, 10, True), (10, 1, False)])
def test_auto_recovery_only_repairs_missing_graph_memories(
    monkeypatch, graph_count, vector_count, should_recover
):
    monitor_module = load_monitor_module()
    monitor = monitor_module.HealthMonitor(auto_recover=True)
    monitor.critical_threshold_percent = 1
    monkeypatch.setattr(
        monitor, "check_falkordb", lambda: {"status": "healthy", "memory_count": graph_count}
    )
    monkeypatch.setattr(
        monitor, "check_qdrant", lambda: {"status": "healthy", "points_count": vector_count}
    )
    monkeypatch.setattr(monitor, "check_api", lambda: {"status": "healthy"})
    recovery = Mock()
    alert = Mock()
    monkeypatch.setattr(monitor, "trigger_recovery", recovery)
    monkeypatch.setattr(monitor, "send_alert", alert)
    monitor.run_check()
    assert recovery.called is should_recover
    if not should_recover:
        alert.assert_called_once()


def test_monitor_warns_about_relationship_limits_and_runs_own_script(monkeypatch, caplog):
    monitor_module = load_monitor_module()
    monitor = monitor_module.HealthMonitor(auto_recover=True)
    monkeypatch.setattr(monitor, "send_alert", Mock())
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr("subprocess.run", run)
    assert monitor.trigger_recovery("Missing graph memories", 90)
    args = run.call_args.args[0]
    assert Path(args[1]) == SCRIPT_PATH.parent / "recover_from_qdrant.py"
    assert "relationship" in caplog.text.lower()
    assert "backup" in caplog.text.lower()
