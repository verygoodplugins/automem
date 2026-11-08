"""Check the effective Compose port bindings, not just the YAML spelling."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

COMPOSE_FILE = Path(__file__).resolve().parents[1] / "docker-compose.yml"


def _compose_config(tmp_path: Path, overrides: dict[str, str] | None = None) -> dict:
    docker = shutil.which("docker")
    if not docker:
        pytest.skip("Docker Compose is not installed")

    # Preserve only values needed to find Docker and its CLI plugins. The empty
    # interpolation env-file and --no-env-resolution keep real credentials out.
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "COMSPEC"}
    }
    if subprocess.run(
        [docker, "compose", "version"], env=env, capture_output=True, check=False
    ).returncode:
        pytest.skip("Docker Compose is not installed")

    env.update(overrides or {})
    env_file = tmp_path / "empty.env"
    env_file.touch()
    result = subprocess.run(
        [
            docker,
            "compose",
            "--env-file",
            str(env_file),
            "-f",
            str(COMPOSE_FILE),
            "config",
            "--no-env-resolution",
            "--format",
            "json",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _ports(config: dict, service: str) -> dict[int, tuple[str | None, str]]:
    return {
        port["target"]: (port.get("host_ip"), str(port["published"]))
        for port in config["services"][service]["ports"]
    }


def test_database_and_browser_ports_bind_to_loopback_by_default(tmp_path: Path) -> None:
    config = _compose_config(tmp_path)

    assert _ports(config, "falkordb") == {
        6379: ("127.0.0.1", "6379"),
        3000: ("127.0.0.1", "3000"),
    }
    assert _ports(config, "qdrant") == {
        6333: ("127.0.0.1", "6333"),
        6334: ("127.0.0.1", "6334"),
    }
    assert _ports(config, "flask-api") == {8001: (None, "8001")}


def test_empty_database_bind_address_still_uses_loopback(tmp_path: Path) -> None:
    config = _compose_config(tmp_path, {"AUTOMEM_DB_BIND_ADDRESS": ""})
    assert {binding[0] for binding in _ports(config, "falkordb").values()} == {"127.0.0.1"}
    assert {binding[0] for binding in _ports(config, "qdrant").values()} == {"127.0.0.1"}


@pytest.mark.parametrize(
    ("bind_address", "resolved_address"), [("0.0.0.0", "0.0.0.0"), ("[::1]", "::1")]
)
def test_database_bind_address_and_host_ports_can_be_explicitly_overridden(
    tmp_path: Path, bind_address: str, resolved_address: str
) -> None:
    config = _compose_config(
        tmp_path,
        {
            "AUTOMEM_DB_BIND_ADDRESS": bind_address,
            "FALKORDB_HOST_PORT": "16379",
            "FALKORDB_BROWSER_HOST_PORT": "13000",
            "QDRANT_HOST_PORT": "16333",
            "QDRANT_GRPC_HOST_PORT": "16334",
        },
    )

    assert _ports(config, "falkordb") == {
        6379: (resolved_address, "16379"),
        3000: (resolved_address, "13000"),
    }
    assert _ports(config, "qdrant") == {
        6333: (resolved_address, "16333"),
        6334: (resolved_address, "16334"),
    }
