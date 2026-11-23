"""Stateful services must not float on a mutable image tag."""

import re
from pathlib import Path

import pytest

COMPOSE_FILE = Path(__file__).resolve().parents[1] / "docker-compose.yml"
SEMVER_TAG = re.compile(r"^v?\d+\.\d+\.\d+$")


def _image_tag(repository: str) -> str:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    match = re.search(rf"^\s*image:\s*{re.escape(repository)}:(\S+)\s*$", text, re.MULTILINE)
    assert match, f"{repository} image not found in docker-compose.yml"
    return match.group(1)


@pytest.mark.parametrize("repository", ["falkordb/falkordb", "qdrant/qdrant"])
def test_stateful_service_images_use_an_exact_version(repository: str) -> None:
    assert SEMVER_TAG.match(_image_tag(repository))
