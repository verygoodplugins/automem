"""Connection settings for the optional content_hash backfill."""

import importlib
import json
import sys
from types import SimpleNamespace

import qdrant_client

from automem.stores.vector_payload import content_fingerprint
from scripts import backfill_content_hash as backfill


class FakeQdrant:
    instances = []

    def __init__(self, *, url, api_key=None):
        self.url = url
        self.api_key = api_key
        self.points = {"legacy": {"content": "Legacy notes"}}
        self.collections = []
        FakeQdrant.instances.append(self)

    def scroll(self, *, collection_name, offset=None, **kwargs):
        self.collections.append(collection_name)
        points = [
            SimpleNamespace(id=point_id, payload=dict(payload))
            for point_id, payload in self.points.items()
            if "content_hash" not in payload
        ]
        return points, None

    def set_payload(self, *, collection_name, points, payload, **kwargs):
        point_id = points.must[0].has_id[0]
        if "content_hash" not in self.points[point_id]:
            self.points[point_id].update(payload)


def test_backfill_connects_with_only_qdrant_host_and_port(monkeypatch, capsys):
    import automem.config as config

    with monkeypatch.context() as mp:
        # Railway sets QDRANT_HOST + QDRANT_PORT and leaves QDRANT_URL unset.
        for name in ("QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION"):
            mp.delenv(name, raising=False)
        mp.setenv("QDRANT_HOST", "qdrant.railway.internal")
        mp.setenv("QDRANT_PORT", "6334")
        mp.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
        importlib.reload(config)
        FakeQdrant.instances.clear()
        mp.setattr(qdrant_client, "QdrantClient", FakeQdrant)
        mp.setattr(sys, "argv", ["backfill_content_hash.py", "--apply"])

        assert backfill.main() == 0

        (client,) = FakeQdrant.instances
        assert client.url == "http://qdrant.railway.internal:6334"
        assert client.api_key is None
        assert client.collections == ["memories"]
        assert client.points["legacy"]["content_hash"] == content_fingerprint("Legacy notes")
        assert json.loads(capsys.readouterr().out) == {
            "scanned": 1,
            "updated": 1,
            "failures": 0,
            "dry_run": False,
        }

    importlib.reload(config)
