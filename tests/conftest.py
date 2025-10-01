import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_falkordb_stub() -> None:
    module = ModuleType("falkordb")

    class FalkorDB:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

        def select_graph(self, name: str) -> SimpleNamespace:
            def _noop_query(*args, **kwargs):
                return SimpleNamespace(result_set=[])

            return SimpleNamespace(query=_noop_query)

    module.FalkorDB = FalkorDB
    sys.modules.setdefault("falkordb", module)


def _install_qdrant_stub() -> None:
    client_module = ModuleType("qdrant_client")

    class QdrantClient:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            self._collections = []

        def get_collections(self):
            return SimpleNamespace(collections=self._collections)

        def create_collection(self, *args, **kwargs):
            self._collections.append(SimpleNamespace(name=kwargs.get("collection_name", "memories")))

        def upsert(self, *args, **kwargs):
            return None

        def search(self, *args, **kwargs):
            return []

        def delete(self, *args, **kwargs):
            return None

    client_module.QdrantClient = QdrantClient
    sys.modules.setdefault("qdrant_client", client_module)

    models_module = ModuleType("qdrant_client.models")

    class Distance:
        COSINE = "Cosine"

    class VectorParams:
        def __init__(self, size: int, distance: str):
            self.size = size
            self.distance = distance

    class PointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

    class MatchAny:
        def __init__(self, any):
            self.any = any

    class MatchValue:
        def __init__(self, value):
            self.value = value

    class FieldCondition:
        def __init__(self, key: str, match):
            self.key = key
            self.match = match

    class Filter:
        def __init__(self, must=None, should=None, must_not=None):
            self.must = must or []
            self.should = should or []
            self.must_not = must_not or []

    class PointIdsList:
        def __init__(self, points):
            self.points = points

    models_module.Distance = Distance
    models_module.VectorParams = VectorParams
    models_module.PointStruct = PointStruct
    models_module.MatchAny = MatchAny
    models_module.MatchValue = MatchValue
    models_module.FieldCondition = FieldCondition
    models_module.Filter = Filter
    models_module.PointIdsList = PointIdsList
    sys.modules.setdefault("qdrant_client.models", models_module)


def _install_openai_stub() -> None:
    module = ModuleType("openai")

    class _Embeddings:
        def create(self, *args, **kwargs):  # pragma: no cover - deterministic stub
            raise RuntimeError("OpenAI client not configured")

    class OpenAI:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            self.embeddings = _Embeddings()

    module.OpenAI = OpenAI
    sys.modules.setdefault("openai", module)


if "falkordb" not in sys.modules:
    _install_falkordb_stub()

if "qdrant_client" not in sys.modules:
    _install_qdrant_stub()

if "openai" not in sys.modules:
    _install_openai_stub()


def pytest_report_header(config):  # pragma: no cover - cosmetic output
    msgs = []
    if not os.getenv("AUTOMEM_RUN_INTEGRATION_TESTS"):
        msgs.append(
            "Integration tests: disabled (set AUTOMEM_RUN_INTEGRATION_TESTS=1 to enable)."
        )
    else:
        base = os.getenv("AUTOMEM_TEST_BASE_URL", "http://localhost:8001")
        msgs.append(f"Integration tests: enabled (base_url={base}).")
        if base.startswith("http://localhost") or base.startswith("http://127.0.0.1"):
            if os.getenv("AUTOMEM_START_DOCKER") == "1":
                msgs.append("Docker: will start via 'docker compose up -d'.")
        else:
            if os.getenv("AUTOMEM_ALLOW_LIVE") == "1":
                msgs.append("Live mode: enabled (AUTOMEM_ALLOW_LIVE=1). Use with caution.")
            else:
                msgs.append(
                    "Live mode: blocked (set AUTOMEM_ALLOW_LIVE=1 to run against non-local endpoints)."
                )
    return "\n".join(msgs)
