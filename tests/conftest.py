import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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

    models_module.Distance = Distance
    models_module.VectorParams = VectorParams
    models_module.PointStruct = PointStruct
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
