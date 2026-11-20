"""Exercise collection validation through the actual startup call chain."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from automem import config
from automem.runtime_wiring import run_default_server
from automem.stores.runtime_clients import ensure_qdrant_collection, init_qdrant
from automem.utils.validation import VectorDimensionMismatchError, get_effective_vector_size


@pytest.fixture
def startup(monkeypatch):
    monkeypatch.setattr(config, "QDRANT_URL", "http://localhost:6333")
    monkeypatch.setattr(config, "QDRANT_API_KEY", None)
    monkeypatch.setattr(config, "VECTOR_SIZE", 1024)
    monkeypatch.setattr(config, "COLLECTION_NAME", "custom-memories")
    monkeypatch.setenv("VECTOR_SIZE_AUTODETECT", "false")
    monkeypatch.setenv("QDRANT_ENSURE_PAYLOAD_INDEXES", "false")
    client = MagicMock()
    client.get_collection.return_value.config.params.vectors.size = 3072
    client.get_collections.return_value.collections = [SimpleNamespace(name="custom-memories")]
    state = SimpleNamespace(qdrant=None, effective_vector_size=1024)
    logger = MagicMock()

    def initialize():
        init_qdrant(
            state=state,
            logger=logger,
            qdrant_client_cls=lambda **kwargs: client,
            ensure_collection_fn=lambda: ensure_qdrant_collection(
                state=state,
                logger=logger,
                collection_name=config.COLLECTION_NAME,
                vector_size_config=config.VECTOR_SIZE,
                get_effective_vector_size_fn=get_effective_vector_size,
                vector_params_cls=MagicMock(),
                hnsw_config_diff_cls=MagicMock(),
                distance_enum=SimpleNamespace(COSINE="Cosine"),
                payload_schema_type_enum=None,
            ),
        )

    return SimpleNamespace(client=client, state=state, logger=logger, initialize=initialize)


def test_strict_mismatch_propagates_through_collection_ensure(startup):
    with pytest.raises(VectorDimensionMismatchError, match="FATAL: Vector dimension mismatch"):
        startup.initialize()
    assert startup.state.qdrant is None
    assert "Qdrant connection established" not in [
        call.args[0] for call in startup.logger.info.call_args_list
    ]


@pytest.mark.parametrize("autodetect,dimension", [("true", 3072), ("false", 1024)])
def test_valid_collection_remains_connected(startup, monkeypatch, autodetect, dimension):
    monkeypatch.setenv("VECTOR_SIZE_AUTODETECT", autodetect)
    startup.client.get_collection.return_value.config.params.vectors.size = dimension
    startup.initialize()
    assert startup.state.qdrant is startup.client
    assert startup.state.effective_vector_size == dimension
    startup.logger.info.assert_any_call("Qdrant connection established")


def test_collection_connectivity_failure_stays_fail_soft_without_success_log(startup):
    startup.client.get_collection.side_effect = ConnectionError("Qdrant unavailable")
    startup.initialize()
    assert startup.state.qdrant is None
    assert "Qdrant connection established" not in [
        call.args[0] for call in startup.logger.info.call_args_list
    ]


def test_default_server_exits_before_serving_on_real_collection_mismatch(startup):
    module = MagicMock()
    module.logger = startup.logger
    module.init_qdrant = startup.initialize
    with pytest.raises(SystemExit) as exc:
        run_default_server(module=module)
    assert exc.value.code == 1
    module.app.run.assert_not_called()
    module.stop_sync_worker.assert_called_once()
    module.stop_consolidation_scheduler.assert_called_once()
