"""Tests for Voyage 4 mixed-model store and recall embeddings."""

import logging
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

from automem.embedding.provider import EmbeddingProvider
from automem.embedding.provider_init import init_embedding_provider
from automem.embedding.runtime_helpers import generate_real_embedding
from automem.embedding.voyage import VoyageEmbeddingProvider
from automem.runtime_wiring import wire_recall_and_blueprints


class _SingleModelProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.calls = []

    def generate_embedding(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 2.0]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.generate_embedding(text) for text in texts]

    def dimension(self) -> int:
        return 2

    def provider_name(self) -> str:
        return "single-model"


def test_base_provider_uses_single_model_for_store_and_recall() -> None:
    provider = _SingleModelProvider()

    assert provider.embed_for_store("memory") == [1.0, 2.0]
    assert provider.embed_for_recall("query") == [1.0, 2.0]
    assert provider.calls == ["memory", "query"]


@patch("automem.embedding.voyage.httpx.Client")
def test_voyage_routes_store_recall_and_batch_to_configured_models(
    mock_httpx_client_class: Mock,
) -> None:
    mock_client = Mock()
    mock_httpx_client_class.return_value = mock_client
    store_vector = [0.1] * 256
    recall_vector = [0.2] * 256
    legacy_vector = [0.3] * 256
    batch_vectors = [[0.4] * 256, [0.5] * 256]

    response = Mock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.side_effect = [
        {"data": [{"embedding": store_vector}]},
        {"data": [{"embedding": recall_vector}]},
        {"data": [{"embedding": legacy_vector}]},
        {"data": [{"embedding": vector} for vector in batch_vectors]},
    ]
    mock_client.post.return_value = response

    provider = VoyageEmbeddingProvider(
        api_key="voyage-test-key",
        model="voyage-4",
        store_model="voyage-4-lite",
        recall_model="voyage-4-large",
        dimension=256,
    )

    assert provider.embed_for_store("memory") == store_vector
    assert provider.embed_for_recall("query") == recall_vector
    assert provider.generate_embedding("legacy store") == legacy_vector
    assert provider.generate_embeddings_batch(["one", "two"]) == batch_vectors
    assert [call.kwargs["json"]["model"] for call in mock_client.post.call_args_list] == [
        "voyage-4-lite",
        "voyage-4-large",
        "voyage-4-lite",
        "voyage-4-lite",
    ]
    assert provider.provider_name() == "voyage:voyage-4"


@patch("automem.embedding.voyage.httpx.Client")
def test_voyage_split_models_independently_fall_back_to_legacy_model(
    mock_httpx_client_class: Mock,
) -> None:
    cases = [
        (None, None, "voyage-4", "voyage-4"),
        ("voyage-4-lite", None, "voyage-4-lite", "voyage-4"),
        (None, "voyage-4-large", "voyage-4", "voyage-4-large"),
    ]

    for store_model, recall_model, expected_store, expected_recall in cases:
        provider = VoyageEmbeddingProvider(
            api_key="voyage-test-key",
            model="voyage-4",
            store_model=store_model,
            recall_model=recall_model,
            dimension=1024,
        )

        assert provider._store_model == expected_store
        assert provider._recall_model == expected_recall


@patch.dict(
    os.environ,
    {
        "EMBEDDING_PROVIDER": "voyage",
        "VOYAGE_API_KEY": "voyage-key",
        "VOYAGE_MODEL": "voyage-4",
        "VOYAGE_STORE_MODEL": "voyage-4-lite",
        "VOYAGE_RECALL_MODEL": "voyage-4-large",
    },
    clear=True,
)
@patch("automem.embedding.voyage.VoyageEmbeddingProvider")
def test_provider_init_passes_voyage_store_and_recall_models(mock_voyage_class: Mock) -> None:
    state = SimpleNamespace(
        embedding_provider=None,
        qdrant=None,
        effective_vector_size=1024,
    )
    mock_voyage_class.return_value.provider_name.return_value = "voyage:voyage-4"

    init_embedding_provider(
        state=state,
        logger=Mock(),
        vector_size_config=1024,
        embedding_model="text-embedding-3-small",
    )

    mock_voyage_class.assert_called_once_with(
        api_key="voyage-key",
        model="voyage-4",
        store_model="voyage-4-lite",
        recall_model="voyage-4-large",
        dimension=1024,
    )


def test_runtime_routes_store_and_recall_to_explicit_provider_methods() -> None:
    class Provider:
        def embed_for_store(self, text: str) -> list[float]:
            assert text == "text"
            return [1.0, 1.0]

        def embed_for_recall(self, text: str) -> list[float]:
            assert text == "text"
            return [2.0, 2.0]

        def provider_name(self) -> str:
            return "split-model"

    state = SimpleNamespace(embedding_provider=Provider(), effective_vector_size=2)
    kwargs = {
        "init_embedding_provider": lambda: None,
        "state": state,
        "logger": logging.getLogger("test_voyage_mixed_model"),
        "placeholder_embedding": lambda _text: [0.0, 0.0],
    }

    assert generate_real_embedding("text", **kwargs) == [1.0, 1.0]
    assert generate_real_embedding("text", for_recall=True, **kwargs) == [2.0, 2.0]


def test_recall_wiring_requests_the_recall_embedding_path() -> None:
    module = Mock()
    module._generate_real_recall_embedding.return_value = [2.0, 2.0]
    configured = {}

    wire_recall_and_blueprints(
        module=module,
        configure_recall_helpers_fn=lambda **kwargs: configured.update(kwargs),
        register_blueprints_fn=lambda **_kwargs: None,
    )

    assert configured["generate_real_embedding"]("query") == [2.0, 2.0]
    module._generate_real_recall_embedding.assert_called_once_with("query")
