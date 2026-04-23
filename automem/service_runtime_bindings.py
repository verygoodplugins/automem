from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from automem.embedding.provider_init import (
    init_embedding_provider as _init_embedding_provider_runtime,
)
from automem.service_runtime import get_memory_graph as _get_memory_graph_runtime
from automem.service_runtime import get_qdrant_client as _get_qdrant_client_runtime
from automem.service_runtime import init_openai as _init_openai_runtime
from automem.stores.runtime_clients import (
    ensure_qdrant_collection as _ensure_qdrant_collection_runtime,
)
from automem.stores.runtime_clients import init_falkordb as _init_falkordb_runtime
from automem.stores.runtime_clients import init_qdrant as _init_qdrant_runtime


@dataclass(frozen=True)
class ServiceRuntimeBindings:
    init_openai: Callable[[], None]
    init_embedding_provider: Callable[[], None]
    init_falkordb: Callable[[], None]
    init_qdrant: Callable[[], None]
    ensure_qdrant_collection: Callable[[], None]
    get_memory_graph: Callable[[], Any]
    get_qdrant_client: Callable[[], Any]


def create_service_runtime(
    *,
    get_state_fn: Callable[[], Any],
    logger: Any,
    openai_cls: Any,
    get_env_fn: Callable[[str], str | None],
    vector_size_config_fn: Callable[[], int],
    embedding_model_fn: Callable[[], str],
    falkordb_cls: Any,
    graph_name: str,
    falkordb_port: int,
    qdrant_client_cls: Any,
    collection_name: str,
    get_effective_vector_size_fn: Callable[[], int],
    vector_params_cls: Any,
    distance_enum: Any,
    payload_schema_type_enum: Any,
    get_init_falkordb_fn: Callable[[], Callable[[], None]],
    get_init_qdrant_fn: Callable[[], Callable[[], None]],
) -> ServiceRuntimeBindings:
    def init_openai() -> None:
        _init_openai_runtime(
            state=get_state_fn(),
            logger=logger,
            openai_cls=openai_cls,
            get_env_fn=get_env_fn,
        )

    def init_embedding_provider() -> None:
        _init_embedding_provider_runtime(
            state=get_state_fn(),
            logger=logger,
            vector_size_config=vector_size_config_fn(),
            embedding_model=embedding_model_fn(),
        )

    def init_falkordb() -> None:
        _init_falkordb_runtime(
            state=get_state_fn(),
            logger=logger,
            falkordb_cls=falkordb_cls,
            graph_name=graph_name,
            falkordb_port=falkordb_port,
        )

    def ensure_qdrant_collection() -> None:
        _ensure_qdrant_collection_runtime(
            state=get_state_fn(),
            logger=logger,
            collection_name=collection_name,
            vector_size_config=vector_size_config_fn(),
            get_effective_vector_size_fn=get_effective_vector_size_fn,
            vector_params_cls=vector_params_cls,
            distance_enum=distance_enum,
            payload_schema_type_enum=payload_schema_type_enum,
        )

    def init_qdrant() -> None:
        _init_qdrant_runtime(
            state=get_state_fn(),
            logger=logger,
            qdrant_client_cls=qdrant_client_cls,
            ensure_collection_fn=ensure_qdrant_collection,
        )

    def get_memory_graph() -> Any:
        return _get_memory_graph_runtime(
            state=get_state_fn(),
            init_falkordb_fn=lambda: get_init_falkordb_fn()(),
        )

    def get_qdrant_client() -> Any:
        return _get_qdrant_client_runtime(
            state=get_state_fn(),
            init_qdrant_fn=lambda: get_init_qdrant_fn()(),
        )

    return ServiceRuntimeBindings(
        init_openai=init_openai,
        init_embedding_provider=init_embedding_provider,
        init_falkordb=init_falkordb,
        init_qdrant=init_qdrant,
        ensure_qdrant_collection=ensure_qdrant_collection,
        get_memory_graph=get_memory_graph,
        get_qdrant_client=get_qdrant_client,
    )
