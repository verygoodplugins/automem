from __future__ import annotations

import os
from typing import Any, Callable


def init_falkordb(
    *,
    state: Any,
    logger: Any,
    falkordb_cls: Any,
    graph_name: str,
    falkordb_port: int,
) -> None:
    """Initialize FalkorDB connection if not already connected."""
    if state.memory_graph is not None:
        return

    host = (
        os.getenv("FALKORDB_HOST")
        or os.getenv("RAILWAY_PRIVATE_DOMAIN")
        or os.getenv("RAILWAY_PUBLIC_DOMAIN")
        or "localhost"
    )
    password = os.getenv("FALKORDB_PASSWORD")

    try:
        logger.info("Connecting to FalkorDB at %s:%s", host, falkordb_port)
        connection_params = {
            "host": host,
            "port": falkordb_port,
        }
        if password:
            connection_params["password"] = password
            connection_params["username"] = "default"

        state.falkordb = falkordb_cls(**connection_params)
        state.memory_graph = state.falkordb.select_graph(graph_name)
        logger.info(
            "FalkorDB connection established (auth: %s)",
            "enabled" if password else "disabled",
        )
    except Exception:  # pragma: no cover
        logger.exception("Failed to initialize FalkorDB connection")
        state.falkordb = None
        state.memory_graph = None


def init_qdrant(
    *,
    state: Any,
    logger: Any,
    qdrant_client_cls: Any,
    ensure_collection_fn: Callable[[], None],
) -> None:
    """Initialize Qdrant connection and ensure the collection exists."""
    if state.qdrant is not None:
        return

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        logger.info("Qdrant URL not provided; skipping client initialization")
        return

    try:
        logger.info("Connecting to Qdrant at %s", url)
        state.qdrant = qdrant_client_cls(url=url, api_key=api_key)
        ensure_collection_fn()
        logger.info("Qdrant connection established")
    except ValueError:
        logger.exception("Invalid Qdrant configuration; running without vector store")
        state.qdrant = None
        return
    except Exception:  # pragma: no cover
        logger.exception("Failed to initialize Qdrant client")
        state.qdrant = None


def ensure_qdrant_collection(
    *,
    state: Any,
    logger: Any,
    collection_name: str,
    vector_size_config: int,
    get_effective_vector_size_fn: Callable[[Any], tuple[int, str]],
    vector_params_cls: Any,
    distance_enum: Any,
    payload_schema_type_enum: Any,
) -> None:
    """Create the Qdrant collection if it does not already exist."""
    if state.qdrant is None:
        return

    try:
        effective_dim, source = get_effective_vector_size_fn(state.qdrant)
        state.effective_vector_size = effective_dim

        if source == "collection":
            logger.info(
                "Using existing collection dimension: %dd (config default: %dd)",
                effective_dim,
                vector_size_config,
            )
        else:
            logger.info("Using configured vector dimension: %dd", effective_dim)

        collections = state.qdrant.get_collections()
        existing = {collection.name for collection in collections.collections}
        if collection_name not in existing:
            logger.info(
                "Creating Qdrant collection '%s' with %dd vectors",
                collection_name,
                effective_dim,
            )
            state.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=vector_params_cls(size=effective_dim, distance=distance_enum.COSINE),
            )

        logger.info("Ensuring Qdrant payload indexes for collection '%s'", collection_name)
        if payload_schema_type_enum:
            state.qdrant.create_payload_index(
                collection_name=collection_name,
                field_name="tags",
                field_schema=payload_schema_type_enum.KEYWORD,
            )
            state.qdrant.create_payload_index(
                collection_name=collection_name,
                field_name="tag_prefixes",
                field_schema=payload_schema_type_enum.KEYWORD,
            )
        else:
            state.qdrant.create_payload_index(
                collection_name=collection_name,
                field_name="tags",
                field_schema="keyword",
            )
            state.qdrant.create_payload_index(
                collection_name=collection_name,
                field_name="tag_prefixes",
                field_schema="keyword",
            )
    except ValueError:
        raise
    except Exception:  # pragma: no cover
        logger.exception("Failed to ensure Qdrant collection; disabling client")
        state.qdrant = None
