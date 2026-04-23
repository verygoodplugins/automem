from __future__ import annotations

from typing import Any, Callable


def init_openai(
    *,
    state: Any,
    logger: Any,
    openai_cls: Any,
    get_env_fn: Callable[[str], str | None],
) -> None:
    """Initialize OpenAI client for memory type classification (not embeddings)."""
    if state.openai_client is not None:
        return

    if openai_cls is None:
        logger.info("OpenAI package not installed (used for memory type classification)")
        return

    api_key = get_env_fn("OPENAI_API_KEY")
    if not api_key:
        logger.info("OpenAI API key not provided (used for memory type classification)")
        return

    try:
        openai_base_url = (get_env_fn("OPENAI_BASE_URL") or "").strip() or None
        client_kwargs = {"api_key": api_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        state.openai_client = openai_cls(**client_kwargs)
        logger.info("OpenAI client initialized for memory type classification")
    except Exception:
        logger.exception("Failed to initialize OpenAI client")
        state.openai_client = None


def get_memory_graph(*, state: Any, init_falkordb_fn: Callable[[], None]) -> Any:
    init_falkordb_fn()
    return state.memory_graph


def get_qdrant_client(*, state: Any, init_qdrant_fn: Callable[[], None]) -> Any:
    init_qdrant_fn()
    return state.qdrant
