from __future__ import annotations

from typing import Any, Callable

from automem.config import MINIMAX_BASE_URL, MINIMAX_DEFAULT_MODEL


def init_openai(
    *,
    state: Any,
    logger: Any,
    openai_cls: Any,
    get_env_fn: Callable[[str], str | None],
) -> None:
    """Initialize OpenAI-compatible client for memory type classification (not embeddings).

    Supports multiple LLM providers via the ``LLM_PROVIDER`` env var:
    - ``auto`` (default): Use OPENAI_API_KEY if set, else MINIMAX_API_KEY
    - ``openai``: Use OpenAI explicitly
    - ``minimax``: Use MiniMax explicitly (OpenAI-compatible API at api.minimax.io)
    """
    if state.openai_client is not None:
        return

    if openai_cls is None:
        logger.info("OpenAI package not installed (used for memory type classification)")
        return

    llm_provider = (get_env_fn("LLM_PROVIDER") or "auto").strip().lower()

    if llm_provider == "minimax":
        api_key = get_env_fn("MINIMAX_API_KEY")
        if not api_key:
            logger.info("LLM_PROVIDER=minimax but MINIMAX_API_KEY not set")
            return
        try:
            state.openai_client = openai_cls(api_key=api_key, base_url=MINIMAX_BASE_URL)
            state.llm_provider = "minimax"
            logger.info(
                "MiniMax client initialized for memory type classification "
                "(model: %s)",
                get_env_fn("CLASSIFICATION_MODEL") or MINIMAX_DEFAULT_MODEL,
            )
        except Exception:
            logger.exception("Failed to initialize MiniMax client")
            state.openai_client = None
        return

    if llm_provider == "openai":
        api_key = get_env_fn("OPENAI_API_KEY")
        if not api_key:
            logger.info("LLM_PROVIDER=openai but OPENAI_API_KEY not set")
            return
        try:
            openai_base_url = (get_env_fn("OPENAI_BASE_URL") or "").strip() or None
            client_kwargs: dict[str, Any] = {"api_key": api_key}
            if openai_base_url:
                client_kwargs["base_url"] = openai_base_url
            state.openai_client = openai_cls(**client_kwargs)
            state.llm_provider = "openai"
            logger.info("OpenAI client initialized for memory type classification")
        except Exception:
            logger.exception("Failed to initialize OpenAI client")
            state.openai_client = None
        return

    # Auto-detection: try OPENAI_API_KEY first, then MINIMAX_API_KEY
    api_key = get_env_fn("OPENAI_API_KEY")
    if api_key:
        try:
            openai_base_url = (get_env_fn("OPENAI_BASE_URL") or "").strip() or None
            client_kwargs = {"api_key": api_key}
            if openai_base_url:
                client_kwargs["base_url"] = openai_base_url
            state.openai_client = openai_cls(**client_kwargs)
            state.llm_provider = "openai"
            logger.info("OpenAI client initialized for memory type classification")
        except Exception:
            logger.exception("Failed to initialize OpenAI client")
            state.openai_client = None
        return

    minimax_key = get_env_fn("MINIMAX_API_KEY")
    if minimax_key:
        try:
            state.openai_client = openai_cls(api_key=minimax_key, base_url=MINIMAX_BASE_URL)
            state.llm_provider = "minimax"
            logger.info(
                "MiniMax client auto-detected for memory type classification "
                "(model: %s)",
                get_env_fn("CLASSIFICATION_MODEL") or MINIMAX_DEFAULT_MODEL,
            )
        except Exception:
            logger.exception("Failed to initialize MiniMax client")
            state.openai_client = None
        return

    logger.info(
        "No LLM API key provided for classification "
        "(set OPENAI_API_KEY or MINIMAX_API_KEY)"
    )


def get_memory_graph(*, state: Any, init_falkordb_fn: Callable[[], None]) -> Any:
    init_falkordb_fn()
    return state.memory_graph


def get_qdrant_client(*, state: Any, init_qdrant_fn: Callable[[], None]) -> Any:
    init_qdrant_fn()
    return state.qdrant
