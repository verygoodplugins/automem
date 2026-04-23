from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class AppHelperBindings:
    normalize_tags: Callable[[Any], List[str]]
    coerce_importance: Callable[[Any], float]
    coerce_embedding: Callable[[Any], Optional[List[float]]]
    generate_placeholder_embedding: Callable[[str], List[float]]
    generate_real_embedding: Callable[[str], List[float]]
    generate_real_embeddings_batch: Callable[[List[str]], List[List[float]]]
    fetch_relations: Callable[[Any, str], List[Dict[str, Any]]]


def create_app_helper_runtime(
    *,
    get_state_fn: Callable[[], Any],
    abort_fn: Callable[..., Any],
    init_embedding_provider_fn: Callable[[], None],
    get_generate_placeholder_embedding_fn: Callable[[], Callable[[str], List[float]]],
    normalize_tags_value_fn: Callable[[Any], List[str]],
    coerce_importance_value_fn: Callable[[Any], float],
    coerce_embedding_value_fn: Callable[[Any, int], Optional[List[float]]],
    generate_placeholder_embedding_value_fn: Callable[[str, int], List[float]],
    generate_real_embedding_value_fn: Callable[..., List[float]],
    generate_real_embeddings_batch_value_fn: Callable[..., List[List[float]]],
    fetch_relations_runtime_fn: Callable[..., List[Dict[str, Any]]],
    relation_limit: int,
    serialize_node_fn: Callable[[Any], Dict[str, Any]],
    summarize_relation_node_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    logger: Any,
) -> AppHelperBindings:
    def normalize_tags(value: Any) -> List[str]:
        try:
            return normalize_tags_value_fn(value)
        except ValueError as exc:
            abort_fn(400, description=str(exc))

    def coerce_importance(value: Any) -> float:
        try:
            return coerce_importance_value_fn(value)
        except ValueError as exc:
            abort_fn(400, description=str(exc))

    def coerce_embedding(value: Any) -> Optional[List[float]]:
        return coerce_embedding_value_fn(value, get_state_fn().effective_vector_size)

    def generate_placeholder_embedding(content: str) -> List[float]:
        return generate_placeholder_embedding_value_fn(
            content, get_state_fn().effective_vector_size
        )

    def generate_real_embedding(content: str) -> List[float]:
        return generate_real_embedding_value_fn(
            content,
            init_embedding_provider=init_embedding_provider_fn,
            state=get_state_fn(),
            logger=logger,
            placeholder_embedding=get_generate_placeholder_embedding_fn(),
        )

    def generate_real_embeddings_batch(contents: List[str]) -> List[List[float]]:
        return generate_real_embeddings_batch_value_fn(
            contents,
            init_embedding_provider=init_embedding_provider_fn,
            state=get_state_fn(),
            logger=logger,
            placeholder_embedding=get_generate_placeholder_embedding_fn(),
        )

    def fetch_relations(graph: Any, memory_id: str) -> List[Dict[str, Any]]:
        return fetch_relations_runtime_fn(
            graph=graph,
            memory_id=memory_id,
            relation_limit=relation_limit,
            serialize_node_fn=serialize_node_fn,
            summarize_relation_node_fn=summarize_relation_node_fn,
            logger=logger,
        )

    return AppHelperBindings(
        normalize_tags=normalize_tags,
        coerce_importance=coerce_importance,
        coerce_embedding=coerce_embedding,
        generate_placeholder_embedding=generate_placeholder_embedding,
        generate_real_embedding=generate_real_embedding,
        generate_real_embeddings_batch=generate_real_embeddings_batch,
        fetch_relations=fetch_relations,
    )
