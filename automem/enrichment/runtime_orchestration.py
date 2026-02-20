from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class FalkorDBUnavailableError(RuntimeError):
    """Raised when enrichment requires FalkorDB but no graph client is available."""


def jit_enrich_lightweight(
    *,
    memory_id: str,
    properties: Dict[str, Any],
    get_memory_graph_fn: Callable[[], Any],
    get_qdrant_client_fn: Callable[[], Any],
    parse_metadata_field_fn: Callable[[Any], Any],
    normalize_tag_list_fn: Callable[[Any], List[str]],
    extract_entities_fn: Callable[[str], Dict[str, List[str]]],
    slugify_fn: Callable[[str], str],
    compute_tag_prefixes_fn: Callable[[List[str]], List[str]],
    enrichment_enable_summaries: bool,
    generate_summary_fn: Callable[[str, Any], Any],
    utc_now_fn: Callable[[], str],
    collection_name: str,
    logger: Any,
) -> Optional[Dict[str, Any]]:
    """Run lightweight JIT enrichment inline during recall."""
    graph = get_memory_graph_fn()
    if graph is None:
        return None

    try:
        check = graph.query(
            "MATCH (m:Memory {id: $id}) RETURN m.enriched, m.processed",
            {"id": memory_id},
        )
        if getattr(check, "result_set", None):
            row = check.result_set[0]
            if row[0] or row[1]:
                logger.debug("JIT skipped for %s (already enriched/processed)", memory_id)
                return None
    except Exception as exc:  # noqa: BLE001 - best-effort guard, proceed if check fails
        logger.debug("JIT state-check failed for %s: %s", memory_id, exc)

    content = properties.get("content", "") or ""
    if not content:
        return None

    entities = extract_entities_fn(content)

    tags = list(dict.fromkeys(normalize_tag_list_fn(properties.get("tags"))))
    entity_tags: Set[str] = set()

    metadata_raw = properties.get("metadata")
    metadata = parse_metadata_field_fn(metadata_raw) or {}
    if not isinstance(metadata, dict):
        metadata = {"_raw_metadata": metadata}

    if entities:
        entities_section = metadata.setdefault("entities", {})
        if not isinstance(entities_section, dict):
            entities_section = {}
        for category, values in entities.items():
            if not values:
                continue
            entities_section[category] = sorted(values)
            for value in values:
                slug = slugify_fn(value)
                if slug:
                    entity_tags.add(f"entity:{category}:{slug}")
        metadata["entities"] = entities_section

    if entity_tags:
        tags = list(dict.fromkeys(tags + sorted(entity_tags)))

    tag_prefixes = compute_tag_prefixes_fn(tags)

    if enrichment_enable_summaries:
        summary = generate_summary_fn(content, properties.get("summary"))
    else:
        summary = properties.get("summary")

    enriched_at = utc_now_fn()

    enrichment_meta = metadata.setdefault("enrichment", {})
    if not isinstance(enrichment_meta, dict):
        enrichment_meta = {}
    enrichment_meta["jit"] = True
    enrichment_meta["jit_at"] = enriched_at
    metadata["enrichment"] = enrichment_meta

    update_payload = {
        "id": memory_id,
        "metadata": json.dumps(metadata, default=str),
        "tags": tags,
        "tag_prefixes": tag_prefixes,
        "summary": summary,
        "enriched_at": enriched_at,
    }
    try:
        graph.query(
            """
            MATCH (m:Memory {id: $id})
            SET m.metadata = $metadata,
                m.tags = $tags,
                m.tag_prefixes = $tag_prefixes,
                m.summary = $summary,
                m.enriched = true,
                m.enriched_at = $enriched_at
            """,
            update_payload,
        )
    except Exception:
        logger.exception("JIT enrichment graph update failed for %s", memory_id)
        return None

    qdrant_client = get_qdrant_client_fn()
    if qdrant_client is not None:
        try:
            qdrant_client.set_payload(
                collection_name=collection_name,
                points=[memory_id],
                payload={
                    "tags": tags,
                    "tag_prefixes": tag_prefixes,
                    "metadata": metadata,
                },
            )
        except Exception as exc:  # noqa: BLE001 - Qdrant client raises multiple exception types
            logger.debug("JIT Qdrant payload sync skipped for %s: %s", memory_id, exc)

    updated = dict(properties)
    updated["tags"] = tags
    updated["tag_prefixes"] = tag_prefixes
    updated["metadata"] = metadata
    updated["summary"] = summary
    updated["enriched"] = True
    updated["enriched_at"] = enriched_at

    logger.debug("JIT-enriched memory %s (entities=%s)", memory_id, bool(entities))
    return updated


def enrich_memory(
    *,
    memory_id: str,
    forced: bool,
    get_memory_graph_fn: Callable[[], Any],
    get_qdrant_client_fn: Callable[[], Any],
    parse_metadata_field_fn: Callable[[Any], Any],
    normalize_tag_list_fn: Callable[[Any], List[str]],
    extract_entities_fn: Callable[[str], Dict[str, List[str]]],
    slugify_fn: Callable[[str], str],
    compute_tag_prefixes_fn: Callable[[List[str]], List[str]],
    find_temporal_relationships_fn: Callable[[Any, str], int],
    detect_patterns_fn: Callable[[Any, str, str], List[Dict[str, Any]]],
    link_semantic_neighbors_fn: Callable[[Any, str], List[Tuple[str, float]]],
    enrichment_enable_summaries: bool,
    generate_summary_fn: Callable[[str, Any], Any],
    utc_now_fn: Callable[[], str],
    collection_name: str,
    unexpected_response_exc: Any,
    logger: Any,
) -> bool:
    """Enrich a memory with relationships, patterns, and entity extraction."""
    graph = get_memory_graph_fn()
    if graph is None:
        raise FalkorDBUnavailableError

    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})

    if not result.result_set:
        logger.debug("Skipping enrichment for %s; memory not found", memory_id)
        return False

    node = result.result_set[0][0]
    properties = getattr(node, "properties", None)
    if not isinstance(properties, dict):
        properties = dict(getattr(node, "__dict__", {}))

    metadata_raw = properties.get("metadata")
    metadata = parse_metadata_field_fn(metadata_raw) or {}
    if not isinstance(metadata, dict):
        metadata = {"_raw_metadata": metadata}

    already_processed = bool(properties.get("processed"))
    if already_processed and not forced:
        return False

    content = properties.get("content", "") or ""
    entities = extract_entities_fn(content)

    tags = list(dict.fromkeys(normalize_tag_list_fn(properties.get("tags"))))
    entity_tags: Set[str] = set()

    if entities:
        entities_section = metadata.setdefault("entities", {})
        if not isinstance(entities_section, dict):
            entities_section = {}
        for category, values in entities.items():
            if not values:
                continue
            entities_section[category] = sorted(values)
            for value in values:
                slug = slugify_fn(value)
                if slug:
                    entity_tags.add(f"entity:{category}:{slug}")
        metadata["entities"] = entities_section

    if entity_tags:
        tags = list(dict.fromkeys(tags + sorted(entity_tags)))

    tag_prefixes = compute_tag_prefixes_fn(tags)

    temporal_links = find_temporal_relationships_fn(graph, memory_id)
    pattern_info = detect_patterns_fn(graph, memory_id, content)
    semantic_neighbors = link_semantic_neighbors_fn(graph, memory_id)

    if enrichment_enable_summaries:
        existing_summary = properties.get("summary")
        summary = generate_summary_fn(content, existing_summary if forced else None)
    else:
        summary = properties.get("summary")

    enrichment_meta = metadata.setdefault("enrichment", {})
    if not isinstance(enrichment_meta, dict):
        enrichment_meta = {}
    enrichment_meta.update(
        {
            "last_run": utc_now_fn(),
            "forced": forced,
            "temporal_links": temporal_links,
            "patterns_detected": pattern_info,
            "semantic_neighbors": [
                {"id": neighbour_id, "score": score} for neighbour_id, score in semantic_neighbors
            ],
        }
    )
    metadata["enrichment"] = enrichment_meta

    update_payload = {
        "id": memory_id,
        "metadata": json.dumps(metadata, default=str),
        "tags": tags,
        "tag_prefixes": tag_prefixes,
        "summary": summary,
        "enriched_at": utc_now_fn(),
    }

    graph.query(
        """
        MATCH (m:Memory {id: $id})
        SET m.metadata = $metadata,
            m.tags = $tags,
            m.tag_prefixes = $tag_prefixes,
            m.summary = $summary,
            m.enriched = true,
            m.enriched_at = $enriched_at,
            m.processed = true
        """,
        update_payload,
    )

    qdrant_client = get_qdrant_client_fn()
    if qdrant_client is not None:
        try:
            qdrant_client.set_payload(
                collection_name=collection_name,
                points=[memory_id],
                payload={
                    "tags": tags,
                    "tag_prefixes": tag_prefixes,
                    "metadata": metadata,
                },
            )
        except unexpected_response_exc as exc:
            if exc.status_code == 404:
                logger.debug(
                    "Qdrant payload sync skipped - point not yet uploaded: %s", memory_id[:8]
                )
            else:
                logger.warning("Qdrant payload sync failed (%d): %s", exc.status_code, memory_id)
        except Exception:
            logger.exception("Failed to sync Qdrant payload for enriched memory %s", memory_id)

    logger.debug(
        "Enriched memory %s (temporal=%s, patterns=%s, semantic=%s)",
        memory_id,
        temporal_links,
        pattern_info,
        len(semantic_neighbors),
    )

    return True
