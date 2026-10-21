from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from automem.utils.entity_quality import validate_entity_value


class FalkorDBUnavailableError(RuntimeError):
    """Raised when enrichment requires FalkorDB but no graph client is available."""


def _validated_entities(entities: Dict[str, List[str]], content: str) -> Dict[str, List[str]]:
    validated: Dict[str, Set[str]] = {
        "tools": set(),
        "projects": set(),
        "people": set(),
        "concepts": set(),
        "organizations": set(),
    }
    for category, values in (entities or {}).items():
        for value in values or []:
            result = validate_entity_value(category, value, context=content)
            if result.accepted and result.name:
                validated[result.category].add(result.name)

    validated["tools"].difference_update(validated["people"])
    return {category: sorted(values) for category, values in validated.items()}


class EnrichmentConflictError(RuntimeError):
    """A concurrent memory edit requires enrichment to start with fresh content."""


def _persist_enrichment(
    *,
    graph: Any,
    memory_id: str,
    content: str,
    entities: Dict[str, List[str]],
    summary: Any,
    previous_summary: Any,
    enrichment_updates: Dict[str, Any],
    processed: bool,
    parse_metadata_field_fn: Callable[[Any], Any],
    normalize_tag_list_fn: Callable[[Any], List[str]],
    slugify_fn: Callable[[str], str],
    compute_tag_prefixes_fn: Callable[[List[str]], List[str]],
    enriched_at: str,
) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    """Merge derived fields into a fresh snapshot, committing only if it is current.

    Expensive extraction happens outside this helper. A concurrent tag/metadata edit
    rebases the derived additions; a content edit invalidates the extraction itself.
    """
    for _ in range(3):
        result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
        if not getattr(result, "result_set", None):
            return None
        node = result.result_set[0][0]
        current = dict(getattr(node, "properties", {}))
        if (current.get("content") or "") != content:
            raise EnrichmentConflictError(f"Memory {memory_id} changed during enrichment")

        metadata = deepcopy(parse_metadata_field_fn(current.get("metadata")) or {})
        if not isinstance(metadata, dict):
            metadata = {"_raw_metadata": metadata}
        entity_updates = {category: values for category, values in entities.items() if values}
        if entity_updates:
            entity_metadata = metadata.get("entities")
            if not isinstance(entity_metadata, dict):
                entity_metadata = {}
            entity_metadata.update(entity_updates)
            metadata["entities"] = entity_metadata
        enrichment_metadata = metadata.get("enrichment")
        if not isinstance(enrichment_metadata, dict):
            enrichment_metadata = {}
        enrichment_metadata.update(enrichment_updates)
        metadata["enrichment"] = enrichment_metadata

        source_tags = current.get("tags") or []
        tags = list(dict.fromkeys(normalize_tag_list_fn(source_tags)))
        for category, values in entity_updates.items():
            for value in values:
                slug = slugify_fn(value)
                tag = f"entity:{category}:{slug}"
                if slug and tag not in tags:
                    tags.append(tag)
        tag_prefixes = compute_tag_prefixes_fn(tags)
        saved_summary = (
            summary if current.get("summary") == previous_summary else current.get("summary")
        )
        update_payload = {
            "id": memory_id,
            "metadata": json.dumps(metadata, default=str),
            "tags": tags,
            "tag_prefixes": tag_prefixes,
            "summary": saved_summary,
            "enriched_at": enriched_at,
            "expected_content": content,
            "expected_tags": source_tags,
            "expected_metadata": current.get("metadata") or "",
            "expected_summary": current.get("summary") or "",
        }
        processed_clause = ", m.processed = true" if processed else ""
        result = graph.query(
            """
            MATCH (m:Memory {id: $id})
            WHERE coalesce(m.content, '') = $expected_content
              AND coalesce(m.tags, []) = $expected_tags
              AND coalesce(m.metadata, '') = $expected_metadata
              AND coalesce(m.summary, '') = $expected_summary
            SET m.metadata = $metadata,
                m.tags = $tags,
                m.tag_prefixes = $tag_prefixes,
                m.summary = $summary,
                m.enriched = true,
                m.enriched_at = $enriched_at
            """
            + processed_clause
            + " RETURN m",
            update_payload,
        )
        if getattr(result, "result_set", None):
            saved_node = result.result_set[0][0]
            updated = dict(getattr(saved_node, "properties", current))
            updated.update(
                metadata=metadata,
                tags=tags,
                tag_prefixes=tag_prefixes,
                summary=saved_summary,
                enriched=True,
                enriched_at=enriched_at,
            )
            if processed:
                updated["processed"] = True
            return updated, source_tags
    raise EnrichmentConflictError(f"Memory {memory_id} kept changing during enrichment")


def _sync_enrichment_payload(
    *,
    client: Any,
    collection_name: str,
    memory_id: str,
    source_tags: List[str],
    updated: Dict[str, Any],
    entities: Dict[str, List[str]],
    enrichment_updates: Dict[str, Any],
    logger: Any,
) -> None:
    """Update only derived metadata; atomically guard whole-array tag replacement.

    PATCH may commit after the graph update and before this sync. Nested payload
    updates preserve caller metadata keys, and a tag filter prevents restoring an
    old tag list over the newer PATCH payload.
    """
    if client is None:
        return
    from qdrant_client import models

    try:
        if updated["tags"] != source_tags:
            tag_conditions = [
                models.FieldCondition(key="tags", match=models.MatchValue(value=tag))
                for tag in set(source_tags)
            ]
            tag_conditions.append(
                models.FieldCondition(
                    key="tags",
                    values_count=models.ValuesCount(gte=len(source_tags), lte=len(source_tags)),
                )
            )
            client.set_payload(
                collection_name=collection_name,
                points=models.Filter(
                    must=[models.HasIdCondition(has_id=[memory_id]), *tag_conditions]
                ),
                payload={"tags": updated["tags"], "tag_prefixes": updated["tag_prefixes"]},
            )
        entity_updates = {category: values for category, values in entities.items() if values}
        if entity_updates:
            client.set_payload(
                collection_name=collection_name,
                points=[memory_id],
                key="metadata.entities",
                payload=entity_updates,
            )
        client.set_payload(
            collection_name=collection_name,
            points=[memory_id],
            key="metadata.enrichment",
            payload=enrichment_updates,
        )
    except Exception:
        logger.exception("Failed to sync Qdrant enrichment fields for %s", memory_id)


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

    entities = _validated_entities(extract_entities_fn(content), content)

    summary = (
        generate_summary_fn(content, properties.get("summary"))
        if enrichment_enable_summaries
        else properties.get("summary")
    )
    enriched_at = utc_now_fn()
    enrichment_updates = {"jit": True, "jit_at": enriched_at}
    try:
        persisted = _persist_enrichment(
            graph=graph,
            memory_id=memory_id,
            content=content,
            entities=entities,
            summary=summary,
            previous_summary=properties.get("summary"),
            enrichment_updates=enrichment_updates,
            processed=False,
            parse_metadata_field_fn=parse_metadata_field_fn,
            normalize_tag_list_fn=normalize_tag_list_fn,
            slugify_fn=slugify_fn,
            compute_tag_prefixes_fn=compute_tag_prefixes_fn,
            enriched_at=enriched_at,
        )
    except EnrichmentConflictError:
        logger.debug("JIT enrichment skipped after concurrent edit of %s", memory_id)
        return None
    except Exception:
        logger.exception("JIT enrichment graph update failed for %s", memory_id)
        return None
    if persisted is None:
        return None
    updated, source_tags = persisted
    _sync_enrichment_payload(
        client=get_qdrant_client_fn(),
        collection_name=collection_name,
        memory_id=memory_id,
        source_tags=source_tags,
        updated=updated,
        entities=entities,
        enrichment_updates=enrichment_updates,
        logger=logger,
    )
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

    if properties.get("processed") and not forced:
        return False
    content = properties.get("content", "") or ""
    entities = _validated_entities(extract_entities_fn(content), content)

    temporal_links = find_temporal_relationships_fn(graph, memory_id)
    pattern_info = detect_patterns_fn(graph, memory_id, content)
    semantic_neighbors = link_semantic_neighbors_fn(graph, memory_id)

    if enrichment_enable_summaries:
        existing_summary = properties.get("summary")
        summary = generate_summary_fn(content, None if forced else existing_summary)
    else:
        summary = properties.get("summary")

    enriched_at = utc_now_fn()
    enrichment_updates = {
        "last_run": enriched_at,
        "forced": forced,
        "temporal_links": temporal_links,
        "patterns_detected": pattern_info,
        "semantic_neighbors": [
            {"id": neighbour_id, "score": score} for neighbour_id, score in semantic_neighbors
        ],
    }
    persisted = _persist_enrichment(
        graph=graph,
        memory_id=memory_id,
        content=content,
        entities=entities,
        summary=summary,
        previous_summary=properties.get("summary"),
        enrichment_updates=enrichment_updates,
        processed=True,
        parse_metadata_field_fn=parse_metadata_field_fn,
        normalize_tag_list_fn=normalize_tag_list_fn,
        slugify_fn=slugify_fn,
        compute_tag_prefixes_fn=compute_tag_prefixes_fn,
        enriched_at=enriched_at,
    )
    if persisted is None:
        return False
    updated, source_tags = persisted
    _sync_enrichment_payload(
        client=get_qdrant_client_fn(),
        collection_name=collection_name,
        memory_id=memory_id,
        source_tags=source_tags,
        updated=updated,
        entities=entities,
        enrichment_updates=enrichment_updates,
        logger=logger,
    )

    logger.debug(
        "Enriched memory %s (temporal=%s, patterns=%s, semantic=%s)",
        memory_id,
        temporal_links,
        pattern_info,
        len(semantic_neighbors),
    )

    return True
