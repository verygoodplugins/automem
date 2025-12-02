from __future__ import annotations

from typing import Any, Callable, List, Dict, Optional, Tuple, Set
from flask import Blueprint, request, abort, jsonify
from pathlib import Path
import json
import time
import re

from automem.config import ALLOWED_RELATIONS, RECALL_RELATION_LIMIT, RECALL_EXPANSION_LIMIT
from automem.utils.graph import _serialize_node

DEFAULT_STYLE_PRIORITY_TAGS: Set[str] = {
    "coding-style",
    "style",
    "style-guide",
    "preferences",
    "preference:coding",
}

CONTEXT_STYLE_KEYWORDS: Set[str] = {
    "style",
    "styles",
    "guideline",
    "guidelines",
    "pep8",
    "formatting",
    "lint",
    "preference",
    "preferences",
}

LANGUAGE_ALIASES: Dict[str, Set[str]] = {
    "python": {"python", "py"},
    "typescript": {"typescript", "ts", "tsx"},
    "javascript": {"javascript", "js", "jsx"},
    "go": {"go", "golang"},
    "rust": {"rust"},
    "java": {"java"},
    "csharp": {"csharp", "c#", "cs"},
    "cpp": {"c++", "cpp", "cc", "cxx"},
    "swift": {"swift"},
}

EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".swift": "swift",
}

# Words to skip when extracting entities from queries
ENTITY_STOPWORDS: Set[str] = {
    'What', 'Would', 'Could', 'Does', 'Did', 'How', 'Why', 'When', 'Where',
    'Which', 'Who', 'Whose', 'Will', 'Can', 'Should', 'Has', 'Have', 'Had',
    'Is', 'Are', 'Was', 'Were', 'Do', 'Been', 'Being', 'The', 'Answer',
    'Yes', 'No', 'Likely', 'Based', 'According', 'Since', 'Because',
    'January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December',
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
    'National', 'American', 'European', 'Asian', 'African',
}


def _extract_query_entities(query: str) -> List[str]:
    """
    Extract named entities (people, places) from a query.
    
    This enables entity-aware recall for questions like:
    "Would Caroline pursue writing?" -> extracts "Caroline"
    
    The recall can then run additional searches for entity-specific memories.
    """
    if not query:
        return []
    
    words = query.split()
    entities = []
    
    for i, word in enumerate(words):
        # Clean punctuation
        clean_word = re.sub(r'[^\w]', '', word)
        
        if len(clean_word) < 2:
            continue
        if clean_word in ENTITY_STOPWORDS:
            continue
        # Skip possessives (handle separately)
        if "'s" in word or "'s" in word:
            continue
        
        # Check for capitalized word (potential name)
        if len(clean_word) > 1 and clean_word[0].isupper() and clean_word[1:].islower():
            # Skip if first word (sentence start)
            if i == 0:
                continue
            # Skip if after sentence boundary
            if i > 0 and words[i-1][-1] in '.?!':
                continue
            entities.append(clean_word)
    
    # Handle possessives like "John's" or "Caroline's"
    possessives = re.findall(r"\b([A-Z][a-z]+)'s\b", query)
    for p in possessives:
        if p not in ENTITY_STOPWORDS and p not in entities:
            entities.append(p)
    
    return list(set(entities))


def _extract_topic_keywords(query: str, exclude_entities: Optional[List[str]] = None) -> List[str]:
    """
    Extract meaningful topic keywords from a query.
    
    For "Would Caroline pursue writing as a career?" extracts:
    ["writing", "career"]
    
    These can be combined with entities for broader searches.
    """
    if not query:
        return []
    
    # Build exclusion set (lowercase entity names to skip)
    exclude_lower = set()
    if exclude_entities:
        exclude_lower = {e.lower() for e in exclude_entities}
    
    # Common question/filler words to skip
    skip_words = {
        'would', 'could', 'should', 'will', 'can', 'may', 'might',
        'does', 'did', 'has', 'have', 'had', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'the', 'a', 'an', 'to', 'for', 'of', 'in',
        'on', 'at', 'by', 'with', 'about', 'as', 'if', 'or', 'and', 'but',
        'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
        'he', 'she', 'his', 'her', 'him', 'likely', 'probably', 'possibly',
        'considered', 'pursue', 'want', 'like', 'prefer', 'interested',
        'still', 'ever', 'more', 'most', 'some', 'any', 'all', 'only',
    }
    
    # Extract words, lowercase, filter
    words = re.findall(r'\b[a-z]{4,}\b', query.lower())
    topics = [w for w in words if w not in skip_words and w not in exclude_lower]
    
    # Return unique topics, preserving order
    seen = set()
    result = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            result.append(t)
    
    return result[:5]  # Limit to 5 topics


def _fingerprint_content(content: str) -> Optional[str]:
    """Lightweight content fingerprint for deduping near-identical memories."""
    if not content:
        return None
    cleaned = (
        re.sub(r"[`*_#>~\-]", " ", str(content).lower())
        .encode("ascii", "ignore")
        .decode("ascii", "ignore")
    )
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    return cleaned[:320]


def _dedupe_results(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Dedupe by memory ID or content fingerprint, keeping highest-score/newest."""
    buckets: Dict[str, Dict[str, Any]] = {}
    removed = 0

    def _mem_id(res: Dict[str, Any]) -> Optional[str]:
        mem = res.get("memory") or {}
        for key in ("id", "memory_id", "uuid"):
            if mem.get(key):
                return str(mem[key])
        if res.get("id"):
            return str(res["id"])
        return None

    # Track fingerprints separately to catch same-content with different IDs
    fp_to_key: Dict[str, str] = {}

    for res in results:
        mem = res.get("memory") or {}
        mid = _mem_id(res)
        fp = _fingerprint_content(mem.get("content") or "")
        key = f"id:{mid}" if mid else (f"fp:{fp}" if fp else None)
        if key is None:
            buckets[f"raw:{len(buckets)}"] = {"item": res, "sources": []}
            continue

        # Check for existing by ID first, then by fingerprint
        existing_key = None
        if key in buckets:
            existing_key = key
        elif fp and fp in fp_to_key:
            # Same content but different ID - merge into existing
            existing_key = fp_to_key[fp]
        
        if existing_key:
            existing = buckets[existing_key]["item"]
            removed += 1

            def _score(item: Dict[str, Any]) -> float:
                return float(item.get("final_score") or item.get("score") or 0.0)

            def _ts(item: Dict[str, Any]) -> str:
                return str((item.get("memory") or {}).get("timestamp") or "")

            choose_new = _score(res) > _score(existing) or (
                _score(res) == _score(existing) and _ts(res) > _ts(existing)
            )
            if choose_new:
                buckets[existing_key]["item"] = res
            buckets[existing_key]["sources"].append(mid or fp or "unknown")
        else:
            buckets[key] = {"item": res, "sources": [mid or fp or "unknown"]}
            # Track this fingerprint -> key mapping
            if fp:
                fp_to_key[fp] = key

    deduped: List[Dict[str, Any]] = []
    for entry in buckets.values():
        item = entry["item"]
        if len(entry["sources"]) > 1:
            item["deduped_from"] = sorted(set(entry["sources"]))
        deduped.append(item)

    return deduped, removed


def _split_multi_value(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(item) for item in raw if item is not None]
    else:
        return []
    normalized: List[str] = []
    for entry in values:
        text = str(entry).strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if parts:
            normalized.extend(parts)
        else:
            normalized.append(text)
    return normalized


def _parse_bool_param(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _tokenize_lower(text: str) -> Set[str]:
    if not text:
        return set()
    return {token for token in re.findall(r"[a-z0-9_\-\+#\.]+", text.lower()) if token}


def _detect_language_hint(
    explicit_language: Optional[str],
    context_label: str,
    query_text: str,
    active_path: str,
) -> Optional[str]:
    def _normalize_candidate(candidate: Optional[str]) -> Optional[str]:
        if not candidate:
            return None
        lowered = candidate.strip().lower()
        return lowered or None

    normalized_explicit = _normalize_candidate(explicit_language)
    if normalized_explicit:
        for lang, aliases in LANGUAGE_ALIASES.items():
            if normalized_explicit == lang or normalized_explicit in aliases:
                return lang

    normalized_context = _normalize_candidate(context_label)
    if normalized_context:
        for lang, aliases in LANGUAGE_ALIASES.items():
            if normalized_context == lang or normalized_context in aliases:
                return lang

    tokens = _tokenize_lower(query_text)
    suffix = ""
    if active_path:
        try:
            suffix = Path(active_path).suffix.lower()
        except Exception:
            suffix = ""
    if suffix and suffix in EXTENSION_LANGUAGE_MAP:
        return EXTENSION_LANGUAGE_MAP[suffix]

    for lang, aliases in LANGUAGE_ALIASES.items():
        if tokens & aliases:
            return lang
    return None


def _build_context_profile(
    manual_tags: List[str],
    manual_types: List[str],
    manual_ids: List[str],
    language_hint: Optional[str],
    context_label: str,
    query_text: str,
) -> Optional[Dict[str, Any]]:
    priority_tags: Set[str] = {tag.strip().lower() for tag in manual_tags if tag and tag.strip()}
    priority_types: Set[str] = {typ.strip().title() for typ in manual_types if typ and typ.strip()}
    priority_ids: Set[str] = {value.strip() for value in manual_ids if value and value.strip()}
    priority_keywords: Set[str] = set()

    style_focus = False
    if language_hint:
        priority_tags.update(
            {
                language_hint,
                f"{language_hint}-style",
                f"{language_hint}:style",
            }
        )
        priority_keywords.add(language_hint)
        style_focus = True

    context_tokens = _tokenize_lower(context_label)
    if context_tokens & CONTEXT_STYLE_KEYWORDS:
        style_focus = True

    query_tokens = _tokenize_lower(query_text)
    if query_tokens & CONTEXT_STYLE_KEYWORDS:
        style_focus = True

    if style_focus:
        priority_tags.update(DEFAULT_STYLE_PRIORITY_TAGS)
        priority_types.update({"Style", "Preference"})
    elif context_label:
        priority_tags.add(context_label.strip().lower())

    if not (priority_tags or priority_types or priority_ids or priority_keywords):
        return None

    return {
        "priority_tags": priority_tags,
        "priority_types": priority_types,
        "priority_ids": priority_ids,
        "priority_keywords": priority_keywords,
        "weights": {
            "tag": 0.45,
            "type": 0.25,
            "keyword": 0.2,
            "anchor": 0.9,
        },
        "language": language_hint,
        "context_label": context_label,
        "require_injection": style_focus or bool(manual_tags),
    }


def _result_matches_context_priority(result: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    memory = result.get("memory", {}) or {}
    priority_ids: Set[str] = profile.get("priority_ids", set())
    priority_tags: Set[str] = profile.get("priority_tags", set())
    priority_types: Set[str] = profile.get("priority_types", set())

    memory_id = str(result.get("id") or memory.get("id") or "")
    if priority_ids and memory_id and memory_id in priority_ids:
        return True

    if priority_tags:
        tags = {
            str(tag).strip().lower()
            for tag in (memory.get("tags") or [])
            if isinstance(tag, str) and tag.strip()
        }
        for tag in tags:
            for priority_tag in priority_tags:
                if tag == priority_tag or tag.startswith(priority_tag) or priority_tag in tag:
                    return True

    if priority_types:
        mem_type = memory.get("type")
        if isinstance(mem_type, str) and mem_type.strip().title() in priority_types:
            return True

    return False


def _inject_priority_memories(
    results: List[Dict[str, Any]],
    graph: Any,
    qdrant_client: Any,
    graph_keyword_search: Callable[..., List[Dict[str, Any]]],
    vector_filter_only_tag_search: Callable[..., List[Dict[str, Any]]],
    context_profile: Dict[str, Any],
    seen_ids: Set[str],
    result_passes_filters: Callable[[Dict[str, Any], Optional[str], Optional[str], Optional[List[str]], str, str], bool],
    start_time: Optional[str],
    end_time: Optional[str],
    tag_mode: str,
    tag_match: str,
    limit: int,
) -> bool:
    priority_tags: Set[str] = context_profile.get("priority_tags") or set()
    if not priority_tags:
        return False

    effective_tag_mode = tag_mode if tag_mode in {"any", "all"} else "any"
    effective_tag_match = tag_match if tag_match in {"exact", "prefix"} else "prefix"
    fetch_limit = max(1, min(limit, 3))
    tag_list = list(priority_tags)

    if graph is not None:
        priority_matches = graph_keyword_search(
            graph,
            "",
            fetch_limit,
            seen_ids,
            start_time=start_time,
            end_time=end_time,
            tag_filters=tag_list,
            tag_mode=effective_tag_mode,
            tag_match=effective_tag_match,
        )
        if priority_matches:
            results.extend(priority_matches)
            return True

    if qdrant_client is not None:
        tag_results = vector_filter_only_tag_search(
            qdrant_client,
            tag_list,
            effective_tag_mode,
            effective_tag_match,
            fetch_limit,
            seen_ids,
        )
        if tag_results:
            filtered_results = [
                record
                for record in tag_results
                if result_passes_filters(
                    record,
                    start_time,
                    end_time,
                    tag_list,
                    effective_tag_mode,
                    effective_tag_match,
                )
            ]
            if filtered_results:
                results.extend(filtered_results)
                return True

    return False


def _results_have_priority(results: List[Dict[str, Any]], profile: Dict[str, Any]) -> bool:
    for result in results:
        if _result_matches_context_priority(result, profile):
            return True
    return False


def _extract_entities_from_results(results: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract entity names (people, places) from seed result metadata.
    
    Returns entity names that can be used to search for related memories.
    Entity tags are created by enrichment as: entity:{category}:{slug}
    """
    entities: Set[str] = set()
    
    for result in results:
        memory = result.get("memory") or result
        
        # Check metadata.entities (populated by enrichment)
        metadata = memory.get("metadata")
        if isinstance(metadata, dict):
            ent_section = metadata.get("entities")
            if isinstance(ent_section, dict):
                # Extract people and places - most useful for multi-hop
                for category in ("people", "places", "organizations"):
                    values = ent_section.get(category, [])
                    if isinstance(values, list):
                        for v in values:
                            if isinstance(v, str) and len(v) > 1:
                                entities.add(v.lower().strip())
        
        # Also check tags for entity: prefixed tags
        tags = memory.get("tags") or []
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("entity:people:"):
                    # entity:people:amanda -> amanda
                    name = tag.split(":")[-1].replace("-", " ").strip()
                    if name:
                        entities.add(name)
    
    return entities


def _expand_entity_memories(
    seed_results: List[Dict[str, Any]],
    seen_ids: Set[str],
    vector_filter_only_tag_search: Callable[..., List[Dict[str, Any]]],
    qdrant_client: Any,
    compute_metadata_score: Callable[[Dict[str, Any], str, List[str], Optional[Dict[str, Any]]], tuple[float, Dict[str, float]]],
    query_text: str,
    query_tokens: List[str],
    context_profile: Optional[Dict[str, Any]],
    limit_per_entity: int,
    total_limit: int,
    logger: Any,
) -> List[Dict[str, Any]]:
    """
    Expand results by finding memories that mention entities found in seed results.
    
    This enables multi-hop reasoning:
    1. Query: "What is Amanda's sister's career?"
    2. Seed result: "Amanda's sister is Rachel"
    3. Entity expansion: Find memories tagged with entity:people:rachel
    4. Returns: "Rachel works as a counselor"
    """
    if qdrant_client is None:
        return []
    
    entities = _extract_entities_from_results(seed_results)
    if not entities:
        return []
    
    logger.debug("Entity expansion: found entities %s", entities)
    
    expanded: Dict[str, Dict[str, Any]] = {}
    total_added = 0
    
    for entity in list(entities)[:5]:  # Limit to 5 entities
        if total_added >= total_limit:
            break
        
        # Search for entity tag (prefix match)
        entity_slug = entity.lower().replace(" ", "-")
        entity_tags = [f"entity:people:{entity_slug}"]
        
        try:
            entity_results = vector_filter_only_tag_search(
                qdrant_client,
                entity_tags,
                "any",  # tag_mode
                "prefix",  # tag_match
                limit_per_entity,
                seen_ids,
            )
        except Exception:
            logger.exception("Entity expansion search failed for %s", entity)
            continue
        
        for result in entity_results:
            result_id = str(result.get("id") or (result.get("memory") or {}).get("id") or "")
            if not result_id or result_id in seen_ids:
                continue
            
            # Mark as entity-expanded result
            result["match_type"] = "entity_expansion"
            result["expanded_from_entity"] = entity
            
            # Score the result
            final_score, components = compute_metadata_score(
                result,
                query_text or "",
                query_tokens,
                context_profile,
            )
            components["entity_expansion"] = 0.15  # Small boost for entity matches
            result.setdefault("score_components", {}).update(components)
            result["final_score"] = final_score + 0.15
            result["score"] = result["final_score"]
            
            if result_id not in expanded:
                expanded[result_id] = result
                seen_ids.add(result_id)
                total_added += 1
                
                if total_added >= total_limit:
                    break
    
    expanded_list = list(expanded.values())
    expanded_list.sort(key=lambda r: -float(r.get("final_score", 0.0)))
    
    logger.debug("Entity expansion: added %d memories from %d entities", len(expanded_list), len(entities))
    
    return expanded_list


def _expand_related_memories(
    graph: Any,
    seed_results: List[Dict[str, Any]],
    seen_ids: Set[str],
    result_passes_filters: Callable[[Dict[str, Any], Optional[str], Optional[str], Optional[List[str]], str, str], bool],
    compute_metadata_score: Callable[[Dict[str, Any], str, List[str], Optional[Dict[str, Any]]], tuple[float, Dict[str, float]]],
    query_text: str,
    query_tokens: List[str],
    context_profile: Optional[Dict[str, Any]],
    start_time: Optional[str],
    end_time: Optional[str],
    tag_filters: Optional[List[str]],
    tag_mode: str,
    tag_match: str,
    per_seed_limit: int,
    expansion_limit: int,
    allowed_relations: Set[str],
    logger: Any,
    seed_score_boost: float = 0.25,
) -> List[Dict[str, Any]]:
    if graph is None or not seed_results or expansion_limit <= 0:
        return []

    relation_types = sorted({rel.upper() for rel in allowed_relations}) if allowed_relations else []
    per_seed_limit = max(1, per_seed_limit)
    expansion_limit = max(1, expansion_limit)

    expansions: Dict[str, Dict[str, Any]] = {}
    total_added = 0

    for seed_rank, seed in enumerate(seed_results):
        if total_added >= expansion_limit:
            break

        memory = seed.get("memory") or {}
        seed_id = str(seed.get("id") or memory.get("id") or memory.get("memory_id") or "").strip()
        if not seed_id:
            continue

        seed_score = float(seed.get("final_score", seed.get("score", 0.0)) or 0.0)

        try:
            rel_filter = " AND type(r) IN $types" if relation_types else ""
            query = f"""
                MATCH (m:Memory {{id: $id}})-[r]-(related:Memory)
                WHERE m.id <> related.id{rel_filter}
                RETURN type(r) as relation_type, r.strength as strength, related
                ORDER BY coalesce(r.updated_at, related.timestamp) DESC
                LIMIT $limit
            """
            params: Dict[str, Any] = {"id": seed_id, "limit": per_seed_limit}
            if relation_types:
                params["types"] = relation_types
            records = graph.query(query, params)
        except Exception:
            logger.exception("Failed to expand relations for seed %s", seed_id)
            continue

        for relation_type, relation_strength, related in getattr(records, "result_set", []) or []:
            if total_added >= expansion_limit:
                break

            data = _serialize_node(related)
            related_id = str(data.get("id") or "")
            if not related_id or related_id in seen_ids:
                continue

            candidate = {"id": related_id, "memory": data}
            if not result_passes_filters(candidate, start_time, end_time, tag_filters, tag_mode, tag_match):
                continue

            relation_strength_val = 0.0
            try:
                relation_strength_val = float(relation_strength) if relation_strength is not None else 0.0
            except (TypeError, ValueError):  # pragma: no cover - defensive
                relation_strength_val = 0.0

            relation_score = relation_strength_val + max(seed_score, 0.0) * seed_score_boost

            edge_info = {
                "type": relation_type,
                "strength": relation_strength_val,
                "from": seed_id,
                "seed_rank": seed_rank,
                "seed_score": seed_score,
            }

            if related_id in expansions:
                existing = expansions[related_id]
                existing["relation_score"] = max(existing.get("relation_score", 0.0), relation_score)
                existing.setdefault("relations", []).append(edge_info)
                existing.setdefault("related_to", []).append(edge_info)
            else:
                record: Dict[str, Any] = {
                    "id": related_id,
                    "match_type": "relation",
                    "source": "graph",
                    "memory": data,
                    "relations": [edge_info],
                    "related_to": [edge_info],
                    "relation_score": relation_score,
                    "match_score": relation_score,
                }
                expansions[related_id] = record
                seen_ids.add(related_id)
                total_added += 1

    expanded_list = list(expansions.values())

    for result in expanded_list:
        final_score, components = compute_metadata_score(
            result,
            query_text or "",
            query_tokens,
            context_profile,
        )
        components["relation"] = result.get("relation_score", 0.0)
        result.setdefault("score_components", {}).update(components)
        result["final_score"] = final_score
        result["score"] = final_score

    expanded_list.sort(key=lambda r: -float(r.get("final_score", 0.0)))
    return expanded_list


def handle_recall(
    get_memory_graph: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    normalize_tag_list: Callable[[Any], List[str]],
    normalize_timestamp: Callable[[str], str],
    parse_time_expression: Callable[[Optional[str]], Tuple[Optional[str], Optional[str]]],
    extract_keywords: Callable[[str], List[str]],
    compute_metadata_score: Callable[[Dict[str, Any], str, List[str], Optional[Dict[str, Any]]], tuple[float, Dict[str, float]]],
    result_passes_filters: Callable[[Dict[str, Any], Optional[str], Optional[str], Optional[List[str]], str, str], bool],
    graph_keyword_search: Callable[..., List[Dict[str, Any]]],
    vector_search: Callable[..., List[Dict[str, Any]]],
    vector_filter_only_tag_search: Callable[..., List[Dict[str, Any]]],
    recall_max_limit: int,
    logger: Any,
    allowed_relations: Optional[Set[str]] = None,
    relation_limit: Optional[int] = None,
    expansion_limit_default: Optional[int] = None,
):
    query_start = time.perf_counter()
    query_text = (request.args.get("query") or "").strip()
    multi_queries = _split_multi_value(
        request.args.getlist("queries") or request.args.get("queries")
    )
    try:
        requested_limit = int(request.args.get("limit", 5))
    except (TypeError, ValueError):
        requested_limit = 5
    limit = max(1, min(requested_limit, recall_max_limit))

    embedding_param = request.args.get("embedding")
    time_query = request.args.get("time_query") or request.args.get("time")
    start_param = request.args.get("start")
    end_param = request.args.get("end")
    tags_param = request.args.getlist("tags") or request.args.get("tags")

    tag_mode = (request.args.get("tag_mode") or "any").strip().lower()
    if tag_mode not in {"any", "all"}:
        tag_mode = "any"

    tag_match = (request.args.get("tag_match") or "prefix").strip().lower()
    if tag_match not in {"exact", "prefix"}:
        tag_match = "prefix"

    time_start, time_end = parse_time_expression(time_query)
    start_time = time_start
    end_time = time_end

    if start_param:
        try:
            start_time = normalize_timestamp(start_param)
        except ValueError as exc:
            abort(400, description=f"Invalid start time: {exc}")
    if end_param:
        try:
            end_time = normalize_timestamp(end_param)
        except ValueError as exc:
            abort(400, description=f"Invalid end time: {exc}")

    tag_filters = normalize_tag_list(tags_param)

    allowed_rel_set: Set[str] = set(allowed_relations) if allowed_relations else set(ALLOWED_RELATIONS)
    relation_limit = relation_limit or RECALL_RELATION_LIMIT
    expansion_limit = expansion_limit_default or RECALL_EXPANSION_LIMIT

    expand_relations = _parse_bool_param(
        request.args.get("expand_relations")
        or request.args.get("expand_associations")
        or request.args.get("expand"),
        False,
    )
    
    # Entity expansion for multi-hop reasoning
    expand_entities = _parse_bool_param(
        request.args.get("expand_entities")
        or request.args.get("entity_expansion"),
        False,
    )

    try:
        relation_limit = max(1, min(int(request.args.get("relation_limit", relation_limit)), 200))
    except (TypeError, ValueError):
        relation_limit = max(1, relation_limit)

    try:
        expansion_limit_param = request.args.get("expansion_limit") or request.args.get("relation_expansion_limit")
        if expansion_limit_param is not None:
            expansion_limit = int(expansion_limit_param)
        expansion_limit = max(1, min(expansion_limit, 500))
    except (TypeError, ValueError):
        expansion_limit = max(1, expansion_limit)

    context_label = (request.args.get("context") or "").strip().lower()
    active_path = (
        request.args.get("active_path")
        or request.args.get("file_path")
        or request.args.get("focus_path")
        or ""
    )
    language_hint_param = (request.args.get("language") or request.args.get("lang") or "").strip()

    context_tags_input = _split_multi_value(
        request.args.getlist("context_tags") or request.args.get("context_tags")
    )
    context_types_input = _split_multi_value(
        request.args.getlist("context_types") or request.args.get("context_types")
    )
    priority_ids_input = _split_multi_value(
        request.args.getlist("priority_ids") or request.args.get("priority_ids")
    )

    context_tags = normalize_tag_list(context_tags_input)
    graph = get_memory_graph()
    qdrant_client = get_qdrant_client()

    def _run_single_query(query_str: str, per_query_limit: int) -> Tuple[List[Dict[str, Any]], bool, Optional[Dict[str, Any]], int]:
        """Run recall for one query string; returns (results, context_injected, context_profile, vector_match_count)."""
        local_seen: set[str] = set()
        language_hint = _detect_language_hint(language_hint_param, context_label, query_str, active_path)
        context_profile = _build_context_profile(
            manual_tags=context_tags,
            manual_types=context_types_input,
            manual_ids=priority_ids_input,
            language_hint=language_hint,
            context_label=context_label,
            query_text=query_str or "",
        )

        local_results: List[Dict[str, Any]] = []
        vector_matches: List[Dict[str, Any]] = []

        if qdrant_client is not None:
            vector_matches = vector_search(
                qdrant_client,
                graph,
                query_str,
                embedding_param,
                per_query_limit,
                local_seen,
                tag_filters,
                tag_mode,
                tag_match,
            )
            if start_time or end_time or tag_filters:
                vector_matches = [
                    res
                    for res in vector_matches
                    if result_passes_filters(res, start_time, end_time, tag_filters, tag_mode, tag_match)
                ]
        local_results.extend(vector_matches[:per_query_limit])

        remaining_slots = max(0, per_query_limit - len(local_results))
        if remaining_slots and graph is not None:
            graph_matches = graph_keyword_search(
                graph,
                query_str,
                remaining_slots,
                local_seen,
                start_time=start_time,
                end_time=end_time,
                tag_filters=tag_filters,
                tag_mode=tag_mode,
                tag_match=tag_match,
            )
            local_results.extend(graph_matches[:remaining_slots])

        tags_only_request = (
            not query_str
            and not (embedding_param and embedding_param.strip())
            and bool(tag_filters)
        )
        if tags_only_request and qdrant_client is not None and len(local_results) < per_query_limit:
            tag_only_results = vector_filter_only_tag_search(
                qdrant_client,
                tag_filters,
                tag_mode,
                tag_match,
                per_query_limit - len(local_results),
                local_seen,
            )
            local_results.extend(tag_only_results)

        context_injected = False
        if context_profile:
            if not _results_have_priority(local_results, context_profile):
                context_injected = _inject_priority_memories(
                    local_results,
                    graph,
                    qdrant_client,
                    graph_keyword_search,
                    vector_filter_only_tag_search,
                    context_profile,
                    local_seen,
                    result_passes_filters,
                    start_time,
                    end_time,
                    tag_mode,
                    tag_match,
                    per_query_limit,
                )

        query_tokens = extract_keywords(query_str.lower()) if query_str else []
        for result in local_results:
            final_score, components = compute_metadata_score(
                result,
                query_str or "",
                query_tokens,
                context_profile,
            )
            result.setdefault("score_components", components)
            result["score_components"].update(components)
            result["final_score"] = final_score
            result["original_score"] = result.get("score", 0.0)
            result["score"] = final_score

        local_results = [
            res
            for res in local_results
            if result_passes_filters(res, start_time, end_time, tag_filters, tag_mode, tag_match)
        ]

        local_results.sort(
            key=lambda r: (
                -float(r.get("final_score", 0.0)),
                r.get("source") != "qdrant",
                -float(r.get("original_score", 0.0)),
                -float((r.get("memory") or {}).get("importance", 0.0) or 0.0),
            )
        )
        if len(local_results) > per_query_limit:
            local_results = local_results[:per_query_limit]

        # Track originating query for debugging/clients
        for res in local_results:
            res["_query"] = query_str

        return local_results, context_injected, context_profile, len(vector_matches)

    # Auto-decompose: extract entities from query and generate supplementary queries
    auto_decompose = _parse_bool_param(request.args.get("auto_decompose"), False)
    decomposed_queries: List[str] = []
    
    if auto_decompose and query_text and not multi_queries:
        entities = _extract_query_entities(query_text)
        topics = _extract_topic_keywords(query_text, exclude_entities=entities)
        
        # Generate entity+topic focused queries for multi-hop reasoning
        # Strategy: Search both entity+specific_topic AND entity+broad_topic
        # E.g., for "Would Caroline pursue writing?" with entities=["Caroline"], topics=["writing", "career"]
        # Generate: ["Caroline", "Caroline career", "Caroline interests", "Caroline writing"]
        for entity in entities[:2]:  # Limit to 2 entities
            # First: entity alone with implicit career/interests context
            decomposed_queries.append(entity)
            
            # Second: entity + each topic for specific focus
            for topic in topics[:3]:
                decomposed_queries.append(f"{entity} {topic}")
            
            # Third: entity + broad related terms to catch alternative answers
            # E.g., "Caroline career writing" might not find "counseling" but "Caroline interests" might
            if "career" in topics or "job" in topics or "work" in topics:
                decomposed_queries.append(f"{entity} interests goals plans")
        
        # Also add topic-only queries for broader coverage
        if topics and not entities:
            for topic in topics[:3]:
                decomposed_queries.append(topic)
    
    is_multi = bool(multi_queries) or bool(decomposed_queries)
    queries_to_run: List[str] = [q for q in multi_queries if q]
    
    # Add decomposed queries if auto_decompose is enabled
    if decomposed_queries:
        # Original query first, then decomposed variations
        queries_to_run = [query_text] + decomposed_queries
    elif not queries_to_run and query_text:
        queries_to_run = [query_text]
    if not queries_to_run:
        queries_to_run = [query_text] if query_text else [""]

    per_query_limit = limit
    try:
        per_query_limit = max(1, min(int(request.args.get("per_query_limit", per_query_limit)), recall_max_limit))
    except (TypeError, ValueError):
        per_query_limit = limit

    aggregated_results: List[Dict[str, Any]] = []
    any_context_profile: Optional[Dict[str, Any]] = None
    any_context_injected = False
    total_vector_matches = 0

    for idx, q in enumerate(queries_to_run):
        single_results, injected, context_profile, vector_count = _run_single_query(q, per_query_limit)
        aggregated_results.extend(single_results)
        total_vector_matches += vector_count
        if any_context_profile is None:
            any_context_profile = context_profile
        any_context_injected = any_context_injected or injected

    deduped_results, dedup_removed = _dedupe_results(aggregated_results)
    deduped_results.sort(
        key=lambda r: (
            -float(r.get("final_score", 0.0)),
            r.get("source") != "qdrant",
            -float(r.get("original_score", 0.0)),
            -float((r.get("memory") or {}).get("importance", 0.0) or 0.0),
        )
    )
    if len(deduped_results) > limit:
        deduped_results = deduped_results[:limit]

    # Graph expansion feature (from upstream branch)
    seed_results = list(deduped_results)
    expansion_results: List[Dict[str, Any]] = []
    entity_expansion_results: List[Dict[str, Any]] = []
    results = deduped_results

    # Track seen IDs for both expansion types
    seen_ids = {str(r.get("id") or (r.get("memory") or {}).get("id") or "") for r in seed_results if r.get("id") or (r.get("memory") or {}).get("id")}
    query_tokens = extract_keywords(query_text.lower()) if query_text else []

    if expand_relations and graph is not None:
        expansion_results = _expand_related_memories(
            graph=graph,
            seed_results=seed_results,
            seen_ids=seen_ids,
            result_passes_filters=result_passes_filters,
            compute_metadata_score=compute_metadata_score,
            query_text=query_text,
            query_tokens=query_tokens,
            context_profile=any_context_profile,
            start_time=start_time,
            end_time=end_time,
            tag_filters=tag_filters,
            tag_mode=tag_mode,
            tag_match=tag_match,
            per_seed_limit=relation_limit,
            expansion_limit=expansion_limit,
            allowed_relations=allowed_rel_set,
            logger=logger,
        )
        results = seed_results + expansion_results

    # Entity-based expansion for multi-hop reasoning
    if expand_entities and qdrant_client is not None:
        entity_expansion_results = _expand_entity_memories(
            seed_results=seed_results + expansion_results,  # Include relation-expanded results too
            seen_ids=seen_ids,
            vector_filter_only_tag_search=vector_filter_only_tag_search,
            qdrant_client=qdrant_client,
            compute_metadata_score=compute_metadata_score,
            query_text=query_text,
            query_tokens=query_tokens,
            context_profile=any_context_profile,
            limit_per_entity=5,
            total_limit=expansion_limit,
            logger=logger,
        )
        results = seed_results + expansion_results + entity_expansion_results

    response = {
        "status": "success",
        "query": query_text,
        "results": results,
        "count": len(results),
        "dedup_removed": dedup_removed,
        "vector_search": {
            "enabled": qdrant_client is not None,
            "matched": bool(total_vector_matches),
        },
    }
    if expand_relations:
        response["expansion"] = {
            "enabled": True,
            "seed_count": len(seed_results),
            "expanded_count": len(expansion_results),
            "relation_limit": relation_limit,
            "expansion_limit": expansion_limit,
        }
    if expand_entities:
        response["entity_expansion"] = {
            "enabled": True,
            "expanded_count": len(entity_expansion_results),
            "entities_found": list(_extract_entities_from_results(seed_results + expansion_results))[:10],
        }
    if is_multi:
        response["queries"] = queries_to_run
    if query_text and not is_multi:
        response["keywords"] = extract_keywords(query_text.lower()) if query_text else []
    if start_time or end_time:
        response["time_window"] = {"start": start_time, "end": end_time}
    if tag_filters:
        response["tags"] = tag_filters
    response["tag_mode"] = tag_mode
    response["tag_match"] = tag_match
    response["query_time_ms"] = round((time.perf_counter() - query_start) * 1000, 2)
    if any_context_profile:
        response["context_priority"] = {
            "language": any_context_profile.get("language"),
            "context": any_context_profile.get("context_label"),
            "priority_tags": sorted(any_context_profile.get("priority_tags") or [])[:10],
            "priority_types": sorted(any_context_profile.get("priority_types") or []),
            "injected": any_context_injected,
        }

    logger.info(
        "recall_complete",
        extra={
            "query": query_text[:100] if query_text else "",
            "results": len(deduped_results),
            "latency_ms": response["query_time_ms"],
            "vector_enabled": qdrant_client is not None,
            "vector_matches": total_vector_matches,
            "has_time_filter": bool(start_time or end_time),
            "has_tag_filter": bool(tag_filters),
            "limit": limit,
            "dedup_removed": dedup_removed,
            "is_multi": is_multi,
            "context_language": (any_context_profile or {}).get("language") if any_context_profile else None,
        },
    )
    return jsonify(response)


def create_recall_blueprint(
    get_memory_graph: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    normalize_tag_list: Callable[[Any], List[str]],
    normalize_timestamp: Callable[[str], str],
    parse_time_expression: Callable[[Optional[str]], Tuple[Optional[str], Optional[str]]],
    extract_keywords: Callable[[str], List[str]],
    compute_metadata_score: Callable[[Dict[str, Any], str, List[str], Optional[Dict[str, Any]]], tuple[float, Dict[str, float]]],
    result_passes_filters: Callable[[Dict[str, Any], Optional[str], Optional[str], Optional[List[str]], str, str], bool],
    graph_keyword_search: Callable[..., List[Dict[str, Any]]],
    vector_search: Callable[..., List[Dict[str, Any]]],
    vector_filter_only_tag_search: Callable[..., List[Dict[str, Any]]],
    recall_max_limit: int,
    logger: Any,
    allowed_relations: List[str] | set[str] | tuple[str, ...] | Any = (),
    relation_limit: int = 5,
    serialize_node: Callable[[Any], Dict[str, Any]] | None = None,
    summarize_relation_node: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
) -> Blueprint:
    bp = Blueprint("recall", __name__)

    @bp.route("/recall", methods=["GET"])
    def recall_memories() -> Any:
        return handle_recall(
            get_memory_graph,
            get_qdrant_client,
            normalize_tag_list,
            normalize_timestamp,
            parse_time_expression,
            extract_keywords,
            compute_metadata_score,
            result_passes_filters,
            graph_keyword_search,
            vector_search,
            vector_filter_only_tag_search,
            recall_max_limit,
            logger,
            allowed_relations=allowed_relations if allowed_relations else set(ALLOWED_RELATIONS),
            relation_limit=relation_limit,
            expansion_limit_default=RECALL_EXPANSION_LIMIT,
        )

    @bp.route("/startup-recall", methods=["GET"])
    def startup_recall() -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        try:
            lesson_query = """
                MATCH (m:Memory)
                WHERE 'critical' IN m.tags OR 'lesson' IN m.tags OR 'ai-assistant' IN m.tags
                RETURN m.id as id, m.content as content, m.tags as tags,
                       m.importance as importance, m.type as type, m.metadata as metadata
                ORDER BY m.importance DESC
                LIMIT 10
            """

            lesson_results = graph.query(lesson_query)
            lessons = []
            if getattr(lesson_results, "result_set", None):
                for row in lesson_results.result_set:
                    lessons.append({
                        'id': row[0],
                        'content': row[1],
                        'tags': row[2] if row[2] else [],
                        'importance': row[3] if row[3] else 0.5,
                        'type': row[4] if row[4] else 'Context',
                        'metadata': json.loads(row[5]) if row[5] else {}
                    })

            system_query = """
                MATCH (m:Memory)
                WHERE 'system' IN m.tags OR 'memory-recall' IN m.tags
                RETURN m.id as id, m.content as content, m.tags as tags
                LIMIT 5
            """

            system_results = graph.query(system_query)
            system_rules = []
            if getattr(system_results, "result_set", None):
                for row in system_results.result_set:
                    system_rules.append({
                        'id': row[0],
                        'content': row[1],
                        'tags': row[2] if row[2] else []
                    })

            response = {
                'status': 'success',
                'critical_lessons': lessons,
                'system_rules': system_rules,
                'lesson_count': len(lessons),
                'has_critical': any(l.get('importance', 0) >= 0.9 for l in lessons),
                'summary': f"Recalled {len(lessons)} lesson(s) and {len(system_rules)} system rule(s)"
            }
            return jsonify(response), 200
        except Exception as e:
            logger.error(f"Startup recall failed: {e}")
            return jsonify({
                "error": "Startup recall failed",
                "details": str(e)
            }), 500

    @bp.route("/analyze", methods=["GET"])
    def analyze_memories() -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")
        analytics = {
            "memory_types": {},
            "patterns": [],
            "preferences": [],
            "temporal_insights": {},
            "entity_frequency": {},
            "confidence_distribution": {},
        }
        try:
            type_result = graph.query(
                """
                MATCH (m:Memory)
                WHERE m.type IS NOT NULL
                RETURN m.type, COUNT(m) as count, AVG(m.confidence) as avg_confidence
                ORDER BY count DESC
                """
            )
            for mem_type, count, avg_conf in getattr(type_result, "result_set", []) or []:
                analytics["memory_types"][mem_type] = {
                    "count": count,
                    "average_confidence": round(avg_conf, 3) if avg_conf else 0,
                }

            pattern_result = graph.query(
                """
                MATCH (p:Pattern)
                WHERE p.confidence > 0.6
                RETURN p.type, p.content, p.confidence, p.observations
                ORDER BY p.confidence DESC
                LIMIT 10
                """
            )
            for p_type, content, confidence, observations in getattr(pattern_result, "result_set", []) or []:
                analytics["patterns"].append({
                    "type": p_type,
                    "description": content,
                    "confidence": round(confidence, 3) if confidence else 0,
                    "observations": observations or 0,
                })

            pref_result = graph.query(
                """
                MATCH (m1:Memory)-[r:PREFERS_OVER]->(m2:Memory)
                RETURN m1.content, m2.content, r.context, r.strength
                ORDER BY r.strength DESC
                LIMIT 10
                """
            )
            for preferred, over, context, strength in getattr(pref_result, "result_set", []) or []:
                analytics["preferences"].append({
                    "prefers": preferred,
                    "over": over,
                    "context": context,
                    "strength": round(strength, 3) if strength else 0,
                })

            try:
                temporal_result = graph.query(
                    """
                    MATCH (m:Memory)
                    WHERE m.timestamp IS NOT NULL
                    RETURN m.timestamp, m.importance
                    LIMIT 100
                    """
                )
                from collections import defaultdict
                hour_data = defaultdict(lambda: {"count": 0, "total_importance": 0})
                for timestamp, importance in getattr(temporal_result, "result_set", []) or []:
                    if timestamp and len(timestamp) > 13:
                        hour_str = timestamp[11:13]
                        if hour_str.isdigit():
                            hour = int(hour_str)
                            hour_data[hour]["count"] += 1
                            hour_data[hour]["total_importance"] += importance or 0.5
                for hour, data in hour_data.items():
                    if data["count"] > 0:
                        analytics["temporal_insights"][f"hour_{hour:02d}"] = {
                            "count": data["count"],
                            "avg_importance": round(data["total_importance"] / data["count"], 3)
                        }
            except Exception:
                pass

            entity_result = graph.query(
                """
                MATCH (m:Memory)
                WHERE m.metadata IS NOT NULL
                RETURN m.metadata
                LIMIT 200
                """
            )
            from collections import Counter
            entities = Counter()
            for (metadata_json,) in getattr(entity_result, "result_set", []) or []:
                try:
                    metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else (metadata_json or {})
                except Exception:
                    metadata = {}
                if not isinstance(metadata, dict):
                    # Skip unsupported metadata shapes (e.g., SimpleNamespace in dummy graphs)
                    continue
                for key in ("entities", "keywords", "topics"):
                    for item in (metadata.get(key) or []):
                        val = str(item).strip().lower()
                        if len(val) >= 3:
                            entities[val] += 1
            analytics["entity_frequency"] = dict(entities.most_common(50))

            conf_result = graph.query(
                """
                MATCH (m:Memory)
                WHERE m.confidence IS NOT NULL
                RETURN m.confidence
                LIMIT 500
                """
            )
            conf_dist: Dict[str, int] = {"low": 0, "medium": 0, "high": 0}
            for (conf,) in getattr(conf_result, "result_set", []) or []:
                try:
                    c = float(conf or 0)
                except Exception:
                    c = 0.0
                if c < 0.4:
                    conf_dist["low"] += 1
                elif c < 0.7:
                    conf_dist["medium"] += 1
                else:
                    conf_dist["high"] += 1
            analytics["confidence_distribution"] = conf_dist
            return jsonify({"status": "success", "analytics": analytics, "elapsed_ms": 0}), 200
        except Exception as e:
            logger.error(f"Analyze failed: {e}")
            return jsonify({"error": "Analyze failed", "details": str(e)}), 500

    @bp.route("/memories/<memory_id>/related", methods=["GET"])
    def get_related_memories(memory_id: str) -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        allowed = set(allowed_relations) if allowed_relations else set()

        rel_types_param = (request.args.get("relationship_types") or "").strip()
        if rel_types_param:
            requested = [part.strip().upper() for part in rel_types_param.split(",") if part.strip()]
            rel_types = [t for t in requested if (t in allowed if allowed else True)]
            if not rel_types and allowed:
                rel_types = sorted(allowed)
        else:
            rel_types = sorted(allowed) if allowed else []

        try:
            max_depth = int(request.args.get("max_depth", 1))
        except (TypeError, ValueError):
            max_depth = 1
        max_depth = max(1, min(max_depth, 3))

        try:
            limit = int(request.args.get("limit", relation_limit))
        except (TypeError, ValueError):
            limit = relation_limit
        limit = max(1, min(limit, 200))

        rel_pattern = ":" + "|".join(rel_types) if rel_types else ""
        query = f"""
            MATCH (m:Memory {{id: $id}}){'-[r' + rel_pattern + f']-' if rel_pattern else '-[r]-'}(related:Memory)
            WHERE m.id <> related.id
            CALL apoc.path.expandConfig(related, {{
                relationshipFilter: '{"|".join(rel_types)}',
                minLevel: 0,
                maxLevel: $max_depth,
                bfs: true,
                filterStartNode: true
            }}) YIELD path
            WITH DISTINCT related
            RETURN related
            ORDER BY coalesce(related.importance, 0.0) DESC, coalesce(related.timestamp, '') DESC
            LIMIT $limit
        """
        fallback_query = f"""
            MATCH (m:Memory {{id: $id}}){'-[r' + rel_pattern + f'*1..$max_depth]-' if rel_pattern else '-[r*1..$max_depth]-'}(related:Memory)
            WHERE m.id <> related.id
            RETURN DISTINCT related
            ORDER BY coalesce(related.importance, 0.0) DESC, coalesce(related.timestamp, '') DESC
            LIMIT $limit
        """
        params = {"id": memory_id, "max_depth": max_depth, "limit": limit}
        try:
            result = graph.query(query, params)
        except Exception:
            try:
                result = graph.query(fallback_query, params)
            except Exception:
                logger.exception("Failed to traverse related memories for %s", memory_id)
                abort(500, description="Failed to fetch related memories")

        related: List[Dict[str, Any]] = []
        for row in getattr(result, "result_set", []) or []:
            node = row[0]
            data = serialize_node(node) if serialize_node else {"value": node}
            if data.get("id") != memory_id:
                if summarize_relation_node:
                    # Present summarized node info
                    related.append(summarize_relation_node(data))
                else:
                    related.append(data)
        return jsonify({
            "status": "success",
            "memory_id": memory_id,
            "count": len(related),
            "related_memories": related,
            "relationship_types": rel_types,
            "max_depth": max_depth,
            "limit": limit,
        })

    return bp
