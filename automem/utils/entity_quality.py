from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_CATEGORY_ALIASES = {
    "person": "people",
    "people": "people",
    "org": "organizations",
    "organization": "organizations",
    "organizations": "organizations",
    "tool": "tools",
    "tools": "tools",
    "project": "projects",
    "projects": "projects",
    "concept": "concepts",
    "concepts": "concepts",
}

_ALLOWED_CATEGORIES = set(_CATEGORY_ALIASES.values())

_ARTICLE_PREFIXES = {"a", "an", "the"}

_GENERIC_ENTITY_SLUGS = {
    "avoid",
    "background",
    "before-after",
    "build",
    "clear",
    "complex",
    "connect",
    "content",
    "dry",
    "env",
    "for",
    "home",
    "memory",
    "metadata",
    "open",
    "plan",
    "post",
    "problem",
    "result",
    "results",
    "session",
    "source",
    "system",
    "tag",
    "tags",
    "task",
    "technical",
    "them",
    "these",
    "they",
    "test",
    "theme",
    "ticket",
    "time",
    "trigger",
    "trusted",
    "unit",
    "universal",
    "url",
    "urls",
    "video",
    "vision",
    "voice",
    "verify",
    "watch",
    "web",
    "week",
    "worker",
    "workflow",
    "workflows",
    "word",
}

_GENERIC_ENTITY_TOKENS = {
    *_GENERIC_ENTITY_SLUGS,
    "about",
    "after",
    "approach",
    "before",
    "deliverable",
    "finding",
    "findings",
    "key",
    "phase",
    "priority",
    "status",
    "track",
}

_ACTION_PREFIXES = {
    "add",
    "build",
    "clean",
    "cleaned",
    "complete",
    "create",
    "deploy",
    "deployed",
    "fix",
    "pull",
    "push",
    "reach",
    "recall",
    "reclaim",
    "remove",
    "reply",
    "retry",
    "run",
    "show",
    "start",
    "started",
    "sync",
    "update",
    "write",
}

_ACTION_STATUS_ROOTS = (
    "accept",
    "approve",
    "build",
    "clean",
    "complete",
    "create",
    "deploy",
    "finish",
    "identify",
    "pass",
    "recommend",
    "reject",
    "select",
    "start",
    "sync",
    "update",
)

_ABSTRACT_SINGLETON_SUFFIXES = ("acy", "ment", "ness")

_MARKDOWN_OR_CODE_TOKENS = {
    "bin",
    "code",
    "config",
    "env",
    "file",
    "json",
    "markdown",
    "md",
    "path",
    "python",
    "settings",
    "tmp",
    "users",
    "venv",
    "yaml",
    "yml",
}

_MARKDOWN_OR_CODE_SECONDARY_TOKENS = {
    "api",
    "bash",
    "cli",
    "css",
    "dockerfile",
    "html",
    "js",
    "m",
    "py",
    "sh",
    "ts",
    "tsx",
    "xml",
}

_NON_PERSON_TECH_TOKENS = {
    "api",
    "app",
    "bot",
    "cli",
    "cloud",
    "compose",
    "db",
    "docker",
    "hub",
    "model",
    "platform",
    "sdk",
    "service",
    "system",
    "tool",
    "tools",
}

_GENERATED_PHRASE_TOKENS = {
    "blog",
    "chronicle",
    "com",
    "comprehensive",
    "conditions",
    "decision",
    "draft",
    "execution",
    "goes",
    "https",
    "identified",
    "issue",
    "kickoff",
    "live",
    "negotiation",
    "passed",
    "quote",
    "round",
    "selected",
    "significance",
    "sprint",
    "terms",
    "wrap",
}

_MONTH_TOKENS = {
    "jan",
    "january",
    "feb",
    "february",
    "mar",
    "march",
    "apr",
    "april",
    "may",
    "jun",
    "june",
    "jul",
    "july",
    "aug",
    "august",
    "sep",
    "sept",
    "september",
    "oct",
    "october",
    "nov",
    "november",
    "dec",
    "december",
}

_PERSON_NAME_PARTICLES = {
    "da",
    "de",
    "del",
    "der",
    "di",
    "du",
    "la",
    "le",
    "st",
    "van",
    "von",
}

_TOOL_OR_ORG_SUFFIXES = (
    "ai",
    "api",
    "app",
    "bot",
    "cli",
    "cloud",
    "corp",
    "db",
    "hub",
    "labs",
    "math",
    "sdk",
)

_TOOL_OR_ORG_CONTEXT_HINTS = (
    "automation",
    "b2b",
    "business",
    "company",
    "data",
    "database",
    "model",
    "pipeline",
    "platform",
    "project",
    "saas",
    "service",
    "services",
    "software",
    "system",
    "tool",
    "tooling",
    "vendor",
)


@dataclass(frozen=True)
class EntityValidationResult:
    accepted: bool
    category: str
    slug: str
    canonical_slug: str
    reason: str
    confidence: float
    name: Optional[str] = None
    original_value: Optional[str] = None

    @property
    def canonical_tag(self) -> str:
        return f"entity:{self.category}:{self.canonical_slug}"


def slugify_entity(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (value or "").lower())
    return cleaned.strip("-")


def name_from_slug(slug: str) -> str:
    return slug.replace("-", " ").title()


def normalize_entity_category(category: str) -> str:
    return _CATEGORY_ALIASES.get((category or "").strip().lower(), (category or "").strip())


def _canonicalize_slug(category: str, slug: str) -> str:
    canonical = re.sub(r"-+", "-", (slug or "").strip().lower()).strip("-")
    if category == "people":
        parts = canonical.split("-")
        if len(parts) >= 3 and parts[-1] in {"s", "a"}:
            canonical = "-".join(parts[:-1])
    return canonical


def _tokens(slug: str) -> list[str]:
    return [token for token in slug.split("-") if token]


def _looks_like_unicode_escape_slug(slug: str) -> bool:
    parts = _tokens(slug)
    if not parts:
        return False
    return all(re.fullmatch(r"u?[0-9a-f]{4,6}", part) for part in parts)


def _contains_unicode_escape_token(slug: str) -> bool:
    return any(re.fullmatch(r"u[0-9a-f]{4,6}", part) for part in _tokens(slug))


def _looks_like_duration_or_count_slug(slug: str) -> bool:
    if re.fullmatch(r"\d+(ms|s|sec|secs|m|min|mins|h|hr|hrs)", slug):
        return True
    if re.fullmatch(r"\d+[a-z]?", slug):
        return True
    if re.fullmatch(r"\d{1,2}-\d{2}(-[a-z][a-z0-9]+)*", slug):
        return True
    if re.fullmatch(r"\d+(-[a-z][a-z0-9]*)+", slug):
        return True
    return False


def _looks_like_action_status_slug(slug: str) -> bool:
    parts = _tokens(slug)
    if len(parts) != 1:
        return False

    token = parts[0]
    for root in _ACTION_STATUS_ROOTS:
        variants = {root, f"{root}s", f"{root}ed", f"{root}ing"}
        if root.endswith("e"):
            variants.add(f"{root}d")
            variants.add(f"{root[:-1]}ing")
        if root.endswith("y"):
            variants.add(f"{root[:-1]}ied")
        if token in variants:
            return True
    return False


def _looks_like_abstract_singleton_slug(slug: str) -> bool:
    parts = _tokens(slug)
    if len(parts) != 1:
        return False
    token = parts[0]
    return len(token) >= 6 and token.endswith(_ABSTRACT_SINGLETON_SUFFIXES)


def _looks_like_generated_fragment_slug(slug: str) -> bool:
    parts = _tokens(slug)
    if any(re.fullmatch(r"[0-9a-f]{6,}", token) for token in parts):
        return True
    if any(re.fullmatch(r"[a-z]\d[a-z0-9]{5,}", token) for token in parts):
        return True
    return False


def _looks_like_markdown_or_code_fragment(tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return False
    if any(token in _MARKDOWN_OR_CODE_TOKENS for token in tokens):
        return True
    code_token_count = sum(
        1
        for token in tokens
        if token in _MARKDOWN_OR_CODE_SECONDARY_TOKENS
        or re.fullmatch(r"[a-z]+\d+", token)
    )
    return code_token_count >= 2


def _looks_like_generated_phrase_slug(category: str, tokens: list[str]) -> bool:
    if category not in {"organizations", "tools", "projects", "concepts"}:
        return False
    if len(tokens) >= 6:
        return True
    if any(token in _MONTH_TOKENS for token in tokens) and any(token.isdigit() for token in tokens):
        return True
    if len(tokens) >= 2 and tokens[0] in {"phase", "round", "sprint", "tier", "track"}:
        return True
    phrase_hits = sum(1 for token in tokens if token in _GENERATED_PHRASE_TOKENS)
    return phrase_hits >= 1 and len(tokens) >= 3


def _has_internal_camelcase(value: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", value or "")
    if not compact or " " in (value or "").strip():
        return False
    return bool(re.search(r"[a-z][A-Z]", compact))


def _looks_tool_or_org_like(value: str, slug: str, context: Optional[str]) -> bool:
    parts = _tokens(slug)
    if _has_internal_camelcase(value):
        return True
    if parts and any(parts[-1].endswith(suffix) for suffix in _TOOL_OR_ORG_SUFFIXES):
        return True

    lowered_context = (context or "").lower()
    if lowered_context and slug in lowered_context.replace(" ", "-"):
        return any(hint in lowered_context for hint in _TOOL_OR_ORG_CONTEXT_HINTS)
    return False


def _has_person_name_shape(tokens: list[str]) -> bool:
    if len(tokens) == 1:
        return True
    if len(tokens) == 2:
        return tokens[0] != tokens[1]
    if len(tokens) == 3 and (
        len(tokens[1]) == 1 or tokens[1] in _PERSON_NAME_PARTICLES
    ):
        return tokens[0] != tokens[-1]
    return False


def _reject(
    *,
    category: str,
    slug: str,
    canonical_slug: str,
    reason: str,
    name: Optional[str] = None,
    original_value: Optional[str] = None,
) -> EntityValidationResult:
    return EntityValidationResult(
        accepted=False,
        category=category,
        slug=slug,
        canonical_slug=canonical_slug,
        reason=reason,
        confidence=0.0,
        name=name,
        original_value=original_value,
    )


def _accepted(
    *,
    category: str,
    slug: str,
    canonical_slug: str,
    name: Optional[str],
    original_value: Optional[str],
) -> EntityValidationResult:
    parts = _tokens(canonical_slug)
    confidence = 0.95
    if category == "people" and len(parts) == 1:
        confidence = 0.6
    if canonical_slug != slug:
        confidence = max(confidence, 0.85)
    return EntityValidationResult(
        accepted=True,
        category=category,
        slug=slug,
        canonical_slug=canonical_slug,
        reason="accepted",
        confidence=confidence,
        name=name,
        original_value=original_value,
    )


def validate_entity_slug(
    category: str,
    slug: str,
    *,
    original_value: Optional[str] = None,
    context: Optional[str] = None,
) -> EntityValidationResult:
    normalized_category = normalize_entity_category(category)
    original_slug = slugify_entity(slug)
    canonical_slug = _canonicalize_slug(normalized_category, original_slug)
    display_name = (
        name_from_slug(canonical_slug)
        if not original_value or slugify_entity(original_value) != canonical_slug
        else original_value.strip()
    )

    def reject(reason: str) -> EntityValidationResult:
        return _reject(
            category=normalized_category,
            slug=original_slug,
            canonical_slug=canonical_slug,
            reason=reason,
            name=display_name,
            original_value=original_value,
        )

    if normalized_category not in _ALLOWED_CATEGORIES:
        return reject("unknown_category")

    if not canonical_slug or len(canonical_slug) < 3:
        return reject("too_short")

    if _looks_like_unicode_escape_slug(canonical_slug) or _contains_unicode_escape_token(
        canonical_slug
    ):
        return reject("unicode_escape_slug")

    tokens = _tokens(canonical_slug)

    if _looks_like_duration_or_count_slug(canonical_slug):
        return reject("duration_or_count_slug")

    if _looks_like_action_status_slug(canonical_slug):
        return reject(
            "low_signal_people_slug"
            if normalized_category == "people"
            else "generic_entity_slug"
        )

    if _looks_like_abstract_singleton_slug(canonical_slug):
        return reject(
            "low_signal_people_slug"
            if normalized_category == "people"
            else "generic_entity_slug"
        )

    if _looks_like_generated_fragment_slug(canonical_slug):
        return reject("generated_fragment_slug")

    if _looks_like_markdown_or_code_fragment(tokens):
        return reject("markdown_or_code_fragment")

    if _looks_like_generated_phrase_slug(normalized_category, tokens):
        return reject("generated_phrase_slug")

    if (
        canonical_slug in _GENERIC_ENTITY_SLUGS
        or (tokens and tokens[0] in _ARTICLE_PREFIXES)
        or (tokens and all(token in _GENERIC_ENTITY_TOKENS for token in tokens))
        or (
            normalized_category in {"organizations", "tools", "projects", "concepts"}
            and tokens
            and tokens[0] in _ACTION_PREFIXES
        )
    ):
        return reject(
            "low_signal_people_slug"
            if normalized_category == "people"
            else "generic_entity_slug"
        )

    if normalized_category == "people":
        if any(not re.fullmatch(r"[a-z]+", token) for token in tokens):
            return reject("non_name_people_slug")
        if any(
            token in _ACTION_PREFIXES
            or token in _GENERIC_ENTITY_TOKENS
            or token in _GENERATED_PHRASE_TOKENS
            or token in _MARKDOWN_OR_CODE_TOKENS
            or token in _MARKDOWN_OR_CODE_SECONDARY_TOKENS
            or token in _NON_PERSON_TECH_TOKENS
            for token in tokens
        ):
            return reject("low_signal_people_slug")
        if not _has_person_name_shape(tokens):
            return reject("non_name_people_slug")
        if _looks_tool_or_org_like(original_value or canonical_slug, canonical_slug, context):
            return reject("tool_or_organization_looking_people")

    return _accepted(
        category=normalized_category,
        slug=original_slug,
        canonical_slug=canonical_slug,
        name=display_name,
        original_value=original_value,
    )


def validate_entity_value(
    category: str,
    value: str,
    *,
    context: Optional[str] = None,
) -> EntityValidationResult:
    return validate_entity_slug(
        category,
        slugify_entity(value),
        original_value=(value or "").strip(),
        context=context,
    )


def validate_entity_tag(tag: str, *, context: Optional[str] = None) -> EntityValidationResult:
    parts = (tag or "").split(":", 2)
    if len(parts) != 3 or parts[0] != "entity":
        normalized = slugify_entity(tag)
        return _reject(
            category="",
            slug=normalized,
            canonical_slug=normalized,
            reason="invalid_entity_tag",
        )
    return validate_entity_slug(parts[1], parts[2], context=context)
