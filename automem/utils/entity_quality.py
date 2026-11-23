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
    "event": "events",
    "events": "events",
    "opportunity": "opportunities",
    "opportunities": "opportunities",
}

ENTITY_CATEGORIES = frozenset(_CATEGORY_ALIASES.values())

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

# "code" is only a weak signal: real tool names end in it (claude-code,
# vs-code, code-server), so it must not condemn a slug on its own.
_MARKDOWN_OR_CODE_SECONDARY_TOKENS = {
    "api",
    "bash",
    "cli",
    "code",
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

# Everyday non-name words that show up in generated people slugs
# (bottom-line, deck-today, email-highlights, claude-desktop). Any one of
# these tokens disqualifies a people slug; none are plausible name parts.
_NON_PERSON_COMMON_TOKENS = {
    "bottom",
    "chrome",
    "deck",
    "desktop",
    "email",
    "emails",
    "highlight",
    "highlights",
    "line",
    "plugin",
    "plugins",
    "today",
    "tomorrow",
    "yesterday",
}

_NON_PERSON_TECH_TOKENS = {
    "api",
    "app",
    "bot",
    "cli",
    "cloud",
    "compose",
    "data",
    "db",
    "docker",
    "hub",
    "model",
    "pipeline",
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
    canonical = (slug or "").strip().lower()
    if "--" in canonical:
        canonical = re.sub(r"-+", "-", canonical)
    canonical = canonical.strip("-")
    if category == "people":
        parts = canonical.split("-")
        if len(parts) >= 3 and parts[-1] in {"s", "a"}:
            canonical = "-".join(parts[:-1])
    return canonical


def _tokens(slug: str) -> list[str]:
    return [token for token in slug.split("-") if token]


def _looks_like_unicode_escape_slug(slug: str) -> bool:
    # One match over the whole slug checks that every token is an escape code.
    return bool(re.fullmatch(r"u?[0-9a-f]{4,6}(-u?[0-9a-f]{4,6})*", slug))


def _contains_unicode_escape_token(parts: list[str]) -> bool:
    return any(part.startswith("u") and re.fullmatch(r"u[0-9a-f]{4,6}", part) for part in parts)


def _looks_like_duration_or_count_slug(slug: str) -> bool:
    # Every pattern below starts with a digit.
    if not slug[:1].isdigit():
        return False
    if re.fullmatch(r"\d+(ms|s|sec|secs|m|min|mins|h|hr|hrs)", slug):
        return True
    if re.fullmatch(r"\d+[a-z]?", slug):
        return True
    if re.fullmatch(r"\d{1,2}-\d{2}(-[a-z][a-z0-9]+)*", slug):
        return True
    if re.fullmatch(r"\d+(-[a-z][a-z0-9]*)+", slug):
        return True
    return False


def _looks_like_action_status_slug(parts: list[str]) -> bool:
    if len(parts) != 1:
        return False

    token = parts[0]
    # Undo each inflection to find the candidate root, e.g. "approving" -> "approve".
    inflections = (
        ("", ""),
        ("s", ""),
        ("ed", ""),
        ("ed", "e"),
        ("ing", ""),
        ("ing", "e"),
        ("ied", "y"),
    )
    for suffix, root_ending in inflections:
        if token.endswith(suffix):
            stem = token[: len(token) - len(suffix)]
            if stem + root_ending in _ACTION_STATUS_ROOTS:
                return True
    return False


def _looks_like_abstract_singleton_slug(parts: list[str]) -> bool:
    if len(parts) != 1:
        return False
    token = parts[0]
    return len(token) >= 6 and token.endswith(_ABSTRACT_SINGLETON_SUFFIXES)


def _looks_like_generated_fragment_slug(parts: list[str]) -> bool:
    # Both patterns need at least six characters; the second needs a digit in second place.
    return any(
        len(token) >= 6
        and (
            re.fullmatch(r"[0-9a-f]{6,}", token)
            or (token[1].isdigit() and re.fullmatch(r"[a-z]\d[a-z0-9]{5,}", token))
        )
        for token in parts
    )


def _looks_like_markdown_or_code_fragment(tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return False
    if not _MARKDOWN_OR_CODE_TOKENS.isdisjoint(tokens):
        return True
    code_token_count = sum(
        1
        for token in tokens
        if token in _MARKDOWN_OR_CODE_SECONDARY_TOKENS
        or (token[-1:].isdigit() and re.fullmatch(r"[a-z]+\d+", token))
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
    # Camel case needs an uppercase letter.
    if (value or "").islower():
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", value or "")
    if not compact or " " in (value or "").strip():
        return False
    return bool(re.search(r"[a-z][A-Z]", compact))


def _looks_tool_or_org_like(
    value: str, slug: str, parts: list[str], context: Optional[str]
) -> bool:
    if _has_internal_camelcase(value):
        return True
    if parts and parts[-1].endswith(_TOOL_OR_ORG_SUFFIXES):
        return True

    # Context hints are too weak to condemn a multi-token person-shaped name:
    # in a technical corpus nearly every memory mentions data/projects/tools,
    # which would reject virtually every real person discussed at work. This
    # must hold for slug-only inputs too (validate_entity_tag has no display
    # value), or stored entity:people tags get re-rejected during repair.
    # Brand-like pairs such as data-dog are handled by the per-token
    # vocabulary checks in validate_entity_slug, not by context hints.
    if len(parts) >= 2 and _has_person_name_shape(parts):
        return False

    lowered_context = (context or "").lower()
    if lowered_context and slug in lowered_context.replace(" ", "-"):
        return any(hint in lowered_context for hint in _TOOL_OR_ORG_CONTEXT_HINTS)
    return False


def _has_person_name_shape(tokens: list[str]) -> bool:
    if len(tokens) == 1:
        return True
    if len(tokens) == 2:
        return tokens[0] != tokens[1]
    if len(tokens) == 3 and (len(tokens[1]) == 1 or tokens[1] in _PERSON_NAME_PARTICLES):
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
    # Stored tags are usually clean already, so skip re-slugifying them.
    if type(slug) is str and re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", slug):
        original_slug = slug
    else:
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

    if normalized_category not in ENTITY_CATEGORIES:
        return reject("unknown_category")

    if not canonical_slug or len(canonical_slug) < 3:
        return reject("too_short")

    # The canonical slug is already slugified, so it has no empty tokens.
    tokens = canonical_slug.split("-")

    if _looks_like_unicode_escape_slug(canonical_slug) or _contains_unicode_escape_token(tokens):
        return reject("unicode_escape_slug")

    if _looks_like_duration_or_count_slug(canonical_slug):
        return reject("duration_or_count_slug")

    if _looks_like_action_status_slug(tokens):
        return reject(
            "low_signal_people_slug" if normalized_category == "people" else "generic_entity_slug"
        )

    if _looks_like_abstract_singleton_slug(tokens):
        return reject(
            "low_signal_people_slug" if normalized_category == "people" else "generic_entity_slug"
        )

    if _looks_like_generated_fragment_slug(tokens):
        return reject("generated_fragment_slug")

    if _looks_like_markdown_or_code_fragment(tokens):
        return reject("markdown_or_code_fragment")

    if _looks_like_generated_phrase_slug(normalized_category, tokens):
        return reject("generated_phrase_slug")

    if (
        canonical_slug in _GENERIC_ENTITY_SLUGS
        or (tokens and tokens[0] in _ARTICLE_PREFIXES)
        or (tokens and _GENERIC_ENTITY_TOKENS.issuperset(tokens))
        or (
            normalized_category in {"organizations", "tools", "projects", "concepts"}
            and tokens
            and tokens[0] in _ACTION_PREFIXES
        )
    ):
        return reject(
            "low_signal_people_slug" if normalized_category == "people" else "generic_entity_slug"
        )

    if normalized_category == "people":
        # The canonical slug has no empty tokens, so one match covers every token.
        if not re.fullmatch(r"[a-z]+(-[a-z]+)*", canonical_slug):
            return reject("non_name_people_slug")
        for vocabulary in (
            _ACTION_PREFIXES,
            _GENERIC_ENTITY_TOKENS,
            _GENERATED_PHRASE_TOKENS,
            _MARKDOWN_OR_CODE_TOKENS,
            _MARKDOWN_OR_CODE_SECONDARY_TOKENS,
            _NON_PERSON_TECH_TOKENS,
            _NON_PERSON_COMMON_TOKENS,
        ):
            if not vocabulary.isdisjoint(tokens):
                return reject("low_signal_people_slug")
        if not _has_person_name_shape(tokens):
            return reject("non_name_people_slug")
        if _looks_tool_or_org_like(
            original_value or canonical_slug, canonical_slug, tokens, context
        ):
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
