from __future__ import annotations

import pytest

import app


def test_issue72_examples_are_not_hardcoded_in_validator_source() -> None:
    """Regression fixtures should exercise general rules, not private deny entries."""
    import inspect

    from automem.utils import entity_quality

    source = inspect.getsource(entity_quality)

    assert "_LOW_SIGNAL_SLUGS" not in source
    for example in ("advocacy", "completed", "involvement", "recommended"):
        assert f'"{example}"' not in source
        assert f"'{example}'" not in source


@pytest.mark.parametrize(
    "slug",
    [
        "completed",
        "advocacy",
        "involvement",
        "key-findings",
        "deployed-automem",
        "config-file-approach",
        "recommended",
        "word",
        "ud83d-udc4d",
    ],
)
@pytest.mark.parametrize("category", ["people", "organizations", "tools", "projects", "concepts"])
def test_issue72_low_quality_entities_are_rejected_across_categories(
    category: str,
    slug: str,
) -> None:
    from automem.utils.entity_quality import validate_entity_slug

    result = validate_entity_slug(category, slug)

    assert result.accepted is False
    assert result.category == category
    assert result.slug == slug
    assert result.reason


@pytest.mark.parametrize(
    ("category", "slug", "reason"),
    [
        ("concepts", "00-00-berlin", "duration_or_count_slug"),
        ("concepts", "400ms", "duration_or_count_slug"),
        ("concepts", "12k", "duration_or_count_slug"),
        ("concepts", "12209-berlin", "duration_or_count_slug"),
        ("concepts", "7bd06aa-ed36b98e", "generated_fragment_slug"),
        ("tools", "ud83c-udd95-starting", "unicode_escape_slug"),
        ("tools", "ud83d-udea7-active-projects", "unicode_escape_slug"),
        ("tools", "venv-bin-python-m", "markdown_or_code_fragment"),
        ("tools", "tmp-settings", "markdown_or_code_fragment"),
        ("tools", "system-settings-wallpaper", "markdown_or_code_fragment"),
        ("tools", "twitter-x-https-x-com-example", "generated_phrase_slug"),
        ("tools", "terms-and-conditions-negotiation", "generated_phrase_slug"),
        ("tools", "sep-22-2025", "generated_phrase_slug"),
        ("concepts", "the-plan", "generic_entity_slug"),
        ("projects", "add-longmemeval", "generic_entity_slug"),
    ],
)
def test_structural_noise_slugs_are_rejected(
    category: str,
    slug: str,
    reason: str,
) -> None:
    from automem.utils.entity_quality import validate_entity_slug

    result = validate_entity_slug(category, slug)

    assert result.accepted is False
    assert result.reason == reason


@pytest.mark.parametrize(
    ("category", "slug"),
    [
        ("organizations", "time"),
        ("organizations", "tags"),
        ("tools", "system"),
        ("tools", "workflow"),
        ("concepts", "before-after"),
        ("people", "docker-compose"),
        ("people", "complete-deliverable"),
    ],
)
def test_generic_and_tooling_noise_is_rejected(category: str, slug: str) -> None:
    from automem.utils.entity_quality import validate_entity_slug

    result = validate_entity_slug(category, slug)

    assert result.accepted is False
    assert result.reason in {
        "generic_entity_slug",
        "low_signal_slug",
        "low_signal_people_slug",
        "markdown_or_code_fragment",
        "non_name_people_slug",
        "tool_or_organization_looking_people",
    }


@pytest.mark.parametrize("slug", ["alex-beck-s", "alex-beck-a"])
def test_possessive_and_suffix_variants_canonicalize_to_base_slug(slug: str) -> None:
    from automem.utils.entity_quality import validate_entity_slug

    result = validate_entity_slug("people", slug)

    assert result.accepted is True
    assert result.canonical_slug == "alex-beck"
    assert result.canonical_tag == "entity:people:alex-beck"
    assert result.confidence >= 0.8


@pytest.mark.parametrize(
    "slug",
    [
        "alex-beck-extra",
        "alex-beck-extra-name",
        "recreated-claude-code",
        "sora-2",
        "config-file",
        "phase-five",
    ],
)
def test_people_slugs_must_have_person_name_shape(slug: str) -> None:
    from automem.utils.entity_quality import validate_entity_slug

    result = validate_entity_slug("people", slug)

    assert result.accepted is False
    assert result.reason in {
        "low_signal_people_slug",
        "markdown_or_code_fragment",
        "non_name_people_slug",
        "tool_or_organization_looking_people",
    }


def test_tool_like_name_is_not_emitted_as_people() -> None:
    from automem.utils.entity_quality import validate_entity_value

    context = "Met with MetricForge about B2B SaaS pipeline automation."
    people_result = validate_entity_value("people", "MetricForge", context=context)
    tool_result = validate_entity_value("tools", "MetricForge", context=context)

    assert people_result.accepted is False
    assert people_result.reason == "tool_or_organization_looking_people"
    assert tool_result.accepted is True


def test_extract_entities_avoids_tool_brand_people_pollution() -> None:
    entities = app.extract_entities("Met with MetricForge about B2B SaaS pipeline automation.")

    assert "MetricForge" not in entities["people"]
    assert "MetricForge" in entities["tools"] or "MetricForge" in entities["organizations"]


@pytest.mark.parametrize(
    ("category", "slug"),
    [
        ("tools", "vectorstorex"),
        ("tools", "graphdbx"),
        ("tools", "containerkit"),
        ("tools", "forgehub"),
        ("tools", "testrunner"),
        ("organizations", "northstarops"),
        ("organizations", "orbitlabs"),
        ("concepts", "recallgraph"),
        ("concepts", "episodicindex"),
    ],
)
def test_single_token_specific_entities_do_not_require_a_curated_allowlist(
    category: str,
    slug: str,
) -> None:
    from automem.utils.entity_quality import validate_entity_slug

    result = validate_entity_slug(category, slug)

    assert result.accepted is True
    assert result.canonical_slug == slug
