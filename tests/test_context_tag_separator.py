"""Regression tests for context_tag separator normalization (mcp-automem #97 bug C).

`context_tags` boosting must treat ``project:foo`` and ``project/foo`` as the same
tag. The recall API canonicalizes hard tag filters via ``_expand_tag_prefixes``
(``[:/]`` -> ``:``), but ``_context_tag_hit`` historically did raw string matching,
so a ``context_tags=["project:foo"]`` boost never landed on a ``project/foo``-tagged
memory (and vice versa).
"""

from automem.utils.scoring import _context_tag_hit

# --- New behavior: cross-separator matches (the bug) ---------------------------


def test_context_tag_hit_matches_colon_priority_against_slash_tag() -> None:
    assert _context_tag_hit({"project/foo"}, {"project:foo"}) is True


def test_context_tag_hit_matches_slash_priority_against_colon_tag() -> None:
    assert _context_tag_hit({"project:foo"}, {"project/foo"}) is True


def test_context_tag_hit_prefix_match_across_separators() -> None:
    # priority "project:foo" should prefix-match a deeper "project/foo/bar" tag
    assert _context_tag_hit({"project/foo/bar"}, {"project:foo"}) is True


# --- Characterization: existing behavior must be preserved ---------------------


def test_context_tag_hit_exact_match_preserved() -> None:
    assert _context_tag_hit({"project:foo"}, {"project:foo"}) is True


def test_context_tag_hit_prefix_match_same_separator_preserved() -> None:
    assert _context_tag_hit({"project:foo:bar"}, {"project:foo"}) is True


def test_context_tag_hit_unrelated_does_not_match() -> None:
    assert _context_tag_hit({"unrelated"}, {"project"}) is False


def test_context_tag_hit_empty_sets_do_not_match() -> None:
    assert _context_tag_hit(set(), {"project:foo"}) is False
    assert _context_tag_hit({"project:foo"}, set()) is False
