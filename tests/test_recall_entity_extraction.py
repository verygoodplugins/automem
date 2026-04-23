from automem.api.recall import _extract_query_entities


def test_extract_query_entities_handles_ascii_possessive_name():
    entities = _extract_query_entities("Would Caroline's sister pursue writing as a career?")
    assert "Caroline" in entities


def test_extract_query_entities_handles_curly_possessive_name():
    entities = _extract_query_entities("Would Caroline’s sister pursue writing as a career?")
    assert "Caroline" in entities
