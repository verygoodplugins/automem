"""Recall must not collapse distinct original memories on a shared prefix or summary."""

from automem.api.recall import _dedupe_results


def result(memory_id, content, original=None, score=1.0):
    memory = {"id": memory_id, "content": content}
    if original is not None:
        memory["metadata"] = {"original_content": original, "was_summarized": True}
    return {"id": memory_id, "memory": memory, "score": score}


def test_different_facts_after_shared_prefix_are_retained():
    prefix = "Routine discussion and planning. " * 20
    rows, removed = _dedupe_results(
        [
            result("a", prefix + "First unique fact is blue."),
            result("b", prefix + "Second unique fact is red."),
        ]
    )
    assert removed == 0
    assert {row["id"] for row in rows} == {"a", "b"}


def test_same_summary_with_distinct_originals_is_not_deduplicated():
    rows, removed = _dedupe_results(
        [
            result("a", "Planning summary", "Decision: launch on Monday."),
            result("b", "Planning summary", "Decision: launch on Friday."),
        ]
    )
    assert removed == 0
    assert {row["id"] for row in rows} == {"a", "b"}


def test_true_duplicates_and_same_id_are_still_merged():
    rows, removed = _dedupe_results(
        [
            result("a", "Identical detail", score=0.8),
            result("b", "Identical detail", score=0.9),
            result("b", "Other representation", score=0.7),
        ]
    )
    assert len(rows) == 1
    assert removed == 2
    assert rows[0]["id"] == "b"


def test_updated_content_with_old_original_is_not_a_duplicate():
    rows, removed = _dedupe_results(
        [
            result("a", "Changed to April", "Old decision was January"),
            result("b", "Changed to May", "Old decision was January"),
        ]
    )
    assert removed == 0
    assert {row["id"] for row in rows} == {"a", "b"}


def test_json_metadata_preserves_distinct_originals():
    rows = [result("a", "Same summary"), result("b", "Same summary")]
    rows[0]["memory"]["metadata"] = '{"original_content": "One"}'
    rows[1]["memory"]["metadata"] = '{"original_content": "Two"}'
    deduped, removed = _dedupe_results(rows)
    assert removed == 0
    assert len(deduped) == 2
