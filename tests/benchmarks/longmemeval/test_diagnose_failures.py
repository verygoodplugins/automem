"""Unit tests for the LongMemEval failure-mode diagnosis harness.

All fixtures are synthetic (synthetic names only); no real benchmark
artifacts or network calls are used, mirroring test_analysis.py.
"""

from types import SimpleNamespace

from tests.benchmarks.judge_preflight import (
    EXIT_AUTH,
    EXIT_OK,
    EXIT_OTHER,
    EXIT_QUOTA,
    classify_judge_error,
    run_preflight,
)
from tests.benchmarks.longmemeval.diagnose_failures import (
    MODE_ANSWER_CONSTRUCTION,
    MODE_CONFLICT,
    MODE_MISSING_DATE,
    MODE_OTHER,
    MODE_OUTDATED,
    MODE_RANKING,
    MODE_RETRIEVAL_GAP,
    answer_rank,
    compute_evidence,
    date_arithmetic_needed,
    diagnose_stage1,
    format_summary,
    is_refusal,
    keyword_tokens,
    parse_session_date,
    run_stage2,
    select_failures,
    suggested_mode,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _dataset_item(**overrides):
    item = {
        "question_id": "q-synth-1",
        "question_type": "knowledge-update",
        "question": "What city does the user live in now?",
        "answer": "Lisbon",
        "question_date": "2023/04/01 (Sat) 10:00",
        "answer_session_ids": ["ans_1"],
        "haystack_session_ids": ["old_1", "ans_1", "noise_1"],
        "haystack_dates": [
            "2023/01/05 (Thu) 10:00",
            "2023/03/10 (Fri) 12:00",
            "2023/02/01 (Wed) 09:00",
        ],
        "haystack_sessions": [
            [{"role": "user", "content": "I live in Porto city right now."}],
            [{"role": "user", "content": "Update: I moved and now live in Lisbon city."}],
            [{"role": "user", "content": "My cat enjoys tuna snacks."}],
        ],
    }
    item.update(overrides)
    return item


def _result_row(**overrides):
    row = {
        "question_id": "q-synth-1",
        "question_type": "knowledge-update",
        "question": "What city does the user live in now?",
        "reference": "Lisbon",
        "hypothesis": "Porto",
        "is_correct": False,
        "is_abstention": False,
        "explanation": "Said Porto, reference is Lisbon.",
        "question_date": "2023/04/01 (Sat) 10:00",
        "answer_session_ids": ["ans_1"],
        "retrieved_session_ids": ["old_1", "ans_1", "noise_1"],
        "recall_hit_at_5": True,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Tokenization and date parsing
# ---------------------------------------------------------------------------


def test_keyword_tokens_drops_stopwords_and_lowercases():
    tokens = keyword_tokens("What city does the User LIVE in now?")
    assert "city" in tokens
    assert "live" in tokens
    assert "what" not in tokens
    assert "the" not in tokens
    assert "in" not in tokens


def test_parse_session_date_handles_longmemeval_format():
    parsed = parse_session_date("2023/05/20 (Sat) 02:21")
    assert parsed is not None
    assert (parsed.year, parsed.month, parsed.day) == (2023, 5, 20)
    assert (parsed.hour, parsed.minute) == (2, 21)


def test_parse_session_date_returns_none_on_garbage():
    assert parse_session_date("not a date") is None
    assert parse_session_date("") is None
    assert parse_session_date(None) is None


# ---------------------------------------------------------------------------
# Evidence primitives
# ---------------------------------------------------------------------------


def test_answer_rank_is_one_based_first_hit():
    assert answer_rank(["s1", "s2", "s3"], ["s2"]) == 2
    assert answer_rank(["s1", "s2", "s3"], ["s3", "s1"]) == 1
    assert answer_rank(["s1", "s2"], ["s9"]) is None
    assert answer_rank([], ["s9"]) is None


def test_is_refusal_detects_dont_know_responses():
    assert is_refusal("I don't know.") is True
    assert is_refusal("There is no information about that in my memory.") is True
    assert is_refusal("Lisbon") is False
    assert is_refusal("") is False


def test_date_arithmetic_needed_markers():
    assert date_arithmetic_needed("How long did the user wait?") is True
    assert date_arithmetic_needed("How many days passed between visits?") is True
    assert date_arithmetic_needed("What happened before the move?") is True
    assert date_arithmetic_needed("When was the first concert?") is True
    assert date_arithmetic_needed("What is the user's favorite color?") is False


# ---------------------------------------------------------------------------
# compute_evidence
# ---------------------------------------------------------------------------


def test_compute_evidence_stale_candidate_above_answer_true():
    # old_1 (non-answer, older, topically overlapping) ranked above ans_1.
    evidence = compute_evidence(_result_row(), _dataset_item())
    assert evidence["answer_rank"] == 2
    assert evidence["abstained_despite_hit"] is False
    assert evidence["stale_candidate_above_answer"] is True
    # noise_1 shares zero question keywords -> 1 of 3 retrieved is noise.
    assert abs(evidence["noise_ratio"] - (1 / 3)) < 1e-9
    # Not a temporal-reasoning question.
    assert evidence["date_arithmetic_needed"] is None
    assert evidence["answer_coverage_top5"] == 1.0


def test_compute_evidence_stale_candidate_false_when_answer_ranked_first():
    row = _result_row(retrieved_session_ids=["ans_1", "old_1", "noise_1"])
    evidence = compute_evidence(row, _dataset_item())
    assert evidence["answer_rank"] == 1
    assert evidence["stale_candidate_above_answer"] is False


def test_compute_evidence_stale_candidate_null_for_other_types():
    row = _result_row(question_type="multi-session")
    item = _dataset_item(question_type="multi-session")
    evidence = compute_evidence(row, item)
    assert evidence["stale_candidate_above_answer"] is None


def test_compute_evidence_abstention_flags():
    row = _result_row(hypothesis="I don't know.")
    evidence = compute_evidence(row, _dataset_item())
    assert evidence["abstained_despite_hit"] is True

    row = _result_row(is_abstention=True, hypothesis="Porto")
    evidence = compute_evidence(row, _dataset_item())
    assert evidence["abstained_despite_hit"] is True


def test_compute_evidence_temporal_question_sets_date_arithmetic():
    row = _result_row(
        question_type="temporal-reasoning",
        question="How many days after the move did the user visit Lisbon?",
    )
    item = _dataset_item(
        question_type="temporal-reasoning",
        question="How many days after the move did the user visit Lisbon?",
    )
    evidence = compute_evidence(row, item)
    assert evidence["date_arithmetic_needed"] is True


def test_compute_evidence_partial_answer_coverage():
    row = _result_row(
        question_type="multi-session",
        answer_session_ids=["ans_1", "missing_ans"],
        retrieved_session_ids=["old_1", "ans_1"],
    )
    item = _dataset_item(
        question_type="multi-session",
        answer_session_ids=["ans_1", "missing_ans"],
    )
    evidence = compute_evidence(row, item)
    assert evidence["answer_coverage_top5"] == 0.5


# ---------------------------------------------------------------------------
# suggested_mode heuristic
# ---------------------------------------------------------------------------


def _evidence(**overrides):
    evidence = {
        "answer_rank": 2,
        "abstained_despite_hit": False,
        "stale_candidate_above_answer": None,
        "noise_ratio": 0.0,
        "date_arithmetic_needed": None,
        "answer_coverage_top5": 1.0,
    }
    evidence.update(overrides)
    return evidence


def test_suggested_mode_abstention_wins_first():
    mode = suggested_mode("multi-session", _evidence(abstained_despite_hit=True))
    assert mode == MODE_ANSWER_CONSTRUCTION


def test_suggested_mode_stale_candidate_maps_to_outdated():
    mode = suggested_mode("knowledge-update", _evidence(stale_candidate_above_answer=True))
    assert mode == MODE_OUTDATED
    mode = suggested_mode("single-session-preference", _evidence(stale_candidate_above_answer=True))
    assert mode == MODE_OUTDATED


def test_suggested_mode_knowledge_update_without_stale_is_conflict():
    mode = suggested_mode("knowledge-update", _evidence(stale_candidate_above_answer=False))
    assert mode == MODE_CONFLICT


def test_suggested_mode_temporal_date_math_maps_to_missing_date_use():
    mode = suggested_mode("temporal-reasoning", _evidence(date_arithmetic_needed=True))
    assert mode == MODE_MISSING_DATE


def test_suggested_mode_multi_session_partial_coverage_is_retrieval_gap():
    mode = suggested_mode("multi-session", _evidence(answer_coverage_top5=0.5))
    assert mode == MODE_RETRIEVAL_GAP


def test_suggested_mode_answer_rank_one_is_answer_construction():
    mode = suggested_mode("single-session-user", _evidence(answer_rank=1))
    assert mode == MODE_ANSWER_CONSTRUCTION


def test_suggested_mode_buried_answer_is_ranking():
    mode = suggested_mode("single-session-user", _evidence(answer_rank=3))
    assert mode == MODE_RANKING


def test_suggested_mode_no_rank_falls_back_to_other():
    mode = suggested_mode("single-session-user", _evidence(answer_rank=None))
    assert mode == MODE_OTHER


# ---------------------------------------------------------------------------
# Selection filter
# ---------------------------------------------------------------------------


def test_select_failures_requires_incorrect_and_recall_hit():
    details = [
        _result_row(question_id="keep"),
        _result_row(question_id="drop-correct", is_correct=True),
        _result_row(question_id="drop-miss", recall_hit_at_5=False),
        _result_row(question_id="drop-unknown", recall_hit_at_5=None),
    ]
    selected = select_failures(details, types=None)
    assert [row["question_id"] for row in selected] == ["keep"]


def test_select_failures_type_filter():
    details = [
        _result_row(question_id="ku", question_type="knowledge-update"),
        _result_row(question_id="ms", question_type="multi-session"),
        _result_row(question_id="ssu", question_type="single-session-user"),
    ]
    selected = select_failures(details, types={"multi-session", "knowledge-update"})
    assert sorted(row["question_id"] for row in selected) == ["ku", "ms"]
    # Default (types=None) keeps every failed-but-retrieved question type.
    assert len(select_failures(details, types=None)) == 3


# ---------------------------------------------------------------------------
# Stage 1 end-to-end on synthetic data
# ---------------------------------------------------------------------------


def test_diagnose_stage1_end_to_end():
    results = {"details": [_result_row(), _result_row(question_id="ok", is_correct=True)]}
    dataset = [_dataset_item(), _dataset_item(question_id="ok")]
    report = diagnose_stage1(results, dataset, types=None)

    assert report["metadata"]["selected"] == 1
    assert report["metadata"]["selected_by_type"] == {"knowledge-update": 1}
    record = report["questions"][0]
    assert record["question_id"] == "q-synth-1"
    assert record["evidence"]["stale_candidate_above_answer"] is True
    assert record["suggested_mode"] == MODE_OUTDATED
    assert report["summary"]["mode_counts"][MODE_OUTDATED] == 1
    assert report["summary"]["mode_counts_by_type"]["knowledge-update"][MODE_OUTDATED] == 1

    text = format_summary(report)
    assert "knowledge-update" in text
    assert MODE_OUTDATED in text


def test_diagnose_stage1_missing_dataset_entry_keeps_rank_evidence():
    results = {"details": [_result_row(question_id="orphan")]}
    report = diagnose_stage1(results, [], types=None)
    record = report["questions"][0]
    assert record["evidence"]["dataset_missing"] is True
    # Dataset-dependent evidence is unavailable...
    assert record["evidence"]["stale_candidate_above_answer"] is None
    assert record["evidence"]["noise_ratio"] is None
    # ...but rank evidence comes from the results row alone (answer at rank 2).
    assert record["suggested_mode"] == MODE_RANKING


# ---------------------------------------------------------------------------
# Judge preflight
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, exc=None, content="OK"):
        self.exc = exc
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, exc=None, content="OK"):
        self.chat = SimpleNamespace(completions=_FakeCompletions(exc=exc, content=content))


def test_classify_judge_error_categories():
    assert classify_judge_error(Exception("Error code: 429 - insufficient_quota")) == "quota"
    assert classify_judge_error(Exception("Rate limit reached for gpt-5.4-mini")) == "quota"
    assert classify_judge_error(Exception("Error code: 401 - invalid_api_key")) == "auth"
    assert classify_judge_error(Exception("authentication failed")) == "auth"
    assert classify_judge_error(Exception("connection reset")) == "other"


def test_run_preflight_success_exits_zero():
    client = _FakeClient()
    code, message = run_preflight(model="gpt-5.4-mini-2026-03-17", client=client)
    assert code == EXIT_OK
    assert "gpt-5.4-mini-2026-03-17" in message
    # gpt-5 family must use max_completion_tokens, not max_tokens.
    call = client.chat.completions.calls[0]
    assert "max_completion_tokens" in call
    assert "max_tokens" not in call


def test_run_preflight_quota_failure():
    client = _FakeClient(exc=Exception("Error code: 429 - insufficient_quota"))
    code, message = run_preflight(model="gpt-5.4-mini-2026-03-17", client=client)
    assert code == EXIT_QUOTA
    assert "quota" in message.lower() or "429" in message


def test_run_preflight_auth_failure():
    client = _FakeClient(exc=Exception("Error code: 401 - invalid_api_key"))
    code, message = run_preflight(model="gpt-5.4-mini-2026-03-17", client=client)
    assert code == EXIT_AUTH
    assert "OPENAI_API_KEY" in message or "auth" in message.lower()


def test_run_preflight_other_failure():
    client = _FakeClient(exc=Exception("connection reset"))
    code, message = run_preflight(model="gpt-5.4-mini-2026-03-17", client=client)
    assert code == EXIT_OTHER
    assert message


# ---------------------------------------------------------------------------
# Stage 2 with a mocked client (no network)
# ---------------------------------------------------------------------------


def test_run_stage2_with_mocked_client_records_labels_and_agreement():
    results = {"details": [_result_row()]}
    dataset = [_dataset_item()]
    report = diagnose_stage1(results, dataset, types=None)
    assert report["questions"][0]["suggested_mode"] == MODE_OUTDATED

    client = _FakeClient(content='```json\n{"mode": "ranking", "rationale": "buried"}\n```')
    run_stage2(report, dataset, model="gpt-5.4-mini-2026-03-17", client=client)

    record = report["questions"][0]
    assert record["llm_mode"] == MODE_RANKING
    assert record["llm_rationale"] == "buried"
    assert record["llm_error"] is None
    agreement = report["agreement"]
    assert agreement["matrix"] == {MODE_OUTDATED: {MODE_RANKING: 1}}
    assert agreement["exact_agreement"] == 0.0
    assert agreement["llm_errors"] == 0
    assert report["metadata"]["llm_stage"] is True
    # Retrieved session content must appear in the prompt sent to the LLM.
    prompt = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "Lisbon" in prompt
    assert "ANSWER SESSION" in prompt


def test_run_stage2_malformed_response_falls_back_to_other():
    results = {"details": [_result_row()]}
    dataset = [_dataset_item()]
    report = diagnose_stage1(results, dataset, types=None)

    client = _FakeClient(content="definitely not json")
    run_stage2(report, dataset, model="gpt-5.4-mini-2026-03-17", client=client)

    record = report["questions"][0]
    assert record["llm_mode"] == MODE_OTHER
    assert record["llm_error"]
    # Error records carry no labeling signal: excluded from agreement metrics.
    agreement = report["agreement"]
    assert agreement["llm_errors"] == 1
    assert agreement["matrix"] == {}
    assert agreement["exact_agreement"] is None


class _FlakyCompletions:
    """Raises on the call indices in ``fail_on``; succeeds otherwise."""

    def __init__(self, fail_on, content):
        self.fail_on = set(fail_on)
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        if index in self.fail_on:
            raise RuntimeError("simulated transport failure")
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FlakyClient:
    def __init__(self, fail_on, content):
        self.chat = SimpleNamespace(completions=_FlakyCompletions(fail_on=fail_on, content=content))


def test_run_stage2_transport_errors_excluded_from_agreement():
    results = {"details": [_result_row(), _result_row(question_id="q-synth-2")]}
    dataset = [_dataset_item(), _dataset_item(question_id="q-synth-2")]
    report = diagnose_stage1(results, dataset, types=None)
    assert [r["suggested_mode"] for r in report["questions"]] == [MODE_OUTDATED, MODE_OUTDATED]

    # First LLM call raises (transport failure); second returns a valid label.
    client = _FlakyClient(fail_on={0}, content='{"mode": "ranking", "rationale": "buried"}')
    run_stage2(report, dataset, model="gpt-5.4-mini-2026-03-17", client=client)

    errored, scored = report["questions"]
    assert errored["llm_mode"] == MODE_OTHER
    assert errored["llm_error"].startswith("RuntimeError")
    assert scored["llm_mode"] == MODE_RANKING
    assert scored["llm_error"] is None

    agreement = report["agreement"]
    assert agreement["llm_errors"] == 1
    # Only the scored record lands in the matrix and the agreement rate.
    assert agreement["matrix"] == {MODE_OUTDATED: {MODE_RANKING: 1}}
    assert agreement["exact_agreement"] == 0.0
    # Per-record llm_mode counts still cover every record, errors included.
    assert agreement["llm_mode_counts"] == {MODE_OTHER: 1, MODE_RANKING: 1}
