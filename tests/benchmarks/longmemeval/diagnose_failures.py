"""LongMemEval failure-mode diagnosis harness (issues #158/#159).

Classifies "failed-but-retrieved" questions — wrong answer despite the
answer session being inside the top-5 retrieved sessions
(``is_correct == false AND recall_hit_at_5 == true``) — so ranking fixes
can be prioritized by failure mode instead of anecdote.

Stage 1 (always, pure code) joins each failure to the dataset's haystack
sessions/dates and emits deterministic per-question evidence plus a
``suggested_mode``. Stage 2 (``--llm``) asks the pinned benchmark judge
model for an independent label and records the stage1-vs-stage2
agreement matrix.

Suggested-mode heuristic (first matching rule wins; auditable on purpose):

    1. abstained_despite_hit                          -> answer-construction
       (evidence was retrieved but the model refused to answer)
    2. type in {knowledge-update, single-session-preference}
       and stale_candidate_above_answer               -> outdated-fact-selection
       (an older, topically-overlapping non-answer session outranked the
       newest answer session)
    3. type == knowledge-update and staleness was
       evaluated as False                             -> conflict-resolution
       (old + new facts both visible; model picked the wrong one)
    4. type == temporal-reasoning
       and date_arithmetic_needed                     -> missing-date-use
       (question requires date math over session dates)
    5. type == multi-session
       and answer_coverage_top5 < 1.0                 -> retrieval-gap-rank6-10
       (some required answer sessions are missing from the top-5;
       with recall depth 10 they sit at rank 6-10 or beyond)
    6. answer_rank == 1                               -> answer-construction
       (retrieval did its job; generation still failed)
    7. answer_rank >= 2                               -> ranking
       (answer session buried under higher-ranked candidates)
    8. otherwise                                      -> other

Type filter default: ALL question types. The canonical full run
(longmemeval_full_gpt5mini_20260425_231308.json) has 58 failed-but-
retrieved questions across six types; the four weak categories
(multi-session, temporal-reasoning, knowledge-update,
single-session-preference) cover 54 of them. Pass ``--types`` to
restrict.

CLI:
    python -m tests.benchmarks.longmemeval.diagnose_failures \
        --results benchmarks/results/longmemeval_full_gpt5mini_20260425_231308.json \
        --dataset tests/benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
        [--types multi-session,temporal-reasoning,...] [--llm] \
        [--llm-model MODEL] [--out failure_modes.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set

# Allow running as a script as well as a module.
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a benchmark dependency
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - only needed for --llm
    OpenAI = None

from tests.benchmarks.judge_policy import CANONICAL_BENCHMARK_JUDGE_MODEL, is_gpt5_family
from tests.benchmarks.longmemeval.analyze_results import _result_details
from tests.benchmarks.longmemeval.evaluator import check_abstention_response

# ---------------------------------------------------------------------------
# Failure-mode labels
# ---------------------------------------------------------------------------

MODE_RANKING = "ranking"
MODE_OUTDATED = "outdated-fact-selection"
MODE_MISSING_DATE = "missing-date-use"
MODE_CONFLICT = "conflict-resolution"
MODE_ANSWER_CONSTRUCTION = "answer-construction"
MODE_RETRIEVAL_GAP = "retrieval-gap-rank6-10"
MODE_JUDGE_ERROR = "judge-error"
MODE_OTHER = "other"

STAGE1_MODES = (
    MODE_RANKING,
    MODE_OUTDATED,
    MODE_MISSING_DATE,
    MODE_CONFLICT,
    MODE_ANSWER_CONSTRUCTION,
    MODE_RETRIEVAL_GAP,
    MODE_OTHER,
)
STAGE2_MODES = STAGE1_MODES[:-1] + (MODE_JUDGE_ERROR, MODE_OTHER)

WEAK_TYPES = (
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-preference",
)
STALE_CHECK_TYPES = frozenset({"knowledge-update", "single-session-preference"})

# ---------------------------------------------------------------------------
# Tokenization / date parsing
# ---------------------------------------------------------------------------

_STOPWORDS: FrozenSet[str] = frozenset(
    """
    a an the is was were are am be been being do does did done i you he she it
    we they my your his her its our their me him them us this that these those
    what which who whom whose when where why how of in on at by for with about
    to from as and or but not no nor so if then than too very can could will
    would shall should may might must have has had having there here also any
    some such own same just only into over under again further once during
    each few more most other s t don
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SESSION_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})(?:\s*\([^)]*\))?\s*(\d{2}):(\d{2})")
# Deliberately over-triggers on temporal-reasoning questions (broad markers
# like \bbefore\b and \bfirst\b); stage 2 (--llm) cross-checks the labels.
_DATE_MATH_RE = re.compile(
    r"\bhow long\b"
    r"|\bhow many\s+(?:days|weeks|months|years)\b"
    r"|\bbefore\b"
    r"|\bafter\b"
    r"|\bfirst\b"
    r"|\blast time\b",
    re.IGNORECASE,
)
_JSON_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(?P<body>.*)\n\s*```\s*$", re.S)


def keyword_tokens(text: Any) -> Set[str]:
    """Lowercase word-set minus stopwords (simple, auditable tokenization)."""
    if not text:
        return set()
    tokens = _TOKEN_RE.findall(str(text).lower())
    return {token for token in tokens if len(token) >= 2 and token not in _STOPWORDS}


def parse_session_date(raw: Any) -> Optional[datetime]:
    """Parse LongMemEval session dates like ``2023/05/20 (Sat) 02:21``."""
    if not raw:
        return None
    match = _SESSION_DATE_RE.search(str(raw))
    if not match:
        return None
    year, month, day, hour, minute = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None


def date_arithmetic_needed(question: Any) -> bool:
    """Heuristic: does the question require date arithmetic?"""
    return bool(_DATE_MATH_RE.search(str(question or "")))


def is_refusal(hypothesis: Any) -> bool:
    """True when the hypothesis is an 'I don't know'-style refusal."""
    if not hypothesis:
        return False
    return check_abstention_response(hypothesis)


def answer_rank(retrieved: Iterable[str], answer_ids: Iterable[str]) -> Optional[int]:
    """1-based rank of the first answer session in the retrieved list."""
    answer_set = set(answer_ids or [])
    for rank, session_id in enumerate(retrieved or [], start=1):
        if session_id in answer_set:
            return rank
    return None


# ---------------------------------------------------------------------------
# Dataset join + evidence extraction
# ---------------------------------------------------------------------------


def _session_text(session: Any) -> str:
    if not isinstance(session, list):
        return ""
    parts = []
    for turn in session:
        if isinstance(turn, dict):
            content = turn.get("content")
            if content:
                parts.append(str(content))
    return "\n".join(parts)


def build_session_index(dataset_item: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map session_id -> {raw_date, date, text, tokens} for one question."""
    index: Dict[str, Dict[str, Any]] = {}
    session_ids = dataset_item.get("haystack_session_ids") or []
    dates = dataset_item.get("haystack_dates") or []
    sessions = dataset_item.get("haystack_sessions") or []
    for position, session_id in enumerate(session_ids):
        raw_date = dates[position] if position < len(dates) else None
        text = _session_text(sessions[position]) if position < len(sessions) else ""
        index[session_id] = {
            "raw_date": raw_date,
            "date": parse_session_date(raw_date),
            "text": text,
            "tokens": keyword_tokens(text),
        }
    return index


def compute_evidence(row: Dict[str, Any], dataset_item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic stage-1 evidence for one failed-but-retrieved question."""
    question_type = str(row.get("question_type") or "unknown")
    question = row.get("question") or ""
    retrieved = list(row.get("retrieved_session_ids") or [])
    answer_ids = set(row.get("answer_session_ids") or [])

    evidence: Dict[str, Any] = {
        "answer_rank": answer_rank(retrieved, answer_ids),
        "abstained_despite_hit": bool(row.get("is_abstention"))
        or is_refusal(row.get("hypothesis")),
        "stale_candidate_above_answer": None,
        "noise_ratio": None,
        "date_arithmetic_needed": (
            date_arithmetic_needed(question) if question_type == "temporal-reasoning" else None
        ),
        "answer_coverage_top5": (
            len(answer_ids & set(retrieved)) / len(answer_ids) if answer_ids else None
        ),
        "newest_answer_session_id": None,
        "newest_answer_rank": None,
        "dataset_missing": dataset_item is None,
    }

    if dataset_item is None:
        return evidence

    session_index = build_session_index(dataset_item)
    question_tokens = keyword_tokens(question)

    # noise_ratio: fraction of retrieved sessions sharing zero question keywords.
    if retrieved:
        noisy = sum(
            1
            for session_id in retrieved
            if not (question_tokens & session_index.get(session_id, {}).get("tokens", set()))
        )
        evidence["noise_ratio"] = noisy / len(retrieved)

    # Newest answer session (by haystack date) and its rank in the top-5.
    dated_answers = [
        (session_index[sid]["date"], sid)
        for sid in answer_ids
        if sid in session_index and session_index[sid]["date"] is not None
    ]
    if dated_answers:
        newest_date, newest_id = max(dated_answers, key=lambda pair: pair[0])
        evidence["newest_answer_session_id"] = newest_id
        newest_rank = retrieved.index(newest_id) + 1 if newest_id in retrieved else None
        evidence["newest_answer_rank"] = newest_rank

        if question_type in STALE_CHECK_TYPES:
            # A retrieved non-answer session that is OLDER than the newest
            # answer session, topically overlapping with the question, and
            # ranked ABOVE the newest answer session. If the newest answer
            # session is absent from the top-5, every retrieved candidate
            # ranks above it by construction.
            cutoff_rank = newest_rank if newest_rank is not None else len(retrieved) + 1
            stale = False
            for rank, session_id in enumerate(retrieved, start=1):
                if rank >= cutoff_rank or session_id in answer_ids:
                    continue
                info = session_index.get(session_id)
                if not info or info["date"] is None:
                    continue
                if info["date"] < newest_date and (question_tokens & info["tokens"]):
                    stale = True
                    break
            evidence["stale_candidate_above_answer"] = stale
    elif question_type in STALE_CHECK_TYPES:
        evidence["stale_candidate_above_answer"] = False

    return evidence


def suggested_mode(question_type: str, evidence: Dict[str, Any]) -> str:
    """Deterministic evidence -> failure-mode mapping (see module docstring)."""
    if evidence.get("abstained_despite_hit"):
        return MODE_ANSWER_CONSTRUCTION
    if question_type in STALE_CHECK_TYPES and evidence.get("stale_candidate_above_answer"):
        return MODE_OUTDATED
    if (
        question_type == "knowledge-update"
        and evidence.get("stale_candidate_above_answer") is False
    ):
        return MODE_CONFLICT
    if question_type == "temporal-reasoning" and evidence.get("date_arithmetic_needed"):
        return MODE_MISSING_DATE
    coverage = evidence.get("answer_coverage_top5")
    if question_type == "multi-session" and coverage is not None and coverage < 1.0:
        return MODE_RETRIEVAL_GAP
    rank = evidence.get("answer_rank")
    if rank == 1:
        return MODE_ANSWER_CONSTRUCTION
    if rank is not None and rank >= 2:
        return MODE_RANKING
    return MODE_OTHER


# ---------------------------------------------------------------------------
# Stage 1 driver
# ---------------------------------------------------------------------------


def select_failures(
    details: Iterable[Dict[str, Any]], types: Optional[Set[str]]
) -> List[Dict[str, Any]]:
    """Filter to failed-but-retrieved rows, optionally restricted by type."""
    selected = []
    for row in details:
        if row.get("is_correct") is not False:
            continue
        if row.get("recall_hit_at_5") is not True:
            continue
        if types is not None and str(row.get("question_type")) not in types:
            continue
        selected.append(row)
    return selected


def diagnose_stage1(
    results: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    types: Optional[Set[str]] = None,
    results_file: Optional[str] = None,
    dataset_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run pure-code stage-1 diagnosis and return the report dict."""
    details = _result_details(results)
    failures = [row for row in details if row.get("is_correct") is False]
    selected = select_failures(details, types)
    dataset_by_id = {item.get("question_id"): item for item in dataset}

    questions = []
    for row in selected:
        question_id = row.get("question_id")
        dataset_item = dataset_by_id.get(question_id)
        evidence = compute_evidence(row, dataset_item)
        mode = suggested_mode(str(row.get("question_type") or "unknown"), evidence)
        session_index = build_session_index(dataset_item) if dataset_item else {}
        answer_ids = set(row.get("answer_session_ids") or [])
        question_tokens = keyword_tokens(row.get("question") or "")
        retrieved_sessions = []
        for rank, session_id in enumerate(row.get("retrieved_session_ids") or [], start=1):
            info = session_index.get(session_id, {})
            retrieved_sessions.append(
                {
                    "rank": rank,
                    "session_id": session_id,
                    "date": info.get("raw_date"),
                    "is_answer": session_id in answer_ids,
                    "question_keyword_overlap": len(question_tokens & info.get("tokens", set())),
                }
            )
        questions.append(
            {
                "question_id": question_id,
                "question_type": row.get("question_type"),
                "question": row.get("question"),
                "question_date": row.get("question_date"),
                "reference": row.get("reference"),
                "hypothesis": row.get("hypothesis"),
                "explanation": row.get("explanation"),
                "answer_session_ids": list(row.get("answer_session_ids") or []),
                "retrieved_session_ids": list(row.get("retrieved_session_ids") or []),
                "retrieved_sessions": retrieved_sessions,
                "evidence": evidence,
                "suggested_mode": mode,
            }
        )

    mode_counts: Counter = Counter(record["suggested_mode"] for record in questions)
    mode_counts_by_type: Dict[str, Dict[str, int]] = defaultdict(dict)
    for record in questions:
        qtype = str(record["question_type"])
        mode_counts_by_type[qtype][record["suggested_mode"]] = (
            mode_counts_by_type[qtype].get(record["suggested_mode"], 0) + 1
        )

    report = {
        "metadata": {
            "results_file": results_file,
            "dataset_file": dataset_file,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "types_filter": sorted(types) if types is not None else None,
            "total_details": len(details),
            "total_failures": len(failures),
            "selected": len(questions),
            "selected_by_type": dict(
                sorted(Counter(str(r["question_type"]) for r in questions).items())
            ),
        },
        "questions": questions,
        "summary": {
            "mode_counts": dict(sorted(mode_counts.items())),
            "mode_counts_by_type": {
                qtype: dict(sorted(modes.items()))
                for qtype, modes in sorted(mode_counts_by_type.items())
            },
        },
    }
    report["summary"]["answer_construction"] = summarize_answer_construction(report)
    return report


def _effective_mode(record: Dict[str, Any]) -> str:
    if record.get("llm_error"):
        return str(record.get("suggested_mode") or MODE_OTHER)
    return str(record.get("llm_mode") or record.get("suggested_mode") or MODE_OTHER)


def summarize_answer_construction(
    report: Dict[str, Any], *, noise_threshold: float = 0.4
) -> Dict[str, Any]:
    """Summarize retrieved-but-unused answer-construction misses."""
    selected = [
        record
        for record in report.get("questions", [])
        if _effective_mode(record) == MODE_ANSWER_CONSTRUCTION
    ]

    by_type: Counter = Counter(str(record.get("question_type") or "unknown") for record in selected)
    by_rank: Counter = Counter()
    abstained: Counter = Counter()
    rank1_abstentions = 0
    high_noise_count = 0
    question_ids: List[str] = []

    for record in selected:
        evidence = record.get("evidence") or {}
        rank = evidence.get("answer_rank")
        by_rank[str(rank) if rank is not None else "missing"] += 1

        abstained_value = bool(evidence.get("abstained_despite_hit"))
        abstained["true" if abstained_value else "false"] += 1
        if rank == 1 and abstained_value:
            rank1_abstentions += 1

        noise_ratio = evidence.get("noise_ratio")
        if isinstance(noise_ratio, (int, float)) and noise_ratio >= noise_threshold:
            high_noise_count += 1

        question_id = record.get("question_id")
        if question_id:
            question_ids.append(str(question_id))

    return {
        "total": len(selected),
        "by_type": dict(sorted(by_type.items())),
        "by_answer_rank": dict(sorted(by_rank.items())),
        "abstained_despite_hit": dict(sorted(abstained.items())),
        "rank1_abstentions": rank1_abstentions,
        "high_noise_count": high_noise_count,
        "high_noise_threshold": noise_threshold,
        "question_ids": question_ids,
    }


def format_summary(report: Dict[str, Any]) -> str:
    """Human-readable mode-by-type summary table."""
    metadata = report["metadata"]
    by_type = report["summary"]["mode_counts_by_type"]
    modes = [mode for mode in STAGE1_MODES if any(mode in row for row in by_type.values())]

    lines = [
        "LongMemEval failure-mode diagnosis (stage 1)",
        f"  results: {metadata.get('results_file')}",
        f"  selected failed-but-retrieved: {metadata['selected']} "
        f"(of {metadata['total_failures']} total failures, "
        f"{metadata['total_details']} questions)",
        f"  types filter: {metadata.get('types_filter') or 'all'}",
        "",
    ]
    type_width = max([len(t) for t in by_type] + [len("question type")]) + 2
    header = "question type".ljust(type_width) + "".join(f"{mode:>26}" for mode in modes)
    header += f"{'total':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for qtype in sorted(by_type):
        row_counts = by_type[qtype]
        line = qtype.ljust(type_width)
        line += "".join(f"{row_counts.get(mode, 0):>26}" for mode in modes)
        line += f"{sum(row_counts.values()):>8}"
        lines.append(line)
    totals = report["summary"]["mode_counts"]
    line = "TOTAL".ljust(type_width)
    line += "".join(f"{totals.get(mode, 0):>26}" for mode in modes)
    line += f"{metadata['selected']:>8}"
    lines.append(line)

    answer_summary = report.get("summary", {}).get("answer_construction")
    if answer_summary:
        lines.extend(
            [
                "",
                "Answer-construction characterization",
                f"  total: {answer_summary['total']}",
                f"  by type: {answer_summary['by_type']}",
                f"  by answer rank: {answer_summary['by_answer_rank']}",
                f"  abstained despite hit: {answer_summary['abstained_despite_hit']}",
                f"  rank-1 abstentions: {answer_summary['rank1_abstentions']}",
                (
                    f"  high-noise cases (noise_ratio >= "
                    f"{answer_summary['high_noise_threshold']}): "
                    f"{answer_summary['high_noise_count']}"
                ),
            ]
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 2 (--llm): independent LLM classification
# ---------------------------------------------------------------------------

_STAGE2_SESSION_CHAR_LIMIT = 2500

_STAGE2_LABEL_GUIDE = """\
- ranking: the answer session was retrieved but ranked below distractors, and the model used a higher-ranked wrong source
- outdated-fact-selection: an older/stale fact was preferred over the newest correct fact
- missing-date-use: the model had the sessions but failed to use session dates / do date arithmetic
- conflict-resolution: conflicting facts were both visible and the model resolved them incorrectly
- answer-construction: retrieval was fine; the model refused, hedged, or composed a wrong answer from correct evidence
- retrieval-gap-rank6-10: required evidence sessions were missing from the provided top-5 context
- judge-error: the hypothesis actually matches the reference; the benchmark judge mis-scored it
- other: none of the above fits"""


def _build_stage2_prompt(record: Dict[str, Any], session_index: Dict[str, Any]) -> str:
    session_blocks = []
    for entry in record["retrieved_sessions"]:
        session_id = entry["session_id"]
        info = session_index.get(session_id, {})
        text = (info.get("text") or "")[:_STAGE2_SESSION_CHAR_LIMIT]
        marker = "ANSWER SESSION" if entry["is_answer"] else "non-answer"
        session_blocks.append(
            f"[Rank {entry['rank']} | session {session_id} | "
            f"{entry.get('date') or 'unknown date'} | {marker}]\n{text}"
        )
    sessions_text = "\n\n".join(session_blocks) or "(no retrieved sessions)"
    return f"""You are diagnosing why a long-term-memory QA system answered incorrectly even though a session containing the answer WAS retrieved in the top-5.

Question (asked {record.get('question_date') or 'unknown date'}): {record.get('question')}
Question type: {record.get('question_type')}
Reference answer: {record.get('reference')}
System's answer: {record.get('hypothesis')}
Benchmark judge explanation: {record.get('explanation')}
Answer session ids: {', '.join(record.get('answer_session_ids') or [])}

Retrieved sessions in rank order:

{sessions_text}

Pick exactly ONE failure-mode label:
{_STAGE2_LABEL_GUIDE}

Respond with ONLY a JSON object:
{{"mode": "<label>", "rationale": "<one sentence>"}}"""


def classify_failure_with_llm(
    client: Any, model: str, record: Dict[str, Any], session_index: Dict[str, Any]
) -> Dict[str, Any]:
    """One chat completion -> {"mode", "rationale", "error"} for one failure."""
    prompt = _build_stage2_prompt(record, session_index)
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if is_gpt5_family(model):
        request_kwargs["max_completion_tokens"] = 300
        request_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "failure_mode",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": list(STAGE2_MODES)},
                        "rationale": {"type": "string"},
                    },
                    "required": ["mode", "rationale"],
                    "additionalProperties": False,
                },
            },
        }
    else:
        request_kwargs["temperature"] = 0
        request_kwargs["max_tokens"] = 300
        request_kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**request_kwargs)
        content = (response.choices[0].message.content or "").strip()
        fence_match = _JSON_FENCE_RE.match(content)
        if fence_match:
            content = fence_match.group("body").strip()
        parsed = json.loads(content)
        mode = str(parsed.get("mode") or "").strip()
        if mode not in STAGE2_MODES:
            return {
                "mode": MODE_OTHER,
                "rationale": str(parsed.get("rationale") or ""),
                "error": f"unexpected label: {mode!r}",
            }
        return {"mode": mode, "rationale": str(parsed.get("rationale") or ""), "error": None}
    except Exception as exc:  # noqa: BLE001 - one bad question must not kill the run
        return {"mode": MODE_OTHER, "rationale": "", "error": f"{type(exc).__name__}: {exc}"}


def run_stage2(
    report: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    model: str,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Annotate the stage-1 report with LLM labels + agreement matrix."""
    if client is None:
        if OpenAI is None:
            raise ImportError("openai package required for --llm stage 2")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for --llm stage 2")
        client = OpenAI(api_key=api_key)

    dataset_by_id = {item.get("question_id"): item for item in dataset}
    agreement: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    matches = 0
    scored = 0
    llm_errors = 0
    for record in report["questions"]:
        dataset_item = dataset_by_id.get(record["question_id"])
        session_index = build_session_index(dataset_item) if dataset_item else {}
        verdict = classify_failure_with_llm(client, model, record, session_index)
        record["llm_mode"] = verdict["mode"]
        record["llm_rationale"] = verdict["rationale"]
        record["llm_error"] = verdict["error"]
        if verdict["error"] is not None:
            # Transport/parse failures carry no labeling signal: keep the
            # per-record llm_error, but exclude the record from the
            # agreement matrix and the exact-agreement rate.
            llm_errors += 1
            continue
        scored += 1
        agreement[record["suggested_mode"]][verdict["mode"]] += 1
        if record["suggested_mode"] == verdict["mode"]:
            matches += 1

    report["agreement"] = {
        "llm_model": model,
        "llm_errors": llm_errors,
        "matrix": {
            stage1: dict(sorted(stage2.items())) for stage1, stage2 in sorted(agreement.items())
        },
        "exact_agreement": matches / scored if scored else None,
        "llm_mode_counts": dict(
            sorted(Counter(r["llm_mode"] for r in report["questions"]).items())
        ),
    }
    report["metadata"]["llm_stage"] = True
    report["metadata"]["llm_model"] = model
    report["summary"]["answer_construction"] = summarize_answer_construction(report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose LongMemEval failed-but-retrieved questions by failure mode."
    )
    parser.add_argument("--results", required=True, type=Path, help="Benchmark results JSON")
    parser.add_argument(
        "--dataset", required=True, type=Path, help="longmemeval_s_cleaned.json dataset"
    )
    parser.add_argument(
        "--types",
        default=None,
        help=(
            "Comma-separated question types to include "
            f"(default: all; weak categories: {','.join(WEAK_TYPES)})"
        ),
    )
    parser.add_argument(
        "--llm", action="store_true", help="Run stage-2 LLM classification per failure"
    )
    parser.add_argument(
        "--llm-model",
        default=CANONICAL_BENCHMARK_JUDGE_MODEL,
        help=f"Stage-2 model (default: {CANONICAL_BENCHMARK_JUDGE_MODEL})",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write report JSON here")
    args = parser.parse_args(argv)

    if load_dotenv is not None:
        load_dotenv()
        load_dotenv(Path.home() / ".config" / "automem" / ".env")

    with args.results.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    with args.dataset.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    types: Optional[Set[str]] = None
    if args.types:
        types = {part.strip() for part in args.types.split(",") if part.strip()}

    report = diagnose_stage1(
        results,
        dataset,
        types=types,
        results_file=str(args.results),
        dataset_file=str(args.dataset),
    )

    if args.llm:
        from tests.benchmarks.judge_preflight import EXIT_OK, run_preflight

        code, message = run_preflight(model=args.llm_model)
        print(message)
        if code != EXIT_OK:
            return code
        run_stage2(report, dataset, model=args.llm_model)

    print(format_summary(report))
    if args.llm and report.get("agreement"):
        agreement = report["agreement"]
        print("")
        print(f"Stage-2 LLM labels ({agreement['llm_model']}):")
        for mode, count in agreement["llm_mode_counts"].items():
            print(f"  {mode:<26} {count}")
        rate = agreement["exact_agreement"]
        if rate is not None:
            print(f"  stage1/stage2 exact agreement: {rate:.1%}")
        print(f"  stage-2 llm errors (excluded from agreement): {agreement.get('llm_errors', 0)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"\nWrote report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
