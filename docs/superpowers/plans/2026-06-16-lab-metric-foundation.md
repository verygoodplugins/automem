# Lab Metric Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the AutoMem Recall Quality Lab so its optimization target is the legible scorecard from the spec — NDCG@10 primary, distractor-precision guardrail, simplicity + latency tiebreakers — with distractor injection, configurable recall params, and a real consolidation step, all unit-tested.

**Architecture:** Extract pure logic from `scripts/lab/run_recall_test.py` into two focused, importable modules — `scripts/lab/lab_metrics.py` (metrics, scorecard, decision rule) and `scripts/lab/lab_corpus.py` (recall-with-params, distractor injection, backdating, consolidation). `run_recall_test.py` becomes thin orchestration that imports both. The matrix harness (Plan B) imports the same modules, so there is one source of truth for scoring.

**Tech Stack:** Python 3.12, `requests`, pytest (pure-Python stats already in-repo, no scipy). HTTP-touching functions take an injectable client (`http_get`/`http_post`) so tests never hit a live server.

## Global Constraints

- Branch: `eval/recall-quality-harness`. **Do NOT stage, commit, stash, or revert** the unrelated pre-existing WIP (`automem/api/memory.py`, `docs/API.md`, `docs/MCP_SSE.md`, `mcp-sse-server/*`, `tests/support/fake_graph.py`, `tests/test_api_endpoints.py`). Only `git add` the exact files each task names.
- Run tests with plugin autoload disabled: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest <path> -v`.
- AutoMem `/recall` nests memory fields under `result["memory"]` (e.g. `result["memory"]["content"]`); the existing `r.get("memory", r)` pattern handles both shapes — preserve it.
- Distractor tag is the bare slug `lab-distractor` (bare-tag convention; no namespace prefix).
- No secrets in code or fixtures. No new network calls in unit tests.
- Format with `black .` and lint with `flake8` before each commit.
- Pure functions only in `lab_metrics.py` (no I/O). All HTTP lives in `lab_corpus.py` behind injectable clients.

---

### Task 1: Extract metrics into `lab_metrics.py` (refactor, no behavior change)

Move the existing pure metric functions out of the CLI script into an importable module, and pin their behavior with characterization tests. No metric math changes in this task.

**Files:**
- Create: `scripts/lab/lab_metrics.py`
- Create: `tests/lab/conftest.py`
- Create: `tests/lab/test_lab_metrics.py`
- Modify: `scripts/lab/run_recall_test.py` (import from the new module instead of defining locally)

**Interfaces:**
- Produces: `recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float`, `mrr(retrieved_ids, expected_ids) -> float`, `ndcg_at_k(retrieved_ids, expected_ids, k: int) -> float`, `paired_ttest(a: list[float], b: list[float]) -> dict` — identical signatures/behavior to the current functions in `run_recall_test.py:47-122`.

- [ ] **Step 1: Write the failing test**

Create `tests/lab/conftest.py`:

```python
import sys
from pathlib import Path

# Make scripts/lab importable without packaging the scripts dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "lab"))
```

Create `tests/lab/test_lab_metrics.py`:

```python
import lab_metrics as m


def test_recall_at_k_counts_hits_in_top_k():
    assert m.recall_at_k(["a", "b", "c"], ["c"], 5) == 1.0
    assert m.recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert m.recall_at_k([], ["c"], 5) == 0.0


def test_mrr_uses_first_hit_rank():
    assert m.mrr(["a", "b", "c"], ["b"]) == 0.5
    assert m.mrr(["a", "b", "c"], ["z"]) == 0.0


def test_ndcg_at_k_rewards_top_rank():
    top = m.ndcg_at_k(["x", "a", "b"], ["x"], 10)
    buried = m.ndcg_at_k(["a", "b", "x"], ["x"], 10)
    assert top == 1.0
    assert 0.0 < buried < top
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lab_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/lab/lab_metrics.py` by moving the four functions verbatim from `run_recall_test.py`:

```python
"""Pure scoring functions for the AutoMem Recall Quality Lab.

No I/O lives here — every function is deterministic and unit-testable.
Imported by run_recall_test.py and by the parallel matrix harness.
"""

import math
from typing import Any, Dict, List


def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    """Fraction of expected IDs found in top-K results."""
    if not expected_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for eid in expected_ids if eid in top_k)
    return hits / len(expected_ids)


def mrr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    """Mean Reciprocal Rank — position of first relevant result."""
    expected_set = set(expected_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in expected_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    expected_set = set(expected_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in expected_set:
            dcg += 1.0 / math.log2(i + 2)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_ids), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def paired_ttest(a: List[float], b: List[float]) -> Dict[str, Any]:
    """Paired t-test + Cohen's d effect size. Pure Python, no scipy needed."""
    n = len(a)
    if n < 2 or n != len(b):
        return {"p_value": 1.0, "t_stat": 0.0, "effect_size": 0.0, "significant": False}

    diffs = [b[i] - a[i] for i in range(n)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    std_d = var_d**0.5 if var_d > 0 else 1e-10

    t_stat = mean_d / (std_d / n**0.5)
    z = abs(t_stat)
    p_value = 2 * (1 - 0.5 * (1 + math.erf(z / 2**0.5)))

    pooled_std = (
        (sum((ai - sum(a) / n) ** 2 for ai in a) + sum((bi - sum(b) / n) ** 2 for bi in b))
        / (2 * n - 2)
    ) ** 0.5
    cohens_d = (sum(b) / n - sum(a) / n) / pooled_std if pooled_std > 0 else 0.0

    effect_label = "negligible"
    if abs(cohens_d) >= 0.8:
        effect_label = "large"
    elif abs(cohens_d) >= 0.5:
        effect_label = "medium"
    elif abs(cohens_d) >= 0.2:
        effect_label = "small"

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "cohens_d": round(cohens_d, 4),
        "effect_size": effect_label,
        "significant": p_value < 0.05,
        "mean_diff": round(mean_d, 4),
    }
```

Then in `scripts/lab/run_recall_test.py`, delete the local definitions of `recall_at_k`, `mrr`, `ndcg_at_k`, `paired_ttest` (lines 47-122) and add an import near the top (after the existing imports, before `API_URL`):

```python
from lab_metrics import mrr, ndcg_at_k, paired_ttest, recall_at_k  # noqa: E402
```

Because `run_recall_test.py` runs from `scripts/lab/`, add this just above that import so it resolves when invoked from the repo root:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_metrics.py -v`
Expected: PASS (3 tests).

Also confirm the CLI still imports cleanly:
Run: `python scripts/lab/run_recall_test.py --help`
Expected: argparse help prints, no ImportError.

- [ ] **Step 5: Commit**

```bash
git add scripts/lab/lab_metrics.py scripts/lab/run_recall_test.py tests/lab/conftest.py tests/lab/test_lab_metrics.py
git commit -m "refactor(lab): extract pure metrics into lab_metrics module"
```

---

### Task 2: Add `distractor_rate_at_k` (the precision guardrail)

**Files:**
- Modify: `scripts/lab/lab_metrics.py`
- Modify: `tests/lab/test_lab_metrics.py`

**Interfaces:**
- Produces: `distractor_rate_at_k(retrieved_ids: list[str], distractor_ids: set[str] | list[str], k: int) -> float` — fraction of the top-k results that are known distractors. Lower is better. Returns 0.0 for empty results or k<=0.

- [ ] **Step 1: Write the failing test**

Append to `tests/lab/test_lab_metrics.py`:

```python
def test_distractor_rate_counts_distractors_in_top_k():
    retrieved = ["good", "d1", "d2", "good2"]
    distractors = {"d1", "d2"}
    assert m.distractor_rate_at_k(retrieved, distractors, 4) == 0.5
    # top-1 is clean -> 0.0
    assert m.distractor_rate_at_k(retrieved, distractors, 1) == 0.0
    # all distractors -> 1.0
    assert m.distractor_rate_at_k(["d1", "d2"], distractors, 10) == 1.0
    # empties are safe
    assert m.distractor_rate_at_k([], distractors, 10) == 0.0
    assert m.distractor_rate_at_k(retrieved, distractors, 0) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_metrics.py::test_distractor_rate_counts_distractors_in_top_k -v`
Expected: FAIL with `AttributeError: module 'lab_metrics' has no attribute 'distractor_rate_at_k'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/lab/lab_metrics.py` (update the typing import line to `from typing import Any, Dict, Iterable, List`):

```python
def distractor_rate_at_k(
    retrieved_ids: List[str], distractor_ids: Iterable[str], k: int
) -> float:
    """Fraction of the top-k results that are known distractors. Lower is better.

    Distractors are memories we injected and labelled as never-relevant, so a
    result that is a distractor is unambiguous noise. This is the precision
    guardrail and the only metric that can see the `forget` consolidation mode
    working (known-item recall is blind to suppression).
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    dset = set(distractor_ids)
    hits = sum(1 for rid in top_k if rid in dset)
    return hits / len(top_k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_metrics.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/lab/lab_metrics.py tests/lab/test_lab_metrics.py
git commit -m "feat(lab): add distractor_rate_at_k precision guardrail metric"
```

---

### Task 3: Add `config_complexity` (the simplicity tiebreaker)

**Files:**
- Modify: `scripts/lab/lab_metrics.py`
- Modify: `tests/lab/test_lab_metrics.py`

**Interfaces:**
- Produces: `config_complexity(config: dict) -> int` — count of "active" knobs. Lower = simpler/more elegant. Rules: a `SEARCH_WEIGHT_*` entry counts if its float value != 0; a known feature flag (`ENRICHMENT_ENABLED`, `JIT_ENRICHMENT_ENABLED`, `ENRICHMENT_ENABLE_SUMMARIES`, `RECALL_RECENCY_BIAS`) counts if not "off/false/0/empty"; a `*_THRESHOLD`/`*_CAP`/`*_GATE` entry counts if its value is > 0 (or, if non-numeric, not "off").

- [ ] **Step 1: Write the failing test**

Append to `tests/lab/test_lab_metrics.py`:

```python
def test_config_complexity_counts_active_knobs():
    baseline = {
        "SEARCH_WEIGHT_VECTOR": "0.35",
        "SEARCH_WEIGHT_KEYWORD": "0.35",
        "SEARCH_WEIGHT_RELEVANCE": "0.0",  # off -> not counted
    }
    assert m.config_complexity(baseline) == 2

    simpler = {
        "SEARCH_WEIGHT_VECTOR": "0.35",
        "SEARCH_WEIGHT_KEYWORD": "0.0",
        "SEARCH_WEIGHT_RELEVANCE": "0.0",
    }
    assert m.config_complexity(simpler) == 1

    flags_and_gates = {
        "SEARCH_WEIGHT_VECTOR": "0.35",        # +1
        "ENRICHMENT_ENABLED": "true",          # +1
        "JIT_ENRICHMENT_ENABLED": "false",     # +0
        "RECALL_RECENCY_BIAS": "off",          # +0
        "RECALL_RELEVANCE_GATE": "0.2",        # +1 (gate > 0)
        "SEARCH_TAG_SCORE_TOKEN_CAP": "0",     # +0
    }
    assert m.config_complexity(flags_and_gates) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_metrics.py::test_config_complexity_counts_active_knobs -v`
Expected: FAIL with `AttributeError: module 'lab_metrics' has no attribute 'config_complexity'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/lab/lab_metrics.py`:

```python
_FLAG_KEYS = {
    "ENRICHMENT_ENABLED",
    "JIT_ENRICHMENT_ENABLED",
    "ENRICHMENT_ENABLE_SUMMARIES",
    "RECALL_RECENCY_BIAS",
}
_OFF_VALUES = {"", "0", "0.0", "off", "false", "no", "none"}


def _is_off(value: Any) -> bool:
    return str(value).strip().lower() in _OFF_VALUES


def config_complexity(config: Dict[str, Any]) -> int:
    """Count 'active' knobs in a config. Lower = simpler/more elegant.

    This is the simplicity tiebreaker: among configs within noise on quality,
    the one with fewer active knobs wins.
    """
    count = 0
    for key, value in config.items():
        ku = str(key).upper()
        if ku.startswith("SEARCH_WEIGHT_"):
            try:
                if float(value) != 0.0:
                    count += 1
            except (TypeError, ValueError):
                continue
        elif ku in _FLAG_KEYS:
            if not _is_off(value):
                count += 1
        elif ku.endswith(("_THRESHOLD", "_CAP", "_GATE")):
            try:
                if float(value) > 0.0:
                    count += 1
            except (TypeError, ValueError):
                if not _is_off(value):
                    count += 1
    return count
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lab/lab_metrics.py tests/lab/test_lab_metrics.py
git commit -m "feat(lab): add config_complexity simplicity metric"
```

---

### Task 4: Add `pick_winner` (the decision rule)

Encode the one-sentence rule from the spec: *highest NDCG@10 that does not regress distractor-precision; break ties toward fewer knobs and lower latency.*

**Files:**
- Modify: `scripts/lab/lab_metrics.py`
- Modify: `tests/lab/test_lab_metrics.py`

**Interfaces:**
- Produces: `pick_winner(cards: list[dict], *, baseline_name: str, ndcg_tol: float = 0.005, distractor_tol: float = 0.01) -> dict` — each card has keys `name, ndcg_10, distractor_rate_10, latency_ms, complexity`. Returns the winning card augmented with a `reason: str`. Configs whose `distractor_rate_10` exceeds baseline + `distractor_tol` are ineligible (precision regression). Among eligible configs within `ndcg_tol` of the best NDCG@10, pick the lowest complexity, then lowest latency.

- [ ] **Step 1: Write the failing test**

Append to `tests/lab/test_lab_metrics.py`:

```python
def _card(name, ndcg, distractor, latency, complexity):
    return {
        "name": name,
        "ndcg_10": ndcg,
        "distractor_rate_10": distractor,
        "latency_ms": latency,
        "complexity": complexity,
    }


def test_pick_winner_prefers_simpler_within_ndcg_tolerance():
    cards = [
        _card("baseline", 0.800, 0.10, 100, 11),
        _card("complex", 0.803, 0.10, 120, 13),   # tiny ndcg gain, more knobs
        _card("simple", 0.801, 0.10, 90, 8),       # within tol, fewer knobs + faster
    ]
    winner = m.pick_winner(cards, baseline_name="baseline")
    assert winner["name"] == "simple"


def test_pick_winner_rejects_precision_regression():
    cards = [
        _card("baseline", 0.800, 0.10, 100, 11),
        _card("greedy", 0.900, 0.30, 100, 11),  # big ndcg but junk floods results
    ]
    winner = m.pick_winner(cards, baseline_name="baseline")
    assert winner["name"] == "baseline"


def test_pick_winner_picks_clear_quality_jump():
    cards = [
        _card("baseline", 0.800, 0.10, 100, 11),
        _card("better", 0.860, 0.09, 100, 11),  # real gain, no regression
    ]
    winner = m.pick_winner(cards, baseline_name="baseline")
    assert winner["name"] == "better"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_metrics.py -k pick_winner -v`
Expected: FAIL with `AttributeError: module 'lab_metrics' has no attribute 'pick_winner'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/lab/lab_metrics.py`:

```python
def pick_winner(
    cards: List[Dict[str, Any]],
    *,
    baseline_name: str,
    ndcg_tol: float = 0.005,
    distractor_tol: float = 0.01,
) -> Dict[str, Any]:
    """Apply the scorecard decision rule and return the winning card + reason.

    Rule: highest NDCG@10 that does not regress distractor-precision vs the
    baseline; break ties (within ndcg_tol) toward fewer knobs, then lower latency.
    """
    baseline = next(c for c in cards if c["name"] == baseline_name)
    ceiling = baseline["distractor_rate_10"] + distractor_tol
    eligible = [c for c in cards if c["distractor_rate_10"] <= ceiling]
    regressed = not eligible
    if regressed:
        eligible = [baseline]

    best_ndcg = max(c["ndcg_10"] for c in eligible)
    contenders = [c for c in eligible if c["ndcg_10"] >= best_ndcg - ndcg_tol]
    winner = dict(min(contenders, key=lambda c: (c["complexity"], c["latency_ms"])))

    if regressed:
        winner["reason"] = "all candidates regressed distractor-precision; held baseline"
    elif winner["name"] == baseline_name:
        winner["reason"] = "no candidate beat baseline NDCG@10 without precision regression"
    else:
        winner["reason"] = (
            f"best NDCG@10 within tolerance, lowest complexity ({winner['complexity']}) "
            f"and latency ({winner['latency_ms']:.0f}ms)"
        )
    return winner
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lab/lab_metrics.py tests/lab/test_lab_metrics.py
git commit -m "feat(lab): add pick_winner scorecard decision rule"
```

---

### Task 5: Create `lab_corpus.py` with parameterized recall

**Files:**
- Create: `scripts/lab/lab_corpus.py`
- Create: `tests/lab/test_lab_corpus.py`

**Interfaces:**
- Produces: `recall(api_url: str, headers: dict, query: str, *, limit: int = 20, expand_relations: bool = False, current_only: bool = True, recency_bias: str | None = None, http_get=requests.get) -> dict` — calls `GET /recall` with the given params and returns the parsed JSON. `expand_relations` and `current_only` are serialized as the lowercase strings `"true"`/`"false"`; `recency_bias` is omitted when `None`.
- Produces: `extract_ids(recall_json: dict) -> list[str]` — pulls memory IDs from a `/recall` response, handling both the nested `result["memory"]["id"]` shape and the flat shape.

- [ ] **Step 1: Write the failing test**

Create `tests/lab/test_lab_corpus.py`:

```python
import lab_corpus as c


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_recall_serializes_params():
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResp({"results": []})

    c.recall(
        "http://x",
        {"Authorization": "Bearer t"},
        "hello",
        limit=10,
        expand_relations=True,
        current_only=False,
        recency_bias="auto",
        http_get=fake_get,
    )
    assert captured["url"].endswith("/recall")
    assert captured["params"]["query"] == "hello"
    assert captured["params"]["limit"] == 10
    assert captured["params"]["expand_relations"] == "true"
    assert captured["params"]["current_only"] == "false"
    assert captured["params"]["recency_bias"] == "auto"


def test_recall_omits_recency_bias_when_none():
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params
        return FakeResp({"results": []})

    c.recall("http://x", {}, "q", http_get=fake_get)
    assert "recency_bias" not in captured["params"]
    assert captured["params"]["current_only"] == "true"


def test_extract_ids_handles_nested_and_flat():
    payload = {
        "results": [
            {"memory": {"id": "a", "content": "x"}},
            {"id": "b", "content": "y"},
        ]
    }
    assert c.extract_ids(payload) == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lab_corpus'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/lab/lab_corpus.py`:

```python
"""Corpus-side helpers for the AutoMem Recall Quality Lab.

All HTTP lives here, behind injectable clients (http_get / http_post) so the
logic is unit-testable without a live server. Imported by run_recall_test.py
and the parallel matrix harness.
"""

from typing import Any, Dict, List, Optional

import requests


def recall(
    api_url: str,
    headers: Dict[str, str],
    query: str,
    *,
    limit: int = 20,
    expand_relations: bool = False,
    current_only: bool = True,
    recency_bias: Optional[str] = None,
    http_get=requests.get,
) -> Dict[str, Any]:
    """GET /recall with explicit recall parameters; returns parsed JSON."""
    params: Dict[str, Any] = {"query": query, "limit": limit}
    if expand_relations:
        params["expand_relations"] = "true"
    params["current_only"] = "true" if current_only else "false"
    if recency_bias is not None:
        params["recency_bias"] = recency_bias
    resp = http_get(f"{api_url}/recall", params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_ids(recall_json: Dict[str, Any]) -> List[str]:
    """Pull memory IDs from a /recall response (nested or flat shape)."""
    results = recall_json.get("results", recall_json.get("memories", []))
    ids: List[str] = []
    for r in results:
        mem = r.get("memory", r)
        mid = str(mem.get("id", r.get("id", "")))
        if mid:
            ids.append(mid)
    return ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_corpus.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/lab/lab_corpus.py tests/lab/test_lab_corpus.py
git commit -m "feat(lab): add lab_corpus with parameterized recall"
```

---

### Task 6: Add distractor injection + backdating

Inject synthetic, aged, low-importance, labelled distractor memories. These are simultaneously the precision guardrail's "known noise" and the `forget` arm's "should-be-archived" targets — one step serves both.

**Files:**
- Modify: `scripts/lab/lab_corpus.py`
- Modify: `tests/lab/test_lab_corpus.py`

**Interfaces:**
- Produces: `iso_days_ago(days: float) -> str` — UTC ISO-8601 timestamp `days` in the past.
- Produces: `make_distractor_memories(n: int, *, age_days: float = 180, importance: float = 0.05, tag: str = "lab-distractor") -> list[dict]` — `POST /memory` payloads, each pre-backdated (`timestamp` and `last_accessed`), low-importance, tagged `lab-distractor`, with `metadata.lab_distractor = True`.
- Produces: `inject_distractors(api_url: str, headers: dict, payloads: list[dict], *, http_post=requests.post) -> list[str]` — POSTs each payload and returns the created memory IDs (reads `memory_id`/`id`/`memory.id`).

- [ ] **Step 1: Write the failing test**

Append to `tests/lab/test_lab_corpus.py`:

```python
def test_make_distractor_memories_are_aged_low_importance_and_tagged():
    payloads = c.make_distractor_memories(3, age_days=200, importance=0.05)
    assert len(payloads) == 3
    for p in payloads:
        assert p["tags"] == ["lab-distractor"]
        assert p["importance"] == 0.05
        assert p["timestamp"] == p["last_accessed"]
        assert p["metadata"]["lab_distractor"] is True
        assert p["timestamp"].endswith("Z")


def test_inject_distractors_returns_created_ids():
    posted = []

    def fake_post(url, json=None, headers=None, timeout=None):
        posted.append(json)
        idx = len(posted)
        return FakeResp({"memory_id": f"id-{idx}"})

    payloads = c.make_distractor_memories(2)
    ids = c.inject_distractors("http://x", {}, payloads, http_post=fake_post)
    assert ids == ["id-1", "id-2"]
    assert len(posted) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_corpus.py -k distractor -v`
Expected: FAIL with `AttributeError: module 'lab_corpus' has no attribute 'make_distractor_memories'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/lab/lab_corpus.py` (add `import time` to the imports):

```python
def iso_days_ago(days: float) -> str:
    """UTC ISO-8601 timestamp `days` in the past."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days * 86400))


def make_distractor_memories(
    n: int,
    *,
    age_days: float = 180,
    importance: float = 0.05,
    tag: str = "lab-distractor",
) -> List[Dict[str, Any]]:
    """Build n aged, low-importance, labelled distractor /memory payloads.

    Pre-backdated so the `forget` consolidation mode treats them as stale; the
    `lab-distractor` tag + metadata flag make them unambiguous noise for the
    distractor-precision metric.
    """
    ts = iso_days_ago(age_days)
    payloads: List[Dict[str, Any]] = []
    for i in range(n):
        payloads.append(
            {
                "content": (
                    f"[lab-distractor #{i}] stale unrelated note about "
                    f"miscellaneous topic {i}; safe to forget."
                ),
                "tags": [tag],
                "importance": importance,
                "timestamp": ts,
                "last_accessed": ts,
                "metadata": {"lab_distractor": True},
            }
        )
    return payloads


def inject_distractors(
    api_url: str,
    headers: Dict[str, str],
    payloads: List[Dict[str, Any]],
    *,
    http_post=requests.post,
) -> List[str]:
    """POST distractor memories; return the created memory IDs."""
    ids: List[str] = []
    for payload in payloads:
        resp = http_post(f"{api_url}/memory", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        mid = str(
            data.get("memory_id")
            or data.get("id")
            or (data.get("memory") or {}).get("id")
            or ""
        )
        if mid:
            ids.append(mid)
    return ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_corpus.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lab/lab_corpus.py tests/lab/test_lab_corpus.py
git commit -m "feat(lab): add aged labelled distractor injection"
```

---

### Task 7: Add real consolidation pass (`run_consolidation`)

**Files:**
- Modify: `scripts/lab/lab_corpus.py`
- Modify: `tests/lab/test_lab_corpus.py`

**Interfaces:**
- Produces: `CONSOLIDATION_ORDER: list[str]` = `["decay", "creative", "cluster", "forget"]` (decay first so creative/cluster see `relevance_score > 0.3`).
- Produces: `run_consolidation(api_url: str, headers: dict, modes: list[str] = CONSOLIDATION_ORDER, *, dry_run: bool = False, http_post=requests.post) -> dict[str, dict]` — POSTs `/consolidate {"mode": mode, "dry_run": dry_run}` for each mode **in order**, returns `{mode: response_json}`. `dry_run` defaults to **False** (a real pass) because the endpoint's own default is True (a no-op).

- [ ] **Step 1: Write the failing test**

Append to `tests/lab/test_lab_corpus.py`:

```python
def test_run_consolidation_sends_dry_run_false_in_order():
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return FakeResp({"mode": json["mode"], "steps": {}})

    out = c.run_consolidation("http://x", {}, http_post=fake_post)
    sent_modes = [call["mode"] for call in calls]
    assert sent_modes == ["decay", "creative", "cluster", "forget"]
    assert all(call["dry_run"] is False for call in calls)
    assert set(out.keys()) == {"decay", "creative", "cluster", "forget"}


def test_run_consolidation_respects_explicit_modes():
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return FakeResp({})

    c.run_consolidation("http://x", {}, modes=["forget"], http_post=fake_post)
    assert [call["mode"] for call in calls] == ["forget"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_corpus.py -k consolidation -v`
Expected: FAIL with `AttributeError: module 'lab_corpus' has no attribute 'run_consolidation'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/lab/lab_corpus.py`:

```python
CONSOLIDATION_ORDER = ["decay", "creative", "cluster", "forget"]


def run_consolidation(
    api_url: str,
    headers: Dict[str, str],
    modes: Optional[List[str]] = None,
    *,
    dry_run: bool = False,
    http_post=requests.post,
) -> Dict[str, Dict[str, Any]]:
    """Run a REAL consolidation pass per mode, in order.

    dry_run defaults to False: the /consolidate endpoint's own default is True,
    which makes creative/cluster/forget silent no-ops. Decay runs first so
    creative/cluster see memories with relevance_score > 0.3.
    """
    if modes is None:
        modes = list(CONSOLIDATION_ORDER)
    out: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        resp = http_post(
            f"{api_url}/consolidate",
            json={"mode": mode, "dry_run": dry_run},
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        out[mode] = resp.json()
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_lab_corpus.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lab/lab_corpus.py tests/lab/test_lab_corpus.py
git commit -m "feat(lab): add real consolidation pass helper"
```

---

### Task 8: Wire the scorecard into `run_recall_test.py`

Make the runner compute the scorecard end to end: per-query distractor rate, configurable recall params, config complexity on the run, and a scorecard in the summary/saved JSON. Add CLI flags so a run can inject distractors, set recall params, and optionally consolidate.

**Files:**
- Modify: `scripts/lab/run_recall_test.py`
- Create: `tests/lab/test_run_recall_test.py`

**Interfaces:**
- Consumes: `lab_metrics.distractor_rate_at_k`, `lab_metrics.config_complexity`; `lab_corpus.recall`, `lab_corpus.extract_ids`, `lab_corpus.make_distractor_memories`, `lab_corpus.inject_distractors`, `lab_corpus.run_consolidation`.
- Produces: `QueryResult.distractor_rate_10: float` field; `TestRunResult.complexity: int` field and `TestRunResult.mean_distractor_rate_10` property; `build_scorecard(result: TestRunResult) -> dict` returning `{config_name, ndcg_10, distractor_rate_10, latency_ms, complexity}`.

- [ ] **Step 1: Write the failing test**

Create `tests/lab/test_run_recall_test.py`:

```python
import run_recall_test as rr


def test_build_scorecard_reads_the_four_axes():
    result = rr.TestRunResult(config_name="cfg")
    result.complexity = 7
    result.query_results = [
        rr.QueryResult(
            query="q1",
            expected_ids=["a"],
            retrieved_ids=["a", "d1"],
            ndcg_10=1.0,
            distractor_rate_10=0.5,
            latency_ms=100.0,
        ),
        rr.QueryResult(
            query="q2",
            expected_ids=["b"],
            retrieved_ids=["b"],
            ndcg_10=0.0,
            distractor_rate_10=0.0,
            latency_ms=200.0,
        ),
    ]
    card = rr.build_scorecard(result)
    assert card["config_name"] == "cfg"
    assert card["ndcg_10"] == 0.5
    assert card["distractor_rate_10"] == 0.25
    assert card["latency_ms"] == 150.0
    assert card["complexity"] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/test_run_recall_test.py -v`
Expected: FAIL — `QueryResult.__init__() got an unexpected keyword argument 'distractor_rate_10'` (or `AttributeError: build_scorecard`).

- [ ] **Step 3: Write minimal implementation**

In `scripts/lab/run_recall_test.py`:

(a) Extend the imports near the top:

```python
from lab_metrics import (  # noqa: E402
    config_complexity,
    distractor_rate_at_k,
    mrr,
    ndcg_at_k,
    paired_ttest,
    recall_at_k,
)
from lab_corpus import (  # noqa: E402
    extract_ids,
    inject_distractors,
    make_distractor_memories,
    recall,
    run_consolidation,
)
```

(b) Add a field to `QueryResult` (after `ndcg_10`):

```python
    distractor_rate_10: float = 0.0
```

(c) Add a field + property to `TestRunResult` (after `timestamp`):

```python
    complexity: int = 0
```

```python
    @property
    def mean_distractor_rate_10(self) -> float:
        vals = [q.distractor_rate_10 for q in self.query_results]
        return sum(vals) / len(vals) if vals else 0.0
```

(d) Replace `run_single_query` so it uses `lab_corpus.recall`/`extract_ids` and scores distractors. New signature and body:

```python
def run_single_query(
    query_data: Dict[str, Any],
    api_url: str,
    *,
    distractor_ids: Optional[set] = None,
    recall_params: Optional[Dict[str, Any]] = None,
) -> QueryResult:
    """Execute a single recall query and compute metrics (incl. distractor rate)."""
    query = query_data["query"]
    expected_ids = query_data.get("expected_ids", [])
    category = query_data.get("category", "unknown")
    distractor_ids = distractor_ids or set()
    recall_params = recall_params or {}

    start = time.perf_counter()
    try:
        data = recall(api_url, get_headers(), query, **recall_params)
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: Query failed: {query[:50]}... — {e}")
        return QueryResult(
            query=query, expected_ids=expected_ids, retrieved_ids=[], category=category
        )

    latency_ms = (time.perf_counter() - start) * 1000
    retrieved_ids = extract_ids(data)

    return QueryResult(
        query=query,
        expected_ids=expected_ids,
        retrieved_ids=retrieved_ids,
        recall_5=recall_at_k(retrieved_ids, expected_ids, 5),
        recall_10=recall_at_k(retrieved_ids, expected_ids, 10),
        recall_20=recall_at_k(retrieved_ids, expected_ids, 20),
        mrr_val=mrr(retrieved_ids, expected_ids),
        ndcg_10=ndcg_at_k(retrieved_ids, expected_ids, 10),
        distractor_rate_10=distractor_rate_at_k(retrieved_ids, distractor_ids, 10),
        latency_ms=latency_ms,
        category=category,
    )
```

(e) Update `run_test` to thread the new args and set complexity:

```python
def run_test(
    config_name: str,
    queries: List[Dict],
    api_url: str,
    *,
    config: Optional[Dict[str, str]] = None,
    distractor_ids: Optional[set] = None,
    recall_params: Optional[Dict[str, Any]] = None,
) -> TestRunResult:
    """Run all test queries with a specific config."""
    result = TestRunResult(
        config_name=config_name, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    result.complexity = config_complexity(config or {})

    for i, q in enumerate(queries):
        qr = run_single_query(
            q, api_url, distractor_ids=distractor_ids, recall_params=recall_params
        )
        result.query_results.append(qr)
        hit = "." if qr.recall_10 > 0 else "x"
        print(hit, end="", flush=True)
        if (i + 1) % 50 == 0:
            print(f" [{i + 1}/{len(queries)}]")

    print(f" [{len(queries)}/{len(queries)}]")
    return result
```

(f) Add `build_scorecard` near `print_summary`:

```python
def build_scorecard(result: TestRunResult) -> Dict[str, Any]:
    """The legible scorecard: NDCG@10 primary, distractor-rate guardrail,
    latency + complexity tiebreakers."""
    return {
        "config_name": result.config_name,
        "ndcg_10": result.mean_ndcg_10,
        "distractor_rate_10": result.mean_distractor_rate_10,
        "latency_ms": result.mean_latency,
        "complexity": result.complexity,
    }
```

(g) Add a scorecard block to `print_summary` (after the NDCG line):

```python
    print(f"  Distractor@10: {result.mean_distractor_rate_10:.3f} (lower better)")
    print(f"  Complexity:   {result.complexity} active knobs")
```

(h) Add the scorecard to the saved JSON in `save_results` (add to the `summary` dict):

```python
            "distractor_rate_10": result.mean_distractor_rate_10,
            "complexity": result.complexity,
```

(i) Add CLI flags in `main` (after the existing `--output-dir` arg):

```python
    parser.add_argument(
        "--distractors", type=int, default=0,
        help="Inject N aged labelled distractor memories before testing",
    )
    parser.add_argument("--expand-relations", action="store_true", help="Recall with expand_relations")
    parser.add_argument(
        "--no-current-only", action="store_true",
        help="Recall with current_only=false (include archived)",
    )
    parser.add_argument(
        "--recency-bias", type=str, default=None, choices=["on", "off", "auto"],
        help="Recall recency_bias override",
    )
    parser.add_argument(
        "--consolidate", action="store_true",
        help="Run a real consolidation pass (dry_run=false) before recall",
    )
```

Then, right after `queries = load_test_set(...)`, build the shared recall params + optional corpus setup:

```python
    recall_params = {
        "expand_relations": args.expand_relations,
        "current_only": not args.no_current_only,
    }
    if args.recency_bias is not None:
        recall_params["recency_bias"] = args.recency_bias

    distractor_ids: set = set()
    if args.distractors > 0:
        payloads = make_distractor_memories(args.distractors)
        distractor_ids = set(inject_distractors(args.api_url, get_headers(), payloads))
        print(f"Injected {len(distractor_ids)} distractor memories")
    if args.consolidate:
        steps = run_consolidation(args.api_url, get_headers())
        print(f"Consolidation pass complete: {list(steps.keys())}")
```

Finally, thread `config=...`, `distractor_ids=distractor_ids`, `recall_params=recall_params` into every `run_test(...)` call in `main` (the sweep, compare, and single-config branches). For example the single-config branch becomes:

```python
        result = run_test(
            args.config, queries, args.api_url,
            config=config, distractor_ids=distractor_ids, recall_params=recall_params,
        )
```

(Apply the same three kwargs to the `run_test` calls in the sweep loop and both compare-mode calls. In the sweep loop pass `config={param_name: val}`; in compare mode pass `config=config_a` / `config=config_b` respectively.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/ -v`
Expected: PASS (all lab tests).

Confirm the CLI still parses:
Run: `python scripts/lab/run_recall_test.py --help`
Expected: help shows the new `--distractors`, `--expand-relations`, `--no-current-only`, `--recency-bias`, `--consolidate` flags.

- [ ] **Step 5: Commit**

```bash
git add scripts/lab/run_recall_test.py tests/lab/test_run_recall_test.py
git commit -m "feat(lab): wire scorecard, distractors, recall params, consolidation into runner"
```

---

### Task 9: Format, lint, and full-suite green

**Files:** none new (cleanup only).

- [ ] **Step 1: Format**

Run: `black scripts/lab/ tests/lab/`
Expected: files reformatted or "All done".

- [ ] **Step 2: Lint**

Run: `flake8 scripts/lab/lab_metrics.py scripts/lab/lab_corpus.py scripts/lab/run_recall_test.py tests/lab/`
Expected: no output (clean). Fix any reported issues inline.

- [ ] **Step 3: Run the full lab suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/lab/ -v`
Expected: all PASS.

- [ ] **Step 4: Confirm no unrelated WIP got staged**

Run: `git status --short`
Expected: the WIP files from Global Constraints still show as unstaged `M`; nothing under `automem/api/` or `mcp-sse-server/` is staged.

- [ ] **Step 5: Commit (only if black/flake8 changed anything)**

```bash
git add scripts/lab/ tests/lab/
git commit -m "style(lab): black + flake8 clean"
```

---

## Self-Review

**Spec coverage (Plan A scope = spec §3 metric, §5 arms' knobs, §8 consolidation hooks, §10 lab build items):**
- NDCG@10 primary → already computed; surfaced in scorecard (Task 8). ✓
- Distractor-precision guardrail → Task 2 metric + Task 6 injection + Task 8 wiring. ✓
- Simplicity (knob-count) → Task 3 + Task 8. ✓
- Latency tiebreaker → existing `mean_latency`, surfaced in scorecard. ✓
- Decision rule → Task 4 `pick_winner`. ✓
- Recall params (expand_relations/current_only/recency_bias) → Task 5 + Task 8 flags. ✓
- Timestamp-backdate → folded into Task 6 (`iso_days_ago` + pre-aged distractors); standalone corpus-aging deferred (YAGNI — the measurable arms inject pre-aged distractors and creative/cluster need only a decay pass, not aged data). Noted as an intentional trim of spec §10.
- Consolidation-in-loop (dry_run:false, order) → Task 7 + Task 8 `--consolidate`. ✓
- Matrix-cell wiring, manifest, isolation, funnel runners, AMB work → **Plan B / Plan C** (out of scope here). ✓

**Placeholder scan:** no TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `distractor_rate_at_k(retrieved, distractors, k)`, `config_complexity(config)`, `pick_winner(cards, baseline_name=...)`, `recall(...)→dict`, `extract_ids(json)→list`, `run_consolidation(...)→dict`, `build_scorecard(result)→dict`, `QueryResult.distractor_rate_10`, `TestRunResult.complexity`/`mean_distractor_rate_10` — names used in Task 8 match their definitions in Tasks 2-7. ✓

---

## Execution Handoff

After saving the plan, the orchestrator offers the execution-mode choice (subagent-driven vs inline). Plans B and C are written next, after Plan A's interfaces are real.
