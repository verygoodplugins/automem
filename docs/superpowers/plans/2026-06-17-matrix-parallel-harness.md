# Matrix Parallel Harness (Plan B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A parallel "matrix" harness that runs N AutoMem config variants — each in its own isolated Docker stack with the config baked in — against a corpus, scores each with the Plan A scorecard, records a provenance manifest, resumes idempotently, and selects a winner via `pick_winner`.

**Architecture:** Live in an isolated git worktree of `automem-evals` (branch `feat/matrix-parallel-harness` off `main`) so the concurrent BEAM agent's `feat/beam-judged-harness` is never touched. The harness imports the scoring primitives (`lab_metrics`, `lab_corpus`) from the automem repo — one source of truth for scoring (DRY). Pure logic (manifest, cell-keying, compose lint, config→override injection, port algebra, concurrency math, orchestration loop with injected provider/scorer) is unit-tested without Docker; one synthetic-corpus task validates the live path end to end.

**Tech Stack:** Python 3.12, `requests`, `pyyaml` (already used by matrix_stack.py), pytest. Docker Compose for stacks. No new external services.

## Global Constraints

- **Worktree:** all code changes happen in the `automem-evals` worktree at `/Users/jgarturo/Projects/OpenAI/automem-evals.worktrees/matrix-harness` on branch `feat/matrix-parallel-harness` (created off `origin/main`). **Never** check out, rebase, reset, or commit on `feat/beam-judged-harness`, and never touch that repo's primary working tree at `/Users/jgarturo/Projects/OpenAI/automem-evals`.
- **Scoring is imported, not reimplemented.** Import `lab_metrics` and `lab_corpus` from the automem repo. Resolve the path via env var `AUTOMEM_DIR` (default `/Users/jgarturo/Projects/OpenAI/automem`), appending `AUTOMEM_DIR/scripts/lab` to `sys.path`. Do NOT copy/duplicate any scoring function.
- **Each config variant = its own stack.** AutoMem reads `SEARCH_WEIGHT_*`/`RECALL_*`/`CONSOLIDATION_*` from env at import time (`automem/config.py`), so config must be baked into the per-stack compose override's `flask-api` `environment:` block at `up` time — never via a shared `.env.bench`.
- **No `container_name`, no fixed/literal host ports** in any generated compose or override. Internal ports stay fixed (`falkordb:6379`, `qdrant:6333`, api `8001`); only host bindings vary, derived deterministically from the cell index.
- **Teardown always:** every provisioned stack is torn down with `docker compose -p <project> down -v --remove-orphans` in a `finally` block.
- **Idempotent resume:** a cell is done iff its result JSON exists; the orchestrator skips done cells.
- Run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest <path> -v`.
- Bare-tag convention (`lab-distractor`); no secrets in code or fixtures.
- Format with `black .`, lint with `flake8` before each commit.

---

### Task 0: Create the isolated worktree (setup, no tests)

**Files:** none committed in this task — environment setup only.

- [ ] **Step 1: Create the worktree off origin/main without disturbing the BEAM branch**

Run (from anywhere; operates on the automem-evals git dir):

```bash
git -C /Users/jgarturo/Projects/OpenAI/automem-evals fetch origin
git -C /Users/jgarturo/Projects/OpenAI/automem-evals worktree add -b feat/matrix-parallel-harness /Users/jgarturo/Projects/OpenAI/automem-evals.worktrees/matrix-harness origin/main
```

Expected: "Preparing worktree ... HEAD is now at <sha>". If the branch already exists, use `git worktree add /Users/jgarturo/Projects/OpenAI/automem-evals.worktrees/matrix-harness feat/matrix-parallel-harness`.

- [ ] **Step 2: Verify isolation**

Run:

```bash
git -C /Users/jgarturo/Projects/OpenAI/automem-evals.worktrees/matrix-harness branch --show-current
git -C /Users/jgarturo/Projects/OpenAI/automem-evals branch --show-current
```

Expected: first prints `feat/matrix-parallel-harness`; second still prints `feat/beam-judged-harness` (the BEAM agent's working tree is untouched).

- [ ] **Step 3: Confirm the lab primitives import from the automem repo**

Run:

```bash
cd /Users/jgarturo/Projects/OpenAI/automem-evals.worktrees/matrix-harness
AUTOMEM_DIR=/Users/jgarturo/Projects/OpenAI/automem python -c "import sys; sys.path.insert(0, '/Users/jgarturo/Projects/OpenAI/automem/scripts/lab'); import lab_metrics, lab_corpus; print('ok', lab_metrics.pick_winner.__name__, lab_corpus.recall.__name__)"
```

Expected: `ok pick_winner recall`. No commit in this task.

All paths below are relative to the worktree root `/Users/jgarturo/Projects/OpenAI/automem-evals.worktrees/matrix-harness`.

---

### Task 1: Cell-keying + provenance manifest

**Files:**
- Create: `runners/matrix/__init__.py` (empty package marker)
- Create: `runners/matrix/manifest.py`
- Create: `tests/matrix/conftest.py`
- Create: `tests/matrix/test_manifest.py`

**Interfaces:**
- Produces: `cell_key(config: dict, automem_commit: str, seed: int, snapshot_id: str) -> str` — sha256 hex (16-char prefix) of the canonical JSON of those four inputs; stable across runs (sorted keys).
- Produces: `ManifestRow` dataclass: `name, key, config, automem_commit, seed, snapshot_id, scorecard (dict), status` and `to_dict()`/`from_dict()`.
- Produces: `result_path(results_dir: str, key: str) -> Path`, `is_cached(results_dir, key) -> bool`, `save_row(results_dir, row) -> Path` (temp-write + atomic rename), `load_rows(results_dir) -> list[ManifestRow]`.

- [ ] **Step 1: Write the failing test**

Create `tests/matrix/conftest.py`:

```python
import sys
from pathlib import Path

# matrix package importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# automem lab primitives importable (single source of truth for scoring)
import os
automem = os.environ.get("AUTOMEM_DIR", "/Users/jgarturo/Projects/OpenAI/automem")
sys.path.insert(0, str(Path(automem) / "scripts" / "lab"))
```

Create `tests/matrix/test_manifest.py`:

```python
from runners.matrix import manifest as mf


def test_cell_key_is_stable_and_order_independent():
    a = mf.cell_key({"A": "1", "B": "2"}, "abc123", 42, "snap1")
    b = mf.cell_key({"B": "2", "A": "1"}, "abc123", 42, "snap1")
    assert a == b
    assert a != mf.cell_key({"A": "9", "B": "2"}, "abc123", 42, "snap1")
    assert len(a) == 16


def test_round_trip_and_cache(tmp_path):
    row = mf.ManifestRow(
        name="baseline",
        key="deadbeefdeadbeef",
        config={"SEARCH_WEIGHT_VECTOR": "0.35"},
        automem_commit="abc123",
        seed=42,
        snapshot_id="snap1",
        scorecard={"name": "baseline", "ndcg_10": 0.8, "distractor_rate_10": 0.1,
                   "latency_ms": 100.0, "complexity": 5},
        status="ok",
    )
    assert mf.is_cached(str(tmp_path), row.key) is False
    mf.save_row(str(tmp_path), row)
    assert mf.is_cached(str(tmp_path), row.key) is True
    rows = mf.load_rows(str(tmp_path))
    assert len(rows) == 1
    assert rows[0].name == "baseline"
    assert rows[0].scorecard["ndcg_10"] == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'runners.matrix'`.

- [ ] **Step 3: Write minimal implementation**

Create `runners/matrix/__init__.py` (empty file).

Create `runners/matrix/manifest.py`:

```python
"""Provenance manifest + cell keying for the parallel matrix harness.

A cell is uniquely identified by (config, automem_commit, seed, snapshot_id).
Its result JSON is the durable record; existence == done (idempotent resume).
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


def cell_key(config: Dict[str, Any], automem_commit: str, seed: int, snapshot_id: str) -> str:
    payload = json.dumps(
        {"config": config, "commit": automem_commit, "seed": seed, "snapshot": snapshot_id},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class ManifestRow:
    name: str
    key: str
    config: Dict[str, Any]
    automem_commit: str
    seed: int
    snapshot_id: str
    scorecard: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "key": self.key,
            "config": self.config,
            "automem_commit": self.automem_commit,
            "seed": self.seed,
            "snapshot_id": self.snapshot_id,
            "scorecard": self.scorecard,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManifestRow":
        return cls(
            name=d["name"],
            key=d["key"],
            config=d.get("config", {}),
            automem_commit=d.get("automem_commit", ""),
            seed=d.get("seed", 0),
            snapshot_id=d.get("snapshot_id", ""),
            scorecard=d.get("scorecard", {}),
            status=d.get("status", "ok"),
        )


def result_path(results_dir: str, key: str) -> Path:
    return Path(results_dir) / f"{key}.json"


def is_cached(results_dir: str, key: str) -> bool:
    return result_path(results_dir, key).exists()


def save_row(results_dir: str, row: ManifestRow) -> Path:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    final = result_path(results_dir, row.key)
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row.to_dict(), indent=2))
    os.replace(tmp, final)
    return final


def load_rows(results_dir: str) -> List[ManifestRow]:
    p = Path(results_dir)
    if not p.exists():
        return []
    return [ManifestRow.from_dict(json.loads(f.read_text())) for f in sorted(p.glob("*.json"))]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_manifest.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add runners/matrix/__init__.py runners/matrix/manifest.py tests/matrix/conftest.py tests/matrix/test_manifest.py
git commit -m "feat(matrix): cell-keying + provenance manifest with idempotent cache"
```

---

### Task 2: Compose lint (no container_name / fixed host ports)

**Files:**
- Create: `runners/matrix/compose_lint.py`
- Create: `tests/matrix/test_compose_lint.py`

**Interfaces:**
- Produces: `lint_compose(yaml_text: str) -> list[str]` — returns a list of human-readable error strings; empty list means clean. Flags any service with a `container_name`, and any `ports` entry whose host side is a fixed literal (e.g. `"6379:6379"`) rather than a variable (`"${FALKOR_PORT}:6379"`).

- [ ] **Step 1: Write the failing test**

Create `tests/matrix/test_compose_lint.py`:

```python
from runners.matrix.compose_lint import lint_compose

CLEAN = """
services:
  falkordb:
    image: falkordb/falkordb
    ports:
      - "${FALKOR_PORT}:6379"
  flask-api:
    build: .
    ports:
      - "${API_PORT}:8001"
"""

DIRTY = """
services:
  falkordb:
    container_name: falkordb
    ports:
      - "6379:6379"
  flask-api:
    ports:
      - "${API_PORT}:8001"
"""


def test_clean_compose_has_no_errors():
    assert lint_compose(CLEAN) == []


def test_dirty_compose_flags_container_name_and_fixed_port():
    errors = lint_compose(DIRTY)
    assert any("container_name" in e for e in errors)
    assert any("6379:6379" in e for e in errors)
    assert len(errors) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_compose_lint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'runners.matrix.compose_lint'`.

- [ ] **Step 3: Write minimal implementation**

Create `runners/matrix/compose_lint.py`:

```python
"""Lint a docker-compose / override for matrix-isolation anti-patterns:
fixed container_name (breaks per-cell isolation) and literal host ports
(breaks parallel stacks). Host ports must be variables like ${API_PORT}.
"""

import re
from typing import List

import yaml

_FIXED_HOST_PORT = re.compile(r"^\s*\d+:\d+\s*$")


def lint_compose(yaml_text: str) -> List[str]:
    errors: List[str] = []
    doc = yaml.safe_load(yaml_text) or {}
    services = doc.get("services", {}) or {}
    for svc_name, svc in services.items():
        if not isinstance(svc, dict):
            continue
        if "container_name" in svc:
            errors.append(f"service '{svc_name}' sets container_name (breaks isolation)")
        for port in svc.get("ports", []) or []:
            if isinstance(port, str) and _FIXED_HOST_PORT.match(port):
                errors.append(f"service '{svc_name}' uses fixed host port '{port}'")
    return errors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_compose_lint.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add runners/matrix/compose_lint.py tests/matrix/test_compose_lint.py
git commit -m "feat(matrix): compose lint for container_name + fixed host ports"
```

---

### Task 3: Port algebra + RAM-based concurrency

**Files:**
- Create: `runners/matrix/resources.py`
- Create: `tests/matrix/test_resources.py`

**Interfaces:**
- Produces: `cell_ports(index: int, base_api: int = 18001) -> dict` — deterministic host ports for cell `index`: `{"api": base_api+index*10, "falkor": base_api+index*10+1, "falkor_ui": base_api+index*10+2, "qdrant": base_api+index*10+3}`. (Stride 10 leaves headroom and keeps blocks non-overlapping for index < 10 internal offsets.)
- Produces: `max_concurrency(total_gb: float, per_stack_gb: float, headroom: float = 0.8) -> int` — `max(1, floor(total_gb*headroom / per_stack_gb))`.

- [ ] **Step 1: Write the failing test**

Create `tests/matrix/test_resources.py`:

```python
from runners.matrix import resources as r


def test_cell_ports_are_deterministic_and_non_overlapping():
    p0 = r.cell_ports(0)
    p1 = r.cell_ports(1)
    assert p0 == {"api": 18001, "falkor": 18002, "falkor_ui": 18003, "qdrant": 18004}
    assert p1["api"] == 18011
    # no overlap between adjacent cells' blocks
    assert set(p0.values()).isdisjoint(set(p1.values()))


def test_max_concurrency_respects_headroom_and_floor():
    assert r.max_concurrency(80, 5) == 12       # floor(80*0.8/5)=12
    assert r.max_concurrency(4, 5) == 1          # floor=0 -> floored to 1
    assert r.max_concurrency(80, 5, headroom=0.5) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_resources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'runners.matrix.resources'`.

- [ ] **Step 3: Write minimal implementation**

Create `runners/matrix/resources.py`:

```python
"""Deterministic host-port allocation and RAM-based concurrency for the matrix."""

import math
from typing import Dict


def cell_ports(index: int, base_api: int = 18001) -> Dict[str, int]:
    base = base_api + index * 10
    return {"api": base, "falkor": base + 1, "falkor_ui": base + 2, "qdrant": base + 3}


def max_concurrency(total_gb: float, per_stack_gb: float, headroom: float = 0.8) -> int:
    if per_stack_gb <= 0:
        return 1
    return max(1, math.floor(total_gb * headroom / per_stack_gb))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_resources.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add runners/matrix/resources.py tests/matrix/test_resources.py
git commit -m "feat(matrix): deterministic port algebra + RAM concurrency cap"
```

---

### Task 4: Bake config env into a per-stack compose override

**Files:**
- Create: `runners/matrix/override.py`
- Create: `tests/matrix/test_override.py`

**Interfaces:**
- Produces: `build_override(ports: dict, config: dict) -> dict` — returns a docker-compose override dict that (a) maps host ports via the `ports` dict using variable-free explicit host:container strings derived from `ports` (e.g. `f"{ports['falkor']}:6379"`), and (b) injects every key/value of `config` into the `flask-api` service's `environment:` mapping (values stringified). Internal container ports stay 6379/6333/8001.
- Produces: `render_override(ports: dict, config: dict) -> str` — YAML text of the override (passes `compose_lint` only for `container_name`; note literal host ports are expected here because this override is per-cell and the lint's fixed-port rule is for the *base* compose — see test).

- [ ] **Step 1: Write the failing test**

Create `tests/matrix/test_override.py`:

```python
import yaml

from runners.matrix.override import build_override


def test_build_override_bakes_config_into_flask_env():
    ports = {"api": 18011, "falkor": 18012, "falkor_ui": 18013, "qdrant": 18014}
    config = {"SEARCH_WEIGHT_VECTOR": "0.5", "RECALL_RELEVANCE_GATE": 0.2}
    ov = build_override(ports, config)

    env = ov["services"]["flask-api"]["environment"]
    assert env["SEARCH_WEIGHT_VECTOR"] == "0.5"
    assert env["RECALL_RELEVANCE_GATE"] == "0.2"  # stringified

    # host ports mapped from the ports dict; internal ports fixed
    assert f"{ports['api']}:8001" in ov["services"]["flask-api"]["ports"]
    assert f"{ports['falkor']}:6379" in ov["services"]["falkordb"]["ports"]
    assert f"{ports['qdrant']}:6333" in ov["services"]["qdrant"]["ports"]

    # no container_name anywhere
    assert "container_name" not in yaml.dump(ov)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_override.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'runners.matrix.override'`.

- [ ] **Step 3: Write minimal implementation**

Create `runners/matrix/override.py`:

```python
"""Per-stack docker-compose override: maps host ports for this cell and bakes
the cell's config into the flask-api environment (AutoMem reads config at boot).
"""

from typing import Any, Dict

import yaml


def build_override(ports: Dict[str, int], config: Dict[str, Any]) -> Dict[str, Any]:
    environment = {str(k): str(v) for k, v in config.items()}
    environment.setdefault("PORT", "8001")
    return {
        "services": {
            "falkordb": {"ports": [f"{ports['falkor']}:6379", f"{ports['falkor_ui']}:3000"]},
            "qdrant": {"ports": [f"{ports['qdrant']}:6333"]},
            "flask-api": {
                "ports": [f"{ports['api']}:8001"],
                "environment": environment,
            },
        }
    }


def render_override(ports: Dict[str, int], config: Dict[str, Any]) -> str:
    return yaml.safe_dump(build_override(ports, config), sort_keys=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_override.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add runners/matrix/override.py tests/matrix/test_override.py
git commit -m "feat(matrix): per-stack override bakes config env + cell ports"
```

---

### Task 5: Score a stack (compose lab_corpus + lab_metrics)

**Files:**
- Create: `runners/matrix/score.py`
- Create: `tests/matrix/test_score.py`

**Interfaces:**
- Consumes: `lab_corpus.recall`, `lab_corpus.extract_ids` (imported from automem repo via the conftest path).
- Consumes: `lab_metrics.ndcg_at_k`, `lab_metrics.distractor_rate_at_k`, `lab_metrics.config_complexity`.
- Produces: `score_stack(api_url, headers, queries, *, distractor_ids=set(), recall_params=None, config=None, recall_fn=lab_corpus.recall) -> dict` — runs every query through `recall_fn`, computes mean NDCG@10, mean distractor_rate@10, mean latency (wall-clock per call), and `config_complexity(config)`; returns a scorecard dict with keys `name`-less core (`ndcg_10, distractor_rate_10, latency_ms, complexity`). `recall_fn` is injectable for tests. The caller adds `name`.

- [ ] **Step 1: Write the failing test**

Create `tests/matrix/test_score.py`:

```python
from runners.matrix.score import score_stack


def make_fake_recall(mapping):
    """mapping: query -> list of result ids."""
    def _recall(api_url, headers, query, **kwargs):
        return {"results": [{"memory": {"id": i}} for i in mapping[query]]}
    return _recall


def test_score_stack_computes_scorecard_axes():
    queries = [
        {"query": "q1", "expected_ids": ["a"], "category": "Decision"},
        {"query": "q2", "expected_ids": ["b"], "category": "Decision"},
    ]
    fake = make_fake_recall({"q1": ["a", "d1"], "q2": ["x", "b"]})
    config = {"SEARCH_WEIGHT_VECTOR": "0.35", "SEARCH_WEIGHT_KEYWORD": "0.0"}
    card = score_stack(
        "http://x", {}, queries,
        distractor_ids={"d1"}, config=config, recall_fn=fake,
    )
    # q1: a at rank1 -> ndcg 1.0 ; q2: b at rank2 -> ndcg ~0.63
    assert 0.7 < card["ndcg_10"] < 0.85
    # q1 has 1 distractor in top10 of 2 results = 0.5 ; q2 has 0 -> mean 0.25
    assert card["distractor_rate_10"] == 0.25
    assert card["complexity"] == 1  # only one nonzero weight
    assert "latency_ms" in card
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'runners.matrix.score'`.

- [ ] **Step 3: Write minimal implementation**

Create `runners/matrix/score.py`:

```python
"""Score one configured stack with the Plan A scorecard primitives.

Imports lab_corpus / lab_metrics from the automem repo (single source of truth).
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import os

_AUTOMEM = os.environ.get("AUTOMEM_DIR", "/Users/jgarturo/Projects/OpenAI/automem")
sys.path.insert(0, str(Path(_AUTOMEM) / "scripts" / "lab"))

import lab_corpus  # noqa: E402
import lab_metrics  # noqa: E402


def score_stack(
    api_url: str,
    headers: Dict[str, str],
    queries: List[Dict[str, Any]],
    *,
    distractor_ids: Optional[Iterable[str]] = None,
    recall_params: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    recall_fn=lab_corpus.recall,
) -> Dict[str, Any]:
    distractor_ids = set(distractor_ids or set())
    recall_params = recall_params or {}
    ndcgs: List[float] = []
    drates: List[float] = []
    latencies: List[float] = []

    for q in queries:
        start = time.perf_counter()
        data = recall_fn(api_url, headers, q["query"], **recall_params)
        latencies.append((time.perf_counter() - start) * 1000)
        retrieved = lab_corpus.extract_ids(data)
        ndcgs.append(lab_metrics.ndcg_at_k(retrieved, q.get("expected_ids", []), 10))
        drates.append(lab_metrics.distractor_rate_at_k(retrieved, distractor_ids, 10))

    n = max(len(queries), 1)
    return {
        "ndcg_10": sum(ndcgs) / n,
        "distractor_rate_10": sum(drates) / n,
        "latency_ms": sum(latencies) / n,
        "complexity": lab_metrics.config_complexity(config or {}),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_score.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add runners/matrix/score.py tests/matrix/test_score.py
git commit -m "feat(matrix): score_stack composing lab_corpus + lab_metrics"
```

---

### Task 6: Orchestrator loop (resume + winner, provider/scorer injected)

The orchestration logic — iterate configs, skip cached cells, score the rest, save manifest rows, select a winner — is unit-tested with an injected fake provider and fake scorer (no Docker). The real provider (Task 7) plugs into the same interface.

**Files:**
- Create: `runners/matrix/orchestrator.py`
- Create: `tests/matrix/test_orchestrator.py`

**Interfaces:**
- Consumes: `manifest.cell_key`, `manifest.is_cached`, `manifest.save_row`, `manifest.load_rows`, `manifest.ManifestRow`; `lab_metrics.pick_winner`.
- Produces: `run_matrix(configs, *, results_dir, automem_commit, snapshot_id, seed, baseline_name, provision, score, teardown) -> dict` where:
  - `configs`: list of `{"name": str, "config": dict}`.
  - `provision(name, config) -> str` returns an api_url (real impl: up stack + restore); `score(api_url, name, config) -> dict` returns a scorecard core (no `name`); `teardown(name) -> None`. All three are injected.
  - Behavior: for each config, compute `key`; if `is_cached`, skip (load existing). Else `provision` → `score` → save `ManifestRow` (scorecard gets `name` added) → `teardown` in `finally`. After all, load all rows, build cards (`{**scorecard, "name": name}`), return `{"winner": pick_winner(cards, baseline_name=baseline_name), "rows": [...]}`.

- [ ] **Step 1: Write the failing test**

Create `tests/matrix/test_orchestrator.py`:

```python
from runners.matrix.orchestrator import run_matrix


def test_run_matrix_scores_resumes_and_picks_winner(tmp_path):
    configs = [
        {"name": "baseline", "config": {"SEARCH_WEIGHT_VECTOR": "0.35"}},
        {"name": "simpler", "config": {"SEARCH_WEIGHT_VECTOR": "0.35"}},
    ]
    scores = {
        "baseline": {"ndcg_10": 0.80, "distractor_rate_10": 0.10, "latency_ms": 100.0, "complexity": 5},
        "simpler": {"ndcg_10": 0.801, "distractor_rate_10": 0.10, "latency_ms": 90.0, "complexity": 3},
    }
    provisioned, torn = [], []

    def provision(name, config):
        provisioned.append(name)
        return f"http://stack/{name}"

    def score(api_url, name, config):
        return scores[name]

    def teardown(name):
        torn.append(name)

    out = run_matrix(
        configs, results_dir=str(tmp_path), automem_commit="abc", snapshot_id="snap",
        seed=42, baseline_name="baseline", provision=provision, score=score, teardown=teardown,
    )
    assert out["winner"]["name"] == "simpler"  # within ndcg tol, fewer knobs + faster
    assert set(torn) == {"baseline", "simpler"}  # always torn down

    # Resume: second run scores nothing (all cached) but still picks winner.
    provisioned.clear()
    out2 = run_matrix(
        configs, results_dir=str(tmp_path), automem_commit="abc", snapshot_id="snap",
        seed=42, baseline_name="baseline", provision=provision, score=score, teardown=teardown,
    )
    assert provisioned == []  # nothing re-provisioned
    assert out2["winner"]["name"] == "simpler"


def test_run_matrix_tears_down_on_score_failure(tmp_path):
    torn = []

    def provision(name, config):
        return "http://x"

    def score(api_url, name, config):
        raise RuntimeError("boom")

    def teardown(name):
        torn.append(name)

    out = run_matrix(
        [{"name": "baseline", "config": {}}], results_dir=str(tmp_path),
        automem_commit="abc", snapshot_id="snap", seed=1, baseline_name="baseline",
        provision=provision, score=score, teardown=teardown,
    )
    assert torn == ["baseline"]  # finally ran despite the failure
    assert out["rows"][0].status == "error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'runners.matrix.orchestrator'`.

- [ ] **Step 3: Write minimal implementation**

Create `runners/matrix/orchestrator.py`:

```python
"""Sequential matrix orchestrator: provision -> score -> teardown per config,
idempotent resume via the manifest, winner via lab_metrics.pick_winner.

Concurrency is intentionally left to the caller/runner; this core is sequential
and deterministic so it is fully unit-testable with injected provider/scorer.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import os

from . import manifest as mf

_AUTOMEM = os.environ.get("AUTOMEM_DIR", "/Users/jgarturo/Projects/OpenAI/automem")
sys.path.insert(0, str(Path(_AUTOMEM) / "scripts" / "lab"))
import lab_metrics  # noqa: E402


def run_matrix(
    configs: List[Dict[str, Any]],
    *,
    results_dir: str,
    automem_commit: str,
    snapshot_id: str,
    seed: int,
    baseline_name: str,
    provision: Callable[[str, Dict[str, Any]], str],
    score: Callable[[str, str, Dict[str, Any]], Dict[str, Any]],
    teardown: Callable[[str], None],
) -> Dict[str, Any]:
    for entry in configs:
        name, config = entry["name"], entry["config"]
        key = mf.cell_key(config, automem_commit, seed, snapshot_id)
        if mf.is_cached(results_dir, key):
            continue
        status, scorecard = "ok", {}
        try:
            api_url = provision(name, config)
            scorecard = score(api_url, name, config)
        except Exception as e:  # noqa: BLE001
            status = "error"
            scorecard = {"error": str(e)}
        finally:
            try:
                teardown(name)
            except Exception:  # noqa: BLE001
                pass
        mf.save_row(
            results_dir,
            mf.ManifestRow(
                name=name, key=key, config=config, automem_commit=automem_commit,
                seed=seed, snapshot_id=snapshot_id,
                scorecard={**scorecard, "name": name}, status=status,
            ),
        )

    rows = mf.load_rows(results_dir)
    cards = [r.scorecard for r in rows if r.status == "ok" and "ndcg_10" in r.scorecard]
    winner = lab_metrics.pick_winner(cards, baseline_name=baseline_name) if cards else None
    return {"winner": winner, "rows": rows}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_orchestrator.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add runners/matrix/orchestrator.py tests/matrix/test_orchestrator.py
git commit -m "feat(matrix): orchestrator loop with resume + winner selection"
```

---

### Task 7: Live provider + synthetic-corpus smoke test

Wire the real `provision`/`score`/`teardown` to `matrix_stack.py` + a synthetic corpus, and prove the whole harness works end to end on real Docker WITHOUT needing the production clone. This task is validated by running, not by a pure unit test.

**Files:**
- Create: `runners/matrix/live.py`
- Create: `runners/matrix/smoke.py`
- Create: `tests/matrix/test_live_helpers.py`

**Interfaces:**
- Consumes: `runners/matrix/resources.cell_ports`, `runners/matrix/override.render_override`, `manifest`, `score.score_stack`.
- Produces (pure, unit-tested): `compose_up_cmd(project, automem_dir, override_path, ports) -> list[str]` and `compose_down_cmd(project) -> list[str]` — the exact docker compose argv (so the argv construction is testable without Docker).
- Produces (live, smoke-only): `LiveProvider(automem_dir, base_api).provision/score/teardown`; `smoke.main()` seeds ~10 synthetic memories + 5 distractors into each stack, scores 2 configs (a baseline and a simpler one), and writes a manifest under `data/results/matrix-smoke/`.

- [ ] **Step 1: Write the failing test (pure argv construction)**

Create `tests/matrix/test_live_helpers.py`:

```python
from runners.matrix.live import compose_up_cmd, compose_down_cmd


def test_compose_up_cmd_uses_project_and_both_files():
    ports = {"api": 18001, "falkor": 18002, "falkor_ui": 18003, "qdrant": 18004}
    cmd = compose_up_cmd("automem_eval_baseline", "/repo/automem", "/tmp/ov.yml", ports)
    assert cmd[:2] == ["docker", "compose"]
    assert "-p" in cmd and "automem_eval_baseline" in cmd
    assert "/repo/automem/docker-compose.yml" in cmd
    assert "/tmp/ov.yml" in cmd
    assert cmd[-2:] == ["up", "-d"]


def test_compose_down_cmd_removes_volumes():
    cmd = compose_down_cmd("automem_eval_baseline")
    assert "down" in cmd and "-v" in cmd and "--remove-orphans" in cmd
    assert "automem_eval_baseline" in cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_live_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'runners.matrix.live'`.

- [ ] **Step 3: Write minimal implementation**

Create `runners/matrix/live.py`:

```python
"""Live provider: provision an isolated AutoMem stack with a baked config,
score it, tear it down. Used by the matrix orchestrator for real runs.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from . import override as ov_mod
from . import resources
from . import score as score_mod

AUTOMEM_DIR = os.environ.get("AUTOMEM_DIR", "/Users/jgarturo/Projects/OpenAI/automem")
API_TOKEN = os.environ.get("AUTOMEM_API_TOKEN", "benchmark-token")


def _project(name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in name.lower())
    return f"automem_eval_{safe}"


def compose_up_cmd(project: str, automem_dir: str, override_path: str, ports: Dict[str, int]) -> List[str]:
    return [
        "docker", "compose", "-p", project,
        "-f", str(Path(automem_dir) / "docker-compose.yml"),
        "-f", override_path,
        "up", "-d",
    ]


def compose_down_cmd(project: str) -> List[str]:
    return ["docker", "compose", "-p", project, "down", "-v", "--remove-orphans"]


def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}


class LiveProvider:
    def __init__(self, automem_dir: str = AUTOMEM_DIR, base_api: int = 18001):
        self.automem_dir = automem_dir
        self.base_api = base_api
        self._ports: Dict[str, Dict[str, int]] = {}
        self._index = 0

    def provision(self, name: str, config: Dict[str, Any]) -> str:
        ports = resources.cell_ports(self._index, base_api=self.base_api)
        self._index += 1
        self._ports[name] = ports
        ov_path = Path(tempfile.gettempdir()) / f"matrix-override-{_project(name)}.yml"
        ov_path.write_text(ov_mod.render_override(ports, config))
        subprocess.run(
            compose_up_cmd(_project(name), self.automem_dir, str(ov_path), ports),
            check=True, capture_output=True, text=True,
        )
        url = f"http://localhost:{ports['api']}"
        self._wait_healthy(url)
        return url

    def _wait_healthy(self, url: str, timeout: int = 120) -> None:
        last = None
        for _ in range(timeout):
            try:
                if requests.get(f"{url}/health", timeout=2).status_code == 200:
                    return
            except Exception as e:  # noqa: BLE001
                last = e
            time.sleep(1)
        raise TimeoutError(f"stack at {url} never became healthy: {last}")

    def score(self, api_url: str, name: str, config: Dict[str, Any], queries=None,
              distractor_ids=None, recall_params=None) -> Dict[str, Any]:
        return score_mod.score_stack(
            api_url, _headers(), queries or [],
            distractor_ids=distractor_ids, recall_params=recall_params, config=config,
        )

    def teardown(self, name: str) -> None:
        subprocess.run(compose_down_cmd(_project(name)), check=False, capture_output=True, text=True)
```

Create `runners/matrix/smoke.py`:

```python
"""End-to-end smoke: 2 stacks on a tiny synthetic corpus, no production clone.

Run:
  AUTOMEM_DIR=/Users/jgarturo/Projects/OpenAI/automem \
  python -m runners.matrix.smoke
"""

import sys
from pathlib import Path

import os

from . import live as live_mod
from . import orchestrator
from . import score as score_mod

_AUTOMEM = os.environ.get("AUTOMEM_DIR", "/Users/jgarturo/Projects/OpenAI/automem")
sys.path.insert(0, str(Path(_AUTOMEM) / "scripts" / "lab"))
import lab_corpus  # noqa: E402

SYNTH_MEMORIES = [
    {"content": f"Synthetic memory {i}: project alpha decision about topic {i}.",
     "tags": ["smoke"], "importance": 0.6} for i in range(10)
]


def _seed(api_url, headers):
    ids = []
    for m in SYNTH_MEMORIES:
        r = lab_corpus.requests.post(f"{api_url}/memory", json=m, headers=headers, timeout=30)
        r.raise_for_status()
        d = r.json()
        ids.append(str(d.get("memory_id") or d.get("id") or (d.get("memory") or {}).get("id")))
    return ids


def main() -> int:
    headers = live_mod._headers()
    provider = live_mod.LiveProvider(automem_dir=_AUTOMEM, base_api=18001)
    results_dir = "data/results/matrix-smoke"

    # Build queries + distractors lazily per stack inside score.
    def provision(name, config):
        url = provider.provision(name, config)
        seeded = _seed(url, headers)
        distractors = set(lab_corpus.inject_distractors(
            url, headers, lab_corpus.make_distractor_memories(5)))
        provider._smoke = {"queries": [
            {"query": SYNTH_MEMORIES[i]["content"][:40], "expected_ids": [seeded[i]],
             "category": "smoke"} for i in range(len(seeded))],
            "distractors": distractors}
        return url

    def score(api_url, name, config):
        s = provider._smoke
        return score_mod.score_stack(api_url, headers, s["queries"],
                                     distractor_ids=s["distractors"], config=config)

    out = orchestrator.run_matrix(
        [
            {"name": "baseline", "config": {"SEARCH_WEIGHT_VECTOR": "0.35", "SEARCH_WEIGHT_KEYWORD": "0.35"}},
            {"name": "simpler", "config": {"SEARCH_WEIGHT_VECTOR": "0.35", "SEARCH_WEIGHT_KEYWORD": "0.0"}},
        ],
        results_dir=results_dir, automem_commit="smoke", snapshot_id="synthetic",
        seed=42, baseline_name="baseline",
        provision=provision, score=score, teardown=provider.teardown,
    )
    print("WINNER:", out["winner"])
    print("ROWS:", [(r.name, r.status, r.scorecard.get("ndcg_10")) for r in out["rows"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the pure test, then the live smoke**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/test_live_helpers.py -v`
Expected: PASS (2 tests).

Then the live smoke (requires Docker; uses synthetic data only):

Run: `AUTOMEM_DIR=/Users/jgarturo/Projects/OpenAI/automem python -m runners.matrix.smoke`
Expected: two stacks come up on ports 18001/18011 blocks, seed + score, print a `WINNER:` line with a `name` and a `reason`, and `ROWS:` showing both configs with non-null `ndcg_10`. A manifest appears under `data/results/matrix-smoke/`. Both stacks are torn down (verify `docker ps` shows no `automem_eval_*` containers afterward).

If the smoke reveals an integration issue (port collision, health timeout, env not applied), fix it in `live.py`/`override.py` and re-run until the WINNER line prints and `docker ps` is clean. Report the exact `WINNER:`/`ROWS:` output.

- [ ] **Step 5: Commit**

```bash
git add runners/matrix/live.py runners/matrix/smoke.py tests/matrix/test_live_helpers.py
git commit -m "feat(matrix): live provider + synthetic-corpus end-to-end smoke"
```

---

### Task 8: Format, lint, full-suite green, and a runnable entrypoint doc

**Files:**
- Create: `runners/matrix/README.md`

- [ ] **Step 1: Format + lint**

Run:
```bash
black runners/matrix/ tests/matrix/
flake8 runners/matrix/ tests/matrix/
```
Expected: black "All done"/unchanged; flake8 no output. Fix any issue inline.

- [ ] **Step 2: Full matrix suite green**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/matrix/ -v`
Expected: all PASS (manifest, compose_lint, resources, override, score, orchestrator, live_helpers).

- [ ] **Step 3: Write the entrypoint doc**

Create `runners/matrix/README.md` documenting: the worktree it lives in; that scoring is imported from `AUTOMEM_DIR`; how to run the smoke (`python -m runners.matrix.smoke`); the manifest/resume model (delete `data/results/<dir>/<key>.json` to force a re-run of one cell); the isolation rules (config baked into the override env; ports via `cell_ports`); and the path to a future full-corpus run (swap the synthetic `_seed` for `clone_production.sh --restore-only` into each stack, supply the real query set from `create_test_queries.py`, and set a real `automem_commit`/`snapshot_id`). Note explicitly that the full-corpus run needs production backup access and is a separate, supervised step.

- [ ] **Step 4: Commit**

```bash
git add runners/matrix/README.md
git commit -m "docs(matrix): entrypoint, isolation model, path to full-corpus run"
```

---

## Self-Review

**Spec coverage (Plan B scope = spec §2 funnel Tier-1, §7 orchestration):**
- Isolated stacks, project naming, dynamic ports, no container_name/fixed ports → Task 3 (ports) + Task 4 (override) + Task 2 (lint) + Task 7 (live). ✓
- Config baked per stack (boot-time env) → Task 4 + Task 7. ✓
- Provenance manifest (config+commit+seed+snapshot) → Task 1. ✓
- Idempotent resume → Task 1 (`is_cached`) + Task 6 (skip logic, tested). ✓
- RAM-based concurrency cap → Task 3 (`max_concurrency`); applied by the runner/caller (the orchestrator core is sequential by design; the concurrency cap is a pure helper the live runner uses to bound waves). Noted: wave-parallel execution is a thin wrapper over the sequential core, deferred to the full-corpus run where it matters.
- Scoring via lab primitives (one source of truth) → Task 5. ✓
- Winner via pick_winner → Task 6. ✓
- Teardown always → Task 6 (`finally`, tested) + Task 7 (`down -v --remove-orphans`). ✓
- Tier-2 usefulness gate + Tier-3 benchmark confirm → out of scope (later plan / Plan C). ✓
- Worktree isolation from the BEAM agent → Task 0. ✓

**Placeholder scan:** no TBD/TODO; pure-logic steps show complete code; the live step gives exact run commands + expected output. ✓

**Type consistency:** `cell_ports`→dict consumed by `build_override`/`compose_up_cmd`; `score_stack`→dict (no `name`) consumed by `run_matrix` which adds `name` before `pick_winner` (which reads `name`) — matches the Plan A contract fixed in commit "align scorecard key contract". `provision/score/teardown` signatures in Task 6 tests match `LiveProvider` in Task 7. ✓

**Scope check:** one coherent deliverable — a parallel corpus-sweep harness that runs configs and picks a winner, smoke-validated on synthetic data. Full-corpus run and the upper funnel tiers are explicitly deferred.

---

## Execution Handoff

Execute via subagent-driven-development in the worktree. Task 0 (worktree) and Task 7-Step-4 (live smoke) are the only steps touching real Docker; all other steps are pure-unit-testable. After Plan B is green, the remaining work is the supervised full-corpus Tier-1 run and Plan C (AMB submission integrity).
