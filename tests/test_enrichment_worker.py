"""Queue-backed worker regressions: retries must survive normal deduplication."""

from queue import Empty, Queue
from threading import Lock
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from automem.enrichment.runtime_worker import enqueue_enrichment, enrichment_worker
from automem.service_state import EnrichmentJob, EnrichmentStats


class WorkerDrained(BaseException):
    """Stop the synchronous test worker without depending on timing or threads."""


class DrainQueue(Queue):
    def __init__(self):
        super().__init__()
        self.completed_counts = []
        self.on_task_done = lambda: None

    def get(self, *args, **kwargs):
        try:
            return super().get(block=False)
        except Empty:
            raise WorkerDrained

    def task_done(self):
        super().task_done()
        self.completed_counts.append(self.unfinished_tasks)
        self.on_task_done()


class InterleavingLock:
    """Run a competing enqueue immediately after the protected transition."""

    def __init__(self, after_release):
        self.lock = Lock()
        self.after_release = after_release

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, *args):
        self.lock.release()
        self.after_release()


def run_jobs(*, forced=False, failures=1, max_attempts=3, interleave_retry=False):
    state = SimpleNamespace(
        enrichment_queue=DrainQueue(),
        enrichment_pending=set(),
        enrichment_inflight=set(),
        enrichment_lock=Lock(),
        enrichment_stats=EnrichmentStats(),
    )
    calls, events, sleeps = [], [], []
    retry_transition = {"armed": False}

    def enqueue(memory_id, forced=False, attempt=0):
        enqueue_enrichment(
            state=state,
            memory_id=memory_id,
            forced=forced,
            attempt=attempt,
            enrichment_job_cls=EnrichmentJob,
        )

    def enrich(memory_id, forced=False):
        calls.append((memory_id, forced))
        assert len(calls) <= max_attempts, "worker exceeded retry bound"
        # New ordinary requests for the same memory must remain deduplicated.
        enqueue(memory_id)
        assert state.enrichment_queue.empty()
        if len(calls) <= failures:
            raise RuntimeError("temporary provider failure")
        return True

    def compete_at_retry_gap():
        if (
            retry_transition["armed"]
            and "memory-1" not in state.enrichment_pending
            and "memory-1" not in state.enrichment_inflight
        ):
            retry_transition["armed"] = False
            enqueue("memory-1")

    def sleep(seconds):
        sleeps.append(seconds)
        if interleave_retry and len(sleeps) == 1:
            retry_transition["armed"] = True

    if interleave_retry:
        state.enrichment_lock = InterleavingLock(compete_at_retry_gap)
        state.enrichment_queue.on_task_done = lambda: retry_transition.update(armed=False)

    enqueue("memory-1", forced=forced)
    enqueue("memory-1")
    assert state.enrichment_queue.qsize() == 1
    with pytest.raises(WorkerDrained):
        enrichment_worker(
            state=state,
            logger=MagicMock(),
            enrichment_idle_sleep_seconds=0,
            enrichment_max_attempts=max_attempts,
            enrichment_failure_backoff_seconds=2,
            empty_exc=Empty,
            enrich_memory_fn=enrich,
            emit_event_fn=lambda name, payload, now: events.append((name, payload)),
            utc_now_fn=lambda: "2026-10-20T00:00:00Z",
            enqueue_enrichment_fn=enqueue,
            perf_counter_fn=lambda: 0,
            sleep_fn=sleep,
        )
    assert state.enrichment_pending == set()
    assert state.enrichment_inflight == set()
    assert state.enrichment_queue.unfinished_tasks == 0
    assert state.enrichment_queue.completed_counts == [1] * (len(calls) - 1) + [0]
    return state.enrichment_stats, calls, events, sleeps


@pytest.mark.parametrize("forced", [False, True])
def test_transient_failure_retries_then_succeeds(forced):
    stats, calls, events, sleeps = run_jobs(forced=forced)
    assert calls == [("memory-1", forced)] * 2
    assert stats.failures == 1
    assert stats.successes == 1
    assert sleeps == [2]
    assert [payload["attempt"] for name, payload in events if name == "enrichment.start"] == [1, 2]
    assert [payload["will_retry"] for name, payload in events if name == "enrichment.failed"] == [
        True
    ]


def test_permanent_failure_exhausts_retry_limit():
    stats, calls, events, sleeps = run_jobs(failures=10)
    assert len(calls) == 3
    assert stats.failures == 3
    assert stats.successes == 0
    assert sleeps == [2, 2]
    assert [payload["will_retry"] for name, payload in events if name == "enrichment.failed"] == [
        True,
        True,
        False,
    ]


def test_no_retry_after_configured_single_attempt():
    stats, calls, events, sleeps = run_jobs(max_attempts=1)
    assert len(calls) == 1
    assert stats.failures == 1
    assert sleeps == []


def test_competing_enqueue_cannot_reset_attempt_counter_during_retry_transition():
    stats, calls, events, sleeps = run_jobs(failures=10, interleave_retry=True)
    assert len(calls) == 3
    assert stats.failures == 3
    assert [payload["attempt"] for name, payload in events if name == "enrichment.start"] == [
        1,
        2,
        3,
    ]
