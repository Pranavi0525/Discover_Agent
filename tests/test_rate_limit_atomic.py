"""Phase 2 / M2.2 abuse limits: atomic slot reservation (no network).

Covers the three limits (per session, global, concurrent) at the storage level, and
the service wiring: rejected requests never reach the planner, finished or failed runs
free their slot, and concurrent requests cannot overshoot a limit.
"""

import threading
from datetime import datetime, timedelta, timezone

import pytest

from discovery_agent import storage
from discovery_agent.storage import SqliteStore


class Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now.isoformat()

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


@pytest.fixture()
def store(tmp_path):
    return SqliteStore(tmp_path / "agent.db")


@pytest.fixture()
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(storage, "_now", c)
    return c


def _rows(store):
    import sqlite3
    with sqlite3.connect(store.path) as conn:
        return conn.execute("SELECT COUNT(*) FROM rate_slots").fetchone()[0]


def _run_threads(n, fn):
    barrier = threading.Barrier(n)
    out, lock = [], threading.Lock()

    def worker(i):
        barrier.wait()
        r = fn(i)
        with lock:
            out.append(r)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out


def test_concurrent_requests_cannot_overshoot_the_session_limit(store):
    sid = store.create_session()
    results = _run_threads(20, lambda i: store.reserve_slot(sid, session_max=5, session_window=60))
    assert sum(1 for slot, _ in results if slot is not None) == 5
    assert {reason for slot, reason in results if slot is None} == {"session"}


def test_global_limit_holds_across_sessions(store):
    sessions = [store.create_session() for _ in range(10)]
    results = _run_threads(
        10, lambda i: store.reserve_slot(sessions[i], session_max=5, global_max=4, global_window=60)
    )
    assert sum(1 for slot, _ in results if slot is not None) == 4
    assert {reason for slot, reason in results if slot is None} == {"global"}


def test_a_new_session_does_not_reset_the_global_cap(store):
    for _ in range(3):
        assert store.reserve_slot(store.create_session(), global_max=3)[0] is not None
    assert store.reserve_slot(store.create_session(), session_max=5, global_max=3) == (None, "global")


def test_concurrent_cap_counts_running_runs_and_frees_on_finish(store, clock):
    sid = store.create_session()
    a, _ = store.reserve_slot(sid, max_concurrent=2)
    b, _ = store.reserve_slot(sid, max_concurrent=2)
    assert store.reserve_slot(sid, max_concurrent=2) == (None, "concurrent")
    store.finish_slot(a)
    assert store.reserve_slot(sid, max_concurrent=2)[0] is not None
    assert b is not None


def test_finishing_does_not_refund_the_window_caps(store, clock):
    sid = store.create_session()
    for _ in range(3):
        slot, _ = store.reserve_slot(sid, session_max=3)
        store.finish_slot(slot)
    assert store.reserve_slot(sid, session_max=3) == (None, "session")
    clock.advance(61)
    assert store.reserve_slot(sid, session_max=3)[0] is not None


def test_a_crashed_run_stops_blocking_after_the_ttl(store, clock):
    sid = store.create_session()
    assert store.reserve_slot(sid, max_concurrent=1, stale_after=300)[0] is not None
    assert store.reserve_slot(sid, max_concurrent=1, stale_after=300) == (None, "concurrent")
    clock.advance(301)
    assert store.reserve_slot(sid, max_concurrent=1, stale_after=300)[0] is not None


def test_a_rejected_request_records_nothing(store):
    sid = store.create_session()
    store.reserve_slot(sid, session_max=1)
    for _ in range(5):
        assert store.reserve_slot(sid, session_max=1) == (None, "session")
    assert _rows(store) == 1


def test_zero_switches_every_limit_off(store):
    sid = store.create_session()
    assert all(store.reserve_slot(sid)[0] is not None for _ in range(50))


def test_slots_survive_a_restart(tmp_path):
    first = SqliteStore(tmp_path / "agent.db")
    sid = first.create_session()
    for _ in range(2):
        first.reserve_slot(sid, session_max=2)
    second = SqliteStore(tmp_path / "agent.db")  # new process, same file
    assert second.reserve_slot(sid, session_max=2) == (None, "session")


# --- service wiring (needs the graph's dependencies) -------------------------------
service = pytest.importorskip("discovery_agent.service")
from discovery_agent.execution import ShopifyMcpExecutor  # noqa: E402

from test_graph import FakeMcpSession, FakePlanner, _search_plan, _settings  # noqa: E402


@pytest.fixture()
def env(tmp_path):
    settings = _settings()
    settings.ledger_path = tmp_path / "ledger.jsonl"
    settings.dedup_window_seconds = 120
    settings.rate_limit_max_runs = 5
    settings.rate_limit_window_seconds = 60
    settings.rate_limit_global_max_runs = 0
    settings.max_concurrent_runs = 0
    store = SqliteStore(tmp_path / "agent.db")
    return settings, store


def _go(settings, store, sid, query, planner=None):
    planner = planner or FakePlanner([_search_plan(query)] * 3)
    executor = ShopifyMcpExecutor(
        session_factory=lambda: FakeMcpSession({query: [{"id": "p1"}]}), settings=settings
    )
    return service.discover(query, sid, None, store=store, settings=settings,
                            planner=planner, executor=executor), planner


class SlowPlanner(FakePlanner):
    """Holds the run open so concurrent requests overlap."""

    def plan(self, query):
        import time
        time.sleep(0.05)
        return super().plan(query)


def test_service_concurrent_distinct_requests_respect_the_session_limit(env):
    settings, store = env
    sid = store.create_session()

    def go(i):
        q = f"running shoes {i}"
        return _go(settings, store, sid, q, SlowPlanner([_search_plan(q)] * 3))

    results = _run_threads(12, go)
    ran = [r for r, p in results if r.status != "rate_limited"]
    assert len(ran) == 5
    assert sum(1 for r, p in results if r.status == "rate_limited" and not p.calls) == 7


def test_service_global_limit_rejects_without_planning_and_stays_retryable(env):
    settings, store = env
    settings.rate_limit_global_max_runs = 2
    for _ in range(2):
        r, _p = _go(settings, store, store.create_session(), "running shoes")
    sid = store.create_session()
    limited, planner = _go(settings, store, sid, "blue shirt", FakePlanner([]))
    assert limited.status == "rate_limited" and limited.error == service.BUSY_MESSAGE
    assert limited.run_id is None and planner.calls == []
    settings.rate_limit_global_max_runs = 0  # operator raises the cap: the same request now runs
    again, planner = _go(settings, store, sid, "blue shirt")
    assert again.status in {"selected", "needs_selection", "clarify"} and planner.calls


def test_service_frees_the_concurrent_slot_after_a_failed_run(env):
    settings, store = env
    settings.max_concurrent_runs = 1
    sid = store.create_session()

    class Boom:
        calls = []

        def plan(self, query):
            raise RuntimeError("planner down")

    failed, _ = _go(settings, store, sid, "running shoes", Boom())
    assert failed.status == "error"
    ok, planner = _go(settings, store, sid, "blue shirt")
    assert ok.status != "rate_limited" and planner.calls


def test_service_concurrent_cap_rejects_overlapping_runs(env):
    settings, store = env
    settings.rate_limit_max_runs = 0
    settings.max_concurrent_runs = 2
    sid = store.create_session()

    def go(i):
        q = f"running shoes {i}"
        return _go(settings, store, sid, q, SlowPlanner([_search_plan(q)] * 3))

    results = _run_threads(8, go)
    assert sum(1 for r, p in results if r.status != "rate_limited") >= 2
    assert all(r.error == service.BUSY_MESSAGE for r, p in results if r.status == "rate_limited")
    assert store.reserve_slot(sid, max_concurrent=1)[0] is not None  # every slot was released
