import json
import sqlite3
import threading

import pytest

from discovery_agent.storage import (
    DB_PATH_ENV,
    SqliteRunTracker,
    SqliteStore,
    get_db_path,
    init_db,
)


@pytest.fixture()
def store(tmp_path):
    return SqliteStore(tmp_path / "nested" / "agent.db")


@pytest.fixture()
def session_id(store):
    return store.create_session()


def test_init_db_is_idempotent_and_sets_wal(tmp_path):
    path = tmp_path / "d" / "agent.db"
    init_db(path)
    init_db(path)  # second call must not fail or drop anything
    conn = sqlite3.connect(path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessions", "runs", "step_events", "feedback"} <= tables
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    conn.close()


def test_db_path_comes_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv(DB_PATH_ENV, raising=False)
    assert str(get_db_path()).replace("\\", "/") == "data/agent.db"
    monkeypatch.setenv(DB_PATH_ENV, str(tmp_path / "x.db"))
    assert get_db_path() == tmp_path / "x.db"


def test_create_session_and_touch(store):
    sid = store.create_session()
    assert store.session_exists(sid)
    assert not store.session_exists("nope")
    store.touch_session(sid)


def test_create_run_defaults(store, session_id):
    run_id = store.create_run(session_id, "red shoes", {"max_price": "5000", "currency": "INR"})
    run = store.get_run(run_id)
    assert run["status"] == "running"
    assert run["partial"] is False
    assert run["query"] == "red shoes"
    assert json.loads(run["form_inputs_json"]) == {"max_price": "5000", "currency": "INR"}
    assert run["decision_json"] is None


def test_create_run_accepts_caller_run_id(store, session_id):
    run_id = store.create_run(session_id, "q", run_id="11111111-1111-1111-1111-111111111111")
    assert run_id == "11111111-1111-1111-1111-111111111111"


def test_foreign_keys_are_enforced(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.create_run("no-such-session", "q")
    with pytest.raises(sqlite3.IntegrityError):
        store.log_step("no-such-run", 1, "plan", "ok")


def test_update_run_json_partial_and_status(store, session_id):
    run_id = store.create_run(session_id, "q")
    store.update_run(
        run_id,
        status="selected",
        partial=True,
        plan_json={"steps": [1, 2]},
        decision_json='{"status": "selected"}',
    )
    run = store.get_run(run_id)
    assert run["status"] == "selected"
    assert run["partial"] is True
    assert json.loads(run["plan_json"]) == {"steps": [1, 2]}
    assert run["decision_json"] == '{"status": "selected"}'
    assert run["updated_at"] >= run["created_at"]


def test_update_run_serializes_pydantic_like_objects(store, session_id):
    class Fake:
        def model_dump_json(self):
            return '{"ok": true}'

    run_id = store.create_run(session_id, "q")
    store.update_run(run_id, intent_json=Fake())
    assert store.get_run(run_id)["intent_json"] == '{"ok": true}'


def test_update_run_rejects_unknown_column_and_unknown_run(store, session_id):
    run_id = store.create_run(session_id, "q")
    with pytest.raises(ValueError):
        store.update_run(run_id, query="hacked")
    with pytest.raises(KeyError):
        store.update_run("missing", status="error")


def test_get_run_missing_returns_none(store):
    assert store.get_run("missing") is None


def test_error_run_keeps_error_text(store, session_id):
    run_id = store.create_run(session_id, "q")
    store.update_run(run_id, status="error", error="planner failed: boom")
    run = store.get_run(run_id)
    assert (run["status"], run["error"]) == ("error", "planner failed: boom")


def test_list_runs_newest_first_scoped_and_limited(store):
    a, b = store.create_session(), store.create_session()
    first = store.create_run(a, "first")
    second = store.create_run(a, "second")
    store.create_run(b, "other session")
    rows = store.list_runs(a)
    assert [r["run_id"] for r in rows] == [second, first]
    assert "decision_json" not in rows[0]
    assert len(store.list_runs(a, limit=1)) == 1


def test_log_and_get_step_events_in_order(store, session_id):
    run_id = store.create_run(session_id, "q")
    store.log_step(run_id, 1, "plan", "ok", latency_ms=12)
    store.log_step(run_id, 1, "get_product_2", "FAILED", error="timeout", payload={"result_count": 0})
    store.log_step(run_id, 2, "plan", "ok")
    events = store.get_step_events(run_id)
    assert [(e["attempt"], e["step"]) for e in events] == [(1, "plan"), (1, "get_product_2"), (2, "plan")]
    assert events[0]["latency_ms"] == 12
    assert events[1]["error"] == "timeout"
    assert json.loads(events[1]["payload_json"]) == {"result_count": 0}
    assert events[2]["latency_ms"] is None


def test_feedback_roundtrip_and_validation(store, session_id):
    run_id = store.create_run(session_id, "q")
    store.add_feedback(run_id, "p1", 1)
    store.add_feedback(run_id, "p2", -1, note="wrong size")
    rows = store.get_feedback(run_id)
    assert [(r["product_id"], r["rating"], r["note"]) for r in rows] == [
        ("p1", 1, None),
        ("p2", -1, "wrong size"),
    ]
    with pytest.raises(ValueError):
        store.add_feedback(run_id, "p3", 5)


def test_data_survives_a_new_store_instance(tmp_path):
    path = tmp_path / "agent.db"
    first = SqliteStore(path)
    sid = first.create_session()
    run_id = first.create_run(sid, "persist me")
    second = SqliteStore(path)  # simulates an app restart
    assert second.get_run(run_id)["query"] == "persist me"
    assert second.list_runs(sid)[0]["run_id"] == run_id


def test_usable_from_another_thread(store, session_id):
    result = {}

    def work():
        result["run_id"] = store.create_run(session_id, "from thread")

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert store.get_run(result["run_id"])["query"] == "from thread"


def test_sqlite_run_tracker_records_audit_event(store, session_id):
    run_id = store.create_run(session_id, "q")
    tracker = SqliteRunTracker(store)
    tracker.record({"run_id": run_id, "status": "selected", "graph_iterations": 2})
    (event,) = store.get_step_events(run_id)
    assert (event["step"], event["status"], event["attempt"]) == ("audit", "selected", 2)
    assert json.loads(event["payload_json"])["graph_iterations"] == 2


def test_sqlite_run_tracker_ignores_unknown_run_and_missing_id(store):
    tracker = SqliteRunTracker(store)
    tracker.record({"run_id": "not-a-run", "status": "selected"})  # must not raise
    tracker.record({"status": "selected"})  # must not raise
