"""M1.6 - SQLite persistence for sessions, runs, step events and feedback.

Design notes:

* Stdlib ``sqlite3`` only.  Pydantic objects are stored as JSON text
  (``model_dump_json()``); this module never imports pydantic, it just calls
  ``model_dump_json`` when an object has one.
* A new connection is opened per call.  Streamlit reruns the script on
  different threads and a ``sqlite3`` connection must not cross threads.
* WAL is enabled once in :func:`init_db` (it is stored in the DB file);
  ``foreign_keys`` is per-connection, so it is set on every connect.
* Timestamps are UTC ISO-8601 strings.
* ``runs.status`` is ``running`` until the run ends, then the decision status
  verbatim (``selected`` / ``needs_selection`` / ``clarify`` / ``failed``) or
  ``error``.  ``partial`` is a separate boolean column.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Mapping
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import pii

DB_PATH_ENV = "DISCOVERY_DB_PATH"
DEFAULT_DB_PATH = "data/agent.db"

RUNNING = "running"
ERROR = "error"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(session_id),
    query            TEXT NOT NULL,
    status           TEXT NOT NULL,
    partial          INTEGER NOT NULL DEFAULT 0,
    form_inputs_json TEXT,
    intent_json      TEXT,
    plan_json        TEXT,
    decision_json    TEXT,
    error            TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS step_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    attempt      INTEGER NOT NULL DEFAULT 1,
    step         TEXT NOT NULL,
    status       TEXT NOT NULL,
    error        TEXT,
    latency_ms   INTEGER,
    payload_json TEXT,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    product_id TEXT NOT NULL,
    rating     INTEGER NOT NULL,
    note       TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS request_dedup (
    key        TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    run_id     TEXT,
    claimed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rate_slots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    claimed_at  TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_slots_session ON rate_slots(session_id, claimed_at);
CREATE INDEX IF NOT EXISTS idx_slots_claimed ON rate_slots(claimed_at);
CREATE INDEX IF NOT EXISTS idx_steps_run ON step_events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_feedback_run ON feedback(run_id);
"""

# Columns update_run may change.  Anything else is a programming error.
_JSON_COLUMNS = {"form_inputs_json", "intent_json", "plan_json", "decision_json"}
_UPDATABLE = _JSON_COLUMNS | {"status", "partial", "error"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db_path() -> Path:
    """DB location from ``DISCOVERY_DB_PATH``, defaulting to ``data/agent.db``."""
    return Path(os.getenv(DB_PATH_ENV) or DEFAULT_DB_PATH)


def _to_json(value: Any) -> str | None:
    """Serialize a Pydantic model / dict / list to JSON text; pass str through."""
    if value is None or isinstance(value, str):
        return value
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    return json.dumps(value, default=str)


def _connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(path: str | Path | None = None) -> Path:
    """Create the DB file, its parent directory and all tables (idempotent)."""
    db_path = Path(path) if path else get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
    return db_path


class SqliteStore:
    """Thin, thread-safe (connection-per-call) wrapper over the M1.6 tables."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = init_db(path)

    # -- sessions ---------------------------------------------------------
    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        now = _now()
        with closing(_connect(self.path)) as conn, conn:
            conn.execute(
                "INSERT INTO sessions(session_id, created_at, last_active_at) VALUES (?,?,?)",
                (session_id, now, now),
            )
        return session_id

    def touch_session(self, session_id: str) -> None:
        with closing(_connect(self.path)) as conn, conn:
            conn.execute(
                "UPDATE sessions SET last_active_at=? WHERE session_id=?", (_now(), session_id)
            )

    def session_exists(self, session_id: str) -> bool:
        with closing(_connect(self.path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return row is not None

    # -- runs -------------------------------------------------------------
    def create_run(
        self,
        session_id: str,
        query: str,
        form_inputs: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        run_id = str(run_id) if run_id else str(uuid.uuid4())
        now = _now()
        query = pii.mask_text(query)  # M2.4: contact PII never reaches the table
        form_inputs = pii.mask_obj(dict(form_inputs), keys=None) if form_inputs else None
        with closing(_connect(self.path)) as conn, conn:
            conn.execute(
                "INSERT INTO runs(run_id, session_id, query, status, partial, form_inputs_json,"
                " created_at, updated_at) VALUES (?,?,?,?,0,?,?,?)",
                (run_id, session_id, query, RUNNING, _to_json(dict(form_inputs) if form_inputs else None), now, now),
            )
            conn.execute("UPDATE sessions SET last_active_at=? WHERE session_id=?", (now, session_id))
        return run_id

    def update_run(self, run_id: str, **fields: Any) -> None:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"update_run cannot change: {sorted(unknown)}")
        if not fields:
            return
        values: dict[str, Any] = {}
        for column, value in fields.items():
            if column in _JSON_COLUMNS:
                values[column] = _to_json(pii.mask_json_value(value))
            elif column == "partial":
                values[column] = 1 if value else 0
            elif column == "error":
                values[column] = pii.mask_text(value)
            else:
                values[column] = value
        values["updated_at"] = _now()
        assignments = ", ".join(f"{column}=?" for column in values)
        with closing(_connect(self.path)) as conn, conn:
            cursor = conn.execute(
                f"UPDATE runs SET {assignments} WHERE run_id=?", (*values.values(), str(run_id))
            )
            if cursor.rowcount == 0:
                raise KeyError(f"unknown run_id: {run_id}")

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with closing(_connect(self.path)) as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (str(run_id),)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["partial"] = bool(data["partial"])
        return data

    def list_runs(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Newest first.  Light rows only (no JSON blobs) for the history sidebar."""
        with closing(_connect(self.path)) as conn:
            rows = conn.execute(
                "SELECT run_id, query, status, partial, error, created_at, updated_at"
                " FROM runs WHERE session_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["partial"] = bool(item["partial"])
            out.append(item)
        return out

    # -- idempotency and rate limiting (Phase 2 / M2.2) -------------------
    def claim_request(self, key: str, session_id: str, window_seconds: float) -> tuple[bool, str | None]:
        """Atomically claim ``key`` for a request.

        Returns ``(True, None)`` when the caller now owns the key and must run
        the search, or ``(False, run_id)`` when an identical request was
        claimed less than ``window_seconds`` ago (``run_id`` is ``None`` while
        that request is still in flight).  A sliding window, not a time bucket,
        so two identical requests either side of a boundary are still caught.
        """
        now = _now()
        conn = _connect(self.path)
        conn.isolation_level = None  # explicit transaction: the check and the claim are one step
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM request_dedup WHERE (julianday(?) - julianday(claimed_at)) * 86400.0 > 86400",
                (now,),
            )
            row = conn.execute(
                "SELECT run_id, (julianday(?) - julianday(claimed_at)) * 86400.0 AS age"
                " FROM request_dedup WHERE key=?",
                (now, key),
            ).fetchone()
            if row is not None and row["age"] < window_seconds:
                conn.execute("COMMIT")
                return False, row["run_id"]
            conn.execute(
                "INSERT OR REPLACE INTO request_dedup(key, session_id, run_id, claimed_at) VALUES (?,?,NULL,?)",
                (key, session_id, now),
            )
            conn.execute("COMMIT")
            return True, None
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def attach_request(self, key: str, run_id: str) -> None:
        """Record which run answered ``key`` so duplicates can return it."""
        with closing(_connect(self.path)) as conn, conn:
            conn.execute("UPDATE request_dedup SET run_id=? WHERE key=?", (str(run_id), key))

    def release_request(self, key: str) -> None:
        """Forget ``key`` (failed runs and rejected requests must stay retryable)."""
        with closing(_connect(self.path)) as conn, conn:
            conn.execute("DELETE FROM request_dedup WHERE key=?", (key,))

    def reserve_slot(
        self,
        session_id: str,
        *,
        session_max: int = 0,
        session_window: float = 60,
        global_max: int = 0,
        global_window: float = 60,
        max_concurrent: int = 0,
        stale_after: float = 300,
    ) -> tuple[int | None, str | None]:
        """Atomically check the three run limits and, if all pass, take a slot.

        Returns ``(slot_id, None)`` on success, or ``(None, reason)`` with reason
        ``"session"`` (this session started too many runs in the window),
        ``"global"`` (all sessions together did) or ``"concurrent"`` (too many runs
        are executing right now).  A limit of 0 is off.

        The checks and the insert share one ``BEGIN IMMEDIATE`` transaction, so
        concurrent callers cannot all pass the check before any of them records
        itself (the check-then-create race the old ``count_recent_runs`` check had).
        A rejected request records nothing.  Slots are rows in SQLite, so the limits
        hold across threads, processes and restarts.  A slot whose run never called
        :meth:`finish_slot` (a crash) stops counting as running after ``stale_after``
        seconds, so a crash cannot lock the app.
        """
        now = _now()
        conn = _connect(self.path)
        conn.isolation_level = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM rate_slots WHERE (julianday(?) - julianday(claimed_at)) * 86400.0 > 86400",
                (now,),
            )

            def _count(where: str, params: tuple, window: float) -> int:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM rate_slots WHERE " + where
                    + " AND (julianday(?) - julianday(claimed_at)) * 86400.0 <= ?",
                    (*params, now, window),
                ).fetchone()
                return int(row["n"])

            reason = None
            if session_max > 0 and _count("session_id=?", (session_id,), session_window) >= session_max:
                reason = "session"
            elif global_max > 0 and _count("1=1", (), global_window) >= global_max:
                reason = "global"
            elif max_concurrent > 0 and _count("finished_at IS NULL", (), stale_after) >= max_concurrent:
                reason = "concurrent"
            if reason:
                conn.execute("COMMIT")
                return None, reason
            cur = conn.execute(
                "INSERT INTO rate_slots(session_id, claimed_at) VALUES (?,?)", (session_id, now)
            )
            slot_id = int(cur.lastrowid)
            conn.execute("COMMIT")
            return slot_id, None
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def finish_slot(self, slot_id: int) -> None:
        """Mark a slot's run as over so it stops counting toward the concurrent cap.

        It still counts toward the per-session and global window caps."""
        with closing(_connect(self.path)) as conn, conn:
            conn.execute(
                "UPDATE rate_slots SET finished_at=? WHERE id=? AND finished_at IS NULL",
                (_now(), int(slot_id)),
            )

    def count_recent_runs(self, session_id: str, window_seconds: float) -> int:
        """Runs this session started within the last ``window_seconds``."""
        with closing(_connect(self.path)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM runs WHERE session_id=?"
                " AND (julianday(?) - julianday(created_at)) * 86400.0 <= ?",
                (session_id, _now(), window_seconds),
            ).fetchone()
        return int(row["n"])

    # -- step events ------------------------------------------------------
    def log_step(
        self,
        run_id: str,
        attempt: int,
        step: str,
        status: str,
        error: str | None = None,
        latency_ms: int | None = None,
        payload: Any = None,
    ) -> int:
        with closing(_connect(self.path)) as conn, conn:
            cursor = conn.execute(
                "INSERT INTO step_events(run_id, attempt, step, status, error, latency_ms,"
                " payload_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (str(run_id), attempt, step, status, pii.mask_text(error), latency_ms,
                 _to_json(pii.mask_json_value(payload)), _now()),
            )
            return int(cursor.lastrowid)

    def get_step_events(self, run_id: str) -> list[dict[str, Any]]:
        with closing(_connect(self.path)) as conn:
            rows = conn.execute(
                "SELECT * FROM step_events WHERE run_id=? ORDER BY id", (str(run_id),)
            ).fetchall()
        return [dict(row) for row in rows]

    # -- feedback ---------------------------------------------------------
    def add_feedback(
        self, run_id: str, product_id: str, rating: int, note: str | None = None
    ) -> int:
        if rating not in (-1, 1):
            raise ValueError("rating must be 1 (thumbs up) or -1 (thumbs down)")
        with closing(_connect(self.path)) as conn, conn:
            cursor = conn.execute(
                "INSERT INTO feedback(run_id, product_id, rating, note, created_at)"
                " VALUES (?,?,?,?,?)",
                (str(run_id), product_id, rating, note, _now()),
            )
            return int(cursor.lastrowid)

    def get_feedback(self, run_id: str) -> list[dict[str, Any]]:
        with closing(_connect(self.path)) as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE run_id=? ORDER BY id", (str(run_id),)
            ).fetchall()
        return [dict(row) for row in rows]


class SqliteRunTracker:
    """``RunTracker`` (ports.py) that mirrors audit events into ``step_events``.

    Runs side by side with ``JsonlRunTracker``.  Best effort: an audit event for
    a run that has no row yet is skipped rather than failing the workflow.
    """

    def __init__(self, store: SqliteStore) -> None:
        self.store = store

    def record(self, event: Mapping[str, object]) -> None:
        run_id = event.get("run_id")
        if not run_id:
            return
        try:
            self.store.log_step(
                str(run_id),
                attempt=int(event.get("graph_iterations") or 1),  # type: ignore[call-overload]
                step="audit",
                status=str(event.get("status") or "recorded"),
                payload=dict(event),
            )
        except sqlite3.IntegrityError:
            return
