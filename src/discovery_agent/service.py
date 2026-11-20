"""M1.6 - the single entry point the UI talks to.

The Streamlit page imports only from this module.  It never touches Gemini,
Shopify, SQLite or the graph directly, and no stack trace ever reaches it:
every failure comes back as a ``WorkflowResult`` with ``status="error"`` and a
plain-language message.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .config import Settings, get_settings
from .graph import GuardBlocked, WorkflowError, run_discovery_workflow
from .log_redaction import install_log_redaction
from .pii import mask_text
from .planner import PlannerConfigurationError
from .schemas import DiscoveryDecision
from .storage import SqliteStore
from .tracking import JsonlRunTracker

logger = logging.getLogger(__name__)
install_log_redaction()

ResultStatus = Literal[
    "running", "selected", "needs_selection", "clarify", "failed", "blocked", "rate_limited", "error"
]


class WorkflowResult(BaseModel):
    run_id: str | None = None
    status: ResultStatus
    partial: bool = False
    decision: DiscoveryDecision | None = None
    error: str | None = None
    # [{"name": ..., "value": ...}] from decision.diagnostics["unverified"]
    unverified: list[dict[str, Any]] = Field(default_factory=list)
    # True when an identical request inside the dedup window returned this run.
    deduplicated: bool = False


def get_store() -> SqliteStore:
    """The SQLite store at ``DISCOVERY_DB_PATH`` (created on first use)."""
    return SqliteStore()


def new_session(store: SqliteStore | None = None) -> str:
    return (store or get_store()).create_session()


def _friendly_error(exc: BaseException) -> str:
    if isinstance(exc, PlannerConfigurationError):
        return (
            "The search assistant is not configured yet (missing or invalid API key or "
            f"package). Details: {mask_text(str(exc))[:300]}"
        )
    if isinstance(exc, WorkflowError):
        return mask_text(str(exc))[:300] or "The search could not be completed."
    return "Something went wrong while searching. Please try again in a moment."


def _unverified(decision: DiscoveryDecision | None) -> list[dict[str, Any]]:
    if decision is None:
        return []
    return list(decision.diagnostics.get("unverified") or [])


RATE_LIMIT_MESSAGE = "You are searching too quickly. Please wait a few seconds and try again."
IN_FLIGHT_MESSAGE = "That search is already running. Please wait for it to finish."
BUSY_MESSAGE = "The assistant is busy right now. Please try again in a few seconds."


def request_key(session_id: str, query: str, form_inputs: Mapping[str, Any] | None) -> str:
    """SHA-256 of who asked, what they typed and every form value.

    Form values are part of the key on purpose: changing the budget, size or
    city is a different request and must not be answered by an older run.
    """
    form = {
        str(k): str(v).strip().casefold()
        for k, v in (form_inputs or {}).items()
        if v not in (None, "")
    }
    payload = json.dumps(
        {"s": session_id, "q": " ".join(query.casefold().split()), "f": form},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def discover(
    query: str,
    session_id: str,
    form_inputs: Mapping[str, Any] | None = None,
    *,
    store: SqliteStore | None = None,
    settings: Settings | None = None,
    planner: Any = None,
    executor: Any = None,
    force: bool = False,
) -> WorkflowResult:
    """Run one discovery and persist it.  Never raises.

    An identical request (same session, query and form values) inside
    ``DEDUP_WINDOW_SECONDS`` returns the existing run (``deduplicated=True``)
    instead of planning again; ``force=True`` skips that.  Failed runs release
    their claim so a retry always runs.  A session may start at most
    ``RATE_LIMIT_MAX_RUNS`` runs per ``RATE_LIMIT_WINDOW_SECONDS``, all sessions
    together at most ``RATE_LIMIT_GLOBAL_MAX_RUNS``, and at most
    ``MAX_CONCURRENT_RUNS`` may execute at once; the check and the reservation
    are one atomic step.

    The keyword-only ``store`` / ``settings`` / ``planner`` / ``executor``
    exist so tests can inject fakes; the UI passes none of them.
    """
    query = (query or "").strip()
    if len(query) < 2:
        return WorkflowResult(status="error", error="Please describe what you are looking for.")

    key: str | None = None
    slot_id: int | None = None
    try:
        store = store or get_store()
        settings = settings or get_settings()

        if not force and settings.dedup_window_seconds > 0:
            key = request_key(session_id, query, form_inputs)
            owned, existing = store.claim_request(key, session_id, settings.dedup_window_seconds)
            if not owned:
                cached = load_run(existing, store) if existing else None
                if cached is not None:
                    return cached.model_copy(update={"deduplicated": True})
                return WorkflowResult(status="running", error=IN_FLIGHT_MESSAGE, deduplicated=True)

        # One atomic step: check the per-session, global and concurrent limits and take a
        # slot.  (Counting runs and creating one later let concurrent requests all pass.)
        if (
            settings.rate_limit_max_runs > 0
            or settings.rate_limit_global_max_runs > 0
            or settings.max_concurrent_runs > 0
        ):
            slot_id, denied = store.reserve_slot(
                session_id,
                session_max=settings.rate_limit_max_runs,
                session_window=settings.rate_limit_window_seconds,
                global_max=settings.rate_limit_global_max_runs,
                global_window=settings.rate_limit_window_seconds,
                max_concurrent=settings.max_concurrent_runs,
                stale_after=settings.rate_limit_slot_ttl_seconds,
            )
            if denied:
                if key:
                    store.release_request(key)
                logger.warning("request rejected by %s limit", denied)
                return WorkflowResult(
                    status="rate_limited",
                    error=RATE_LIMIT_MESSAGE if denied == "session" else BUSY_MESSAGE,
                )

        trackers = [JsonlRunTracker(settings.ledger_path)]
        tracker = _Fanout(trackers)
        state = run_discovery_workflow(
            query, planner=planner, executor=executor, settings=settings, tracker=tracker,
            store=store, session_id=session_id, form_inputs=dict(form_inputs) if form_inputs else None,
        )
    except GuardBlocked as exc:
        _keep_claim(store, key, exc.run_id)
        return WorkflowResult(run_id=exc.run_id, status="blocked", error=str(exc))
    except WorkflowError as exc:
        _release_claim(store, key)
        return WorkflowResult(run_id=exc.run_id, status="error", error=_friendly_error(exc))
    except Exception as exc:  # noqa: BLE001 - the UI must never see a stack trace
        logger.exception("discovery failed")
        _release_claim(store, key)
        return WorkflowResult(
            run_id=getattr(exc, "run_id", None), status="error", error=_friendly_error(exc)
        )

    finally:
        _finish_slot(store, slot_id)

    run_id = state["run_id"]
    _keep_claim(store, key, run_id)
    run = store.get_run(run_id)
    decision = state.get("decision")
    return WorkflowResult(
        run_id=run_id,
        status=(run or {}).get("status") or (decision.status if decision else "error"),
        partial=bool((run or {}).get("partial")),
        decision=decision,
        unverified=_unverified(decision),
    )


def _finish_slot(store: SqliteStore | None, slot_id: int | None) -> None:
    if store is not None and slot_id is not None:
        try:
            store.finish_slot(slot_id)
        except Exception:  # noqa: BLE001 - the slot expires on its own after the TTL
            logger.warning("could not release run slot %s", slot_id, exc_info=True)


def _keep_claim(store: SqliteStore | None, key: str | None, run_id: str | None) -> None:
    """Point the claim at the finished run so duplicates return it."""
    if store is not None and key and run_id:
        try:
            store.attach_request(key, run_id)
        except Exception:  # noqa: BLE001 - bookkeeping must never break a finished search
            logger.warning("could not record request %s", key[:8], exc_info=True)


def _release_claim(store: SqliteStore | None, key: str | None) -> None:
    if store is not None and key:
        try:
            store.release_request(key)
        except Exception:  # noqa: BLE001
            logger.warning("could not release request %s", key[:8], exc_info=True)


def load_run(run_id: str, store: SqliteStore | None = None) -> WorkflowResult | None:
    """Rebuild a past run from SQLite alone.  No Gemini, no Shopify, no network."""
    store = store or get_store()
    run = store.get_run(run_id)
    if run is None:
        return None
    decision = None
    if run["decision_json"]:
        try:
            decision = DiscoveryDecision.model_validate_json(run["decision_json"])
        except ValidationError:
            logger.warning("stored decision for %s no longer validates", run_id)
    return WorkflowResult(
        run_id=run["run_id"],
        status=run["status"],
        partial=run["partial"],
        decision=decision,
        error=run["error"],
        unverified=_unverified(decision),
    )


def load_run_meta(run_id: str, store: SqliteStore | None = None) -> dict[str, Any] | None:
    """Query and saved form values of a run (for headers and re-runs)."""
    run = (store or get_store()).get_run(run_id)
    if run is None:
        return None
    import json

    return {
        "query": run["query"],
        "form_inputs": json.loads(run["form_inputs_json"]) if run["form_inputs_json"] else {},
        "created_at": run["created_at"],
    }


def list_history(session_id: str, limit: int = 20, store: SqliteStore | None = None) -> list[dict[str, Any]]:
    return (store or get_store()).list_runs(session_id, limit)


def get_steps(run_id: str, store: SqliteStore | None = None) -> list[dict[str, Any]]:
    return (store or get_store()).get_step_events(run_id)


def record_feedback(
    run_id: str, product_id: str, rating: int, note: str | None = None,
    store: SqliteStore | None = None,
) -> None:
    (store or get_store()).add_feedback(run_id, product_id, rating, note)


def get_feedback(run_id: str, store: SqliteStore | None = None) -> list[dict[str, Any]]:
    return (store or get_store()).get_feedback(run_id)


class _Fanout:
    """Send each audit event to every tracker; one failing sink never blocks the others."""

    def __init__(self, trackers: list[Any]) -> None:
        self.trackers = trackers

    def record(self, event: Mapping[str, object]) -> None:
        for tracker in self.trackers:
            try:
                tracker.record(event)
            except Exception:  # noqa: BLE001
                logger.warning("tracker %s failed", type(tracker).__name__, exc_info=True)


# --- helpers for the UI (so it never imports config / storage directly) ---------
def ensure_session(session_id: str | None, store: SqliteStore | None = None) -> str:
    """Reuse ``session_id`` if it exists in SQLite (e.g. after an app restart), else start one."""
    store = store or get_store()
    if session_id and store.session_exists(session_id):
        store.touch_session(session_id)
        return session_id
    return store.create_session()


def planner_configured() -> bool:
    """False when no Gemini API key is set, so the UI can explain instead of failing."""
    try:
        return bool(get_settings().google_api_key)
    except Exception:  # noqa: BLE001
        return False


def max_clarifications() -> int:
    """Bound on clarification re-runs in a row (MAX_REFINEMENTS)."""
    return get_settings().max_refinements


def guard_status() -> dict[str, Any]:
    """Configured vs. active screening (``active``, ``requested``, ``warning``) for the UI."""
    try:
        from .guard import guard_status as _status

        return _status(get_settings())
    except Exception:  # noqa: BLE001 - never break the page over a status line
        logger.warning("could not determine guard status", exc_info=True)
        return {"active": "unknown", "requested": None, "fail_closed": False,
                "warning": "The guard status could not be determined."}


def privacy_warnings() -> list[str]:
    """Sentences for every PII protection that is off or weak (also logged once per process)."""
    try:
        from . import pii

        return pii.log_config_warnings()
    except Exception:  # noqa: BLE001 - never break the page over a status line
        logger.warning("could not determine PII settings", exc_info=True)
        return []


def warm_guard() -> None:
    """Load the guard classifier once at app start so the first search is not slowed by it.

    Safe to call repeatedly: ``build_classifier`` returns one cached instance per model and
    the model itself loads only on the first ``score``.  Never raises; if the model cannot
    load, the guard falls back to its normal per-request behaviour.
    """
    privacy_warnings()  # log weakened PII settings at start-up, even if the guard is off
    try:
        settings = get_settings()
        if not getattr(settings, "guard_enabled", True):
            return
        from .guard import build_classifier

        classifier = build_classifier(settings.guard_classifier)
        if classifier is not None:
            classifier.score("warm up")
        status = guard_status()
        if status["warning"]:
            logger.error("GUARD: %s", status["warning"])
        else:
            logger.info("GUARD active: %s", status["active"])
    except Exception:  # noqa: BLE001 - warm-up is an optimisation, never a failure
        logger.error(
            "GUARD: the classifier could not be loaded; requests run on the rule layer only "
            "unless GUARD_FAIL_CLOSED=1.", exc_info=True,
        )