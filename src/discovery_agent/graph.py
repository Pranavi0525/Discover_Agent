"""M1.4 - Planner/Executor workflow, wired together with LangGraph.

This module is the state machine described in ``notes/implementation-plan.md``:

    request -> plan (Gemini) -> validate (Pydantic + deterministic guardrails)
            -> execute (Shopify MCP, executor-owned tool calls)
            -> rank + decide -> select / ask user / clarify
                                     |
                                     +-- bounded refinement loop back to plan

Design decisions, on purpose:

* The planner (``planner.GeminiPlanner``) never touches an MCP session. It
  only ever returns a Pydantic ``ExecutionPlan``. That plan is re-validated
  here with ``ExecutionPlan.model_validate`` (defense in depth: the object
  already passed validation once inside the planner, but nothing upstream of
  this graph is trusted) before any tool call is allowed to happen.
* The executor (``execution.ShopifyMcpExecutor``) is the only thing that
  opens an MCP session and calls tools. It already runs the bounded,
  allowlisted ``search_catalog`` -> parallel ``get_product`` sequence with
  per-tool retries and timeouts (see ``execution.py``), so this graph treats
  one ``execute`` node as one atomic, awaited unit of work instead of
  re-implementing tool dispatch as individual graph nodes/edges.
* Every bound (`MAX_PLAN_STEPS`, `MAX_PRODUCT_DETAILS`, `MAX_TOOL_CALLS`,
  `MAX_RETRIES_PER_TOOL`, `MAX_GRAPH_ITERATIONS`, `TOOL_TIMEOUT_SECONDS`) is
  read from ``Settings`` in plain Python and enforced before or during a
  tool call. No LLM response can raise any of these limits - a plan that
  tries to is rejected in ``_validate_node`` before ``execute`` ever runs.
* The refinement loop (`plan -> validate -> execute -> decide -> refine ->
  plan`) is bounded twice over: ``refinement_count`` (business limit,
  ``MAX_REFINEMENTS`` / ``Settings.max_refinements``) and
  ``graph_iterations`` (hard structural limit, ``MAX_GRAPH_ITERATIONS``) so a
  misbehaving planner cannot spin the graph forever even if it kept
  returning technically-valid "clarify" results.
* Which query each attempt searches is decided in plain Python by the query
  ladder (``query_refinement.build_query_ladder``): specific, core, broaden,
  synonym.  The planner's text only supplies the intent; the ladder supplies
  the catalog query, and the user's hard constraints are restored on every
  attempt.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any, Literal, TypedDict
from uuid import UUID

from pydantic import ValidationError

from . import pii
from .config import Settings, get_settings
from .budget import budget_currency_missing, resolve_currency
from .categories import clarify_missing_attributes, missing_required
from .constraints import UnverifiedConstraint, city_market_note, unverified_constraints
from .execution import ShopifyMcpExecutor
from .forms import intent_overrides, normalize_form_inputs, planner_line
from .guard import BLOCK_MESSAGE, InputGuard, build_guard
from .orchestration import attach_unverified, decide, no_match_message
from .planner import GeminiPlanner
from .policy_check import PolicyChecker, apply_policy, summarize_policy
from .providers.shopify_ucp import buyer_country
from .ports import RunTracker
from .query_refinement import (
    STRATEGY_SYNONYM,
    build_query_ladder,
    ladder_from_state,
    ladder_to_state,
    refinement_note,
)
from .ranking import filter_and_rank_with_stats, unverified_from_stats
from .refinement import broaden_query, carry_over_constraints
from .schemas import DiscoveryDecision, ExecutionPlan, ExecutionReport, UserIntent
from .semantic import RelevanceScorer, RunScorer, get_default_scorer
from .storage import SqliteStore

logger = logging.getLogger(__name__)

PROVIDER_NAME = "shopify-mcp"

WorkflowStatus = Literal[
    "planning",
    "planned",
    "needs_clarification",
    "validated",
    "executed",
    "decided",
    "refining",
    "blocked",
    "error",
    "done",
]


class AgentState(TypedDict, total=False):
    """LangGraph state threaded through every node.

    Every key is optional (``total=False``) because the graph is built
    incrementally: a fresh run only has ``raw_query``, ``refinement_count``,
    and ``graph_iterations`` populated.
    """

    raw_query: str
    refined_query: str | None
    plan: ExecutionPlan
    report: ExecutionReport
    decision: DiscoveryDecision
    refinement_count: int
    graph_iterations: int
    status: WorkflowStatus
    error: str | None
    # The FIRST validated intent.  Its max_price / currency / merchant are
    # immutable for the whole run; refinements may only change the query.
    original_intent: UserIntent
    # MAX_TOOL_CALLS is one budget for the whole run, shared by all attempts.
    tool_calls_used: int
    attempt_log: list[dict]
    # Notes the planner made when it overrode or filled in the user's request
    # (e.g. "assumed INR"), collected across every plan attempt.
    corrections: list[str]
    # M1.6 persistence.  Absent unless a store was supplied.
    run_id: str
    session_id: str
    # Values the caller (form) decided; applied to the plan intent (Python wins).
    intent_overrides: dict[str, Any]
    # "Form inputs: ..." line appended to the text the planner sees.
    form_line: str
    # Phase 2: the input guard's verdict (GuardResult as a dict).
    guard: dict[str, Any]
    # M1.4: the catalog queries to try, in order (query_refinement ladder).
    ladder: list[dict[str, str]]
    # M1.4: 0 = the user has not been asked for size/gender yet; >= 1 = asked once.
    clarify_round: int


def _clarify_enabled(settings: Settings) -> bool:
    """CLARIFY_MISSING_ATTRIBUTES: on by default, off with 0/false/no/off."""
    value = getattr(settings, "clarify_missing_attributes", None)
    if value is None:
        raw = os.getenv("CLARIFY_MISSING_ATTRIBUTES", "1").strip().lower()
        return raw not in {"0", "false", "no", "off", ""}
    return bool(value)


def _clarify_round(form_inputs: Mapping[str, Any] | None) -> int:
    try:
        return max(0, int((form_inputs or {}).get("clarify_round") or 0))
    except (TypeError, ValueError, AttributeError):
        return 0


def _annotate_decision(
    decision: DiscoveryDecision,
    intent: UserIntent,
    corrections: list[str],
    policy: dict[str, Any] | None = None,
    missing: list[str] | None = None,
) -> None:
    """Make the result say what was assumed and what was NOT verified.

    A constraint the catalog cannot answer (delivery city, returns, ...) must be
    shown as unverified, never silently dropped, and an assumption such as a
    default currency must be visible.  Both are added to ``diagnostics`` (for
    the UI / SQLite) and, in words, to ``message`` (for any plain consumer).
    """
    unverified = unverified_constraints(intent)
    # A constraint that at least one SHOWN product's store policy confirms is
    # reported as confirmed (per-product badges say which); anything no store
    # confirmed stays unverified.
    counts = summarize_policy(policy, [c.product.id for c in decision.candidates]) if policy else {}
    policy_key = {"delivery_city": "delivery", "return_required": "returns"}
    confirmed_lines = {
        "returns": "returns/exchange",
        "delivery": f"delivery to {intent.delivery_city}",
    }
    unverified = [
        item for item in unverified
        if not counts.get(policy_key.get(item.name, ""), {}).get("confirmed")
    ]
    # A requested size that the variant data lists as buyable for a shown
    # product is confirmed for that product; otherwise it stays unverified.
    size_confirmed = [c for c in decision.candidates if c.size_status == "available"]
    if intent.size and intent.size.lower() != "any" and size_confirmed:
        unverified = [item for item in unverified if item.name != "size"]
    # A detail the category needs but the user never gave (after the one
    # question was asked) is reported, not silently ignored.
    if "size" in (missing or []):
        unverified.append(UnverifiedConstraint(
            "size", "size not specified",
            "You did not give a size, so fit was not checked."))
    if "gender" in (missing or []):
        unverified.append(UnverifiedConstraint(
            "gender", "audience not specified",
            "You did not give who it is for (men, women or unisex)."))
    decision.diagnostics["policy_summary"] = counts
    decision.diagnostics["corrections"] = list(corrections)
    decision.diagnostics["unverified"] = [asdict(item) for item in unverified]
    if decision.status == "failed":
        return
    extra: list[str] = []
    if corrections:
        extra.append("Assumptions made: " + "; ".join(corrections) + ".")
    confirmed = [
        f"{confirmed_lines[key]} for {c['confirmed']} of {c['total']} results"
        for key, c in counts.items() if c["confirmed"]
    ]
    if confirmed:
        extra.append("Confirmed by store policy (see the badges): " + "; ".join(confirmed) + ".")
    if intent.size and intent.size.lower() != "any" and size_confirmed:
        extra.append(
            f"Confirmed from variant data: size {intent.size} listed as buyable for "
            f"{len(size_confirmed)} of {len(decision.candidates)} results."
        )
    if unverified:
        extra.append(
            "Not verified against catalog data: "
            + "; ".join(
                f"delivery to {item.value}" if item.name == "delivery_city" else item.value
                for item in unverified
            )
            + "."
        )
    if extra:
        decision.message = f"{decision.message} {' '.join(extra)}"


class GuardBlocked(RuntimeError):
    """The input guard refused the request; nothing was planned or executed.

    Not a ``WorkflowError``: a blocked request is a normal, expected outcome,
    not a failure, so callers can treat it separately.
    """

    run_id: str | None = None


class WorkflowError(RuntimeError):
    """Raised when the graph finishes in an error state and the caller asked to raise.

    ``run_id`` is set when a store was supplied, so callers can still find the
    saved error row.
    """

    run_id: str | None = None


def _tag_run_id(exc: BaseException, run_id: str | None) -> None:
    """Attach the run ID to an exception (best effort) so callers can find its row."""
    if run_id:
        try:
            exc.run_id = run_id  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


# Kept under its historical name; the implementation now lives in refinement.py
# so the Streamlit path (DiscoveryAgent) and this graph refine identically.
_broaden_query = broaden_query


_PARTIAL_FAILED = {"FAILED", "TIMED_OUT", "SKIPPED"}


def compute_partial(report: ExecutionReport | None) -> bool:
    """True when the result is incomplete: a get_product step FAILED / TIMED_OUT /
    SKIPPED, or some details fell back to the search payload."""
    if report is None:
        return False
    if any(s.action == "get_product" and s.status in _PARTIAL_FAILED for s in report.steps):
        return True
    return int(report.diagnostics.get("details_from_search_payload") or 0) > 0


def apply_form_overrides(
    plan: ExecutionPlan, overrides: Mapping[str, Any], default_currency: str | None
) -> tuple[ExecutionPlan, list[str]]:
    """Impose the form's values on ``plan.intent`` (Python wins over the LLM).

    Returns the new plan and one note per value that changed, for
    ``corrections``.  A form budget whose currency is still unknown after the
    planner had its say is resolved like the planner does (market default) or
    turned into a clarification; it never reaches the executor without one.
    """
    if not overrides:
        return plan, []
    notes: list[str] = []
    for field, value in overrides.items():
        if getattr(plan.intent, field, None) != value:
            notes.append(f"{field} set from the form")
    intent = plan.intent.model_copy(update=dict(overrides))
    plan = plan.model_copy(update={"intent": intent})
    if (
        "max_price" in overrides
        and not plan.requires_clarification
        and plan.steps[0].action != "ask_clarification"
        and budget_currency_missing(intent)
    ):
        intent, extra = resolve_currency(intent, default_currency)
        notes.extend(extra)
        plan = plan.model_copy(update={"intent": intent})
        if budget_currency_missing(intent):
            plan = GeminiPlanner._ask_for_currency(intent)
    return plan, notes


def _adopt_run_id(decision: DiscoveryDecision, state: Mapping[str, Any]) -> None:
    """Make the decision carry the run's ID so JSONL and SQLite share one ID."""
    run_id = state.get("run_id")
    if run_id:
        decision.run_id = UUID(str(run_id))


def _safely(action: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Persistence must never break a discovery run; log and carry on."""
    try:
        return action(*args, **kwargs)
    except Exception:  # noqa: BLE001
        logger.warning("SQLite persistence failed", exc_info=True)
        return None


def build_workflow(
    planner: GeminiPlanner,
    executor: ShopifyMcpExecutor,
    settings: Settings,
    tracker: RunTracker | None = None,
    store: SqliteStore | None = None,
    policy_checker: PolicyChecker | None = None,
    guard: InputGuard | None = None,
    scorer: RelevanceScorer | None = None,
):
    """Compile the planner/executor LangGraph state machine.

    Returns a compiled LangGraph graph. Callers normally want
    :func:`run_discovery_workflow` / :func:`arun_discovery_workflow` instead
    of calling this directly.
    """
    from langgraph.graph import END, START, StateGraph

    if guard is None:
        guard = build_guard(settings)  # None when GUARD_ENABLED=0

    # One relevance scorer is pinned for the whole run, so keyword and
    # embedding scores are never mixed against the same decision tiers.
    base_scorer = scorer if scorer is not None else get_default_scorer()
    run_scorer = base_scorer if isinstance(base_scorer, RunScorer) else RunScorer(base_scorer)

    def _checkpoint(name: str, state: AgentState, update: dict[str, Any]) -> None:
        """Write the run-row checkpoints for one finished node."""
        run_id = state.get("run_id")
        if not run_id or update.get("status") == "error":
            return
        if name == "validate" and "plan" in update:
            plan = update["plan"]
            store.update_run(run_id, plan_json=plan, intent_json=plan.intent)
        elif name == "execute" and "report" in update:
            attempt = state.get("graph_iterations", 0) or 1
            for step in update["report"].steps:
                # No per-step latency exists in the report, so it stays NULL.
                store.log_step(
                    run_id, attempt, step.step_id, step.status, error=step.error,
                    payload={"action": step.action, "result_count": step.result_count,
                             "retries": step.retries},
                )
        elif name in {"decide", "finalize_clarify"} and "decision" in update:
            decision = update["decision"]
            report = update.get("report") or state.get("report")
            store.update_run(
                run_id,
                decision_json=decision,
                partial=compute_partial(report) if name == "decide" else False,
                status=decision.status,
            )

    def _record(name: str, state: AgentState, update: dict[str, Any] | None,
                started: float, exc: BaseException | None = None) -> None:
        run_id = state.get("run_id")
        if not run_id:
            return
        latency_ms = int((time.perf_counter() - started) * 1000)
        update = update or {}
        attempt = update.get("graph_iterations", state.get("graph_iterations", 0)) or 1
        failed = exc is not None or update.get("status") == "error"
        error = f"{type(exc).__name__}: {exc}" if exc is not None else update.get("error")
        outcome = "error" if failed else ("blocked" if update.get("status") == "blocked" else "ok")
        # Guard reasons go in the payload, not the error column the UI shows.
        payload = update.get("guard") if name == "guard" else None
        _safely(store.log_step, run_id, attempt, name, outcome,
                error=error, latency_ms=latency_ms, payload=payload)
        if exc is None:
            _safely(_checkpoint, name, state, update)

    def _persisted(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a graph node so each visit becomes a ``step_events`` row.

        Nodes report failure as ``{"status": "error"}`` rather than raising, so
        the returned dict is inspected as well as exceptions.  With no store the
        node is returned unchanged.
        """
        if store is None:
            return fn

        if inspect.iscoroutinefunction(fn):
            async def async_wrapper(state: AgentState) -> dict[str, Any]:
                started = time.perf_counter()
                try:
                    update = await fn(state)
                except Exception as exc:
                    _record(name, state, None, started, exc)
                    raise
                _record(name, state, update, started)
                return update

            return async_wrapper

        def sync_wrapper(state: AgentState) -> dict[str, Any]:
            started = time.perf_counter()
            try:
                update = fn(state)
            except Exception as exc:
                _record(name, state, None, started, exc)
                raise
            _record(name, state, update, started)
            return update

        return sync_wrapper

    def _guard_node(state: AgentState) -> dict[str, Any]:
        # Everything user-typed that the planner would see: the query plus the
        # free-text form line.  Runs once per request (refinements re-plan from
        # text this system generated, so they do not pass through here again).
        text = state["raw_query"]
        if state.get("form_line"):
            text = f"{text}\n{state['form_line']}"
        result = guard.check(text)
        update: dict[str, Any] = {"guard": result.model_dump()}
        if result.blocked:
            update.update(status="blocked", error=BLOCK_MESSAGE)
        return update

    def _plan_node(state: AgentState) -> dict[str, Any]:
        iterations = state.get("graph_iterations", 0) + 1
        if iterations > settings.max_graph_iterations:
            return {
                "graph_iterations": iterations,
                "status": "error",
                "error": "MAX_GRAPH_ITERATIONS exceeded before a plan could be produced",
            }
        query = state.get("refined_query") or state["raw_query"]
        if state.get("form_line"):
            # Tell the planner what the form already answered.
            query = f"{query}\n{state['form_line']}"
        query = pii.redact_for_model(query)  # M2.4: contact PII never leaves the machine
        try:
            plan = planner.plan(query)
        except Exception as exc:  # noqa: BLE001 - surfaced as a bounded workflow error
            return {"graph_iterations": iterations, "status": "error", "error": f"planner failed: {exc}"}
        update: dict[str, Any] = {}
        # Keep every planner note (deduplicated, in order) for the final result.
        notes = list(state.get("corrections", []))
        for note in getattr(planner, "last_corrections", None) or []:
            if note not in notes:
                notes.append(note)
        # Form values are applied BEFORE original_intent is stored, so
        # carry_over_constraints preserves them on every refinement.
        plan, form_notes = apply_form_overrides(
            plan, state.get("intent_overrides") or {}, settings.default_currency
        )
        for note in form_notes:
            if note not in notes:
                notes.append(note)
        update["corrections"] = notes
        is_search = plan.steps[0].action == "search_catalog"
        original = state.get("original_intent")
        if original is None:
            # The first validated intent keeps the user's own query: it is what
            # every product is scored against.  The catalog query comes from the
            # ladder (most specific first).
            update["original_intent"] = plan.intent
            if is_search:
                ladder = build_query_ladder(plan.intent, settings.max_refinements)
                update["ladder"] = ladder_to_state(ladder)
                plan = plan.model_copy(
                    update={"intent": plan.intent.model_copy(update={"query": ladder[0].query})}
                )
        else:
            # A refinement re-plans from shortened TEXT that no longer contains
            # the budget.  Restore the user's original hard constraints so the
            # budget can never be dropped or relaxed by a re-plan.
            count = state.get("refinement_count", 0)
            intent = carry_over_constraints(plan.intent, original, count)
            ladder = ladder_from_state(state.get("ladder"))
            if ladder and is_search:
                rung = ladder[min(count, len(ladder) - 1)]
                intent = intent.model_copy(update={"query": rung.query})
            plan = plan.model_copy(update={"intent": intent})
        return {"plan": plan, "graph_iterations": iterations, "status": "planned", **update}

    def _validate_node(state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        # Re-validate at the graph boundary: nothing upstream of this node is
        # trusted, even though GeminiPlanner already validated once.
        try:
            validated = ExecutionPlan.model_validate(plan.model_dump())
        except ValidationError as exc:
            return {"status": "error", "error": f"plan failed Pydantic validation: {exc}"}

        if len(validated.steps) > settings.max_plan_steps:
            return {"status": "error", "error": "planner returned more steps than MAX_PLAN_STEPS"}

        product_detail_count = sum(
            len(step.product_ids) for step in validated.steps if step.action == "get_product"
        )
        if product_detail_count > settings.max_product_details:
            return {"status": "error", "error": "planner returned more product details than MAX_PRODUCT_DETAILS"}

        allowed_actions = {"search_catalog", "get_product", "ask_clarification"}
        if any(step.action not in allowed_actions for step in validated.steps):
            return {"status": "error", "error": "plan contains an unapproved action"}

        if validated.requires_clarification or validated.steps[0].action == "ask_clarification":
            return {"plan": validated, "status": "needs_clarification"}

        # Ask ONCE for a size / gender the category needs, before any Shopify
        # call.  A refinement or an answered round never asks again.
        round_ = max(state.get("clarify_round", 0), 1 if state.get("refinement_count", 0) else 0)
        ask = clarify_missing_attributes(
            validated, clarify_round=round_, enabled=_clarify_enabled(settings)
        )
        if ask is not None:
            return {"plan": ask, "status": "needs_clarification"}
        return {"plan": validated, "status": "validated"}

    def _finalize_clarify_node(state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        reason = plan.steps[0].reason if plan.steps else "The request needs more detail before searching."
        decision = DiscoveryDecision(status="clarify", provider="planner", message=reason)
        _adopt_run_id(decision, state)
        return {"decision": decision, "status": "done"}

    async def _execute_node(state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        used = state.get("tool_calls_used", 0)
        remaining = max(0, settings.max_tool_calls - used)
        try:
            report = await executor.aexecute_plan(plan, tool_call_budget=remaining)
        except Exception as exc:  # noqa: BLE001 - surfaced as a bounded workflow error
            return {"status": "error", "error": f"executor failed: {exc}"}
        return {"report": report, "status": "executed", "tool_calls_used": used + report.tool_calls}

    def _decide_node(state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        report = state["report"]
        # Hard filters are identical on every attempt; scoring always uses the
        # ORIGINAL user query (plus the synonym when this attempt was one).
        original = state.get("original_intent", plan.intent)
        ladder = ladder_from_state(state.get("ladder"))
        count = state.get("refinement_count", 0)
        step = ladder[count] if ladder and count < len(ladder) else None
        extra = [plan.intent.query] if step is not None and step.strategy == STRATEGY_SYNONYM else None
        ranked, stats = filter_and_rank_with_stats(
            original, report.products, scorer=run_scorer, extra_queries=extra,
            max_unverified=getattr(settings, "max_unverified_shown", 3),
        )
        unverified = unverified_from_stats(stats)
        relevance = stats["relevance"]
        attempt_log = list(state.get("attempt_log", [])) + [{
            "attempt": state.get("graph_iterations", 1),
            "query": plan.intent.query,
            "refinement": count,
            "fetched": len(report.products),
            "kept": stats["kept"],
            "unverified": stats.get("unverified", 0),
            "dropped": stats["dropped"],
            "currencies_seen": stats["currencies_seen"],
            "products": stats.get("products", []),
            "relevance": relevance,
            "error": None,
            "shopify": report.diagnostics,
        }]
        attempts = max(1, state.get("graph_iterations", 1))
        policy: dict[str, Any] = {}
        if ranked and policy_checker is not None:
            ranked, policy = apply_policy(
                ranked, city=original.delivery_city, return_required=original.return_required,
                checker=policy_checker,
            )
        if ranked:
            decision = decide(PROVIDER_NAME, ranked, plan.intent, run_scorer.name)
            if count > 0 and plan.intent.query != original.query:
                if step is not None:
                    decision.message += refinement_note(original.query, step)
                else:
                    decision.message += (
                        f" No products matched \u201c{original.query}\u201d, so the search was"
                        f" broadened to \u201c{plan.intent.query}\u201d; your budget was not changed."
                    )
            decision.attempts = attempts
        else:
            failed_steps = [step_ for step_ in report.steps if step_.status in {"FAILED", "TIMED_OUT"}]
            # A failed detail call is only fatal when it left us with nothing.
            fatal = bool(failed_steps) and not report.products
            decision = DiscoveryDecision(
                status="failed" if fatal else "clarify",
                provider=PROVIDER_NAME,
                message=(
                    "; ".join(step_.error or step_.status for step_ in failed_steps)
                    if fatal
                    else no_match_message(original, attempt_log)
                ),
                attempts=attempts,
            )
        _adopt_run_id(decision, state)
        attach_unverified(decision, unverified)
        decision.diagnostics = {"attempt_log": attempt_log, "relevance": relevance}
        _annotate_decision(
            decision, original, state.get("corrections", []), policy,
            missing=missing_required(original) if _clarify_enabled(settings) else None,
        )
        if policy:
            decision.diagnostics["policy"] = policy
        note = city_market_note(original.delivery_city, buyer_country(settings, original.currency))
        if note:
            decision.diagnostics["market_note"] = note
            if decision.status != "failed":
                decision.message = f"{decision.message} {note}"
        if tracker:
            tracker.record({
                "run_id": str(decision.run_id),
                "status": decision.status,
                "provider": decision.provider,
                "raw_candidate_count": len(report.products),
                "ranked_candidate_count": len(decision.candidates),
                "unverified_candidate_count": len(decision.unverified_candidates),
                "selected_product_id": decision.selected.product.id if decision.selected else None,
                "graph_iterations": state.get("graph_iterations", 0),
                "refinement_count": state.get("refinement_count", 0),
                "attempts": attempts,
                "final_query": plan.intent.query,
                "attempt_log": attempt_log,
                "corrections": decision.diagnostics.get("corrections", []),
                "unverified": decision.diagnostics.get("unverified", []),
                "relevance": relevance,
            })
        return {"decision": decision, "status": "decided", "attempt_log": attempt_log}

    def _refine_node(state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        count = state.get("refinement_count", 0) + 1
        ladder = ladder_from_state(state.get("ladder"))
        if ladder and count < len(ladder):
            next_query = ladder[count].query
        else:
            next_query = _broaden_query(plan.intent.query)
        return {
            "refinement_count": count,
            "refined_query": next_query,
            "status": "refining",
            # The user is never asked for size/gender again after a refinement.
            "clarify_round": max(1, state.get("clarify_round", 0)),
        }

    def _route_after_guard(state: AgentState) -> str:
        return END if state.get("status") == "blocked" else "plan"

    def _route_after_plan(state: AgentState) -> str:
        return END if state.get("status") == "error" else "validate"

    def _route_after_validate(state: AgentState) -> str:
        status = state.get("status")
        if status == "error":
            return END
        if status == "needs_clarification":
            return "finalize_clarify"
        return "execute"

    def _route_after_execute(state: AgentState) -> str:
        return END if state.get("status") == "error" else "decide"

    def _route_after_decide(state: AgentState) -> str:
        decision = state.get("decision")
        # Refine only when NOTHING was found.  Weak-but-real matches are shown
        # to the user, not silently replaced by a broader search.
        if decision is None or decision.status != "clarify" or decision.candidates:
            return END
        plan = state["plan"]
        ladder = state.get("ladder")
        if ladder:
            if state.get("refinement_count", 0) + 1 >= len(ladder):
                return END  # every rung of the ladder has been tried
        elif broaden_query(plan.intent.query) == plan.intent.query.strip():
            return END  # nothing left to broaden
        if state.get("refinement_count", 0) >= settings.max_refinements:
            return END
        if state.get("graph_iterations", 0) >= settings.max_graph_iterations:
            return END
        return "refine"

    graph = StateGraph(AgentState)
    if guard is not None:
        graph.add_node("guard", _persisted("guard", _guard_node))
    graph.add_node("plan", _persisted("plan", _plan_node))
    graph.add_node("validate", _persisted("validate", _validate_node))
    graph.add_node("finalize_clarify", _persisted("finalize_clarify", _finalize_clarify_node))
    graph.add_node("execute", _persisted("execute", _execute_node))
    graph.add_node("decide", _persisted("decide", _decide_node))
    graph.add_node("refine", _persisted("refine", _refine_node))

    if guard is not None:
        graph.add_edge(START, "guard")
        graph.add_conditional_edges("guard", _route_after_guard, {"plan": "plan", END: END})
    else:
        graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", _route_after_plan, {"validate": "validate", END: END})
    graph.add_conditional_edges(
        "validate", _route_after_validate, {"finalize_clarify": "finalize_clarify", "execute": "execute", END: END}
    )
    graph.add_edge("finalize_clarify", END)
    graph.add_conditional_edges("execute", _route_after_execute, {"decide": "decide", END: END})
    graph.add_conditional_edges("decide", _route_after_decide, {"refine": "refine", END: END})
    graph.add_edge("refine", "plan")

    return graph.compile()


async def arun_discovery_workflow(
    query: str,
    planner: GeminiPlanner | None = None,
    executor: ShopifyMcpExecutor | None = None,
    settings: Settings | None = None,
    tracker: RunTracker | None = None,
    store: SqliteStore | None = None,
    session_id: str | None = None,
    form_inputs: Mapping[str, Any] | None = None,
    policy_checker: PolicyChecker | None = None,
    guard: InputGuard | None = None,
    scorer: RelevanceScorer | None = None,
) -> AgentState:
    """Run the planner/executor graph end-to-end and return the final state.

    Raises :class:`WorkflowError` if the run ended in an error state
    (planner failure, invalid plan, or executor failure) so callers do not
    have to remember to check ``state["status"]`` themselves.

    With a ``store`` the run row is created up front (so a ``run_id`` exists
    even on error paths) and finalized before any exception is raised.
    """
    resolved_settings = settings or get_settings()
    initial_state: AgentState = {"raw_query": query, "refinement_count": 0, "graph_iterations": 0}
    form = normalize_form_inputs(form_inputs)
    if overrides := intent_overrides(form):
        initial_state["intent_overrides"] = overrides
    if line := planner_line(form):
        initial_state["form_line"] = line
    if round_ := _clarify_round(form_inputs):
        initial_state["clarify_round"] = round_
    run_id: str | None = None
    if store is not None:
        # Created before anything can fail (even planner construction, e.g. a
        # missing API key) so every failure has a row and a run ID.
        session_id = session_id or store.create_session()
        run_id = str(uuid.uuid4())
        store.create_run(session_id, query, form_inputs, run_id=run_id)
        initial_state["run_id"] = run_id
        initial_state["session_id"] = session_id
    try:
        resolved_planner = planner or GeminiPlanner(resolved_settings)
        resolved_executor = executor or ShopifyMcpExecutor(settings=resolved_settings)
        if policy_checker is None and getattr(resolved_settings, "policy_check", False):
            policy_checker = PolicyChecker()
        workflow = build_workflow(
            resolved_planner, resolved_executor, resolved_settings, tracker, store,
            policy_checker=policy_checker, guard=guard, scorer=scorer,
        )
        # Structural ceiling on total node visits: plan/validate/execute/decide/refine
        # per iteration, bounded by MAX_GRAPH_ITERATIONS, plus headroom for the
        # terminal edge. Keeps a misconfigured limit from tripping LangGraph's
        # own generic recursion guard instead of our explicit error state.
        recursion_limit = max(10, resolved_settings.max_graph_iterations * 6)
        final_state: AgentState = await workflow.ainvoke(
            initial_state, config={"recursion_limit": recursion_limit}
        )
    except Exception as exc:
        if store is not None and run_id:
            _safely(store.update_run, run_id, status="error", error=f"{type(exc).__name__}: {exc}")
            _tag_run_id(exc, run_id)
        raise

    if final_state.get("status") == "blocked":
        message = final_state.get("error") or BLOCK_MESSAGE
        if store is not None and run_id:
            _safely(store.update_run, run_id, status="blocked", error=message)
        blocked = GuardBlocked(message)
        blocked.run_id = run_id
        raise blocked
    if final_state.get("status") == "error":
        message = final_state.get("error") or "discovery workflow failed"
        if store is not None and run_id:
            # Saved BEFORE raising so the failure is never lost.
            _safely(store.update_run, run_id, status="error", error=message)
        error = WorkflowError(message)
        error.run_id = run_id
        raise error
    if store is not None and run_id:
        decision = final_state.get("decision")
        _safely(
            store.update_run, run_id,
            status=decision.status if decision else "error",
            partial=compute_partial(final_state.get("report")),
            **({"decision_json": decision} if decision else {"error": "workflow ended without a decision"}),
        )
        _safely(store.touch_session, session_id)
    return final_state


def run_discovery_workflow(
    query: str,
    planner: GeminiPlanner | None = None,
    executor: ShopifyMcpExecutor | None = None,
    settings: Settings | None = None,
    tracker: RunTracker | None = None,
    store: SqliteStore | None = None,
    session_id: str | None = None,
    form_inputs: Mapping[str, Any] | None = None,
    policy_checker: PolicyChecker | None = None,
    guard: InputGuard | None = None,
    scorer: RelevanceScorer | None = None,
) -> AgentState:
    """Synchronous convenience wrapper around :func:`arun_discovery_workflow`."""
    return asyncio.run(
        arun_discovery_workflow(
            query, planner=planner, executor=executor, settings=settings, tracker=tracker,
            store=store, session_id=session_id, form_inputs=form_inputs,
            policy_checker=policy_checker, guard=guard, scorer=scorer,
        )
    )
