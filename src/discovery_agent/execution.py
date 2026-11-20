from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .budget import budget_currency_missing
from .config import Settings, get_settings
from .providers.shopify_ucp import (
    ShopifyRateLimited,
    ShopifyToolError,
    ShopifyUcpMcpClient,
    product_arguments,
    search_arguments,
    to_minor_units,
)
from .currency import minor_unit_exponent
from .schemas import ExecutionPlan, ExecutionReport, Product, ProductVariant, StepExecution


class McpToolSession(Protocol):
    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        ...


class ExecutorError(RuntimeError):
    """Raised when a plan cannot be executed safely."""


class AgentState(BaseModel):
    """LangGraph state with explicit bounded counters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: ExecutionPlan
    report: ExecutionReport | None = None
    product_ids: list[str] = Field(default_factory=list)
    graph_iterations: int = 0
    tool_calls: int = 0
    search_done: bool = False
    details_done: bool = False
    error: str | None = None
    session: Any | None = None
    # Per-run tool-call ceiling; never larger than Settings.max_tool_calls.
    tool_call_limit: int | None = None
    # Raw top-level products from search_catalog, keyed by product id.  Used
    # only as a fallback when get_product fails for that id.
    search_products: dict[str, Any] = Field(default_factory=dict)
    # One entry per MCP attempt (see _log_attempt); explains tool_calls.
    call_log: list[dict[str, Any]] = Field(default_factory=list)


class ShopifyMcpExecutor:
    """LangGraph executor with Python-enforced limits, retries and parallel MCP calls."""

    ALLOWED_TOOLS = {"search_catalog", "get_product"}

    def __init__(
        self,
        session_factory: Callable[[], Any] | None = None,
        settings: Settings | None = None,
    ) -> None:
        load_dotenv()
        self.settings = settings or get_settings()
        self._session_factory = session_factory

    def execute_plan(
        self, plan: ExecutionPlan, tool_call_budget: int | None = None
    ) -> ExecutionReport:
        """Synchronous convenience wrapper around :meth:`aexecute_plan`."""
        return asyncio.run(self.aexecute_plan(plan, tool_call_budget))

    async def aexecute_plan(
        self, plan: ExecutionPlan, tool_call_budget: int | None = None
    ) -> ExecutionReport:
        # Critical boundary: validate the complete plan BEFORE opening an MCP session.
        validated = self._validate_and_bound_plan(plan)
        return await self._execute_graph(validated, tool_call_budget)

    async def _execute_graph(
        self, plan: ExecutionPlan, tool_call_budget: int | None = None
    ) -> ExecutionReport:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise ExecutorError(
                "LangGraph is required at execution time. Run: python -m pip install -e ."
            ) from exc

        # A caller may pass a smaller remaining budget (refinement attempts share
        # MAX_TOOL_CALLS); it can never raise the configured ceiling.
        limit = self.settings.max_tool_calls
        if tool_call_budget is not None:
            limit = max(0, min(int(tool_call_budget), limit))
        initial = AgentState(plan=plan, tool_call_limit=limit)

        graph = StateGraph(AgentState)
        graph.add_node("route", self._route_node)
        graph.add_node("search", self._search_node)
        graph.add_node("details", self._details_node)
        graph.add_node("finish", self._finish_node)

        graph.set_entry_point("route")
        graph.add_conditional_edges(
            "route",
            self._route_after_validation,
            {
                "search": "search",
                "clarify": "finish",
                "finish": "finish",
            },
        )
        graph.add_conditional_edges(
            "search",
            self._route_after_search,
            {
                "details": "details",
                "finish": "finish",
            },
        )
        graph.add_edge("details", "finish")
        graph.add_edge("finish", END)

        app = graph.compile()

        if self._session_factory is None:
            async with ShopifyUcpMcpClient(self.settings) as session:
                initial.session = session  # dynamic private attribute used only inside graph nodes
                result = await app.ainvoke(initial)
        else:
            session = self._session_factory()
            if hasattr(session, "__aenter__"):
                async with session as active:
                    initial.session = active
                    result = await app.ainvoke(initial)
            else:
                initial.session = session
                result = await app.ainvoke(initial)

        state = AgentState.model_validate(result)
        if state.report is None:
            raise ExecutorError("graph completed without an execution report")
        return state.report

    def _route_after_validation(self, state: AgentState) -> str:
        if state.plan.requires_clarification or state.plan.steps[0].action == "ask_clarification":
            return "clarify"
        return "search"

    async def _route_node(self, state: AgentState) -> dict[str, Any]:
        return {"graph_iterations": self._next_iteration(state)}

    async def _search_node(self, state: AgentState) -> dict[str, Any]:
        self._check_iteration(state)
        session = state.session
        search_step = next(s for s in state.plan.steps if s.action == "search_catalog")
        record = StepExecution(step_id=search_step.step_id, action="search_catalog", status="RUNNING")
        intent = state.plan.intent

        request = search_arguments(
            self.settings,
            intent.query,
            self.settings.max_products,
            intent.max_price,
            intent.currency,
        )
        diagnostics: dict[str, Any] = {
            "search_request": request["catalog"],
            "max_price_minor_units": (
                to_minor_units(intent.max_price, intent.currency)
                if intent.max_price is not None
                else None
            ),
        }

        try:
            raw = await self._call_with_limits(state, session, "search_catalog", request)
            unwrapped = _unwrap_mcp_result(raw)
            listed = _top_level_products(unwrapped)
            messages = _extract_messages(unwrapped)
            diagnostics["shopify_messages"] = messages[:5]

            error_messages = [m for m in messages if m.get("type") == "error"]
            if error_messages and not listed:
                detail = "; ".join(
                    str(m.get("content") or m.get("code") or "error") for m in error_messages
                )
                raise ExecutorError(f"Shopify UCP reported an error: {detail}"[:500])

            all_ids = _extract_product_ids(unwrapped)
            ids = all_ids[: self.settings.max_product_details]
            listed = listed[: max(self.settings.max_products * 5, 50)]
            by_id = {
                str(candidate): product
                for product in listed
                if (candidate := _product_identifier(product))
            }
            by_id = {key: value for key, value in by_id.items() if key in ids}
            diagnostics.update(
                {
                    "search_returned": len(listed),
                    "ids_extracted": len(ids),
                    "tool_calls": state.tool_calls,
                    "graph_iterations": state.graph_iterations,
                }
            )
            record.status = "SUCCEEDED"
            record.result_count = len(ids)
            state.report = ExecutionReport(
                plan=state.plan,
                steps=[record],
                products=[],
                graph_iterations=state.graph_iterations,
                tool_calls=state.tool_calls,
                diagnostics=diagnostics,
            )
            return {
                "product_ids": ids,
                "search_products": by_id,
                "search_done": True,
                "report": state.report,
                # LangGraph gives each node its own copy of the state, so a
                # counter mutated inside a node is lost unless it is returned.
                "tool_calls": state.tool_calls,
                "call_log": state.call_log,
            }
        except asyncio.TimeoutError:
            record.status = "TIMED_OUT"
            record.error = "search_catalog timed out"
        except Exception as exc:
            record.status = "FAILED"
            record.error = _safe_error(exc)

        diagnostics["tool_calls"] = state.tool_calls
        diagnostics["call_ledger"] = _ledger_summary(state.call_log)
        diagnostics["search_error"] = record.error
        state.report = ExecutionReport(
            plan=state.plan,
            steps=[record],
            products=[],
            graph_iterations=state.graph_iterations,
            tool_calls=state.tool_calls,
            diagnostics=diagnostics,
        )
        return {
            "search_done": True,
            "report": state.report,
            "error": record.error,
            "tool_calls": state.tool_calls,
            "call_log": state.call_log,
        }

    def _route_after_search(self, state: AgentState) -> str:
        return "details" if state.product_ids and not state.error else "finish"

    async def _details_node(self, state: AgentState) -> dict[str, Any]:
        self._check_iteration(state)
        session = state.session
        ids = list(dict.fromkeys(state.product_ids))[: self.settings.max_product_details]

        # One search + N detail calls must never exceed the per-run total.
        remaining = max(0, self._tool_call_limit(state) - state.tool_calls)
        fetch_ids = ids[:remaining]
        skipped_ids = ids[remaining:]

        results = await asyncio.gather(
            *(self._fetch_product(state, session, pid) for pid in fetch_ids),
            return_exceptions=True,
        )

        report = state.report or ExecutionReport(plan=state.plan)
        details_ok = details_failed = from_search = 0
        failure_reasons: Counter[str] = Counter()
        step_number = 0

        def add_step(status: str, error: str | None, count: int) -> None:
            nonlocal step_number
            step_number += 1
            report.steps.append(
                StepExecution(
                    step_id=f"get_product_{step_number}",
                    action="get_product",
                    status=status,
                    error=error,
                    result_count=count,
                )
            )

        for product_id, result in zip(fetch_ids, results, strict=True):
            failure: str | None = None
            failure_kind: str | None = None
            product: Product | None = None
            if isinstance(result, _TimedOut):
                failure = str(result)
                failure_kind = _failure_kind(result)
            elif isinstance(result, Exception):
                failure = _safe_error(result)
                failure_kind = _failure_kind(result)
            else:
                product = _normalize_product(result, product_id)
                if product is None:
                    failure = "MCP result did not contain a valid product"
                    failure_kind = "invalid_result"

            if product is not None:
                details_ok += 1
                report.products.append(_with_source(product, "detail"))
                add_step("SUCCEEDED", None, 1)
                continue

            details_failed += 1
            kind = failure_kind or "service_error"
            failure_reasons[kind] += 1
            # search_catalog already returned a complete UCP product (title, url,
            # price_range, variants).  Use it rather than losing a real match
            # because one detail call failed - but record that it happened.
            fallback = _normalize_product(state.search_products.get(product_id), product_id)
            if fallback is not None:
                from_search += 1
                report.products.append(_with_source(fallback, "search_payload", kind))
                add_step("SUCCEEDED", f"get_product failed ({failure}); used search_catalog data", 1)
            else:
                status = "TIMED_OUT" if isinstance(result, _TimedOut) else "FAILED"
                add_step(status, failure, 0)

        for product_id in skipped_ids:
            failure_reasons["budget_skipped"] += 1
            fallback = _normalize_product(state.search_products.get(product_id), product_id)
            if fallback is not None:
                from_search += 1
                report.products.append(_with_source(fallback, "search_payload", "budget_skipped"))
                add_step("SKIPPED", "MAX_TOOL_CALLS budget reached; used search_catalog data", 1)
            else:
                add_step("SKIPPED", "MAX_TOOL_CALLS budget reached", 0)

        report.products = _dedupe_products(report.products)[: self.settings.max_products]
        report.graph_iterations = state.graph_iterations
        report.tool_calls = state.tool_calls
        report.diagnostics.update(
            {
                "details_requested": len(fetch_ids),
                "details_ok": details_ok,
                "details_failed": details_failed,
                "details_from_search_payload": from_search,
                "details_failure_reasons": dict(failure_reasons),
                "call_ledger": _ledger_summary(state.call_log),
                "products_normalized": len(report.products),
                "tool_calls": state.tool_calls,
                "graph_iterations": state.graph_iterations,
            }
        )
        return {
            "details_done": True,
            "report": report,
            "tool_calls": state.tool_calls,
            "call_log": state.call_log,
        }

    async def _fetch_product(self, state: AgentState, session: McpToolSession, product_id: str) -> Any:
        try:
            return await self._call_with_limits(
                state,
                session,
                "get_product",
                product_arguments(self.settings, product_id, state.plan.intent.currency),
            )
        except asyncio.TimeoutError:
            return _TimedOut(f"get_product timed out for {product_id}")
        except Exception as exc:
            return exc

    async def _call_with_limits(
        self,
        state: AgentState,
        session: McpToolSession,
        logical_tool: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        if logical_tool not in self.ALLOWED_TOOLS:
            raise ExecutorError(f"tool '{logical_tool}' is not allowlisted")
        if state.tool_calls >= self._tool_call_limit(state):
            raise ExecutorError("MAX_TOOL_CALLS exceeded")

        # Resolve logical names to the configured live MCP names.
        tool_name = (
            self.settings.shopify_search_tool
            if logical_tool == "search_catalog"
            else self.settings.shopify_product_tool
        )

        # Increment before the call: even a timeout/error consumes a call budget.
        state.tool_calls += 1
        last_exc: Exception | None = None
        for attempt in range(self.settings.max_retries_per_tool + 1):
            try:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments),
                    timeout=self.settings.tool_timeout_seconds,
                )
                if getattr(result, "isError", False) or (
                    isinstance(result, dict) and result.get("isError") is True
                ):
                    raise ExecutorError(f"MCP tool {tool_name} returned an error")
                _log_attempt(state, logical_tool, arguments, attempt, "ok")
                return result
            except asyncio.TimeoutError:
                last_exc = asyncio.TimeoutError(f"{tool_name} timed out")
                _log_attempt(state, logical_tool, arguments, attempt, "timeout")
            except ShopifyRateLimited as exc:
                _log_attempt(state, logical_tool, arguments, attempt, "rate_limited")
                # Never an immediate blind retry.  One bounded wait is allowed
                # only for the single serial search call (no thundering herd
                # from the parallel get_product fan-out).  Everything else
                # fails fast and keeps its remaining tool-call budget.
                if not self._may_wait_and_retry(state, logical_tool, attempt, exc):
                    raise
                await asyncio.sleep(exc.retry_after or 0.0)
                state.tool_calls += 1
                continue
            except ShopifyToolError:
                # Shopify rejected the request itself; retrying the identical
                # request cannot succeed and would only burn tool-call budget.
                _log_attempt(state, logical_tool, arguments, attempt, "tool_error")
                raise
            except Exception as exc:
                last_exc = exc
                _log_attempt(state, logical_tool, arguments, attempt, _failure_kind(exc))

            if attempt < self.settings.max_retries_per_tool:
                # Each retry is a separate MCP call and therefore consumes budget.
                if state.tool_calls >= self._tool_call_limit(state):
                    raise ExecutorError("MAX_TOOL_CALLS reached while retrying")
                state.tool_calls += 1

        assert last_exc is not None
        raise last_exc

    def _may_wait_and_retry(
        self,
        state: AgentState,
        logical_tool: str,
        attempt: int,
        exc: ShopifyRateLimited,
    ) -> bool:
        cap = float(getattr(self.settings, "max_retry_after_seconds", 10.0))
        return (
            logical_tool == "search_catalog"
            and attempt == 0
            and exc.retry_after is not None
            and exc.retry_after <= cap
            and state.tool_calls < self._tool_call_limit(state)
        )

    def _validate_and_bound_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        try:
            validated = ExecutionPlan.model_validate(plan.model_dump(mode="python"))
        except ValidationError as exc:
            raise ExecutorError(f"invalid ExecutionPlan: {exc}") from exc

        if len(validated.steps) > self.settings.max_plan_steps:
            raise ExecutorError("execution exceeds MAX_PLAN_STEPS")
        if validated.intent.refinement_count > self.settings.max_refinements:
            raise ExecutorError("execution exceeds MAX_REFINEMENTS")
        if any(step.action == "get_product" and step.product_ids for step in validated.steps):
            raise ExecutorError("planner cannot supply product IDs")
        if validated.requires_clarification and validated.steps[0].action != "ask_clarification":
            raise ExecutorError("requires_clarification must route to ask_clarification")
        if validated.steps[0].action != "ask_clarification" and budget_currency_missing(
            validated.intent
        ):
            # Last line of defence: never search with a budget whose currency is
            # unknown; Shopify would apply the number in some default currency.
            raise ExecutorError(
                "the budget has no currency; refusing to search with a budget that "
                "cannot be enforced (the planner must resolve the currency or ask)"
            )
        return validated

    def _tool_call_limit(self, state: AgentState) -> int:
        if state.tool_call_limit is None:
            return self.settings.max_tool_calls
        return min(state.tool_call_limit, self.settings.max_tool_calls)

    def _next_iteration(self, state: AgentState) -> int:
        value = state.graph_iterations + 1
        if value > self.settings.max_graph_iterations:
            raise ExecutorError("MAX_GRAPH_ITERATIONS exceeded")
        return value

    def _check_iteration(self, state: AgentState) -> None:
        if state.graph_iterations > self.settings.max_graph_iterations:
            raise ExecutorError("MAX_GRAPH_ITERATIONS exceeded")

    async def _finish_node(self, state: AgentState) -> dict[str, Any]:
        report = state.report or ExecutionReport(
            plan=state.plan,
            steps=[
                StepExecution(
                    step_id=state.plan.steps[0].step_id,
                    action=state.plan.steps[0].action,
                    status="WAITING_FOR_USER",
                    error="Clarification required",
                )
            ],
        )
        report.graph_iterations = state.graph_iterations
        report.tool_calls = state.tool_calls
        return {"report": report}


class _TimedOut(Exception):
    pass


def _safe_error(exc: Exception) -> str:
    return str(exc)[:500] or exc.__class__.__name__


def _failure_kind(exc: Exception) -> str:
    """Why a get_product call failed, from the exception TYPE (never its text)."""
    if isinstance(exc, (_TimedOut, asyncio.TimeoutError)):
        return "timeout"
    if isinstance(exc, ShopifyRateLimited):
        return "rate_limited"
    if isinstance(exc, ShopifyToolError):
        return "tool_error"
    if isinstance(exc, ExecutorError):
        return "limit"
    return "service_error"


_CALL_LOG_CAP = 60  # far above MAX_TOOL_CALLS; only a guard against runaway growth

# Catalog data is untrusted.  These caps bound what one hostile or broken
# response can make the executor hold, store or display.
_MAX_MCP_TEXT_CHARS = 2_000_000
_MAX_ID_CHARS = 200
_MAX_TITLE_CHARS = 300
_MAX_MERCHANT_CHARS = 200
_MAX_VARIANTS = 100


def _as_bool(value: Any) -> bool | None:
    """Strict boolean from catalog data, or None when it is not clearly one.

    ``bool("false")`` is True, so a string flag must never go through bool().
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes"):
            return True
        if text in ("false", "0", "no"):
            return False
    return None


def _log_attempt(
    state: AgentState,
    tool: str,
    arguments: Mapping[str, Any],
    attempt: int,
    outcome: str,
) -> None:
    """Record ONE MCP attempt.  Every attempt increments tool_calls exactly once."""
    if len(state.call_log) >= _CALL_LOG_CAP:
        return
    catalog = arguments.get("catalog") if isinstance(arguments, Mapping) else None
    target = catalog.get("id") if isinstance(catalog, Mapping) else None
    state.call_log.append(
        {
            "tool": tool,
            "attempt": attempt,
            "outcome": outcome,
            "target": str(target)[-8:] if target else None,
        }
    )


def _ledger_summary(call_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact form for the run record: outcomes per attempt, grouped by call."""
    search = [e["outcome"] for e in call_log if e.get("tool") == "search_catalog"]
    products: dict[str, list[str]] = {}
    for entry in call_log:
        if entry.get("tool") == "get_product":
            products.setdefault(entry.get("target") or "?", []).append(entry["outcome"])
    return {"attempts": len(call_log), "search": search, "get_product": products}


def _with_source(product: Product, source: str, reason: str | None = None) -> Product:
    """A copy of ``product`` stamped with where its data came from and, for a
    search-payload fallback, why get_product did not supply it."""
    update: dict[str, Any] = {"detail_source": source}
    if reason is not None:
        update["detail_failure_kind"] = reason
    return product.model_copy(update=update)


def _unwrap_mcp_result(value: Any) -> Any:
    """Normalize MCP/JSON-RPC/structured-content result shapes."""
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump()
        except Exception:
            pass

    if isinstance(value, str):
        if len(value) > _MAX_MCP_TEXT_CHARS:
            return {"content": ""}  # oversized: treated as "no usable result"
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {"content": value}

    # JSON-RPC response shape:
    # {"result": {"structuredContent": {...}}}
    if isinstance(value, dict):
        result = value.get("result")
        if isinstance(result, dict):
            if "structuredContent" in result:
                return result["structuredContent"]
            if "content" in result:
                value = result["content"]

        if "structuredContent" in value:
            return value["structuredContent"]

    structured = getattr(value, "structuredContent", None)
    if structured is not None:
        return structured

    content = getattr(value, "content", None)
    if content:
        texts = [getattr(item, "text", "") for item in content]
        text = "\n".join(t for t in texts if isinstance(t, str) and t)
        if len(text) > _MAX_MCP_TEXT_CHARS:
            return {"content": ""}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"content": text}

    return value


def _top_level_products(value: Any) -> list[dict[str, Any]]:
    """Return ONLY the top-level ``products`` array of a UCP response.

    Nested variant IDs are never collected.
    """
    value = _unwrap_mcp_result(value)
    if not isinstance(value, dict):
        return []

    products = value.get("products")
    if not isinstance(products, list):
        # Be tolerant of an MCP envelope that was not unwrapped fully.
        result = value.get("result")
        if isinstance(result, dict):
            structured = result.get("structuredContent")
            if isinstance(structured, dict):
                products = structured.get("products")

    if not isinstance(products, list):
        return []
    return [product for product in products if isinstance(product, dict)]


def _product_identifier(product: dict[str, Any]) -> str | None:
    candidate = product.get("id") or product.get("product_id") or product.get("productId")
    if candidate and isinstance(candidate, (str, int)) and not isinstance(candidate, bool):
        text = str(candidate).strip()
        # The id is sent back to the MCP server as a tool argument: keep it short
        # and free of control characters.
        if text and len(text) <= _MAX_ID_CHARS and text.isprintable():
            return text
    return None


def _extract_messages(value: Any) -> list[dict[str, Any]]:
    """UCP ``messages`` (info/warning/error) from a structured response."""
    value = _unwrap_mcp_result(value)
    if not isinstance(value, dict):
        return []
    messages = value.get("messages")
    if not isinstance(messages, list):
        return []
    return [
        {key: (str(item.get(key))[:200] if item.get(key) is not None else None)
         for key in ("type", "code", "content")}
        for item in messages
        if isinstance(item, dict)
    ]


def _extract_product_ids(value: Any) -> list[str]:
    """Extract only top-level product IDs from a UCP search response.

    UCP search products contain nested variant IDs. Recursively collecting every
    ``id`` causes variants to be sent to ``get_product`` as though they were
    products, which can multiply detail calls and break the executor budget.
    """
    identifiers = [
        identifier
        for product in _top_level_products(value)
        if (identifier := _product_identifier(product))
    ]
    return list(dict.fromkeys(identifiers))


def _normalize_product(value: Any, product_id: str) -> Product | None:
    """Product from untrusted catalog data, or None.  Never raises: one malformed
    entry must not abort the run or discard the valid products beside it."""
    try:
        return _normalize_product_unchecked(value, product_id)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, ArithmeticError):
        return None


def _normalize_product_unchecked(value: Any, product_id: str) -> Product | None:
    value = _unwrap_mcp_result(value)
    if not isinstance(value, dict):
        return None

    product = value.get("product") if isinstance(value.get("product"), dict) else value
    if not isinstance(product, dict) or not isinstance(product.get("title"), str) \
            or not product["title"].strip():
        return None

    variants = product.get("variants")
    variants = variants[:_MAX_VARIANTS] if isinstance(variants, list) else []
    first_variant = variants[0] if variants and isinstance(variants[0], dict) else {}

    url = (
        product.get("url")
        or product.get("product_url")
        or product.get("link")
        or first_variant.get("url")
        or first_variant.get("checkout_url")
    )
    if not url:
        return None

    raw_price = product.get("price") or product.get("price_amount")
    currency = product.get("currency")

    if isinstance(raw_price, dict):
        currency = raw_price.get("currency") or currency
        raw_price = raw_price.get("amount")
    if isinstance(raw_price, bool):
        raw_price = None  # True/False is not an amount

    if raw_price is None:
        price_range = product.get("price_range")
        if isinstance(price_range, dict):
            minimum = price_range.get("min")
            if isinstance(minimum, dict):
                currency = minimum.get("currency") or currency
                raw_price = minimum.get("amount")

    # UCP prices are integer minor currency units; the scale is per currency
    # (JPY has no decimals, KWD has three), not always 100.
    price = raw_price
    if isinstance(raw_price, (int, float)):
        from decimal import Decimal

        price = Decimal(str(raw_price)).scaleb(-minor_unit_exponent(currency))

    # A missing or malformed currency is never defaulted (it used to become
    # "USD").  The product keeps currency=None and the hard filter treats it as
    # unverifiable.
    currency = _clean_currency(currency)

    seller = product.get("seller")
    if not isinstance(seller, dict):
        # Global Catalog puts the seller on each variant, not on the product.
        seller = next(
            (
                variant["seller"]
                for variant in variants
                if isinstance(variant, dict) and isinstance(variant.get("seller"), dict)
            ),
            None,
        )
    merchant = (
        product.get("merchant")
        or product.get("vendor")
        or (seller.get("name") if isinstance(seller, dict) else None)
    )
    merchant = merchant.strip()[:_MAX_MERCHANT_CHARS] if isinstance(merchant, str) else None

    description = product.get("description") or ""
    if isinstance(description, dict):
        description = (
            description.get("plain")
            or description.get("text")
            or description.get("html")
            or ""
        )

    image_url = product.get("image_url") or product.get("imageUrl")
    if not image_url:
        media = product.get("media")
        if isinstance(media, list) and media and isinstance(media[0], dict):
            image_url = media[0].get("url")

    available = _as_bool(product.get("available"))
    if available is None:
        available = _as_bool(product.get("availableForSale"))
    if available is None:
        # A product is purchasable when ANY variant is available, not just the
        # first one.
        flags = [
            variant["availability"].get("available")
            for variant in variants
            if isinstance(variant, dict)
            and isinstance(variant.get("availability"), dict)
            and variant["availability"].get("available") is not None
        ]
        if flags:
            available = any(flags)

    product_variants = _normalize_variants(variants)
    if available is None:
        # Variant flags under other keys (available / availableForSale) count too.
        known_flags = [v.available for v in product_variants if v.availability_known]
        if known_flags:
            available = any(known_flags)
    # A missing flag is "unknown", never "in stock": the product is marked
    # unavailable AND availability_known=False, so ranking can say why.
    availability_known = available is not None
    if available is None:
        available = False

    # Variant prices carry their OWN currency and are the source of truth.  The
    # product-level price_range of an aggregated (multi-seller) product has been
    # seen to stamp a EUR amount with the buyer's currency (EUR 52.00 -> INR 52.00),
    # so it is only used when no variant has a price.
    price_note: str | None = None
    priced = [
        (v.price, v.currency, v.available)
        for v in product_variants
        if v.price is not None and v.currency
    ]
    if priced:
        codes = sorted({code for _, code, _ in priced})
        if len(codes) == 1:
            buyable = [a for a, _, ok in priced if ok] or [a for a, _, _ in priced]
            price, currency = min(buyable), codes[0]
        else:
            # No single currency: leave it unset.  (It used to be the
            # alphabetically first code, which labelled a mixed product "INR".)
            price, currency = None, None
            price_note = (
                "Variants are priced in different currencies ("
                + ", ".join(codes)
                + "); no single price can be checked."
            )

    if price is not None and currency is None and price_note is None:
        price_note = "The catalog did not state a currency; this amount is unverified."

    try:
        return Product(
            id=str(product.get("id") or product_id)[:_MAX_ID_CHARS],
            title=product["title"].strip()[:_MAX_TITLE_CHARS],
            url=url,
            price=price,
            currency=currency,
            price_note=price_note,
            available=bool(available),
            availability_known=availability_known,
            merchant=merchant,
            description=str(description)[:1000],
            image_url=image_url,
            source="shopify-mcp",
            variants=product_variants,
        )
    except ValidationError:
        # One malformed catalog entry (bad URL, negative price...) must not
        # abort the whole run.
        return None


def _clean_currency(value: Any) -> str | None:
    """An ISO-4217-shaped code (three letters, upper-cased) or None; never a guess."""
    if not isinstance(value, str):
        return None
    code = value.strip().upper()
    return code if len(code) == 3 and code.isalpha() else None


def _variant_options(raw: Any) -> dict[str, str]:
    """Option name -> value from the shapes catalogs use.

    [{"name": "Size", "value": "9"}], [{"name": "Size", "label": "9"}] or {"Size": "9"}.
    Names are lower-cased ("Size" -> "size").
    """
    pairs: list[tuple[Any, Any]] = []
    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                value = item.get("value")
                if value is None:
                    value = item.get("label")
                pairs.append((item.get("name") or item.get("key"), value))
    options: dict[str, str] = {}
    for name, value in pairs:
        if isinstance(value, dict):
            value = value.get("label") or value.get("value")
        if name is None or value is None:
            continue
        name, value = str(name).strip().lower(), str(value).strip()
        if name and value:
            options[name] = value
    return options


def _variant_money(raw: dict) -> tuple[Any, str | None]:
    """(major-unit price, ISO currency) of one variant, or (None, None)."""
    from decimal import Decimal

    money = raw.get("price")
    if not isinstance(money, dict):
        return None, None
    amount, code = money.get("amount"), money.get("currency")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not code:
        return None, None
    code = str(code).strip().upper()
    if len(code) != 3:
        return None, None
    return Decimal(str(amount)).scaleb(-minor_unit_exponent(code)), code


def _variant_identity(raw: dict) -> dict[str, str | None]:
    """id, title, seller name and own URL of one variant (None when absent)."""

    def text(value: Any) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    seller = raw.get("seller")
    seller_name = seller.get("name") if isinstance(seller, dict) else seller
    return {
        "id": text(raw.get("id")),
        "title": text(raw.get("title")),
        "seller": text(seller_name),
        "url": text(raw.get("url") or raw.get("checkout_url")),
    }


def _normalize_variants(raw_variants: Any) -> list[ProductVariant]:
    """Variants with their options and whether each can be bought."""
    result: list[ProductVariant] = []
    if not isinstance(raw_variants, list):
        return result
    for raw in raw_variants[:100]:
        if not isinstance(raw, dict):
            continue
        options = _variant_options(
            raw.get("selected_options") or raw.get("selectedOptions") or raw.get("options")
        )
        availability = raw.get("availability")
        flag = availability.get("available") if isinstance(availability, dict) else None
        if flag is None:
            flag = raw.get("available", raw.get("availableForSale"))
        price, price_currency = _variant_money(raw)
        parsed_flag = _as_bool(flag)  # "false" is False; unparseable is unknown
        result.append(
            ProductVariant(
                options=options,
                available=bool(parsed_flag),
                availability_known=parsed_flag is not None,
                price=price,
                currency=price_currency,
                **_variant_identity(raw),
            )
        )
    return result


def _dedupe_products(products: list[Product]) -> list[Product]:
    seen: set[str] = set()
    result: list[Product] = []
    for product in products:
        if product.id in seen:
            continue
        seen.add(product.id)
        result.append(product)
    return result
