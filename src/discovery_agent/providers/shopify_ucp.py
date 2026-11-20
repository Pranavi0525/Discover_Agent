"""Shopify UCP Catalog MCP integration over Shopify's MCP HTTP endpoint.

The Shopify Global Catalog currently returns UCP tool results with
``result.structuredContent`` and may omit the MCP ``content`` field.
The generic LangChain MCP adapter version used by this project validates
``CallToolResult`` strictly and therefore rejects that otherwise-valid
Shopify response.  This wrapper calls the MCP JSON-RPC endpoint directly,
while preserving the MCP tool interface expected by the executor.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from ..config import Settings
from ..currency import CURRENCY_TO_COUNTRY, minor_unit_exponent
from ..ports import DiscoveryProvider
from ..schemas import ExecutionPlan, PlanStep, SearchResult, UserIntent


class ShopifyToolError(RuntimeError):
    """Shopify answered, but the MCP tool result itself was an error.

    MCP reports tool failures as ``result.isError = true`` (HTTP 200, no
    JSON-RPC ``error``).  Treating that as "no products" made an invalid
    request indistinguishable from a genuinely empty catalog.
    """


# One catalog response is untrusted network data.  Anything larger than this is
# refused while it is being read, so a hostile or broken server cannot make the
# process hold an unbounded body in memory.
MAX_RESPONSE_BYTES = 2_000_000


class ResponseTooLarge(ShopifyToolError):
    """The response body passed MAX_RESPONSE_BYTES.  Not retried: the same
    request would return the same oversized body."""


class ShopifyRateLimited(RuntimeError):
    """Shopify answered HTTP 429 (too many requests).

    Deliberately NOT a ShopifyToolError subclass: the executor treats the two
    differently.  ``retry_after`` is the server's Retry-After in seconds, or
    None when the header is absent or unparseable.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Retry-After is either delta-seconds or an HTTP-date (RFC 9110 10.2.3)."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - (now or datetime.now(timezone.utc))).total_seconds()
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return None
    return max(0.0, seconds)


def _parse_mcp_http_body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON-RPC envelope from a plain-JSON or SSE response body."""
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        payloads = [
            line[5:].strip()
            for line in response.text.splitlines()
            if line.startswith("data:") and line[5:].strip()
        ]
        if not payloads:
            raise RuntimeError("Shopify UCP MCP sent an empty event stream")
        return json.loads(payloads[-1])
    return response.json()


def _tool_error_text(result: dict[str, Any]) -> str:
    texts: list[str] = []
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("text"):
            texts.append(str(block["text"]))
    structured = result.get("structuredContent")
    if not texts and structured is not None:
        texts.append(json.dumps(structured)[:300])
    return " ".join(texts)[:500] or "no error detail returned"


async def _read_capped(response: httpx.Response, limit: int = MAX_RESPONSE_BYTES) -> httpx.Response:
    """Read a streamed response, stopping at ``limit`` bytes, and return a plain
    Response holding the (already decoded) body for the existing parser."""
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise ResponseTooLarge(f"Shopify response too large ({declared} bytes; limit {limit})")
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body += chunk
        if len(body) > limit:
            raise ResponseTooLarge(f"Shopify response exceeded {limit} bytes")
    return httpx.Response(
        response.status_code,
        headers={"content-type": response.headers.get("content-type", "")},
        content=bytes(body),
        request=response.request,
    )


class ShopifyUcpMcpClient:
    """Small async MCP JSON-RPC client for Shopify UCP Streamable HTTP."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ShopifyUcpMcpClient":
        self._client = httpx.AsyncClient(
            # Bounded: read timeout follows TOOL_TIMEOUT_SECONDS; connecting
            # must not hang either.  (asyncio.wait_for in the executor is a
            # second, outer bound.)
            timeout=httpx.Timeout(
                float(getattr(self.settings, "tool_timeout_seconds", 20.0)),
                connect=10.0,
            ),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if self._client is not None:
            await self._client.aclose()
        self._client = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
        if self._client is None:
            raise RuntimeError("Shopify UCP MCP client is not connected")

        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 1,
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }

        if hasattr(self._client, "stream"):
            async with self._client.stream(
                "POST", self.settings.shopify_ucp_mcp_url, json=payload
            ) as streamed:
                status, retry_header = streamed.status_code, streamed.headers.get("retry-after")
                if status == 429 or status >= 400:
                    response = httpx.Response(
                        status, headers={"retry-after": retry_header or ""} if retry_header else None,
                        request=streamed.request,
                    )  # the error body is never read
                else:
                    response = await _read_capped(streamed)
        else:  # a test double without streaming
            response = await self._client.post(
                self.settings.shopify_ucp_mcp_url,
                json=payload,
            )
        if response.status_code == 429:
            retry_after = parse_retry_after(response.headers.get("retry-after"))
            raise ShopifyRateLimited(
                "Shopify catalog rate limit (HTTP 429)"
                + (
                    f"; Retry-After {retry_after:g}s"
                    if retry_after is not None
                    else "; no Retry-After header"
                ),
                retry_after=retry_after,
            )
        response.raise_for_status()

        data = _parse_mcp_http_body(response)
        if not isinstance(data, dict):
            # Retrying the identical request would return the identical body.
            raise ShopifyToolError("Shopify UCP MCP returned a response that is not a JSON object")

        if "error" in data:
            error = data["error"]
            if isinstance(error, dict):
                message = error.get("message") or str(error)
            else:
                message = str(error)
            raise RuntimeError(f"Shopify UCP MCP error: {message}")

        # Shopify's UCP endpoint returns:
        # {"jsonrpc":"2.0","id":1,"result":{"structuredContent":{...}}}
        # Return the structured UCP payload directly so the executor can
        # consume it without passing through CallToolResult validation.
        result = data.get("result")
        if isinstance(result, dict):
            if result.get("isError"):
                raise ShopifyToolError(
                    f"Shopify UCP tool '{name}' returned an error: "
                    f"{_tool_error_text(result)}"
                )
            if "structuredContent" in result:
                return result["structuredContent"]
            if "content" in result:
                return result["content"]

        return result if result is not None else data


# Buyer country used to localize the Global Catalog when SHOPIFY_BUYER_COUNTRY
# is not configured.  The map lives in currency.py so config and requests agree.
_CURRENCY_TO_COUNTRY = CURRENCY_TO_COUNTRY


def buyer_country(settings: Settings, currency: str | None) -> str | None:
    configured = getattr(settings, "shopify_buyer_country", None)
    if configured:
        return str(configured).upper()
    if currency:
        return _CURRENCY_TO_COUNTRY.get(str(currency).upper())
    return None


def to_minor_units(amount, currency: str | None = None) -> int:
    """Convert a major-unit amount (INR 4000) to UCP minor units (400000).

    UCP amounts are integers in ISO 4217 minor units, so the scale depends on
    the currency's exponent: USD/INR 2 (x100), JPY 0 (x1), KWD 3 (x1000).
    Without a currency the common 2-decimal scale is used.
    """
    scaled = Decimal(str(amount)).scaleb(minor_unit_exponent(currency))
    return int(scaled.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def search_arguments(
    settings: Settings,
    query: str,
    limit: int,
    max_price=None,
    currency: str | None = None,
) -> dict:
    catalog: dict[str, Any] = {
        "query": query,
        "pagination": {
            # Global Catalog search_catalog accepts a maximum page size of 50.
            "limit": min(limit, 50),
        },
        "context": {},
        "filters": {
            "available": True,
        },
    }

    if max_price is not None:
        # UCP puts buyer currency in catalog.context; price bounds are
        # integer minor currency units and do not contain a currency field.
        catalog["filters"]["price"] = {"max": to_minor_units(max_price, currency)}

    if currency:
        catalog["context"]["currency"] = str(currency).upper()

    country = buyer_country(settings, currency)
    if country:
        catalog["context"]["address_country"] = country

    if settings.shopify_catalog_id:
        catalog["catalog_id"] = settings.shopify_catalog_id

    return {
        "meta": {
            "ucp-agent": {
                "profile": settings.shopify_ucp_agent_profile,
            }
        },
        "catalog": catalog,
    }


def product_arguments(settings: Settings, product_id: str, currency: str | None = None) -> dict:
    """Build the current UCP get_product request.

    ``search_catalog`` sends ``catalog.context.currency`` so Shopify prices
    and filters the search results in the buyer's requested currency. If
    ``get_product`` omits that same buyer context, Shopify has no reason to
    return the product priced in that currency and instead returns each
    product's native/default catalog currency (commonly USD). Downstream,
    ``ranking.filter_and_rank`` rejects any product whose ``currency``
    doesn't exactly match ``UserIntent.currency`` (correctly refusing to
    guess an FX conversion for a budget comparison) which silently drops
    every product for any non-USD request (e.g. INR) even though the search
    itself found real, in-budget matches. Sending the same buyer currency
    context on both calls keeps them consistent and fixes that.
    """
    catalog: dict[str, Any] = {"id": product_id}
    context: dict[str, Any] = {}
    if currency:
        context["currency"] = str(currency).upper()
    country = buyer_country(settings, currency)
    if country:
        context["address_country"] = country
    if context:
        catalog["context"] = context

    return {
        "meta": {
            "ucp-agent": {
                "profile": settings.shopify_ucp_agent_profile,
            }
        },
        "catalog": catalog,
    }


class ShopifyUcpCatalogProvider:
    """DiscoveryProvider adapter backed by the bounded LangGraph executor."""

    name = "shopify-ucp"

    def __init__(self, settings: Settings, executor=None) -> None:
        from ..execution import ShopifyMcpExecutor

        self.settings = settings
        self._executor = executor or ShopifyMcpExecutor(settings=settings)
        self._tool_calls_used = 0

    def begin_run(self) -> None:
        """Start a new discovery run: MAX_TOOL_CALLS is a per-run budget shared
        by the first search and every bounded refinement."""
        self._tool_calls_used = 0

    def search(self, intent: UserIntent) -> SearchResult:
        remaining = self.settings.max_tool_calls - self._tool_calls_used
        if remaining <= 0:
            return SearchResult(
                provider=self.name,
                products=[],
                query=intent.query,
                error="MAX_TOOL_CALLS exhausted for this discovery run",
                diagnostics={"stopped": "MAX_TOOL_CALLS", "tool_calls": self._tool_calls_used},
            )

        plan = ExecutionPlan(
            intent=intent,
            steps=[
                PlanStep(
                    step_id="search_catalog",
                    action="search_catalog",
                    reason="Search Shopify UCP catalog for the validated user intent",
                )
            ],
        )
        report = self._executor.execute_plan(plan, tool_call_budget=remaining)
        self._tool_calls_used += report.tool_calls
        failed = [s for s in report.steps if s.status in {"FAILED", "TIMED_OUT"}]
        if not report.products and failed:
            raise RuntimeError("; ".join(s.error or s.status for s in failed))
        return SearchResult(
            provider=self.name,
            products=report.products,
            query=intent.query,
            diagnostics=dict(report.diagnostics),
        )
