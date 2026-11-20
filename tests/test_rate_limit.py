"""HTTP 429 handling and HTTP timeout (open item 2)."""
import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from discovery_agent.config import Settings
from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.providers.shopify_ucp import (
    ShopifyRateLimited,
    ShopifyUcpMcpClient,
    parse_retry_after,
)
from discovery_agent.schemas import ExecutionPlan, PlanStep, UserIntent


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        intent=UserIntent(query="running shoes"),
        steps=[PlanStep(step_id="s1", action="search_catalog", reason="search")],
    )


class FakeSession:
    """Scripted session: one outcome per call, last outcome repeats."""

    def __init__(self, search_outcomes, detail_outcome=None):
        self.search_outcomes = list(search_outcomes)
        self.detail_outcome = detail_outcome
        self.calls: list[str] = []

    async def call_tool(self, name, arguments):
        self.calls.append(name)
        if name == "get_product":
            outcome = self.detail_outcome
        else:
            idx = min(self.calls.count(name) - 1, len(self.search_outcomes) - 1)
            outcome = self.search_outcomes[idx]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _run(session, monkeypatch, **settings_attrs):
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("discovery_agent.execution.asyncio.sleep", fake_sleep)
    settings = Settings()
    for key, value in settings_attrs.items():
        setattr(settings, key, value)
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    report = asyncio.run(executor.aexecute_plan(_plan()))
    return report, sleeps


EMPTY = {"products": []}


def test_search_429_waits_retry_after_once_then_succeeds(monkeypatch):
    s = FakeSession([ShopifyRateLimited("429", retry_after=3.0), EMPTY])
    report, sleeps = _run(s, monkeypatch)
    assert s.calls == ["search_catalog", "search_catalog"]
    assert sleeps == [3.0]
    assert report.steps[0].status == "SUCCEEDED"
    assert report.tool_calls == 2


def test_search_429_persisting_is_retried_only_once(monkeypatch):
    s = FakeSession([ShopifyRateLimited("Shopify catalog rate limit (HTTP 429)", retry_after=1.0)])
    report, sleeps = _run(s, monkeypatch)
    assert s.calls == ["search_catalog", "search_catalog"]  # 1 try + 1 retry, no more
    assert sleeps == [1.0]
    assert report.steps[0].status == "FAILED"
    assert "429" in report.steps[0].error
    assert "get_product" not in s.calls  # no detail fan-out after a failed search


@pytest.mark.parametrize("retry_after", [None, 60.0])
def test_search_429_without_usable_retry_after_fails_fast(monkeypatch, retry_after):
    s = FakeSession([ShopifyRateLimited("429", retry_after=retry_after)])
    report, sleeps = _run(s, monkeypatch)
    assert s.calls == ["search_catalog"]
    assert sleeps == []
    assert report.steps[0].status == "FAILED"
    assert report.tool_calls == 1


def test_retry_after_cap_is_configurable(monkeypatch):
    s = FakeSession([ShopifyRateLimited("429", retry_after=60.0), EMPTY])
    report, sleeps = _run(s, monkeypatch, max_retry_after_seconds=120.0)
    assert sleeps == [60.0]
    assert report.steps[0].status == "SUCCEEDED"


def test_wait_does_not_overrun_tool_call_budget(monkeypatch):
    s = FakeSession([ShopifyRateLimited("429", retry_after=1.0), EMPTY])
    settings_budget = {"max_tool_calls": 1}
    report, sleeps = _run(s, monkeypatch, **settings_budget)
    assert s.calls == ["search_catalog"]
    assert sleeps == []


def test_get_product_429_is_never_retried(monkeypatch):
    search = {"products": [{"id": "gid://shopify/p/1", "title": "Shoe"}]}
    s = FakeSession([search], detail_outcome=ShopifyRateLimited("429", retry_after=1.0))
    report, sleeps = _run(s, monkeypatch)
    assert s.calls.count("get_product") == 1
    assert sleeps == []
    assert report.diagnostics["details_failed"] == 1


# ---- client -----------------------------------------------------------------

def _client_with(handler) -> ShopifyUcpMcpClient:
    client = ShopifyUcpMcpClient(Settings())
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def test_client_raises_rate_limited_with_numeric_retry_after():
    client = _client_with(lambda req: httpx.Response(429, headers={"Retry-After": "7"}))
    with pytest.raises(ShopifyRateLimited) as info:
        asyncio.run(client.call_tool("search_catalog", {}))
    assert info.value.retry_after == 7.0


def test_client_429_without_header_has_none():
    client = _client_with(lambda req: httpx.Response(429))
    with pytest.raises(ShopifyRateLimited) as info:
        asyncio.run(client.call_tool("search_catalog", {}))
    assert info.value.retry_after is None


def test_client_success_path_unchanged():
    body = {"jsonrpc": "2.0", "id": 1, "result": {"structuredContent": {"products": []}}}
    client = _client_with(lambda req: httpx.Response(200, json=body))
    assert asyncio.run(client.call_tool("search_catalog", {})) == {"products": []}


def test_client_other_http_errors_still_raise_status_error():
    client = _client_with(lambda req: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.call_tool("search_catalog", {}))


def test_client_has_bounded_timeouts():
    async def go():
        settings = Settings()
        settings.tool_timeout_seconds = 12.0
        async with ShopifyUcpMcpClient(settings) as c:
            return c._client.timeout

    t = asyncio.run(go())
    assert t.read == 12.0 and t.connect == 10.0 and t.pool == 12.0


def test_parse_retry_after_forms():
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after(" 2.5 ") == 2.5
    assert parse_retry_after("-3") == 0.0
    assert parse_retry_after("nan") is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("") is None
    assert parse_retry_after(None) is None
    now = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after(format_datetime(now + timedelta(seconds=30), usegmt=True), now) == 30.0
    assert parse_retry_after(format_datetime(now - timedelta(seconds=30), usegmt=True), now) == 0.0
