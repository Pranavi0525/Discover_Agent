"""Every increment of tool_calls is explained by one ledger entry.

The fake session counts the REAL external calls, so three numbers must agree:
calls the fake saw == report.tool_calls == call_ledger["attempts"].
"""
from __future__ import annotations

from decimal import Decimal

from discovery_agent.config import Settings
from discovery_agent.execution import ShopifyMcpExecutor, _ledger_summary
from discovery_agent.providers.shopify_ucp import ShopifyRateLimited, ShopifyToolError
from discovery_agent.schemas import ExecutionPlan, PlanStep, UserIntent


def _raw(name):
    return {
        "id": f"gid://shopify/p/{name}",
        "title": f"Cotton shirt {name}",
        "url": f"https://{name.lower()}.example/p",
        "variants": [
            {
                "id": f"gid://shopify/ProductVariant/{name}",
                "seller": {"name": name},
                "price": {"amount": 120000, "currency": "INR"},
                "availability": {"available": True},
                "url": f"https://{name.lower()}.example/p?variant=1",
            }
        ],
    }


class _Scripted:
    """Each get_product id has a script of exceptions to raise, then it succeeds."""

    def __init__(self, settings, search_script, product_scripts):
        self.settings = settings
        self.search_script = list(search_script)
        self.scripts = {k: list(v) for k, v in product_scripts.items()}
        self.calls: list[tuple[str, str | None]] = []

    async def call_tool(self, name, arguments):
        if name == self.settings.shopify_search_tool:
            self.calls.append(("search", None))
            if self.search_script:
                raise self.search_script.pop(0)
            return {"products": [_raw(n) for n in self.scripts]}
        target = arguments["catalog"]["id"].rsplit("/", 1)[-1]
        self.calls.append(("get", target))
        if self.scripts[target]:
            raise self.scripts[target].pop(0)
        return {"product": _raw(target)}


def _run(search_script, product_scripts):
    settings = Settings()
    settings.max_retries_per_tool = 2
    settings.max_tool_calls = 30
    session = _Scripted(settings, search_script, product_scripts)
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    plan = ExecutionPlan(
        intent=UserIntent(query="shirt", currency="INR", max_price=Decimal(1500)),
        steps=[PlanStep(step_id="search_catalog", action="search_catalog", reason="test")],
    )
    return executor.execute_plan(plan), session


def _blip():
    return RuntimeError("Shopify UCP MCP error: Service error. Please try again later.")


def _limited():
    return ShopifyRateLimited("Shopify catalog rate limit (HTTP 429)", retry_after=1.0)


def test_ledger_explains_every_tool_call():
    report, session = _run(
        [],
        {
            "RETRY2XX": [_blip(), _blip(), _limited()],  # two generic retries, then a 429
            "FIRST429": [_limited()],                    # a 429 is never retried
            "BLIPONCE": [_blip()],                       # one retry, then it works
            "GOODGOOD": [],
            "TOOLERRX": [ShopifyToolError("rejected")],  # never retried either
        },
    )
    ledger = report.diagnostics["call_ledger"]

    assert ledger["search"] == ["ok"]
    assert ledger["get_product"] == {
        "RETRY2XX": ["service_error", "service_error", "rate_limited"],
        "FIRST429": ["rate_limited"],
        "BLIPONCE": ["service_error", "ok"],
        "GOODGOOD": ["ok"],
        "TOOLERRX": ["tool_error"],
    }
    # 1 search + 5 first attempts + 2 + 1 retries
    assert len(session.calls) == report.tool_calls == ledger["attempts"] == 9
    assert report.diagnostics["tool_calls"] == 9


def test_a_get_product_429_never_adds_a_retry():
    report, session = _run([], {f"PRODUCT{i}": [_limited()] for i in range(4)})
    ledger = report.diagnostics["call_ledger"]
    assert all(v == ["rate_limited"] for v in ledger["get_product"].values())
    assert len(session.calls) == report.tool_calls == ledger["attempts"] == 5


def test_search_429_waits_once_and_is_one_extra_call():
    report, session = _run([ShopifyRateLimited("429", retry_after=0.0)], {"GOODGOOD": []})
    ledger = report.diagnostics["call_ledger"]
    assert ledger["search"] == ["rate_limited", "ok"]
    assert len(session.calls) == report.tool_calls == ledger["attempts"] == 3


def test_ledger_summary_is_compact_and_bounded():
    log = [{"tool": "get_product", "attempt": 0, "outcome": "ok", "target": "ABCDEFGH"}] * 3
    summary = _ledger_summary(log)
    assert summary == {"attempts": 3, "search": [], "get_product": {"ABCDEFGH": ["ok", "ok", "ok"]}}
