"""Regression tests for the live 'No matching products were found' behaviour.

Every test is deterministic (no network): Shopify is replaced by a fake MCP
session or an httpx MockTransport carrying documented UCP response shapes.
"""
import asyncio
import json
from decimal import Decimal

import httpx
import pytest

from ucp_fakes import ScriptedSession, ucp_product
from discovery_agent.config import Settings
from discovery_agent.execution import ShopifyMcpExecutor, _extract_product_ids, _normalize_product
from discovery_agent.orchestration import DiscoveryAgent
from discovery_agent.providers.shopify_ucp import ShopifyUcpCatalogProvider
from discovery_agent.schemas import ExecutionPlan, UserIntent
from discovery_agent.tracking import JsonlRunTracker

INR_4000 = UserIntent(query="running shoes", max_price=Decimal("4000"), currency="INR")


def _settings(**overrides) -> Settings:
    settings = Settings()
    settings.max_retries_per_tool = 0
    settings.shopify_catalog_id = None
    settings.shopify_buyer_country = None
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def _plan(intent: UserIntent = INR_4000) -> ExecutionPlan:
    return ExecutionPlan(
        intent=intent,
        steps=[{"step_id": "s1", "action": "search_catalog", "reason": "search"}],
    )


def _run(session, settings=None, intent=INR_4000, **kwargs):
    settings = settings or _settings()
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    return executor.execute_plan(_plan(intent), **kwargs)


# ---------------------------------------------------------------- product ids
def test_only_top_level_product_ids_are_sent_to_get_product_not_variant_ids():
    products = [ucp_product(1), ucp_product(2), ucp_product(3)]
    session = ScriptedSession(products)

    report = _run(session)

    detail_ids = [c[1]["catalog"]["id"] for c in session.calls if c[0] == "get_product"]
    assert detail_ids == [p["id"] for p in products]
    assert not any("ProductVariant" in i for i in detail_ids)
    assert [c[0] for c in session.calls].count("get_product") == 3  # 6 variants exist
    assert len(report.products) == 3


def test_extract_product_ids_unwraps_json_rpc_structured_content():
    envelope = {"result": {"structuredContent": {"products": [ucp_product(1), ucp_product(2)]}}}
    assert _extract_product_ids(envelope) == ["gid://shopify/p/prod1", "gid://shopify/p/prod2"]


# ------------------------------------------------- MAX_TOOL_CALLS is enforced
def test_max_tool_calls_counts_the_search_call_and_the_detail_calls():
    """Before the fix the counter never left the search node, so MAX_TOOL_CALLS=4
    allowed 1 search + 4 details = 5 calls and report.tool_calls was 0."""
    settings = _settings(max_tool_calls=4)
    session = ScriptedSession([ucp_product(i) for i in range(1, 8)])

    report = _run(session, settings)

    assert len(session.calls) == 4
    assert report.tool_calls == 4
    assert report.graph_iterations == 1


def test_a_smaller_run_budget_is_respected_and_cannot_be_raised():
    session = ScriptedSession([ucp_product(i) for i in range(1, 6)])
    report = _run(session, tool_call_budget=2)
    assert len(session.calls) == 2

    session = ScriptedSession([ucp_product(i) for i in range(1, 6)])
    _run(session, _settings(max_tool_calls=3), tool_call_budget=99)
    assert len(session.calls) == 3


def test_products_beyond_the_budget_still_use_the_search_payload():
    session = ScriptedSession([ucp_product(i) for i in range(1, 6)])
    report = _run(session, tool_call_budget=2)  # search + 1 detail
    assert len(report.products) == 5
    assert [s.status for s in report.steps].count("SKIPPED") == 4


# --------------------------------------------- get_product failure fallback
def test_failed_get_product_falls_back_to_the_search_payload_and_says_so():
    session = ScriptedSession([ucp_product(1, amount=250000)], fail_details=True)

    report = _run(session)

    assert len(report.products) == 1
    assert report.products[0].price == Decimal("2500")
    assert report.products[0].currency == "INR"
    assert report.diagnostics["details_failed"] == 1
    assert report.diagnostics["details_from_search_payload"] == 1
    assert "used search_catalog data" in (report.steps[1].error or "")


def test_one_malformed_product_does_not_abort_the_run():
    bad = ucp_product(1)
    bad["url"] = "not a url"
    bad["variants"] = []
    session = ScriptedSession([bad, ucp_product(2)])

    report = _run(session)

    assert [p.id for p in report.products] == ["gid://shopify/p/prod2"]


# -------------------------------------------------- Shopify reports an error
def test_shopify_error_message_with_no_products_is_a_failure_not_an_empty_result():
    payload = {"products": [], "messages": [{"type": "error", "code": "invalid_input", "content": "bad filter"}]}
    session = ScriptedSession([], search_payload=payload)

    report = _run(session)

    assert report.steps[0].status == "FAILED"
    assert "bad filter" in report.steps[0].error
    assert report.diagnostics["shopify_messages"][0]["code"] == "invalid_input"


# -------------------------------------------------------- request + diagnostics
def test_search_request_carries_minor_unit_price_and_inr_context_and_is_reported():
    session = ScriptedSession([ucp_product(1)])

    report = _run(session)

    sent = session.calls[0][1]["catalog"]
    assert sent["query"] == "running shoes"
    assert sent["filters"]["price"] == {"max": 400000}
    assert sent["context"]["currency"] == "INR"
    assert sent["context"]["address_country"] == "IN"
    assert report.diagnostics["search_request"]["filters"]["price"] == {"max": 400000}
    assert report.diagnostics["max_price_minor_units"] == 400000
    assert report.diagnostics["search_returned"] == 1


@pytest.mark.parametrize(
    "rupees, minor",
    [(500, 50000), (4000, 400000), (8000, 800000), (5000, 500000)],
)
def test_inr_budgets_become_the_expected_minor_units(rupees, minor):
    session = ScriptedSession([])
    _run(session, intent=UserIntent(query="shoes", max_price=Decimal(rupees), currency="INR"))
    assert session.calls[0][1]["catalog"]["filters"]["price"]["max"] == minor


# ------------------------------------------------------------ normalization
def test_normalization_reads_minor_units_seller_and_any_available_variant():
    product = ucp_product(1, amount=399900, seller="Acme Sports")
    product["variants"][0]["availability"] = {"available": False}
    product["variants"][1]["availability"] = {"available": True}

    normalized = _normalize_product({"product": product}, "fallback-id")

    assert normalized.price == Decimal("3999")
    assert normalized.currency == "INR"
    assert normalized.merchant == "Acme Sports"
    assert normalized.available is True
    assert str(normalized.image_url) == "https://cdn.example/1.jpg"
    assert normalized.description.startswith("Lightweight running shoes")


# ------------------------------------------------ budget is never relaxed
def test_a_4500_rupee_product_is_never_returned_for_a_4000_rupee_budget():
    session = ScriptedSession([ucp_product(1, amount=450000)])
    provider = ShopifyUcpCatalogProvider(_settings(), executor=ShopifyMcpExecutor(
        session_factory=lambda: session, settings=_settings()))

    decision = DiscoveryAgent(primary=provider, max_refinements=0).discover(INR_4000)

    assert decision.status == "clarify"
    assert decision.candidates == []
    assert "1 over your budget" in decision.message
    assert "Your budget was not changed" in decision.message


def test_currency_mismatch_is_reported_truthfully_instead_of_generic_no_results(tmp_path):
    session = ScriptedSession([ucp_product(i, amount=9000, currency="USD") for i in range(1, 6)])
    settings = _settings()
    provider = ShopifyUcpCatalogProvider(settings, executor=ShopifyMcpExecutor(
        session_factory=lambda: session, settings=settings))
    tracker = JsonlRunTracker(tmp_path / "runs.jsonl")

    decision = DiscoveryAgent(primary=provider, tracker=tracker, max_refinements=0).discover(INR_4000)

    assert decision.status == "clarify"
    assert "5 priced in USD instead of INR" in decision.message
    ledger = json.loads((tmp_path / "runs.jsonl").read_text().splitlines()[-1])
    # The old ledger hard-coded 0 here, hiding that Shopify DID return products.
    assert ledger["raw_candidate_count"] == 5


# ------------------------------------------- real client: isError, SSE, envelope
def _patch_transport(monkeypatch, handler):
    real = httpx.AsyncClient

    class Patched(real):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Patched)


def _rpc(result):
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def _provider():
    settings = _settings()
    return ShopifyUcpCatalogProvider(settings)


def test_tool_level_is_error_is_surfaced_and_not_reported_as_no_products(monkeypatch):
    def handler(request):
        return httpx.Response(200, json=_rpc({
            "isError": True, "content": [{"type": "text", "text": "filters.price.max must be an integer"}]}))
    _patch_transport(monkeypatch, handler)

    decision = DiscoveryAgent(primary=_provider(), max_refinements=2).discover(INR_4000)

    assert decision.status == "failed"
    assert "filters.price.max must be an integer" in decision.message
    assert decision.attempts == 1  # an error is not refined into more calls


def test_structured_content_over_streamable_http_end_to_end(monkeypatch):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        name = body["params"]["name"]
        if name == "search_catalog":
            return httpx.Response(200, json=_rpc({"structuredContent": {"products": [ucp_product(1), ucp_product(2)]}}))
        product = ucp_product(1) if body["params"]["arguments"]["catalog"]["id"].endswith("prod1") else ucp_product(2)
        return httpx.Response(200, json=_rpc({"structuredContent": {"product": product}}))
    _patch_transport(monkeypatch, handler)

    decision = DiscoveryAgent(primary=_provider()).discover(INR_4000)

    assert decision.status in {"selected", "needs_selection"}
    assert len(decision.candidates) == 2
    names = [b["params"]["name"] for b in seen]
    assert names == ["search_catalog", "get_product", "get_product"]
    search_catalog = seen[0]["params"]["arguments"]["catalog"]
    assert search_catalog["filters"]["price"] == {"max": 400000}
    assert seen[0]["params"]["arguments"]["meta"]["ucp-agent"]["profile"]


def test_server_sent_event_response_is_parsed(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        name = body["params"]["name"]
        result = ({"structuredContent": {"products": [ucp_product(1)]}} if name == "search_catalog"
                  else {"structuredContent": {"product": ucp_product(1)}})
        sse = "event: message\ndata: " + json.dumps(_rpc(result)) + "\n\n"
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
    _patch_transport(monkeypatch, handler)

    decision = DiscoveryAgent(primary=_provider(), max_refinements=0).discover(INR_4000)

    assert len(decision.candidates) == 1
