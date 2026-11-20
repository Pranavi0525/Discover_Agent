import asyncio
import json
from decimal import Decimal

from discovery_agent.config import Settings
from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.orchestration import DiscoveryAgent
from discovery_agent.providers.shopify import ShopifyProvider
from discovery_agent.providers.shopify_ucp import ShopifyUcpCatalogProvider
from discovery_agent.providers.tavily import TavilyProvider
from discovery_agent.schemas import UserIntent
from discovery_agent.tracking import JsonlRunTracker


def product_payload(title="Black running shorts", price="29.99", available=True):
    return {
        "id": "p-1",
        "title": title,
        "url": "https://example.com/products/p-1",
        "price": price,
        "currency": "USD",
        "available": available,
        "merchant": "Example Shop",
        "description": "Lightweight black running shorts",
    }


def test_primary_provider_selects_high_confidence_result(tmp_path):
    tracker = JsonlRunTracker(tmp_path / "runs.jsonl")
    agent = DiscoveryAgent(
        primary=ShopifyProvider(lambda intent: [product_payload()]),
        tracker=tracker,
    )

    decision = agent.discover(UserIntent(query="black running shorts", max_price=Decimal("40"), currency="USD"))

    assert decision.status == "selected"
    assert decision.selected.product.id == "p-1"
    assert '"provider": "shopify"' in (tmp_path / "runs.jsonl").read_text()


def test_tracker_records_ordered_discovery_steps_with_one_run_id(tmp_path):
    log_path = tmp_path / "runs.jsonl"
    agent = DiscoveryAgent(
        primary=ShopifyProvider(lambda intent: [product_payload()]),
        tracker=JsonlRunTracker(log_path),
    )

    decision = agent.discover(UserIntent(query="black running shorts", max_price=Decimal("40"), currency="USD"))

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "discovery_started",
        "provider_search_started",
        "provider_search_completed",
        "filter_and_rank_completed",
        "decision_created",
        "discovery_completed",
    ]
    assert {record["run_id"] for record in records} == {str(decision.run_id)}
    assert records[3]["raw_product_count"] == 1
    assert records[3]["ranked_product_count"] == 1
    assert records[-1]["status"] == "selected"


def test_primary_miss_falls_back_to_tavily():
    tavily = TavilyProvider(search=lambda intent: [product_payload(title="Blue trail shoes")])
    agent = DiscoveryAgent(
        primary=ShopifyProvider(lambda intent: []),
        fallback=tavily,
    )

    decision = agent.discover(UserIntent(query="blue trail shoes", max_price=Decimal("40"), currency="USD"))

    assert decision.provider == "tavily"
    assert decision.status == "selected"


def test_hard_filters_remove_unavailable_and_over_budget_products():
    agent = DiscoveryAgent(
        primary=ShopifyProvider(lambda intent: [
            product_payload(price="60"),
            product_payload(available=False),
        ])
    )

    decision = agent.discover(UserIntent(query="black running shorts", max_price=Decimal("40"), currency="USD"))

    assert decision.status == "clarify"
    assert decision.candidates == []


def test_missing_tavily_key_is_a_failed_decision():
    agent = DiscoveryAgent(primary=TavilyProvider())

    decision = agent.discover(UserIntent(query="running shorts"))

    assert decision.status == "failed"
    assert "TAVILY_API_KEY" in decision.message


class CurrencyAwareMcpSession:
    """Simulates the live Shopify UCP endpoint end-to-end for the exact
    Streamlit path (ShopifyUcpCatalogProvider -> DiscoveryAgent.discover),
    for an INR query such as 'running shoes for 4000 rupees'."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "search_catalog":
            return {"products": [{"id": "p1"}]}
        await asyncio.sleep(0)
        catalog = arguments["catalog"]
        currency = catalog.get("context", {}).get("currency", "USD")
        return {
            "product": {
                "id": catalog["id"],
                "title": "Running Shoes - Trail Runner",
                "description": "Lightweight running shoes for everyday training",
                "url": "https://shop.example/p1",
                "price": {"amount": 350000, "currency": currency},
                "currency": currency,
                "available": True,
            }
        }


def test_shopify_ucp_provider_selects_a_product_for_an_inr_budget_query():
    """Regression test for the live 'No matching products were found' bug
    on INR queries, through the exact provider/orchestration path the
    Streamlit UI uses."""
    settings = Settings()
    settings.max_retries_per_tool = 0
    executor = ShopifyMcpExecutor(session_factory=lambda: CurrencyAwareMcpSession(), settings=settings)
    provider = ShopifyUcpCatalogProvider(settings, executor=executor)
    agent = DiscoveryAgent(primary=provider)

    decision = agent.discover(UserIntent(query="running shoes", max_price=Decimal("4000"), currency="INR"))

    assert decision.status == "selected"
    assert decision.selected.product.currency == "INR"
    assert decision.selected.product.price <= Decimal("4000")
