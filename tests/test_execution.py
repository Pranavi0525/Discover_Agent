import asyncio

from discovery_agent.config import Settings
from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.schemas import ExecutionPlan, UserIntent


class FakeMcpSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "search_catalog":
            return {"products": [{"id": "p1"}, {"id": "p2"}]}
        await asyncio.sleep(0)
        product_id = arguments["catalog"]["id"]
        return {
            "product": {
                "id": product_id,
                "title": f"Product {product_id}",
                "url": f"https://shop.example/{product_id}",
                "price": {"amount": 2999, "currency": "USD"},
                "currency": "USD",
                "available": True,
            }
        }


class CurrencyAwareMcpSession:
    """Simulates a real UCP endpoint: get_product prices the product in
    whatever buyer currency the request's catalog.context.currency asks for.
    Used to prove the executor now sends that context on get_product too."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "search_catalog":
            return {"products": [{"id": "p1"}]}
        await asyncio.sleep(0)
        catalog = arguments["catalog"]
        product_id = catalog["id"]
        currency = catalog.get("context", {}).get("currency", "USD")
        return {
            "product": {
                "id": product_id,
                "title": f"Product {product_id}",
                "url": f"https://shop.example/{product_id}",
                "price": {"amount": 400000, "currency": currency},
                "currency": currency,
                "available": True,
            }
        }


def test_executor_requests_get_product_in_the_intent_currency():
    """Regression test: previously get_product omitted buyer currency
    context entirely, so a live Shopify UCP catalog would return USD
    pricing for an INR request. ranking.filter_and_rank then rejected
    every product on a currency mismatch, producing a false
    'No matching products were found' for any non-USD query."""
    session = CurrencyAwareMcpSession()
    settings = Settings()
    settings.max_retries_per_tool = 0
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    plan = ExecutionPlan(
        intent=UserIntent(query="running shoes", max_price=4000, currency="INR"),
        steps=[{"step_id": "s1", "action": "search_catalog", "reason": "Find products"}],
    )

    report = executor.execute_plan(plan)

    get_product_calls = [call for call in session.calls if call[0] == "get_product"]
    assert get_product_calls[0][1]["catalog"]["context"]["currency"] == "INR"
    assert len(report.products) == 1
    assert report.products[0].currency == "INR"


def test_executor_calls_search_then_parallel_product_details():
    session = FakeMcpSession()
    settings = Settings()
    settings.shopify_search_tool = "search_catalog"
    settings.shopify_product_tool = "get_product"
    settings.max_retries_per_tool = 0
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    plan = ExecutionPlan(
        intent=UserIntent(query="black running shorts", max_price=40, currency="USD"),
        steps=[{"step_id": "s1", "action": "search_catalog", "reason": "Find products"}],
    )

    report = executor.execute_plan(plan)

    assert [call[0] for call in session.calls] == [
        "search_catalog",
        "get_product",
        "get_product",
    ]
    assert [call[1]["catalog"]["id"] for call in session.calls[1:]] == ["p1", "p2"]
    assert len(report.products) == 2
    assert all(step.status == "SUCCEEDED" for step in report.steps)
