import asyncio

import pytest

from discovery_agent.config import Settings
from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.providers.shopify_live import LiveShopifyMcpProvider
from discovery_agent.schemas import UserIntent


class FakeMcpSession:
    def __init__(self, fail_search: bool = False):
        self.calls = []
        self.fail_search = fail_search

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "search_catalog":
            if self.fail_search:
                raise RuntimeError("catalog unavailable")
            return {"products": [{"id": "p1"}]}
        await asyncio.sleep(0)
        return {
            "product": {
                "id": arguments["catalog"]["id"],
                "title": f"Product {arguments['catalog']['id']}",
                "url": f"https://shop.example/{arguments['catalog']['id']}",
                "price": "29.99",
                "currency": "USD",
                "available": True,
            }
        }


def _settings() -> Settings:
    settings = Settings()
    settings.shopify_search_tool = "search_catalog"
    settings.shopify_product_tool = "get_product"
    settings.max_retries_per_tool = 0
    return settings


def test_live_provider_returns_normalized_products():
    settings = _settings()
    session = FakeMcpSession()
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    provider = LiveShopifyMcpProvider(executor=executor, settings=settings)

    result = provider.search(UserIntent(query="black running shorts", max_price=40, currency="USD"))

    assert result.provider == "shopify-mcp"
    assert result.error is None
    assert len(result.products) == 1
    assert result.products[0].title == "Product p1"


def test_live_provider_raises_on_search_failure_with_no_products():
    settings = _settings()
    session = FakeMcpSession(fail_search=True)
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    provider = LiveShopifyMcpProvider(executor=executor, settings=settings)

    with pytest.raises(RuntimeError, match="Shopify MCP search_catalog failed"):
        provider.search(UserIntent(query="black running shorts"))
