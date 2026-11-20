from discovery_agent.providers.tavily import TavilyProvider
from discovery_agent.schemas import UserIntent


def test_tavily_normalizes_compact_product_fields():
    provider = TavilyProvider(search=lambda intent: [{
        "title": "Black shorts",
        "url": "https://shop.example/shorts",
        "price": "$24.50",
        "description": "Running shorts",
        "content": "This long raw page body should not be preferred",
    }])

    result = provider.search(UserIntent(query="black shorts"))

    assert result.products[0].price == 24.50
    assert result.products[0].source == "tavily"
    assert result.products[0].description == "Running shorts"
