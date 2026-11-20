from ..ports import ProviderCallable
from ..schemas import Product, SearchResult, UserIntent
from .tavily import TavilyProvider


class ShopifyProvider:
    """Adapter seam for Akanksha's MCP or Storefront GraphQL callable."""

    name = "shopify"

    def __init__(self, search: ProviderCallable) -> None:
        self._search = search

    def search(self, intent: UserIntent) -> SearchResult:
        products = [TavilyProvider._normalize(item, index).model_copy(update={"source": self.name})
                    for index, item in enumerate(self._search(intent))]
        return SearchResult(provider=self.name, products=products, query=intent.query)
