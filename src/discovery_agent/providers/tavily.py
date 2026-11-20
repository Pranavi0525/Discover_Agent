from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from ..ports import ProviderCallable
from ..schemas import Product, SearchResult, UserIntent


class TavilyProvider:
    name = "tavily"

    def __init__(self, search: ProviderCallable | None = None, api_key: str | None = None) -> None:
        self._search = search
        self._api_key = api_key

    def search(self, intent: UserIntent) -> SearchResult:
        if self._search is None:
            if not self._api_key:
                raise RuntimeError("TAVILY_API_KEY is required to use the Tavily provider")
            try:
                from tavily import TavilyClient
            except ImportError as exc:
                raise RuntimeError("Install the 'tavily' optional dependency to use Tavily") from exc
            client = TavilyClient(api_key=self._api_key)
            raw_results = client.search(query=intent.query, search_depth="basic", max_results=10).get("results", [])
        else:
            raw_results = self._search(intent)
        products = [self._normalize(item, index) for index, item in enumerate(raw_results)]
        return SearchResult(provider=self.name, products=products, query=intent.query)

    @staticmethod
    def _normalize(raw: Mapping[str, Any], index: int) -> Product:
        price = raw.get("price")
        try:
            price = Decimal(str(price).replace("$", "").strip()) if price is not None else None
        except (InvalidOperation, ValueError):
            price = None
        return Product(
            id=str(raw.get("id") or raw.get("url") or f"tavily-{index}"),
            title=str(raw.get("title") or "Untitled product"),
            url=raw["url"],
            price=price,
            currency=raw.get("currency") or None,
            available=bool(raw.get("available", True)),
            merchant=raw.get("merchant"),
            description=str(raw.get("description") or raw.get("content") or "")[:500],
            image_url=raw.get("image_url"),
            source="tavily",
        )
