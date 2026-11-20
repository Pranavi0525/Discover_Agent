"""Shared builders that mirror the documented Shopify Global Catalog UCP shapes."""
from __future__ import annotations

import asyncio
from typing import Any


def ucp_product(index: int, amount: int = 300000, currency: str = "INR", *, title: str | None = None,
                variants: int = 2, available: bool = True, seller: str = "Example Running") -> dict[str, Any]:
    """A top-level UCP product with NESTED variant ids, as search_catalog returns."""
    pid = f"gid://shopify/p/prod{index}"
    return {
        "id": pid,
        "title": title or f"Running Shoe {index}",
        "description": {"plain": "Lightweight running shoes for daily training"},
        "url": f"https://shop{index}.example/products/shoe-{index}",
        "price_range": {
            "min": {"amount": amount, "currency": currency},
            "max": {"amount": amount + 50000, "currency": currency},
        },
        "media": [{"type": "image", "url": f"https://cdn.example/{index}.jpg"}],
        "variants": [
            {
                "id": f"gid://shopify/ProductVariant/{index}{v}",
                "price": {"amount": amount, "currency": currency},
                "availability": {"available": available},
                "seller": {"name": seller},
            }
            for v in range(variants)
        ],
    }


class ScriptedSession:
    """Fake MCP session: search returns ``products``; get_product returns the
    matching product wrapped as ``{"product": ...}`` unless ``fail_details``."""

    def __init__(self, products: list[dict[str, Any]], *, fail_details: bool = False,
                 search_payload: dict[str, Any] | None = None) -> None:
        self.products = products
        self.fail_details = fail_details
        self.search_payload = search_payload
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: Any) -> Any:
        self.calls.append((name, dict(arguments)))
        await asyncio.sleep(0)
        if name == "search_catalog":
            if self.search_payload is not None:
                return self.search_payload
            return {"products": self.products}
        if self.fail_details:
            raise RuntimeError("get_product exploded")
        wanted = arguments["catalog"]["id"]
        for product in self.products:
            if product["id"] == wanted:
                return {"product": product}
        return {}
