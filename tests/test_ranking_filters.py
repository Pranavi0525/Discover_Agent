"""Hard filters in ranking.filter_and_rank_with_stats: merchant restriction."""
from decimal import Decimal

from discovery_agent.orchestration import no_match_message
from discovery_agent.ranking import filter_and_rank_with_stats
from discovery_agent.schemas import Product, UserIntent


def _host(merchant: str | None) -> str:
    # Merchant identity is proven by the URL host, never by the seller's own name.
    if merchant and merchant.lower() == "acme sports":
        return "www.acmesports.com"
    return "shop.example"


def _product(product_id: str, merchant: str | None) -> Product:
    return Product(
        id=product_id,
        title="Running shoes",
        url=f"https://{_host(merchant)}/{product_id}",
        price="100",
        currency="INR",
        merchant=merchant,
        description="running shoes",
        source="test",
    )


def test_a_merchant_restriction_drops_other_and_unidentified_merchants():
    intent = UserIntent(
        query="running shoes", max_price=Decimal("500"), currency="INR", merchant="Acme Sports"
    )
    products = [
        _product("exact", "Acme Sports"),
        _product("case-insensitive", "acme sports"),
        _product("other", "Other Shop"),
        _product("unknown", None),
    ]

    ranked, stats = filter_and_rank_with_stats(intent, products)

    assert sorted(item.product.id for item in ranked) == ["case-insensitive", "exact"]
    assert stats["dropped"]["other_merchant"] == 2


def test_a_seller_name_alone_never_satisfies_a_merchant_restriction():
    intent = UserIntent(
        query="running shoes", max_price=Decimal("500"), currency="INR", merchant="Acme Sports"
    )
    spoof = Product(
        id="spoof", title="Running shoes", url="https://shop.evil.example/spoof",
        price="100", currency="INR", merchant="Acme Sports",
        description="running shoes", source="test",
    )

    ranked, stats = filter_and_rank_with_stats(intent, [spoof])

    assert ranked == []
    assert stats["dropped"]["merchant_unverified"] == 1


def test_without_a_merchant_restriction_every_merchant_is_kept():
    intent = UserIntent(query="running shoes", max_price=Decimal("500"), currency="INR")
    products = [_product("a", "Acme Sports"), _product("b", None)]

    ranked, stats = filter_and_rank_with_stats(intent, products)

    assert len(ranked) == 2
    assert stats["dropped"]["other_merchant"] == 0


def test_the_no_result_message_names_merchant_drops_and_keeps_the_budget():
    intent = UserIntent(
        query="running shoes", max_price=Decimal("500"), currency="INR", merchant="Acme Sports"
    )
    log = [{
        "query": "running shoes", "error": None, "fetched": 2,
        "dropped": {"other_merchant": 2}, "currencies_seen": {"INR": 2},
    }]

    message = no_match_message(intent, log)

    assert "2 from another or unidentified merchant" in message
    assert "Your budget was not changed" in message
