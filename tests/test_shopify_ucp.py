from decimal import Decimal

from discovery_agent.config import Settings
from discovery_agent.providers.shopify_ucp import product_arguments, search_arguments


def test_ucp_search_request_uses_context_currency_and_minor_unit_price():
    settings = Settings()
    settings.shopify_catalog_id = None

    payload = search_arguments(
        settings=settings,
        query="shoes",
        limit=10,
        max_price=Decimal("500"),
        currency="INR",
    )

    catalog = payload["catalog"]

    assert catalog["query"] == "shoes"
    assert catalog["pagination"]["limit"] == 10
    assert catalog["context"]["currency"] == "INR"
    assert catalog["filters"]["available"] is True
    assert catalog["filters"]["price"]["max"] == 50000
    assert "currency" not in catalog["filters"]["price"]


def test_ucp_search_request_caps_global_catalog_page_size_at_50():
    settings = Settings()
    payload = search_arguments(settings, "shoes", 250, None, "INR")
    assert payload["catalog"]["pagination"]["limit"] == 50


def test_ucp_get_product_request_uses_catalog_id_and_agent_profile():
    settings = Settings()
    payload = product_arguments(settings, "gid://shopify/p/test")

    assert payload["meta"]["ucp-agent"]["profile"] == settings.shopify_ucp_agent_profile
    assert payload["catalog"]["id"] == "gid://shopify/p/test"


def test_ucp_get_product_request_sends_the_same_buyer_currency_as_search():
    """Regression test for the live 'No matching products' bug.

    search_arguments() sends catalog.context.currency so Shopify prices the
    search results in the buyer's currency. get_product must send the same
    buyer context, or Shopify returns each product's native/default
    currency instead, which then fails the exact-currency match in
    ranking.filter_and_rank() and silently drops every result.
    """
    settings = Settings()
    payload = product_arguments(settings, "gid://shopify/p/test", currency="inr")

    assert payload["catalog"]["context"]["currency"] == "INR"


def test_ucp_get_product_request_without_a_currency_sends_none_not_usd():
    settings = Settings()
    settings.shopify_buyer_country = None
    payload = product_arguments(settings, "gid://shopify/p/test")

    assert "context" not in payload["catalog"]  # no invented currency or country
