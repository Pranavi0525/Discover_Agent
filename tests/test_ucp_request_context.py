from decimal import Decimal

import pytest

from discovery_agent.config import Settings
from discovery_agent.providers.shopify_ucp import (
    buyer_country, product_arguments, search_arguments, to_minor_units,
)


def _settings(country=None) -> Settings:
    settings = Settings()
    settings.shopify_catalog_id = None
    settings.shopify_buyer_country = country
    return settings


@pytest.mark.parametrize(
    "amount, minor",
    [("500", 50000), ("4000", 400000), ("8000", 800000), ("4999.99", 499999), (4000, 400000)],
)
def test_to_minor_units(amount, minor):
    assert to_minor_units(Decimal(str(amount))) == minor


def test_inr_search_sends_indian_buyer_context_and_a_currency_free_price_filter():
    catalog = search_arguments(_settings(), "running shoes", 10, Decimal("4000"), "INR")["catalog"]

    assert catalog["context"] == {"currency": "INR", "address_country": "IN"}
    assert catalog["filters"]["price"] == {"max": 400000}
    assert "currency" not in catalog["filters"]["price"]
    assert catalog["filters"]["available"] is True


def test_get_product_sends_the_same_buyer_context_as_search():
    catalog = product_arguments(_settings(), "gid://shopify/p/x", "inr")["catalog"]
    assert catalog["id"] == "gid://shopify/p/x"
    assert catalog["context"] == {"currency": "INR", "address_country": "IN"}


def test_configured_buyer_country_wins_and_ambiguous_currencies_send_none():
    assert buyer_country(_settings("ae"), "INR") == "AE"
    assert buyer_country(_settings(), "EUR") is None
    catalog = search_arguments(_settings(), "shoes", 10, None, "EUR")["catalog"]
    assert "address_country" not in catalog["context"]
    assert "price" not in catalog["filters"]
