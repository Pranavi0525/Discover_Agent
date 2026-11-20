"""A missing currency means 'unverifiable', never 'USD' (open item 3)."""
from discovery_agent.execution import _normalize_product
from discovery_agent.orchestration import no_match_message
from discovery_agent.providers.tavily import TavilyProvider
from discovery_agent.ranking import filter_and_rank_with_report
from discovery_agent.schemas import Product, UserIntent


def _raw(price=None, currency=None):
    raw = {"id": "p1", "title": "Running Shoes", "url": "https://example.com/p"}
    if price is not None:
        raw["price"] = {"amount": price} if currency is None else {"amount": price, "currency": currency}
    return raw


def _product(price, currency):
    return Product(id="p", title="Running Shoes", url="https://example.com/p",
                   price=price, currency=currency, source="t")


def test_product_has_no_default_currency():
    p = Product(id="p", title="x", url="https://example.com/p", source="t")
    assert p.currency is None


def test_currency_is_still_upper_cased():
    assert _product(1, "inr").currency == "INR"


def test_normalize_keeps_missing_currency_as_none_with_a_note():
    p = _normalize_product(_raw(price=499900), "p1")
    assert p is not None
    assert p.currency is None
    assert p.price is not None
    assert "currency" in (p.price_note or "").lower()


def test_normalize_malformed_currency_becomes_none_not_usd():
    assert _normalize_product(_raw(price=100, currency="US"), "p1").currency is None
    assert _normalize_product(_raw(price=100, currency="12!"), "p1").currency is None


def test_normalize_stated_currency_is_kept():
    p = _normalize_product(_raw(price=499900, currency="inr"), "p1")
    assert (p.currency, str(p.price)) == ("INR", "4999.00")
    assert p.price_note is None


def _run(intent, products):
    ranked, stats, _ = filter_and_rank_with_report(intent, products)
    return ranked, stats["dropped"]


def test_unknown_currency_is_excluded_from_a_budget():
    intent = UserIntent(query="running shoes", max_price=6000, currency="INR")
    ranked, dropped = _run(intent, [_product(100, None)])
    assert ranked == []
    assert dropped["currency_unknown"] == 1
    assert dropped["currency_mismatch"] == 0
    assert dropped["over_budget"] == 0


def test_unknown_currency_is_excluded_when_a_currency_is_requested():
    intent = UserIntent(query="running shoes", currency="INR")
    ranked, dropped = _run(intent, [_product(100, None)])
    assert ranked == [] and dropped["currency_unknown"] == 1


def test_unknown_currency_without_budget_or_currency_is_kept():
    intent = UserIntent(query="running shoes")
    ranked, dropped = _run(intent, [_product(100, None)])
    assert len(ranked) == 1 and dropped["currency_unknown"] == 0


def test_no_price_and_no_currency_is_not_a_currency_mismatch():
    intent = UserIntent(query="running shoes", currency="INR")
    ranked, dropped = _run(intent, [_product(None, None)])
    assert len(ranked) == 1 and dropped["currency_mismatch"] == 0


def test_budget_still_enforced_for_known_currency():
    intent = UserIntent(query="running shoes", max_price=6000, currency="INR")
    ranked, dropped = _run(intent, [_product(7000, "INR"), _product(5000, "INR"), _product(1, "EUR")])
    assert [s.product.price for s in ranked] == [5000]
    assert dropped["over_budget"] == 1 and dropped["currency_mismatch"] == 1


def test_currencies_seen_labels_missing_currency_unknown():
    intent = UserIntent(query="running shoes", currency="INR")
    _, stats, _ = filter_and_rank_with_report(intent, [_product(1, None)])
    assert stats["currencies_seen"] == {"unknown": 1}


def test_no_match_message_explains_missing_currency_without_blaming_a_currency():
    intent = UserIntent(query="running shoes", max_price=6000, currency="INR")
    log = [{"query": "running shoes", "error": None, "fetched": 3,
            "dropped": {"currency_unknown": 3}, "currencies_seen": {"unknown": 3}}]
    text = no_match_message(intent, log)
    assert "no stated currency" in text
    assert "priced in" not in text
    assert "Your budget was not changed." in text


def test_tavily_without_currency_is_not_usd():
    provider = TavilyProvider(search=lambda intent: [{"url": "https://e.com/a", "title": "x", "price": "10"}])
    product = provider.search(UserIntent(query="running shoes")).products[0]
    assert product.currency is None


def test_tavily_stated_currency_is_kept():
    provider = TavilyProvider(search=lambda intent: [
        {"url": "https://e.com/a", "title": "x", "price": "10", "currency": "eur"}])
    assert provider.search(UserIntent(query="running shoes")).products[0].currency == "EUR"
