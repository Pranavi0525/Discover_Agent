"""Currency-aware offer selection.

The fixture mirrors a real get_product response: one Global Catalog product with a
USD offer from Fashion Nova and an INR offer from a different seller, whose
product-level price_range stamps the USD amount with INR (1500 INR "min").
"""
from __future__ import annotations

from decimal import Decimal

from discovery_agent.execution import _normalize_product
from discovery_agent.ranking import filter_and_rank_with_stats
from discovery_agent.schemas import UserIntent
from discovery_agent.semantic import LexicalScorer

FN_URL = "https://www.fashionnova.com/products/ryland-short-sleeve-button-up-shirt-white"
FG_URL = "https://www.fashiongotti.shop/products/ryland-short-sleeve-button-up-shirt-black?variant=67892535722276"
FN_ID = "gid://shopify/ProductVariant/12204031017084"
FG_ID = "gid://shopify/ProductVariant/67892535722276"


def _variant(vid, title, amount, cur, seller, url, avail=True, size="S"):
    return {
        "id": vid,
        "title": title,
        "price": {"amount": amount, "currency": cur},
        "availability": {"available": avail},
        "seller": {"name": seller},
        "selected_options": [{"name": "Size", "label": size}],
        "url": url,
    }


def _payload(variants, url=None):
    return {
        **({"url": url} if url else {}),
        "id": "gid://shopify/p/5ztEf0eUUlQ2WgS4WxxEqb",
        "title": "Ryland Short Sleeve Button Up Shirt - White",
        "price_range": {"min": {"amount": 1500, "currency": "INR"},
                        "max": {"amount": 200000, "currency": "INR"}},
        "variants": variants,
    }


def _mixed(inr_available=True, inr_url=FG_URL):
    return _normalize_product(_payload([
        _variant(FN_ID, "Ryland Short Sleeve Button Up Shirt - White", 1500, "USD", "Fashion Nova", FN_URL),
        _variant(FG_ID, "Ryland Short Sleeve Button Up Shirt - Black", 200000, "INR",
                 "FASHIONGOTTI", inr_url, avail=inr_available),
    ]), "x")


def _rank(product, currency, budget=None):
    intent = UserIntent(query="shirt", currency=currency,
                        max_price=Decimal(budget) if budget is not None else None)
    ranked, stats = filter_and_rank_with_stats(intent, [product], scorer=LexicalScorer())
    return ranked, stats["dropped"]


# ------------------------------------------------------------ normalization
def test_variant_identity_is_kept():
    p = _mixed()
    usd, inr = p.variants
    assert (usd.id, usd.seller, usd.url) == (FN_ID, "Fashion Nova", FN_URL)
    assert (inr.id, inr.seller, inr.url) == (FG_ID, "FASHIONGOTTI", FG_URL)
    assert inr.title.endswith("Black")


def test_mixed_product_has_no_price_and_no_currency():
    p = _mixed()
    assert p.price is None and p.currency is None and p.price_note
    assert p.selected_variant_id is None


def test_variants_without_identity_still_normalize():
    p = _normalize_product({
        "id": "p", "title": "Shirt", "url": "https://example.com/p",
        "variants": [{"price": {"amount": 5200, "currency": "EUR"}}]}, "p")
    v = p.variants[0]
    assert (v.id, v.title, v.seller, v.url) == (None, None, None, None)


# ------------------------------------------------------------ selection
def test_inr_intent_uses_the_inr_variant_and_its_own_seller_and_url():
    ranked, dropped = _rank(_mixed(), "INR", budget=2500)
    assert dropped["no_price"] == 0
    p = ranked[0].product
    assert (p.price, p.currency) == (Decimal("2000.00"), "INR")
    assert p.selected_variant_id == FG_ID
    assert p.merchant == "FASHIONGOTTI"
    assert str(p.url) == FG_URL
    assert [v.seller for v in p.variants] == ["FASHIONGOTTI"]


def test_usd_variant_never_passes_an_inr_budget_as_if_it_were_rupees():
    # The mislabelled price_range minimum reads "1500 INR" (=15.00); the only
    # real INR offer is 2000.00, so a 1500 budget must reject the product.
    ranked, dropped = _rank(_mixed(), "INR", budget=1500)
    assert ranked == []
    assert dropped["over_budget"] == 1


def test_usd_intent_selects_the_usd_offer():
    ranked, _ = _rank(_mixed(), "USD", budget=20)
    p = ranked[0].product
    assert (p.price, p.currency) == (Decimal("15.00"), "USD")
    assert p.selected_variant_id == FN_ID and p.merchant == "Fashion Nova"
    assert str(p.url) == FN_URL


def test_unavailable_inr_variant_is_excluded():
    ranked, dropped = _rank(_mixed(inr_available=False), "INR", budget=5000)
    assert ranked == []
    assert dropped["unavailable"] == 1


def test_no_variant_in_the_requested_currency_is_a_currency_mismatch():
    ranked, dropped = _rank(_mixed(), "EUR", budget=100)
    assert ranked == []
    assert dropped["currency_mismatch"] == 1


def test_no_intent_currency_keeps_a_mixed_product_unverified():
    ranked, dropped = _rank(_mixed(), None)
    p = ranked[0].product
    assert p.price is None and p.currency is None and p.selected_variant_id is None


def test_no_intent_currency_with_a_budget_is_never_verified():
    ranked, dropped = _rank(_mixed(), None, budget=5000)
    assert ranked == []
    assert dropped["budget_currency_unknown"] == 1


def test_cheapest_available_inr_variant_wins_and_ties_go_to_the_first():
    p = _normalize_product(_payload([
        _variant("a", "A", 300000, "INR", "S1", "https://s1.example/a"),
        _variant("b", "B", 250000, "INR", "S2", "https://s2.example/b", size="M"),
        _variant("c", "C", 250000, "INR", "S3", "https://s3.example/c", size="L"),
        _variant("d", "D", 100000, "INR", "S4", "https://s4.example/d", avail=False),
    ]), "x")
    chosen = _rank(p, "INR", budget=5000)[0][0].product
    assert chosen.selected_variant_id == "b" and chosen.price == Decimal("2500.00")
    assert chosen.merchant == "S2"


def test_malformed_variant_url_is_skipped_not_replaced_by_another_sellers_url():
    p = _normalize_product(_payload([
        _variant("a", "A", 100000, "INR", "S1", "not a url"),
        _variant("b", "B", 200000, "INR", "S2", "https://s2.example/b"),
    ], url="https://example.com/p"), "x")
    chosen = _rank(p, "INR", budget=5000)[0][0].product
    assert chosen.selected_variant_id == "b" and chosen.merchant == "S2"


def test_all_matching_offers_malformed_leaves_the_product_unverified():
    p = _normalize_product(_payload([
        _variant("a", "A", 100000, "INR", "S1", "not a url"),
    ], url="https://example.com/p"), "x")
    # Verified-offer selection fails, so the product is kept but unverified; with
    # a budget set it is then dropped as no_price, never shown with a price.
    ranked, dropped = _rank(p, "INR", budget=5000)
    assert ranked == [] and dropped["no_price"] == 1
    ranked, _ = _rank(p, "INR")
    assert ranked[0].product.price is None and ranked[0].product.price_note


def test_size_availability_only_counts_the_selected_offer():
    p = _normalize_product(_payload([
        _variant(FN_ID, "W", 1500, "USD", "Fashion Nova", FN_URL, size="M"),
        _variant(FG_ID, "B", 200000, "INR", "FASHIONGOTTI", FG_URL, size="S"),
    ]), "x")
    intent = UserIntent(query="shirt", currency="INR", max_price=Decimal(5000), size="M")
    ranked, _ = filter_and_rank_with_stats(intent, [p], scorer=LexicalScorer())
    # Size M exists only in the USD offer, so it must not be reported available.
    assert ranked[0].size_status != "available"


def test_products_without_priced_variants_and_unknown_availability_are_excluded():
    p = _normalize_product({
        "id": "p",
        "title": "Shirt",
        "url": "https://example.com/p",
        "price_range": {
            "min": {"amount": 100000, "currency": "INR"}
        },
        "variants": []
    }, "p")

    ranked, dropped = _rank(p, "INR", budget=5000)

    assert ranked == []
    assert dropped["availability_unknown"] == 1