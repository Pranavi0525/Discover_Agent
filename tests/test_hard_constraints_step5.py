"""Step 5: hard constraints may not be bypassed by the unverified lane, a seller's
self-declared name, or missing availability data."""

from decimal import Decimal
from itertools import product as cartesian

import pytest

from discovery_agent.execution import _normalize_product
from discovery_agent.orchestration import no_match_message
from discovery_agent.ranking import filter_and_rank_with_stats
from discovery_agent.schemas import Product, ProductVariant, UserIntent

NIKE = "https://www.nike.com/in/p/shoe-1"
EVIL = "https://shop.evil.example/p/1"
OTHER = "https://www.adidas.com/in/p/2"


def variant(price="4000", currency="INR", available=True, known=True, seller="Nike",
            url=NIKE, vid=None):
    return ProductVariant(
        price=None if price is None else Decimal(price), currency=currency,
        available=available, availability_known=known, seller=seller, url=url, id=vid,
    )


def prod(pid="gid://p/1", variants=None, source="detail", kind=None, merchant="Nike",
         url=NIKE, price=None, currency=None, available=True, known=True):
    return Product(
        id=pid, title="Nike running shoes", url=url, price=price, currency=currency,
        available=available, availability_known=known, merchant=merchant,
        source="shopify-mcp", variants=variants or [],
        detail_source=source, detail_failure_kind=kind,
    )


def payload(variants=None, **kw):
    return prod(variants=variants, source="search_payload", kind="timeout", **kw)


def intent(max_price="5000", currency="INR", merchant=None):
    return UserIntent(
        query="running shoes",
        max_price=None if max_price is None else Decimal(max_price),
        currency=currency, merchant=merchant,
    )


def run(it, *products):
    ranked, stats = filter_and_rank_with_stats(it, list(products))
    return ranked, stats, stats["unverified_candidates"], stats["dropped"]


# ------------------------------------------------------------ 1. availability
def test_explicitly_available_offer_is_recommended():
    ranked, *_ = run(intent(), prod(variants=[variant()]))
    assert len(ranked) == 1


def test_unknown_variant_availability_is_not_recommended():
    ranked, _, _, dropped = run(intent(), prod(variants=[variant(available=False, known=False)]))
    assert ranked == [] and dropped["availability_unknown"] == 1


def test_unknown_product_level_availability_is_not_recommended():
    ranked, _, _, dropped = run(
        intent(currency=None, max_price=None),
        prod(price=Decimal("10"), currency="INR", available=False, known=False),
    )
    assert ranked == [] and dropped["availability_unknown"] == 1


def test_explicit_unavailable_stays_unavailable():
    ranked, _, _, dropped = run(intent(), prod(variants=[variant(available=False)]))
    assert ranked == [] and dropped["unavailable"] == 1


def test_known_available_variant_beats_unknown_sibling():
    ranked, *_ = run(intent(), prod(variants=[
        variant(price="3000", available=False, known=False, vid="a"),
        variant(price="4500", vid="b"),
    ]))
    assert len(ranked) == 1 and ranked[0].product.selected_variant_id == "b"


@pytest.mark.parametrize("raw", [
    {},                                                    # no flag anywhere
    {"available": "maybe"},                                # unparseable
    {"variants": [{"price": {"amount": 100, "currency": "INR"}}]},
])
def test_executor_marks_missing_availability_unknown(raw):
    base = {"id": "p1", "title": "Nike shoe", "url": NIKE, "seller": {"name": "Nike"}}
    product = _normalize_product({**base, **raw}, "p1")
    assert product is not None
    assert product.availability_known is False and product.available is False


@pytest.mark.parametrize("flag,expected", [(True, True), ("true", True), (False, False), ("false", False)])
def test_executor_reads_explicit_flags_strictly(flag, expected):
    base = {"id": "p1", "title": "Nike shoe", "url": NIKE, "available": flag}
    product = _normalize_product(base, "p1")
    assert product.availability_known is True and product.available is expected


def test_executor_variant_flag_string_false_is_false_not_true():
    raw = {"id": "p1", "title": "Nike shoe", "url": NIKE,
           "variants": [{"availableForSale": "false", "price": {"amount": 100, "currency": "INR"}}]}
    product = _normalize_product(raw, "p1")
    variant_ = product.variants[0]
    assert variant_.availability_known is True and variant_.available is False
    assert product.available is False and product.availability_known is True


# ------------------------------------------------------- 2. merchant identity
def test_merchant_verified_by_url_host_is_recommended():
    ranked, *_ = run(intent(merchant="Nike"), prod(variants=[variant()]))
    assert len(ranked) == 1


def test_seller_name_alone_never_matches_merchant():
    v = variant(url=EVIL)
    ranked, _, _, dropped = run(intent(merchant="Nike"), prod(variants=[v], url=EVIL))
    assert ranked == [] and dropped["merchant_unverified"] == 1


def test_other_merchant_is_dropped():
    v = variant(seller="Adidas", url=OTHER)
    ranked, _, _, dropped = run(intent(merchant="Nike"), prod(variants=[v], merchant="Adidas", url=OTHER))
    assert ranked == [] and dropped["other_merchant"] == 1


def test_merchant_domain_request_needs_matching_host():
    ranked, _, _, dropped = run(intent(merchant="nike.com"),
                                prod(variants=[variant(url="https://nike.com.evil.example/x")],
                                     url="https://nike.com.evil.example/x"))
    assert ranked == [] and dropped["other_merchant"] == 1


def test_no_merchant_requested_changes_nothing():
    ranked, *_ = run(intent(merchant=None), prod(variants=[variant(url=EVIL)], url=EVIL))
    assert len(ranked) == 1


# --------------------------------------------------- 3. unverified lane drops
def test_unverified_over_budget_payload_is_dropped():
    ranked, _, shown, dropped = run(intent(), payload(variants=[variant(price="90000")]))
    assert ranked == [] and shown == [] and dropped["over_budget"] == 1


def test_unverified_other_currency_payload_is_dropped():
    _, _, shown, dropped = run(intent(), payload(variants=[variant(price="20", currency="USD")]))
    assert shown == [] and dropped["currency_mismatch"] == 1


def test_unverified_explicitly_unavailable_payload_is_dropped():
    _, _, shown, dropped = run(intent(), payload(variants=[variant(available=False)]))
    assert shown == [] and dropped["unavailable"] == 1


def test_unverified_other_merchant_payload_is_dropped():
    v = variant(seller="Adidas", url=OTHER)
    _, _, shown, dropped = run(intent(merchant="Nike"), payload(variants=[v], merchant="Adidas", url=OTHER))
    assert shown == [] and dropped["other_merchant"] == 1


def test_unverified_keeps_only_offers_nothing_contradicts():
    good = variant(price="4000", vid="good")
    bad = variant(price="90000", seller="Other Store", url=EVIL, vid="bad")
    _, _, shown, _ = run(intent(), payload(variants=[bad, good]))
    assert len(shown) == 1
    assert shown[0]["candidate_price"] == "4000.00" or Decimal(shown[0]["candidate_price"]) == Decimal("4000")
    assert len(shown[0]["listed_offers"]) == 2  # listing kept as returned
    assert len(shown[0]["product"]["variants"]) == 1


def test_unverified_unknown_data_is_still_shown_for_clarification():
    _, _, shown, _ = run(intent(), payload(variants=[variant(price=None, currency=None, known=False, available=False)]))
    assert len(shown) == 1 and shown[0]["budget_status"] == "unverified"


def test_unverified_price_with_no_currency_is_not_compared_or_dropped():
    _, _, shown, _ = run(intent(), payload(variants=[variant(price="90000", currency=None)]))
    assert len(shown) == 1 and shown[0]["requested_currency_status"] == "unknown"


def test_unverified_merchant_status_is_never_claimed_without_url_proof():
    name_only = payload(variants=[variant(url=EVIL)], url=EVIL)
    proven = payload(pid="gid://p/2", variants=[variant(url=NIKE)])
    _, _, shown, _ = run(intent(merchant="Nike"), name_only, proven)
    status = {s["product"]["id"]: s["merchant_status"] for s in shown}
    assert status == {"gid://p/1": "unverified", "gid://p/2": "verified"}


def test_unverified_merchant_not_requested_status():
    _, _, shown, _ = run(intent(merchant=None), payload(variants=[variant()]))
    assert shown[0]["merchant_status"] == "not_requested"


def test_unverified_prefers_url_verified_offer_over_name_only_offer():
    spoof = variant(price="1000", url=EVIL, vid="spoof")
    real = variant(price="4000", url=NIKE, vid="real")
    _, _, shown, _ = run(intent(merchant="Nike"), payload(variants=[spoof, real]))
    assert shown[0]["merchant_status"] == "verified"
    assert [v["id"] for v in shown[0]["product"]["variants"]] == ["real"]


def test_payload_product_level_over_budget_without_variants_is_dropped():
    _, _, shown, dropped = run(intent(), payload(price=Decimal("90000"), currency="INR"))
    assert shown == [] and dropped["over_budget"] == 1


# ------------------------------------------------- 4. regression: no bypass
BUDGETS = [None, "5000"]
CURRENCIES = ["INR", None]
MERCHANTS = [None, "Nike"]
PRICES = [None, "100", "4000", "5000", "5000.01", "90000"]
PRICE_CCY = ["INR", "USD", None]
AVAIL = [(True, True), (False, True), (False, False)]
SELLERS = [("Nike", NIKE), ("Nike", EVIL), ("Other", OTHER)]


@pytest.mark.parametrize("budget,ccy,merchant", list(cartesian(BUDGETS, CURRENCIES, MERCHANTS)))
@pytest.mark.parametrize("lane", ["detail", "search_payload"])
def test_no_lane_can_show_a_hard_constraint_violation(budget, ccy, merchant, lane):
    it = intent(max_price=budget, currency=ccy, merchant=merchant)
    products = []
    for n, (price, pccy, (avail, known), (seller, url)) in enumerate(
        cartesian(PRICES, PRICE_CCY, AVAIL, SELLERS)
    ):
        products.append(prod(
            pid=f"gid://p/{n}", source=lane, kind="timeout" if lane == "search_payload" else None,
            merchant=seller, url=url,
            variants=[variant(price=price, currency=pccy, available=avail, known=known,
                              seller=seller, url=url, vid=f"v{n}")],
        ))
    ranked, stats, shown, _ = run(it, *products)

    def check_offer(price, currency, available, known, seller_url, *, verified_lane):
        if it.currency and currency and currency != it.currency:
            return "currency"
        if verified_lane and not (available and known):
            return "availability"
        if known and not available:
            return "availability"
        if it.max_price is not None and price is not None and currency == it.currency and price > it.max_price:
            return "budget"
        return None

    for item in ranked:
        v = item.product
        assert v.available and v.availability_known
        if it.max_price is not None:
            assert v.price is not None and v.currency == it.currency and v.price <= it.max_price
        if it.merchant:
            assert it.merchant.lower() in v.url.host  # URL host, not the seller name

    for item in shown:
        cand = item["product"]
        for variant_ in cand["variants"]:
            price = None if variant_["price"] is None else Decimal(str(variant_["price"]))
            assert check_offer(price, variant_["currency"], variant_["available"],
                               variant_["availability_known"], None, verified_lane=False) is None
        if it.merchant:
            assert item["merchant_status"] in ("verified", "unverified")
            for variant_ in cand["variants"]:
                assert "adidas" not in (variant_["url"] or "")
        if lane == "detail":
            raise AssertionError("verified-lane products must never become unverified candidates")


# ------------------------------------------------------------- 5. messages
def test_no_match_message_explains_new_drop_reasons():
    it = intent(merchant="Nike")
    attempt = [{
        "query": "running shoes", "error": None, "fetched": 3, "currencies_seen": {"INR": 3},
        "dropped": {"merchant_unverified": 2, "availability_unknown": 1},
    }]
    text = no_match_message(it, attempt)
    assert "could not be verified as sold by Nike" in text
    assert "no confirmed availability" in text
    assert "budget was not changed" in text.lower()


# The unknown flag must win even when a caller left ``available`` True.
def test_unknown_flag_wins_over_available_true_on_variant():
    ranked, _, _, dropped = run(intent(), prod(variants=[variant(available=True, known=False)]))
    assert ranked == [] and dropped["availability_unknown"] == 1


def test_unknown_flag_wins_over_available_true_on_product():
    ranked, _, _, dropped = run(
        intent(currency=None, max_price=None),
        prod(price=Decimal("10"), currency="INR", available=True, known=False),
    )
    assert ranked == [] and dropped["availability_unknown"] == 1


def test_unverified_candidate_keeps_the_full_search_listing_but_prices_from_safe_offers():
    good = variant(price="4000", vid="good")
    bad = variant(price="90000", seller="Other Store", url=EVIL, vid="bad")
    usd = variant(price="20", currency="USD", vid="usd")
    _, _, shown, _ = run(intent(), payload(variants=[bad, usd, good]))
    (c,) = shown
    assert Decimal(c["candidate_price"]) == Decimal("4000") and c["candidate_currency"] == "INR"
    assert [v["id"] for v in c["product"]["variants"]] == ["good"]   # only the safe offer
    assert len(c["listed_offers"]) == 3                               # listing as returned
