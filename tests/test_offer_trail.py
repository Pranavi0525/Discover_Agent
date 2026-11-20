"""Per-product trail: what happened to every fetched product, as logged per attempt.

The trail is diagnostics only: the existing drop counters and the ranking must not change.
"""
from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal

from discovery_agent.ranking import filter_and_rank_with_stats
from discovery_agent.schemas import Product, ProductVariant, UserIntent
from discovery_agent.semantic import LexicalScorer


def _v(vid, cur, price, seller, size="M", avail=True):
    return ProductVariant(
        id=f"gid://shopify/ProductVariant/{vid}",
        price=Decimal(price),
        currency=cur,
        seller=seller,
        url=f"https://{seller.lower().replace(' ', '')}.example/p?variant={vid}",
        available=avail,
        options={"size": size},
    )


def _p(pid, title, variants, **extra):
    return Product(
        id=f"gid://shopify/p/{pid}",
        title=title,
        url=f"https://example.com/{pid}",
        source="test",
        variants=variants,
        **extra,
    )


INR_OK = _p("AAAAAAAAAAAA1", "Cotton shirt", [_v("1001", "INR", "1200.00", "Shop One", size="S"),
                                              _v("1002", "INR", "1400.00", "Shop One", size="M")],
            price=Decimal("1200.00"), currency="INR")
MIXED_OVER = _p("BBBBBBBBBBBB2", "Linen shirt", [_v("2001", "USD", "15.00", "Fashion Nova"),
                                                  _v("2002", "INR", "2000.00", "Gotti")])
USD_ONLY = _p("CCCCCCCCCCCC3", "Denim shirt", [_v("3001", "USD", "20.00", "US Shop")],
              price=Decimal("20.00"), currency="USD")
SOLD_OUT = _p("DDDDDDDDDDDD4", "Silk shirt", [_v("4001", "INR", "900.00", "Shop Four", avail=False)],
              price=Decimal("900.00"), currency="INR", available=False)
UNRELATED = _p("EEEEEEEEEEEE5", "Kitchen Blender", [_v("5001", "INR", "800.00", "Shop Five")],
               price=Decimal("800.00"), currency="INR")

PRODUCTS = [INR_OK, MIXED_OVER, USD_ONLY, SOLD_OUT, UNRELATED]


def _run(products=PRODUCTS, **intent_kw):
    kw = dict(query="shirt", currency="INR", max_price=Decimal(1500))
    kw.update(intent_kw)
    return filter_and_rank_with_stats(UserIntent(**kw), products, scorer=LexicalScorer())


def test_one_entry_per_fetched_product_in_input_order():
    _, stats = _run()
    trail = stats["products"]
    assert [e["id"] for e in trail] == [p.id[-12:] for p in PRODUCTS]
    assert [e["outcome"] for e in trail] == [
        "kept", "over_budget", "currency_mismatch", "unavailable", "irrelevant",
    ]


def test_trail_agrees_with_the_drop_counters():
    ranked, stats = _run()
    outcomes = Counter(e["outcome"] for e in stats["products"])
    assert outcomes["kept"] == stats["kept"] == len(ranked) == 1
    for reason, count in stats["dropped"].items():
        assert outcomes[reason] == count, reason


def test_counters_and_ranking_are_what_they_were():
    ranked, stats = _run()
    assert stats["dropped"] == {
        "unavailable": 1, "currency_mismatch": 1, "currency_unknown": 0, "no_price": 0,
        "budget_currency_unknown": 0, "over_budget": 1, "other_merchant": 0,
        "merchant_unverified": 0, "availability_unknown": 0, "irrelevant": 1,
    }
    assert stats["currencies_seen"] == {"INR": 3, "unknown": 1, "USD": 1}
    assert [s.product.id for s in ranked] == [INR_OK.id]


def test_kept_entry_describes_the_offer_that_was_shown():
    ranked, stats = _run()
    kept = stats["products"][0]
    assert kept["price"] == "1200.00" and kept["currency"] == "INR"
    assert kept["merchant"] == "Shop One"
    assert kept["variant"] == "Variant/1001"  # last 12 chars of the variant id
    assert kept["variant_size"] == "S"
    assert kept["in_currency_span"] == ["1200.00", "1400.00"]
    assert kept["rank"] == 1
    assert kept["size_status"] == "unknown"
    assert 0 <= kept["score"] <= 1
    assert ranked[0].product.selected_variant_id.endswith("1001")


def test_dropped_mixed_product_shows_the_offer_it_was_judged_on():
    _, stats = _run()
    entry = stats["products"][1]
    assert entry["outcome"] == "over_budget"
    assert (entry["price"], entry["currency"], entry["merchant"]) == ("2000.00", "INR", "Gotti")
    assert entry["listed_currency"] is None
    assert entry["variant_currencies"] == ["INR", "USD"]
    assert entry["in_currency_span"] == ["2000.00", "2000.00"]
    assert "rank" not in entry and "score" not in entry


def test_currency_mismatch_entry_has_no_invented_price():
    _, stats = _run()
    entry = stats["products"][2]
    assert entry["outcome"] == "currency_mismatch"
    assert entry["variant_currencies"] == ["USD"]
    assert entry["in_currency_span"] is None
    # judged on its own USD price; it is never relabelled as INR
    assert (entry["price"], entry["currency"]) == ("20.00", "USD")


def test_ranks_follow_the_returned_order():
    a = _p("FFFFFFFFFFFF6", "Shirt", [_v("6001", "INR", "700.00", "S6", size="M")],
           price=Decimal("700.00"), currency="INR")
    b = _p("GGGGGGGGGGGG7", "Long sleeve blue shirt for men", [_v("7001", "INR", "800.00", "S7", size="L")],
           price=Decimal("800.00"), currency="INR")
    ranked, stats = _run([b, a], size="M")
    entries = [e for e in stats["products"] if e["outcome"] == "kept"]
    by_rank = sorted(entries, key=lambda e: e["rank"])
    assert [e["id"] for e in by_rank] == [item.product.id[-12:] for item in ranked]
    assert [e["rank"] for e in by_rank] == [1, 2]
    assert {e["id"]: e["size_status"] for e in entries} == {
        a.id[-12:]: "available", b.id[-12:]: "unavailable",
    }


def test_trail_is_json_serialisable_and_bounded():
    long = _p("HHHHHHHHHHHH8", "shirt " + "x" * 200, [_v("8001", "INR", "100.00", "S8")],
              price=Decimal("100.00"), currency="INR")
    _, stats = _run([long], query="shirt")
    text = json.dumps(stats["products"])
    entry = stats["products"][0]
    assert len(entry["title"]) == 50 and len(entry["id"]) == 12
    assert len(text) < 700


def test_without_an_intent_currency_nothing_is_invented():
    plain = _p("IIIIIIIIIIII9", "Plain shirt", [], price=Decimal("5.00"), currency="USD")
    _, stats = _run([plain], currency=None, max_price=None)
    entry = stats["products"][0]
    assert entry["outcome"] == "kept"
    assert entry["in_currency_span"] is None and entry["variant"] is None
    assert (entry["price"], entry["currency"]) == ("5.00", "USD")


def test_no_products_gives_an_empty_trail():
    ranked, stats = _run([])
    assert ranked == [] and stats["products"] == []
