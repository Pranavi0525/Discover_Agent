from decimal import Decimal

from discovery_agent.orchestration import decide, no_match_message
from discovery_agent.ranking import filter_and_rank_with_stats
from discovery_agent.schemas import Product, ScoredProduct, UserIntent


def _product(pid, title, price="1.00", description="", currency="INR"):
    return Product(
        id=pid, title=title, url=f"https://example.com/p/{pid}", price=Decimal(price),
        currency=currency, available=True, merchant="M", description=description,
        source="test",
    )


def _intent(**kw):
    return UserIntent(query="running shoes", max_price=Decimal("500"), currency="INR", **kw)


def test_product_sharing_no_word_with_the_search_is_dropped_as_irrelevant():
    products = [
        _product("1", "Running Shoe X shoes"),
        _product("2", "Skid Washer for snow blowers", description="Aftermarket part"),
    ]
    ranked, stats = filter_and_rank_with_stats(_intent(), products)

    assert [item.product.id for item in ranked] == ["1"]
    assert stats["dropped"]["irrelevant"] == 1
    assert stats["kept"] == 1


def test_any_word_overlap_survives_the_floor():
    ranked, stats = filter_and_rank_with_stats(_intent(), [_product("1", "Running Shoe")])

    assert len(ranked) == 1 and ranked[0].score == 0.5
    assert stats["dropped"]["irrelevant"] == 0


def test_older_drop_reasons_win_over_irrelevant_and_are_counted_unchanged():
    products = [
        _product("1", "Snow blower part", price="9999"),   # over budget AND irrelevant
        _product("2", "Snow blower part", currency="USD"),  # wrong currency AND irrelevant
    ]
    _, stats = filter_and_rank_with_stats(_intent(), products)

    assert stats["dropped"]["over_budget"] == 1
    assert stats["dropped"]["currency_mismatch"] == 1
    assert stats["dropped"]["irrelevant"] == 0


def test_budget_is_untouched_by_the_relevance_floor():
    intent = _intent()
    ranked, _ = filter_and_rank_with_stats(intent, [_product("1", "Running shoes", price="400")])

    assert intent.max_price == Decimal("500") and len(ranked) == 1


def test_no_match_message_names_irrelevant_results():
    log = [{
        "attempt": 1, "query": "running shoes", "error": None, "fetched": 3, "kept": 0,
        "dropped": {"irrelevant": 3}, "currencies_seen": {"INR": 3},
    }]
    message = no_match_message(_intent(), log)

    assert "3 shared no word with your search" in message
    assert "Your budget was not changed." in message


def test_single_partial_match_is_not_described_as_several():
    only = ScoredProduct(product=_product("1", "Running Shoe"), score=0.5)
    decision = decide("shopify-mcp", [only], _intent())

    assert decision.status == "needs_selection"
    assert "Several" not in decision.message
    assert "One possible match" in decision.message


def test_two_partial_matches_still_say_several():
    ranked = [ScoredProduct(product=_product(str(i), "Running Shoe"), score=0.5) for i in (1, 2)]

    assert decide("shopify-mcp", ranked, _intent()).message.startswith("Several products")
def test_a_weak_match_below_the_selection_threshold_asks_for_clarification():
    weak = ScoredProduct(product=_product("1", "Garden shoes"), score=1 / 3)
    decision = decide("shopify-mcp", [weak], _intent())

    assert decision.status == "clarify"
    assert decision.candidates == [weak]
    assert "weak matches" in decision.message


def test_a_strong_match_is_selected():
    strong = ScoredProduct(product=_product("1", "Running shoes"), score=1.0)
    decision = decide("shopify-mcp", [strong], _intent())

    assert decision.status == "selected" and decision.selected == strong