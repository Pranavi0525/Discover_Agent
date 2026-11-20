"""Search-payload fallback products are unverified candidates in a separate lane.

An unverified price never counts as a verified budget check: the product is kept out of
the verified ranking, shown as a display-only candidate, and never auto-selected.  Only
detail-sourced (or unstamped) products are hard-filtered, ranked and selected.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal

from discovery_agent.config import Settings
from discovery_agent.execution import (
    ShopifyMcpExecutor,
    _failure_kind,
    _TimedOut,
    _with_source,
)
from discovery_agent.orchestration import decide
from discovery_agent.providers.shopify_ucp import ShopifyRateLimited, ShopifyToolError
from discovery_agent.ranking import filter_and_rank_with_stats, unverified_from_stats
from discovery_agent.schemas import (
    ExecutionPlan,
    PlanStep,
    Product,
    ProductVariant,
    ScoredProduct,
    UserIntent,
)
from discovery_agent.semantic import LexicalScorer


def _v(vid, cur, price, seller="Shop One", size="M", avail=True):
    return ProductVariant(
        id=f"gid://shopify/ProductVariant/{vid}",
        price=Decimal(price),
        currency=cur,
        seller=seller,
        url=f"https://{seller.lower().replace(' ', '')}.example/p?variant={vid}",
        available=avail,
        options={"size": size},
    )


def _p(pid, title="Cotton shirt", variants=None, source=None, kind=None, **extra):
    return Product(
        id=f"gid://shopify/p/{pid}",
        title=title,
        url=f"https://example.com/{pid}",
        source="test",
        variants=variants if variants is not None else [_v(pid, "INR", "1200.00")],
        detail_source=source,
        detail_failure_kind=kind,
        **extra,
    )


def _run(products, **intent_kw):
    kw = dict(query="shirt", currency="INR", max_price=Decimal(1500))
    kw.update(intent_kw)
    return filter_and_rank_with_stats(UserIntent(**kw), products, scorer=LexicalScorer())


# ---------------------------------------------------------------- ranking
def test_fallback_product_is_an_unverified_candidate_not_a_ranked_result():
    ranked, stats = _run([_p("FALLBACK0001", source="search_payload", kind="rate_limited")])
    assert ranked == []  # never in the verified ranking
    (candidate,) = unverified_from_stats(stats)
    assert candidate.product.detail_source == "search_payload"
    assert candidate.candidate_price == Decimal("1200.00") and candidate.candidate_currency == "INR"
    assert candidate.unverified_reason == "rate_limited"
    assert stats["kept"] == 0 and stats["unverified"] == 1
    assert stats["products"][0]["outcome"] == "unverified_candidate"
    assert stats["products"][0]["detail_source"] == "search_payload"


def test_verified_product_is_ranked_and_has_no_unverified_lane_entry():
    ranked, stats = _run([_p("DETAIL000001", source="detail")])
    assert ranked[0].product.price_note is None
    assert unverified_from_stats(stats) == []
    assert stats["products"][0]["detail_source"] == "detail"


def test_unstamped_product_is_treated_as_verified():
    ranked, stats = _run([_p("UNSTAMPED001")])
    assert len(ranked) == 1 and ranked[0].product.price_note is None
    assert stats["products"][0]["detail_source"] is None


def test_verified_and_unverified_never_mix_whatever_the_input_order():
    unverified = _p("FALLBACK0002", title="Shirt", source="search_payload", kind="timeout")
    verified = _p("DETAIL000002", title="Long sleeve blue shirt for men", source="detail")
    for order in ([unverified, verified], [verified, unverified]):
        ranked, stats = _run(order)
        assert [s.product.detail_source for s in ranked] == ["detail"]
        assert [c.product.detail_source for c in unverified_from_stats(stats)] == ["search_payload"]


def test_a_fallback_product_is_dropped_only_when_its_own_payload_contradicts_a_constraint():
    # Over budget in INR and USD-only (with INR requested) are contradicted by the payload
    # itself, so they are dropped.  A product with no price at all is unknown, not a
    # violation, so it stays visible as an unverified candidate.
    over = _p("FALLBACK0003", variants=[_v("3", "INR", "2000.00")], source="search_payload", kind="timeout")
    usd = _p("FALLBACK0004", variants=[_v("4", "USD", "15.00")], source="search_payload", kind="timeout")
    no_price = _p("FALLBACK0005", variants=[], source="search_payload", kind="timeout")
    ranked, stats = _run([over, usd, no_price])
    assert ranked == []
    assert [e["outcome"] for e in stats["products"]] == [
        "over_budget", "currency_mismatch", "unverified_candidate"]
    assert stats["dropped"]["over_budget"] == 1 and stats["dropped"]["currency_mismatch"] == 1
    assert stats["dropped"]["no_price"] == 0
    got = {c.product.id[-12:]: (c.candidate_price, c.candidate_currency) for c in unverified_from_stats(stats)}
    assert set(got) == {"FALLBACK0005"}  # only the product nothing contradicts is shown
    assert got["FALLBACK0005"] == (None, None)


def test_counters_do_not_change_for_unmarked_products():
    ranked, stats = _run([_p("PLAIN0000001")])
    assert stats["dropped"]["over_budget"] == 0 and stats["kept"] == len(ranked) == 1


# --------------------------------------------------------------- decide()
def _scored(score):
    return ScoredProduct(product=_p("DECIDE000001", source="detail"), score=score, size_status="available")


def _intent():
    return UserIntent(query="shirt", currency="INR", max_price=Decimal(1500))


def test_verified_best_match_is_still_auto_selected():
    assert decide("shopify-ucp", [_scored(0.9)], _intent()).status == "selected"


def test_decide_only_ever_sees_verified_products_and_its_messages_are_unchanged():
    decision = decide("shopify-ucp", [_scored(0.6)], _intent())
    assert decision.status == "needs_selection"
    assert "could not be verified" not in decision.message


def test_unverified_only_result_is_never_selected():
    ranked, stats = _run([_p("FALLBACK0006", source="search_payload", kind="rate_limited")])
    assert ranked == []  # nothing for decide() to select from
    assert unverified_from_stats(stats)


# ----------------------------------------------------------- execution.py
def test_failure_kind_is_decided_by_exception_type_not_text():
    assert _failure_kind(ShopifyRateLimited("anything", retry_after=1.0)) == "rate_limited"
    assert _failure_kind(ShopifyToolError("rate limit")) == "tool_error"
    assert _failure_kind(_TimedOut("get_product timed out")) == "timeout"
    assert _failure_kind(asyncio.TimeoutError()) == "timeout"
    assert _failure_kind(RuntimeError("Shopify UCP MCP error: Service error")) == "service_error"


def test_with_source_stamps_a_copy_and_records_the_reason():
    original = _p("COPY00000001")
    stamped = _with_source(original, "search_payload", "rate_limited")
    assert stamped.detail_source == "search_payload" and stamped.detail_failure_kind == "rate_limited"
    assert original.detail_source is None and original.detail_failure_kind is None
    assert _with_source(original, "detail").detail_failure_kind is None


def _raw(pid, title, seller):
    return {
        "id": f"gid://shopify/p/{pid}",
        "title": title,
        "url": f"https://{seller}.example/{pid}",
        "variants": [
            {
                "id": f"gid://shopify/ProductVariant/{pid}",
                "seller": {"name": seller},
                "price": {"amount": 120000, "currency": "INR"},
                "availability": {"available": True},
                "url": f"https://{seller}.example/{pid}?variant=1",
            }
        ],
    }


class _FakeSession:
    """search returns four products; get_product behaves differently for each."""

    def __init__(self, settings):
        self.settings = settings
        self.raw = {
            "GOOD": _raw("GOOD", "Cotton shirt good", "shopa"),
            "LIMITED": _raw("LIMITED", "Cotton shirt limited", "shopb"),
            "TOOLERR": _raw("TOOLERR", "Cotton shirt toolerr", "shopc"),
            "EMPTY": _raw("EMPTY", "Cotton shirt empty", "shopd"),
        }

    async def call_tool(self, name, arguments):
        if name == self.settings.shopify_search_tool:
            return {"products": list(self.raw.values())}
        pid = arguments["catalog"]["id"].rsplit("/", 1)[-1]
        if pid == "GOOD":
            return {"product": self.raw["GOOD"]}
        if pid == "LIMITED":
            raise ShopifyRateLimited("Shopify catalog rate limit (HTTP 429)", retry_after=1.0)
        if pid == "TOOLERR":
            raise ShopifyToolError("Shopify rejected the request")
        return {"product": {}}  # no title -> not a valid product


def test_details_node_stamps_sources_reasons_and_counts_failures():
    settings = Settings()
    executor = ShopifyMcpExecutor(session_factory=lambda: _FakeSession(settings), settings=settings)
    intent = UserIntent(query="shirt", currency="INR", max_price=Decimal(1500))
    plan = ExecutionPlan(
        intent=intent,
        steps=[PlanStep(step_id="search_catalog", action="search_catalog", reason="test")],
    )
    report = executor.execute_plan(plan)

    by_id = {p.id.rsplit("/", 1)[-1]: p for p in report.products}
    assert {k: p.detail_source for k, p in by_id.items()} == {
        "GOOD": "detail", "LIMITED": "search_payload", "TOOLERR": "search_payload", "EMPTY": "search_payload",
    }
    assert {k: p.detail_failure_kind for k, p in by_id.items()} == {
        "GOOD": None, "LIMITED": "rate_limited", "TOOLERR": "tool_error", "EMPTY": "invalid_result",
    }
    diag = report.diagnostics
    assert diag["details_ok"] == 1 and diag["details_failed"] == 3
    assert diag["details_from_search_payload"] == 3
    assert diag["details_failure_reasons"] == {"rate_limited": 1, "tool_error": 1, "invalid_result": 1}

    # end to end through ranking: the verified product is ranked, the other three are
    # unverified candidates with their reasons, and none of them is budget-checked.
    ranked, stats = filter_and_rank_with_stats(intent, report.products, scorer=LexicalScorer())
    assert [s.product.detail_source for s in ranked] == ["detail"]
    candidates = unverified_from_stats(stats)
    assert {c.unverified_reason for c in candidates} == {"rate_limited", "tool_error", "invalid_result"}
