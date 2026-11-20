"""Store-policy check: "replaceable" and "deliverable in <city>" read from policy pages."""
import asyncio
from decimal import Decimal

import pytest

from discovery_agent.config import Settings
from discovery_agent.constraints import city_country, city_market_note
from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.graph import arun_discovery_workflow
from discovery_agent.policy_check import (
    CONFIRMED,
    CONTRADICTED,
    NOT_FOUND,
    UNCHECKED,
    HttpxFetcher,
    PolicyChecker,
    analyze_returns,
    analyze_shipping,
    apply_policy,
    html_to_text,
)
from discovery_agent.schemas import ExecutionPlan, PlanStep, Product, ScoredProduct, UserIntent
from ucp_fakes import ScriptedSession, ucp_product


# ------------------------------------------------------------------ returns
@pytest.mark.parametrize(
    "text, status",
    [
        ("We offer a 30-day exchange policy on all unworn shoes.", CONFIRMED),
        ("Items can be returned within 15 days of delivery for a full refund.", CONFIRMED),
        ("If your shoes do not fit, we will happily replace them with another size.", CONFIRMED),
        ("Free returns with no questions asked.", CONFIRMED),           # 'no' is not a refusal here
        ("Returns are free, no restocking fee.", CONFIRMED),
        ("All sales are final. Thank you for shopping with us.", NOT_FOUND),  # no return term at all
        ("We do not accept returns or exchanges.", CONTRADICTED),
        ("Returns are not accepted on any item.", CONTRADICTED),
        ("Nothing here about anything relevant.", NOT_FOUND),
    ],
)
def test_analyze_returns_states(text, status):
    assert analyze_returns(text).status == status


def test_no_returns_but_exchange_only_counts_as_exchange():
    finding = analyze_returns("No returns, exchange only within 7 days.")
    assert finding.status == CONFIRMED
    assert "Exchange" in finding.detail
    assert "7 days" in finding.detail


def test_exceptions_are_not_read_as_a_blanket_refusal():
    finding = analyze_returns("We do not accept returns except for defective or damaged items.")
    assert finding.status == NOT_FOUND
    assert "conditions" in finding.detail


def test_window_and_evidence_are_reported():
    finding = analyze_returns("You may exchange any pair within 30 days of delivery.")
    assert finding.status == CONFIRMED
    assert "30 days" in finding.detail
    assert "exchange any pair" in finding.evidence


def test_real_shaped_html_page():
    page = """
    <html><head><style>.x{color:red}</style><script>var refund="no returns";</script></head>
    <body><h1>Refund policy</h1>
    <p>Our policy lasts 30 days. If 30 days have gone by since your purchase,
    unfortunately we can&#39;t offer you a refund or exchange.</p>
    <p>To be eligible for a return, your item must be unused.</p></body></html>
    """
    text = html_to_text(page)
    assert "var refund" not in text and "color:red" not in text
    assert "can't offer" in text
    finding = analyze_returns(text)
    # "eligible for a return" is positive; the 30-day sentence is a negative about day 31.
    assert finding.status == CONFIRMED


# ----------------------------------------------------------------- shipping
def test_shipping_names_the_city():
    finding = analyze_shipping("We deliver to Mumbai, Delhi and Pune in 3-5 days.", city="Mumbai")
    assert finding.status == CONFIRMED and "Mumbai" in finding.detail


def test_shipping_accepts_a_misspelt_city_and_bombay():
    assert analyze_shipping("Orders ship to Bombay within two days.", city="mumbia").status == CONFIRMED


@pytest.mark.parametrize(
    "text",
    ["We ship all over India.", "Delivery across India takes 4-7 days.", "We ship worldwide."],
)
def test_shipping_country_or_worldwide_is_confirmed(text):
    assert analyze_shipping(text, city="Mumbai").status == CONFIRMED


def test_city_mention_beats_country_mention():
    finding = analyze_shipping("We ship across India. Express delivery to Mumbai in 1 day.", city="Mumbai")
    assert "Mumbai" in finding.detail


def test_shipping_refusal_is_contradicted():
    finding = analyze_shipping("We do not ship to India.", city="Mumbai")
    assert finding.status == CONTRADICTED


def test_shipping_that_never_mentions_the_place_is_not_found():
    finding = analyze_shipping("We ship within the United States only.", city="Mumbai")
    assert finding.status == NOT_FOUND


def test_other_cities_are_not_matched():
    assert analyze_shipping("We deliver to Pune.", city="Mumbai").status == NOT_FOUND


# ---------------------------------------------------- constraints.py additions
def test_city_country_and_market_note():
    assert city_country("mumbia") == "IN"
    assert city_country("Atlantis") is None
    assert city_market_note("Mumbai", "IN") is None
    assert city_market_note("Mumbai", None) is None
    assert "Mumbai is in IN" in city_market_note("Mumbai", "US")


# ------------------------------------------------------------------ checker
class FakeFetcher:
    def __init__(self, pages):
        self.pages = pages
        self.urls = []

    def fetch(self, url):
        self.urls.append(url)
        return self.pages.get(url)


def _product(i, host=None):
    return Product(id=f"p{i}", title=f"Running Shoe {i}", description="running shoes",
                   url=f"https://{host or f'shop{i}.example'}/products/s{i}",
                   price=Decimal("5000"), currency="INR", source="test")


def test_pages_are_cached_per_store():
    fetcher = FakeFetcher({"https://a.example/policies/refund-policy": "You can exchange within 10 days."})
    checker = PolicyChecker(fetcher)
    products = [_product(1, "a.example"), _product(2, "a.example"), _product(3, "a.example")]

    result = checker.check_products(products, city=None, return_required=True)

    assert len(fetcher.urls) == 1
    assert all(result[p.id]["returns"]["status"] == CONFIRMED for p in products)
    assert "delivery" not in result["p1"]           # only what was asked for


def test_failed_fetch_is_unchecked_never_confirmed():
    checker = PolicyChecker(FakeFetcher({}))
    result = checker.check_products([_product(1)], city="Mumbai", return_required=True)
    assert result["p1"]["returns"]["status"] == UNCHECKED
    assert result["p1"]["delivery"]["status"] == UNCHECKED


def test_hosts_beyond_the_limit_are_marked_unchecked():
    checker = PolicyChecker(FakeFetcher({}), max_hosts=1)
    result = checker.check_products([_product(1), _product(2)], city=None, return_required=True)
    assert result["p2"]["returns"]["detail"].startswith("Store policy was not checked")


# ------------------------------------------------------------------ ranking
def _scored(i, score):
    return ScoredProduct(product=_product(i), score=score)


def _pages():
    return {
        "https://shop1.example/policies/refund-policy": "No returns, all sales final. We do not accept returns.",
        "https://shop2.example/policies/refund-policy": "We accept returns within 30 days.",
        "https://shop3.example/policies/refund-policy": "Welcome to our store.",
    }


def test_confirmed_moves_up_contradicted_is_dropped_rest_keep_order():
    ranked = [_scored(1, 0.9), _scored(3, 0.8), _scored(2, 0.7)]
    out, info = apply_policy(ranked, city=None, return_required=True,
                             checker=PolicyChecker(FakeFetcher(_pages())))
    assert [s.product.id for s in out] == ["p2", "p3"]          # p1 dropped, confirmed p2 first
    assert info["summary"]["dropped_contradicted"] == ["p1"]
    assert info["products"]["p2"]["returns"]["status"] == CONFIRMED


def test_if_everything_is_contradicted_nothing_is_dropped():
    pages = {f"https://shop{i}.example/policies/refund-policy": "We do not accept returns." for i in (1, 2)}
    out, info = apply_policy([_scored(1, 0.9), _scored(2, 0.8)], city=None, return_required=True,
                             checker=PolicyChecker(FakeFetcher(pages)))
    assert len(out) == 2
    assert info["summary"]["kept_all_because_all_contradicted"] is True


def test_no_constraints_means_no_fetching_at_all():
    fetcher = FakeFetcher({})
    ranked = [_scored(1, 0.9)]
    out, info = apply_policy(ranked, city=None, return_required=False, checker=PolicyChecker(fetcher))
    assert out is ranked and info == {} and fetcher.urls == []


def test_a_crashing_checker_never_breaks_the_search():
    class Boom:
        def check_products(self, *a, **k):
            raise RuntimeError("network down")

    ranked = [_scored(1, 0.9)]
    out, info = apply_policy(ranked, city="Mumbai", return_required=True, checker=Boom())
    assert out == ranked and info["summary"]["error"]


# -------------------------------------------------------------- fetch safety
@pytest.mark.parametrize(
    "url", ["http://shop.example/policies/refund-policy", "https://127.0.0.1/x",
            "https://localhost/x", "https://db.internal/x", "file:///etc/passwd"],
)
def test_fetcher_refuses_unsafe_urls(url):
    assert HttpxFetcher().fetch(url) is None


# -------------------------------------------------------------- whole graph
class _Planner:
    def plan(self, text):
        intent = UserIntent(query="running shoes", currency="INR", max_price=Decimal("8000"),
                            delivery_city="Mumbai", return_required=True)
        return ExecutionPlan(intent=intent, steps=[PlanStep(step_id="s1", action="search_catalog", reason="r")])


class _Factory:
    def __init__(self, products):
        self.products = products

    def __call__(self):
        return ScriptedSession(self.products)


def _run_graph(checker, *, buyer_country="IN"):
    settings = Settings()
    settings.max_retries_per_tool = 0
    settings.shopify_buyer_country = buyer_country
    settings.policy_check = False  # a real .env with POLICY_CHECK=1 must not change these tests
    products = [ucp_product(1, 500000), ucp_product(2, 500000)]
    executor = ShopifyMcpExecutor(session_factory=_Factory(products), settings=settings)
    return asyncio.run(arun_discovery_workflow(
        "running shoes under 8000 rupees, replaceable, deliverable in mumbai",
        planner=_Planner(), executor=executor, settings=settings, policy_checker=checker))


def test_workflow_attaches_policy_findings_and_prefers_the_confirmed_store():
    pages = {
        "https://shop1.example/policies/refund-policy": "Welcome.",
        "https://shop2.example/policies/refund-policy": "Exchange within 14 days.",
        "https://shop2.example/policies/shipping-policy": "We ship across India.",
    }
    state = _run_graph(PolicyChecker(FakeFetcher(pages)))
    decision = state["decision"]

    assert decision.selected.product.url.host == "shop2.example"
    findings = decision.diagnostics["policy"]["products"]
    shop2 = next(v for k, v in findings.items() if v["returns"]["status"] == CONFIRMED)
    assert shop2["delivery"]["status"] == CONFIRMED


def test_workflow_without_a_checker_behaves_as_before():
    decision = _run_graph(None)["decision"]
    assert "policy" not in decision.diagnostics
    assert decision.diagnostics["unverified"]         # still shown as unverified


def test_workflow_warns_when_city_is_outside_the_search_market():
    decision = _run_graph(None, buyer_country="US")["decision"]
    assert "Mumbai is in IN" in decision.diagnostics["market_note"]
    assert "Mumbai is in IN" in decision.message


# ------------------------------------------------------ banner reflects policy
def test_summarize_policy_counts_only_shown_products():
    from discovery_agent.policy_check import summarize_policy
    policy = {"products": {
        "a": {"returns": {"status": CONFIRMED}, "delivery": {"status": NOT_FOUND}},
        "b": {"returns": {"status": NOT_FOUND}, "delivery": {"status": NOT_FOUND}},
        "z": {"returns": {"status": CONFIRMED}, "delivery": {"status": CONFIRMED}},
    }}
    assert summarize_policy(policy, ["a", "b"]) == {
        "returns": {"confirmed": 1, "total": 2}, "delivery": {"confirmed": 0, "total": 2}}
    assert summarize_policy(None, ["a"]) == {}


def test_banner_drops_confirmed_constraints_and_keeps_unconfirmed_ones():
    pages = {
        "https://shop1.example/policies/refund-policy": "Exchange within 14 days.",
        "https://shop2.example/policies/refund-policy": "Welcome.",
        "https://shop1.example/policies/shipping-policy": "Welcome.",
        "https://shop2.example/policies/shipping-policy": "Welcome.",
    }
    decision = _run_graph(PolicyChecker(FakeFetcher(pages)))["decision"]

    names = [u["name"] for u in decision.diagnostics["unverified"]]
    assert names == ["delivery_city"]                      # returns confirmed, Mumbai not
    assert "returns/exchange for 1 of 2 results" in decision.message
    assert "Not verified against catalog data: delivery to Mumbai" in decision.message
    assert "returns / exchange" not in decision.message.split("Not verified")[-1]


def test_banner_unchanged_when_no_store_confirms_anything():
    decision = _run_graph(PolicyChecker(FakeFetcher({})))["decision"]   # every page unreadable
    assert [u["name"] for u in decision.diagnostics["unverified"]] == ["delivery_city", "return_required"]
    assert "Confirmed by store policy" not in decision.message
