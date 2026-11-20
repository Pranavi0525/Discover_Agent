"""Verified offers vs unverified candidates: ranking, trail, executor, graph, SQLite.

Every test is deterministic (no network).  Shopify is replaced by the repo's
ScriptedSession; nothing here needs a Gemini key.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from decimal import Decimal

import pytest

from ucp_fakes import ScriptedSession, ucp_product
from discovery_agent import graph as graph_module
from discovery_agent import service
from discovery_agent.config import Settings
from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.graph import arun_discovery_workflow
from discovery_agent.orchestration import DiscoveryAgent
from discovery_agent.providers.shopify_ucp import ShopifyUcpCatalogProvider
from discovery_agent.ranking import filter_and_rank_with_stats, unverified_from_stats
from discovery_agent.schemas import (
    DiscoveryDecision,
    ExecutionPlan,
    PlanStep,
    Product,
    ProductVariant,
    UserIntent,
)
from discovery_agent.semantic import LexicalScorer
from discovery_agent.storage import SqliteStore


# ------------------------------------------------------------------ builders
def _v(vid, cur, price, seller="Shop One", size="M", avail=True):
    return ProductVariant(
        id=f"gid://shopify/ProductVariant/{vid}", price=Decimal(price), currency=cur, seller=seller,
        url=f"https://{seller.lower().replace(' ', '')}.example/p?variant={vid}",
        available=avail, options={"size": size},
    )


def _p(pid, title="Cotton shirt", variants=None, source=None, kind=None, **extra):
    return Product(
        id=f"gid://shopify/p/{pid}", title=title, url=f"https://example.com/{pid}", source="test",
        variants=variants if variants is not None else [_v(pid, "INR", "1200.00")],
        detail_source=source, detail_failure_kind=kind, **extra,
    )


def _payload(pid, title="Cotton shirt", variants=None, kind="rate_limited", **extra):
    return _p(pid, title, variants, source="search_payload", kind=kind, **extra)


def _run(products, **intent_kw):
    kw = dict(query="shirt", currency="INR", max_price=Decimal(1500))
    kw.update(intent_kw)
    max_unverified = kw.pop("max_unverified", 3)
    return filter_and_rank_with_stats(
        UserIntent(**kw), products, scorer=LexicalScorer(), max_unverified=max_unverified
    )


# ------------------------------------------------------- ranking: the two lanes
def test_verified_and_unverified_products_land_in_separate_lanes():
    good = _p("DETAIL000001", source="detail")
    cand = _payload("PAYLOAD00001")
    ranked, stats = _run([cand, good])
    assert [s.product.id for s in ranked] == [good.id]
    assert [c.product.id for c in unverified_from_stats(stats)] == [cand.id]
    assert stats["kept"] == 1 and stats["unverified"] == 1


def test_an_unverified_price_never_establishes_budget_compliance():
    under = _payload("PAYLOAD00002", variants=[_v("a", "INR", "1200.00")])
    over = _payload("PAYLOAD00003", variants=[_v("b", "INR", "2000.00")])
    ranked, stats = _run([under, over])
    assert ranked == []  # not ranked, so never selectable
    candidates = unverified_from_stats(stats)
    assert {c.budget_status for c in candidates} == {"unverified"}
    assert [c.product.id[-12:] for c in candidates] == ["PAYLOAD00002"]  # only the 1200 offer
    assert stats["dropped"]["over_budget"] == 1  # the 2000 offer contradicts the budget in INR
    # the very same offers WITH details are judged normally
    detail_over = _p("DETAIL000002", variants=[_v("c", "INR", "2000.00")], source="detail")
    ranked, stats = _run([detail_over])
    assert ranked == [] and stats["dropped"]["over_budget"] == 1


def test_a_usd_only_payload_is_dropped_as_a_currency_mismatch():
    cand = _payload("PAYLOAD00004", variants=[_v("d", "USD", "15.00")], kind="service_error")
    ranked, stats = _run([cand])
    assert unverified_from_stats(stats) == []  # a USD-only payload contradicts an INR request
    assert stats["dropped"]["currency_mismatch"] == 1
    assert stats["products"][0]["outcome"] == "currency_mismatch"
    # with details the same USD-only product is a verified currency mismatch
    ranked, stats = _run([_p("DETAIL000003", variants=[_v("d", "USD", "15.00")], source="detail")])
    assert ranked == [] and stats["dropped"]["currency_mismatch"] == 1


def test_a_payload_price_with_no_currency_is_kept_unconverted_and_unlabelled():
    cand = _payload("PAYLOAD00005", variants=[], price=Decimal("5.00"), currency=None)
    ranked, stats = _run([cand])
    (c,) = unverified_from_stats(stats)
    assert c.candidate_price == Decimal("5.00") and c.candidate_currency is None
    assert c.payload_lists_requested_currency is None
    assert stats["dropped"]["currency_unknown"] == 0
    # a DETAIL product with a price and no currency is still dropped when a budget is set
    detail = _p("DETAIL000004", variants=[], source="detail", price=Decimal("5.00"), currency=None)
    ranked, stats = _run([detail])
    assert ranked == [] and stats["dropped"]["currency_unknown"] == 1


def test_incomplete_search_offers_are_preserved_as_listed():
    unpriced = ProductVariant(price=None, currency=None, seller="Shop Two")
    mixed = [_v("m1", "USD", "10.99", seller="TFashion Mart"), _v("m2", "INR", "1400.00", seller="Shop Two")]
    ranked, stats = _run([
        _payload("PAYLOAD00006", variants=[unpriced]),
        _payload("PAYLOAD00007", variants=[]),
        _payload("PAYLOAD00008", variants=mixed),
    ])
    assert ranked == []
    by_id = {c.product.id[-12:]: c for c in unverified_from_stats(stats)}
    a, b, c = by_id["PAYLOAD00006"], by_id["PAYLOAD00007"], by_id["PAYLOAD00008"]
    assert a.candidate_price is None and len(a.listed_offers) == 1 and a.listed_offers[0].price is None
    assert b.candidate_price is None and b.listed_offers == []
    # mixed currencies: the USD offer contradicts the INR request, so the shown price comes
    # only from the INR offer; both offers are still listed as the search returned them
    assert (c.candidate_price, c.candidate_currency) == (Decimal("1400.00"), "INR")
    assert [(o.currency, o.price) for o in c.listed_offers] == [("USD", Decimal("10.99")), ("INR", Decimal("1400.00"))]
    assert c.payload_lists_requested_currency is True  # lists INR, still NOT verified or ranked


def test_a_verified_inr_offer_is_ranked_with_its_own_price_and_seller():
    detail = _p("DETAIL000005", variants=[_v("v", "INR", "1200.00", seller="TheOGshop.com")], source="detail")
    ranked, stats = _run([detail])
    shown = ranked[0].product
    assert (shown.price, shown.currency, shown.merchant) == (Decimal("1200.00"), "INR", "TheOGshop.com")
    assert shown.price_note is None and unverified_from_stats(stats) == []


def test_merchant_filter_is_enforced_on_unverified_candidates():
    other = _payload("PAYLOAD00009", variants=[_v("o", "INR", "1200.00", seller="Other Seller")])
    ranked, stats = _run([other], merchant="Shop One")
    assert unverified_from_stats(stats) == []  # the evidence points to another merchant
    assert stats["dropped"]["other_merchant"] == 1
    detail = _p("DETAIL000006", variants=[_v("o", "INR", "1200.00", seller="Other Seller")], source="detail")
    ranked, stats = _run([detail], merchant="Shop One")
    assert ranked == [] and stats["dropped"]["other_merchant"] == 1


def test_at_most_three_unverified_candidates_are_shown():
    items = [_payload(f"PAYLOAD1000{i}", title=f"Cotton shirt {i}") for i in range(5)]
    ranked, stats = _run(items)
    assert len(unverified_from_stats(stats)) == 3 and stats["unverified"] == 3
    assert stats["unverified_not_shown"] == 2
    outcomes = Counter(e["outcome"] for e in stats["products"])
    assert outcomes["unverified_candidate"] == 3 and outcomes["unverified_not_shown"] == 2
    ranks = sorted(e["unverified_rank"] for e in stats["products"] if "unverified_rank" in e)
    assert ranks == [1, 2, 3]
    _, stats = _run(items, max_unverified=1)
    assert stats["unverified"] == 1 and stats["unverified_not_shown"] == 4


def test_unverified_candidates_are_ordered_by_relevance_and_junk_is_dropped():
    best = _payload("PAYLOAD20001", title="Long sleeve blue shirt for men shirt")
    weak = _payload("PAYLOAD20002", title="Shirt")
    junk = _payload("PAYLOAD20003", title="Kitchen Blender")
    ranked, stats = _run([weak, junk, best])
    got = unverified_from_stats(stats)
    assert [c.relevance for c in got] == sorted((c.relevance for c in got), reverse=True)
    assert stats["dropped"]["irrelevant"] == 1
    assert next(e for e in stats["products"] if e["title"] == "Kitchen Blender")["outcome"] == "irrelevant"


def test_the_audit_trail_records_candidate_price_currency_status_and_reason():
    cand = _payload("PAYLOAD30001", variants=[_v("t", "INR", "1100.00", seller="Fashion Nova")], kind="timeout")
    skipped = _payload("PAYLOAD30002", kind="budget_skipped")
    _, stats = _run([cand, skipped])
    first, second = stats["products"]
    assert first["outcome"] == "unverified_candidate" and first["detail_source"] == "search_payload"
    assert (first["candidate_price"], first["candidate_currency"]) == ("1100.00", "INR")
    assert first["candidate_merchant"] == "Fashion Nova"
    assert first["verification_status"] == "unverified_detail_failed" and first["unverified_reason"] == "timeout"
    assert first["price"] is None and first["currency"] is None  # never mistaken for a verified price
    assert first["payload_lists_requested_currency"] is True and first["n_listed_offers"] == 1
    assert second["verification_status"] == "unverified_detail_skipped"
    assert second["unverified_reason"] == "budget_skipped"
    json.dumps(stats["products"])  # serialisable


def test_a_size_mismatch_and_an_unverified_size_are_different_things():
    only_small = [_v("s1", "INR", "1200.00", size="Small")]
    verified = _p("DETAIL000007", variants=only_small, source="detail")
    ranked, stats = _run([verified], size="M")
    assert ranked[0].size_status == "unavailable"  # checked: no Medium exists on this verified offer
    cand = _payload("PAYLOAD40001", variants=only_small)
    ranked, stats = _run([cand], size="M")
    assert ranked == []
    (c,) = unverified_from_stats(stats)
    assert c.size_status == "unverified"  # never "unavailable": the details were never loaded
    no_size_info = _p("DETAIL000008", variants=[], source="detail", price=Decimal("1200.00"), currency="INR")
    ranked, _ = _run([no_size_info], size="M")
    assert ranked[0].size_status == "unknown"  # verified listing without sizes


# ---------------------------------------------------------------- executor
def _settings(**overrides) -> Settings:
    settings = Settings()
    settings.max_retries_per_tool = 0
    settings.max_refinements = 2
    settings.max_graph_iterations = 3
    settings.shopify_buyer_country = None
    settings.shopify_catalog_id = None
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


INR_4000 = UserIntent(query="running shoes", max_price=Decimal("4000"), currency="INR")


def _plan(intent=INR_4000):
    return ExecutionPlan(intent=intent, steps=[{"step_id": "s1", "action": "search_catalog", "reason": "search"}])


def test_executor_records_why_each_fallback_happened():
    session = ScriptedSession([ucp_product(i) for i in range(1, 6)])
    report = ShopifyMcpExecutor(session_factory=lambda: session, settings=_settings()).execute_plan(
        _plan(), tool_call_budget=2)  # search + 1 detail, the other four are skipped
    kinds = Counter(p.detail_failure_kind for p in report.products if p.detail_source == "search_payload")
    assert kinds == {"budget_skipped": 4}
    assert [p.detail_failure_kind for p in report.products if p.detail_source == "detail"] == [None]

    session = ScriptedSession([ucp_product(1), ucp_product(2)], fail_details=True)
    report = ShopifyMcpExecutor(session_factory=lambda: session, settings=_settings()).execute_plan(_plan())
    assert all(p.detail_source == "search_payload" and p.detail_failure_kind for p in report.products)


# ------------------------------------------------------------- graph (LangGraph)
class _Planner:
    """Re-plans from text like Gemini would: it only knows the budget the text states."""

    def plan(self, text):
        match = re.search(r"(\d+)\s*rupees", text)
        query = re.sub(r"\s*(for|under)?\s*\d+\s*rupees", "", text).strip()
        intent = UserIntent(query=query, currency="INR", max_price=Decimal(match.group(1)) if match else None)
        return ExecutionPlan(intent=intent, steps=[PlanStep(step_id="s1", action="search_catalog", reason="r")])


class _Factory:
    def __init__(self, products, session_cls=ScriptedSession, **kwargs):
        self.sessions, self.products, self.cls, self.kwargs = [], list(products), session_cls, kwargs

    def __call__(self):
        session = self.cls(self.products, **self.kwargs)
        self.sessions.append(session)
        return session

    @property
    def searches(self):
        return [c[1]["catalog"] for s in self.sessions for c in s.calls if c[0] == "search_catalog"]


class _MixedSession(ScriptedSession):
    """get_product works for every product except prod2."""

    async def call_tool(self, name, arguments):
        if name == "get_product" and arguments["catalog"]["id"].endswith("prod2"):
            raise RuntimeError("Shopify UCP MCP error: Service error")
        return await super().call_tool(name, arguments)


def _workflow(factory, settings=None, **kwargs):
    settings = settings or _settings()
    executor = ShopifyMcpExecutor(session_factory=factory, settings=settings)
    return asyncio.run(arun_discovery_workflow(
        "running shoes for 4000 rupees", planner=_Planner(), executor=executor, settings=settings, **kwargs))


def test_unverified_only_result_keeps_the_refinement_behaviour_and_the_candidates():
    factory = _Factory([ucp_product(1, amount=300000), ucp_product(2, amount=300000)], fail_details=True)

    state = _workflow(factory)

    decision = state["decision"]
    assert 1 < len(factory.searches) <= 3  # refined exactly as before (nothing verified was found)
    assert all(s["filters"]["price"] == {"max": 400000} for s in factory.searches)  # budget kept
    assert decision.status == "clarify" and decision.candidates == [] and decision.selected is None
    assert len(decision.unverified_candidates) == 2  # kept from the final attempt
    assert {c.unverified_reason for c in decision.unverified_candidates} <= {
        "service_error", "tool_error", "rate_limited", "timeout", "invalid_result"}
    assert "listed separately" in decision.message and "budget compliance" in decision.message
    assert state["refinement_count"] <= 2 and state["graph_iterations"] <= 3


def test_verified_and_unverified_results_coexist_and_only_the_verified_one_can_be_selected():
    factory = _Factory([ucp_product(1, amount=300000), ucp_product(2, amount=300000)], session_cls=_MixedSession)

    state = _workflow(factory)

    decision = state["decision"]
    assert [c.product.id.rsplit("/", 1)[-1] for c in decision.candidates] == ["prod1"]
    assert [c.product.id.rsplit("/", 1)[-1] for c in decision.unverified_candidates] == ["prod2"]
    if decision.selected is not None:
        assert decision.selected.product.id.endswith("prod1")
    assert "listed separately" in decision.message
    attempt = decision.diagnostics["attempt_log"][-1]
    assert attempt["kept"] == 1 and attempt["unverified"] == 1
    assert {e["outcome"] for e in attempt["products"]} >= {"kept", "unverified_candidate"}


def test_unverified_candidates_never_reach_the_policy_check(monkeypatch):
    seen = []

    def spy(ranked, **kwargs):
        seen.append([s.product.id for s in ranked])
        return ranked, {}

    monkeypatch.setattr(graph_module, "apply_policy", spy)
    factory = _Factory([ucp_product(1, amount=300000), ucp_product(2, amount=300000)], session_cls=_MixedSession)

    state = _workflow(factory, policy_checker=object())

    assert seen and all(ids == ["gid://shopify/p/prod1"] for ids in seen)
    assert state["decision"].unverified_candidates


def test_the_legacy_discover_path_also_keeps_unverified_candidates():
    session = ScriptedSession([ucp_product(1, amount=300000)], fail_details=True)
    settings = _settings()
    provider = ShopifyUcpCatalogProvider(
        settings, executor=ShopifyMcpExecutor(session_factory=lambda: session, settings=settings))

    decision = DiscoveryAgent(primary=provider, max_refinements=0).discover(INR_4000)

    assert decision.status == "clarify" and decision.candidates == []
    assert len(decision.unverified_candidates) == 1 and "listed separately" in decision.message


# --------------------------------------------------------- SQLite persistence
@pytest.fixture()
def stored(tmp_path):
    settings = _settings(ledger_path=tmp_path / "ledger.jsonl")
    store = SqliteStore(tmp_path / "agent.db")
    session_id = store.create_session()
    factory = _Factory([ucp_product(1, amount=300000), ucp_product(2, amount=300000)], session_cls=_MixedSession)
    executor = ShopifyMcpExecutor(session_factory=factory, settings=settings)
    result = service.discover(
        "running shoes for 4000 rupees", session_id, store=store, settings=settings,
        planner=_Planner(), executor=executor)
    return store, result


def test_unverified_candidates_survive_the_sqlite_round_trip(stored):
    store, result = stored
    assert result.decision.unverified_candidates  # the run really produced some
    assert "unverified_candidates" in store.get_run(result.run_id)["decision_json"]

    loaded = service.load_run(result.run_id, store=SqliteStore(store.path))  # fresh store = app restart

    assert loaded.decision.model_dump(mode="json") == result.decision.model_dump(mode="json")
    got = loaded.decision.unverified_candidates[0]
    want = result.decision.unverified_candidates[0]
    assert (got.candidate_price, got.candidate_currency, got.candidate_merchant) == (
        want.candidate_price, want.candidate_currency, want.candidate_merchant)
    assert got.unverified_reason == want.unverified_reason and got.product.detail_source == "search_payload"
    assert [c.product.id for c in loaded.decision.candidates] == [c.product.id for c in result.decision.candidates]


def test_runs_saved_before_this_feature_still_load(stored, monkeypatch):
    store, result = stored
    old = json.loads(store.get_run(result.run_id)["decision_json"])
    old.pop("unverified_candidates")
    for item in old["candidates"]:
        item["product"].pop("detail_failure_kind", None)
        item["product"].pop("detail_source", None)
    old_json = json.dumps(old)
    real = store.get_run

    def old_row(run_id):
        row = dict(real(run_id))
        row["decision_json"] = old_json
        return row

    monkeypatch.setattr(store, "get_run", old_row)

    loaded = service.load_run(result.run_id, store=store)

    assert loaded is not None and loaded.decision is not None
    assert loaded.decision.unverified_candidates == []
    assert [c.product.id for c in loaded.decision.candidates] == [c.product.id for c in result.decision.candidates]
    assert DiscoveryDecision.model_validate_json(old_json).unverified_candidates == []
