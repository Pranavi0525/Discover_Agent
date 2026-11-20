"""M1.4 bounded query refinement: at most MAX_REFINEMENTS, budget never touched."""
from decimal import Decimal

import pytest

from discovery_agent.config import Settings
from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.orchestration import DiscoveryAgent
from discovery_agent.providers.shopify_ucp import ShopifyUcpCatalogProvider
from discovery_agent.refinement import broaden_query, carry_over_constraints, refine_intent
from discovery_agent.schemas import Product, SearchResult, UserIntent
from ucp_fakes import ScriptedSession, ucp_product


class RecordingProvider:
    """Returns products only for queries listed in ``hits``; records every intent."""

    name = "recording"

    def __init__(self, hits=None, error=None):
        self.hits = hits or {}
        self.error = error
        self.intents: list[UserIntent] = []
        self.began = 0

    def begin_run(self):
        self.began += 1

    def search(self, intent):
        self.intents.append(intent)
        if self.error:
            raise RuntimeError(self.error)
        return SearchResult(provider=self.name, query=intent.query,
                            products=list(self.hits.get(intent.query, [])))


def _product(title="Running shoes", price="3000", currency="INR"):
    return Product(id="p1", title=title, url="https://shop.example/p1", price=price,
                   currency=currency, description="running shoes for daily training", source="test")


INTENT = UserIntent(query="blue running shoes", max_price=Decimal("4000"), currency="INR")


def test_broaden_query_keeps_the_head_noun():
    assert broaden_query("blue running shoes size 9") == "blue running shoes"
    assert broaden_query("blue running shoes") == "running shoes"
    assert broaden_query("running shoes") == "shoes"
    assert broaden_query("shoes") == "shoes"
    assert broaden_query("size 9") == "size 9"


def test_at_most_two_refinements_then_a_truthful_no_result():
    provider = RecordingProvider()
    decision = DiscoveryAgent(primary=provider, max_refinements=2, max_graph_iterations=3).discover(INTENT)

    assert [i.query for i in provider.intents] == [
        "blue running shoes", "running shoes", "blue running sneakers"]
    assert [i.refinement_count for i in provider.intents] == [0, 1, 2]
    assert decision.status == "clarify"
    assert decision.attempts == 3
    assert "Shopify returned no products" in decision.message
    assert "Your budget was not changed" in decision.message
    assert decision.candidates == []


def test_every_refinement_keeps_the_original_budget_and_currency():
    provider = RecordingProvider()
    DiscoveryAgent(primary=provider).discover(INTENT)

    assert {i.max_price for i in provider.intents} == {Decimal("4000")}
    assert {i.currency for i in provider.intents} == {"INR"}


def test_max_graph_iterations_caps_searches_even_if_more_refinements_are_allowed():
    provider = RecordingProvider()
    DiscoveryAgent(primary=provider, max_refinements=2, max_graph_iterations=2).discover(INTENT)
    assert len(provider.intents) == 2

    provider = RecordingProvider()
    DiscoveryAgent(primary=provider, max_refinements=50, max_graph_iterations=99).discover(INTENT)
    assert len(provider.intents) == 3  # the schema itself caps refinement_count at 2


def test_zero_refinements_means_exactly_one_search():
    provider = RecordingProvider()
    decision = DiscoveryAgent(primary=provider, max_refinements=0).discover(INTENT)
    assert len(provider.intents) == 1
    assert decision.attempts == 1


def test_refinement_stops_as_soon_as_a_broader_query_finds_products():
    provider = RecordingProvider(hits={"running shoes": [_product()]})
    decision = DiscoveryAgent(primary=provider).discover(INTENT)

    assert [i.query for i in provider.intents] == ["blue running shoes", "running shoes"]
    assert decision.status in {"selected", "needs_selection", "clarify"}
    assert decision.candidates and decision.attempts == 2
    assert "broadened to “running shoes”" in decision.message
    assert "budget was not changed" in decision.message


def test_a_found_product_over_budget_is_still_rejected_after_refinement():
    provider = RecordingProvider(hits={"shoes": [_product(price="4500")]})
    decision = DiscoveryAgent(primary=provider).discover(
        UserIntent(query="running shoes", max_price=Decimal("4000"), currency="INR"))

    assert decision.candidates == []
    assert decision.status == "clarify"
    assert "over your budget" in decision.message


def test_a_provider_error_is_not_refined_into_more_calls():
    provider = RecordingProvider(error="Shopify UCP tool 'search_catalog' returned an error: nope")
    decision = DiscoveryAgent(primary=provider).discover(INTENT)

    assert len(provider.intents) == 1
    assert decision.status == "failed"
    assert "returned an error: nope" in decision.message


def test_an_unbroadenable_query_is_searched_once():
    provider = RecordingProvider()
    DiscoveryAgent(primary=provider).discover(UserIntent(query="shoes", max_price=Decimal("500"), currency="INR"))
    # "shoes" cannot be broadened, but the ladder still tries its two synonyms.
    assert [i.query for i in provider.intents] == ["shoes", "sneakers", "trainers"]


def test_begin_run_is_called_so_the_tool_budget_is_per_run():
    provider = RecordingProvider()
    DiscoveryAgent(primary=provider).discover(INTENT)
    assert provider.began == 1


def test_scoring_uses_the_original_query_not_the_broadened_one():
    off_topic = _product(title="Garden hose")
    off_topic = off_topic.model_copy(update={"description": "hose for garden"})
    provider = RecordingProvider(hits={"shoes": [off_topic]})
    decision = DiscoveryAgent(primary=provider).discover(
        UserIntent(query="running shoes", max_price=Decimal("4000"), currency="INR"))

    # Against the original "running shoes" a garden hose scores 0 -> weak match,
    # it must not be auto-selected just because "shoes" was searched.
    assert decision.status == "clarify"
    assert decision.selected is None


def _over_budget_provider(settings, sessions):
    """Provider whose every search returns 2 real products that are over the 4000 INR budget,
    so DiscoveryAgent keeps wanting to refine; ``sessions`` records one fake session per search."""

    def factory():
        session = ScriptedSession([ucp_product(1, amount=900000), ucp_product(2, amount=900000)])
        sessions.append(session)
        return session

    return ShopifyUcpCatalogProvider(
        settings, executor=ShopifyMcpExecutor(session_factory=factory, settings=settings))


def test_all_refinements_share_one_max_tool_calls_budget():
    """Search 1 costs 3 calls (search + 2 details).  With MAX_TOOL_CALLS=5 only 2 calls are
    left for search 2, and none for search 3.  Were the budget per attempt instead of per
    run, three searches of 3 calls each (9) would be made."""
    settings = Settings()
    settings.max_retries_per_tool = 0
    settings.max_tool_calls = 5
    settings.shopify_buyer_country = None
    sessions = []

    decision = DiscoveryAgent(
        primary=_over_budget_provider(settings, sessions), max_refinements=2
    ).discover(INTENT)

    assert [len(s.calls) for s in sessions] == [3, 2]
    assert sum(len(s.calls) for s in sessions) <= settings.max_tool_calls
    assert decision.attempts == 2
    assert decision.candidates == []


def test_tool_budget_exhaustion_stops_searching_and_is_reported():
    settings = Settings()
    settings.max_retries_per_tool = 0
    settings.max_tool_calls = 2  # search + one detail, nothing left for refinements
    settings.shopify_buyer_country = None
    sessions = []

    decision = DiscoveryAgent(
        primary=_over_budget_provider(settings, sessions), max_refinements=2
    ).discover(INTENT)

    # Exactly one real search happened: 1 search + 1 detail call, then the budget was gone.
    assert len(sessions) == 1 and len(sessions[0].calls) == 2
    assert decision.status == "clarify"
    assert decision.candidates == []
    assert decision.attempts == 1
    log = decision.diagnostics["attempt_log"]
    assert [bool(entry.get("stopped")) for entry in log] == [False, True]
    # The stop is reported truthfully: the broadened query was never searched.
    assert "Searched 1 time(s): \u201cblue running shoes\u201d." in decision.message
    assert "MAX_TOOL_CALLS" in decision.message
    assert "Your budget was not changed" in decision.message


def test_refine_intent_and_carry_over_never_change_the_constraints():
    refined = refine_intent(INTENT, 2)
    assert refined.query == "running shoes" and refined.refinement_count == 1
    assert (refined.max_price, refined.currency) == (Decimal("4000"), "INR")
    assert refine_intent(refined.model_copy(update={"refinement_count": 2}), 2) is None

    replanned = UserIntent(query="shoes")  # what a text-only re-plan would produce
    restored = carry_over_constraints(replanned, INTENT, 1)
    assert (restored.max_price, restored.currency, restored.refinement_count) == (Decimal("4000"), "INR", 1)
