"""M1.5: MiniLM relevance behind an injectable scorer.  No model is ever loaded here."""
import asyncio
import importlib.machinery
import sys
import types
from decimal import Decimal

import pytest

from discovery_agent import semantic
from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.graph import arun_discovery_workflow
from discovery_agent.orchestration import (
    AUTO_SELECT_THRESHOLD, NEEDS_SELECTION_THRESHOLD, DiscoveryAgent, decide,
)
from discovery_agent.ranking import filter_and_rank_with_report, filter_and_rank_with_stats
from discovery_agent.schemas import Product, ProductVariant, ScoredProduct, SearchResult, UserIntent
from discovery_agent.semantic import (
    EmbeddingScorer, MiniLMScorer, RunScorer, get_default_scorer,
)
from test_graph import FakeMcpSession, FakePlanner, _search_plan, _settings


def _product(pid="p1", title="Running shoes", price="1000", currency="INR", available=True,
             merchant="M", description="", variants=None, url=None):
    return Product(
        id=pid,
        title=title,
        url=url or f"https://shop.example/{pid}",
        price=Decimal(price),
        currency=currency,
        available=available,
        merchant=merchant,
        description=description,
        source="test",
        variants=variants or []
    )
def _intent(**kw):
    kw.setdefault("query", "running shoes")
    return UserIntent(max_price=Decimal("4000"), currency="INR", **kw)


class FakeScorer:
    """Deterministic injectable scorer; records exactly what it was asked to score."""

    semantic = True

    def __init__(self, scores=None, default=0.9, name="minilm"):
        self.name = name
        self.scores = scores or {}
        self.default = default
        self.calls: list[tuple[list[str], list[str]]] = []

    def score(self, queries, products):
        self.calls.append((list(queries), [p.id for p in products]))
        return [self.scores.get(p.id, self.default) for p in products]


class RecordingProvider:
    name = "recording"

    def __init__(self, hits):
        self.hits = hits
        self.intents: list[UserIntent] = []

    def search(self, intent):
        self.intents.append(intent)
        return SearchResult(provider=self.name, query=intent.query,
                            products=list(self.hits.get(intent.query, [])))


class ListTracker:
    def __init__(self):
        self.events: list[dict] = []

    def record(self, event):
        self.events.append(dict(event))


# --- A / B / C: decision tiers (thresholds unchanged) ---------------------------------

def _decide(score, scorer_name="minilm"):
    ranked = [ScoredProduct(product=_product(), score=score)]
    return decide("p", ranked, _intent(), scorer_name)


def test_thresholds_are_the_original_ones():
    assert (AUTO_SELECT_THRESHOLD, NEEDS_SELECTION_THRESHOLD) == (0.80, 0.50)


def test_A_exactly_0_80_is_auto_selected():
    assert _decide(0.80).status == "selected"


def test_B_exactly_0_50_asks_the_user_to_choose():
    assert _decide(0.50).status == "needs_selection"
    assert _decide(0.7999).status == "needs_selection"


def test_C_below_0_50_clarifies():
    assert _decide(0.4999).status == "clarify"


@pytest.mark.parametrize("score,status", [(0.80, "selected"), (0.50, "needs_selection"), (0.49, "clarify")])
def test_tiers_hold_through_the_whole_pipeline(score, status):
    ranked, _ = filter_and_rank_with_stats(_intent(), [_product()], scorer=FakeScorer(default=score))
    assert decide("p", ranked, _intent(), "minilm").status == status


# --- D: explicit fallback -------------------------------------------------------------

def test_D_missing_minilm_falls_back_explicitly_and_never_claims_semantic(monkeypatch):
    monkeypatch.setenv(semantic.SCORER_ENV_VAR, "auto")
    monkeypatch.setattr(semantic.importlib.util, "find_spec", lambda name: None)
    scorer = get_default_scorer()
    assert scorer.name == "lexical_fallback" and not scorer.semantic

    provider = RecordingProvider({"running shoes": [_product()]})
    decision = DiscoveryAgent(primary=provider).discover(_intent())

    rel = decision.diagnostics["relevance"]
    assert rel["scorer"] == "lexical_fallback"
    assert rel["lexical_fallback"] is True and rel["minilm_active"] is False
    assert "not installed" in rel["fallback_reason"]
    assert decision.status == "selected"
    assert "not semantic confidence" in decision.message
    assert "embedding" not in decision.message.lower()


def test_D_midrun_scorer_failure_degrades_the_run_to_lexical_and_says_why():
    class Broken(FakeScorer):
        def score(self, queries, products):
            raise RuntimeError("model download failed")

    run_scorer = RunScorer(Broken())
    ranked, stats = filter_and_rank_with_stats(_intent(), [_product()], scorer=run_scorer)

    rel = stats["relevance"]
    assert rel["scorer"] == "lexical_fallback" and rel["minilm_active"] is False
    assert "model download failed" in rel["fallback_reason"]
    assert ranked and ranked[0].score == 1.0
    # pinned: a later call in the same run stays on the fallback
    assert filter_and_rank_with_stats(_intent(), [_product()], scorer=run_scorer)[1]["relevance"]["scorer"] == "lexical_fallback"


def test_deliberate_lexical_choice_is_not_called_a_fallback(monkeypatch):
    monkeypatch.setenv(semantic.SCORER_ENV_VAR, "lexical")
    assert get_default_scorer().name == "lexical"


# --- E / K: no model download, lazy loading ---------------------------------------------

def _trap_module(monkeypatch, record):
    class Trap:
        def __init__(self, *args, **kwargs):
            record.append(args)
            raise AssertionError("MiniLM must not be instantiated in tests")

    module = types.ModuleType("sentence_transformers")
    module.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    module.SentenceTransformer = Trap
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_K_building_the_production_scorer_does_not_instantiate_minilm(monkeypatch):
    built: list = []
    _trap_module(monkeypatch, built)
    monkeypatch.setattr(semantic, "_MINILM", None)
    monkeypatch.setenv(semantic.SCORER_ENV_VAR, "auto")

    scorer = get_default_scorer()

    assert isinstance(scorer, MiniLMScorer)
    assert scorer.load_count == 0 and built == []
    assert get_default_scorer() is scorer  # one shared scorer, so one load per process


def test_E_a_failing_model_load_falls_back_instead_of_crashing(monkeypatch):
    built: list = []
    _trap_module(monkeypatch, built)
    run_scorer = RunScorer(MiniLMScorer())
    _, stats = filter_and_rank_with_stats(_intent(), [_product()], scorer=run_scorer)
    assert stats["relevance"]["scorer"] == "lexical_fallback"


def test_K_model_loads_once_and_is_reused(monkeypatch):
    constructed: list = []

    class Array(list):
        def tolist(self):
            return list(self)

    class FakeModel:
        def __init__(self, name):
            constructed.append(name)

        def encode(self, texts, **kwargs):
            return Array([[1.0, 0.0] for _ in texts])

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    scorer = MiniLMScorer()
    assert constructed == []
    scorer.score(["running shoes"], [_product()])
    scorer.score(["running shoes"], [_product("p2")])
    assert constructed == ["sentence-transformers/all-MiniLM-L6-v2"] and scorer.load_count == 1


# --- F / G: original query is the anchor; best score over synonym ----------------------

def test_F_scoring_anchors_on_the_original_query_not_a_broadened_one():
    original = UserIntent(query="blue running shoes", max_price=Decimal("4000"), currency="INR")
    provider = RecordingProvider({"running shoes": [_product()]})  # only the BROADEN rung hits
    scorer = FakeScorer()

    decision = DiscoveryAgent(primary=provider, scorer=scorer).discover(original)

    assert [i.query for i in provider.intents] == ["blue running shoes", "running shoes"]
    assert scorer.calls == [(["blue running shoes"], ["p1"])]
    assert decision.diagnostics["relevance"]["queries"] == ["blue running shoes"]


def test_G_synonym_rung_scores_original_plus_synonym_and_keeps_the_best():
    vectors = {"running shoes": [1.0, 0.0], "running sneakers": [0.0, 1.0],
               "Running sneakers": [0.0, 1.0]}
    scorer = EmbeddingScorer(lambda texts: [vectors.get(t, [0.0, 1.0]) for t in texts])
    product = _product(title="Running sneakers")

    ranked, stats = filter_and_rank_with_stats(
        _intent(), [product], extra_queries=["running sneakers"], scorer=scorer)

    assert stats["relevance"]["queries"] == ["running shoes", "running sneakers"]
    assert ranked[0].score == pytest.approx(1.0)  # synonym matched even though the original did not


def test_G_synonym_rung_through_the_agent_passes_both_queries():
    original = UserIntent(query="blue running shoes", max_price=Decimal("4000"), currency="INR")
    provider = RecordingProvider({"blue running sneakers": [_product()]})
    scorer = FakeScorer()

    DiscoveryAgent(primary=provider, scorer=scorer).discover(original)

    assert scorer.calls == [(["blue running shoes", "blue running sneakers"], ["p1"])]


def test_original_query_is_embedded_once_across_calls():
    embedded: list[str] = []

    def embed(texts):
        embedded.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    scorer = EmbeddingScorer(embed)
    scorer.score(["running shoes"], [_product()])
    scorer.score(["running shoes"], [_product("p2")])
    assert embedded.count("running shoes") == 1


# --- H: budget is never touched ----------------------------------------------------------

def test_H_refinement_and_semantic_scoring_never_change_the_budget():
    original = UserIntent(query="blue running shoes", max_price=Decimal("4000"), currency="INR",
                          merchant="M")
    provider = RecordingProvider({"blue running sneakers": [_product(price="3999")]})

    DiscoveryAgent(primary=provider, scorer=FakeScorer()).discover(original)

    assert len(provider.intents) == 3
    assert all(i.max_price == Decimal("4000") and i.currency == "INR" and i.merchant == "M"
               for i in provider.intents)
    assert original.max_price == Decimal("4000")


# --- I: hard filters beat any similarity -------------------------------------------------

def test_I_high_similarity_cannot_bring_back_a_filtered_product():
    # A one-letter merchant name can never be verified (step 5), so use a real name.
    intent = _intent(merchant="Acme")
    products = [
        _product("ok", merchant="Acme", url="https://acme.com/ok"),
        _product("over", price="9000"),
        _product("gone", available=False),
        _product("usd", currency="USD"),
        _product("other", merchant="Other"),
        _product("nomerchant", merchant=None),
    ]
    scorer = FakeScorer(default=1.0)

    ranked, stats = filter_and_rank_with_stats(intent, products, scorer=scorer)

    assert [r.product.id for r in ranked] == ["ok"]
    assert scorer.calls == [(["running shoes"], ["ok"])]  # filtered products are never scored
    assert stats["dropped"]["over_budget"] == 1 and stats["dropped"]["unavailable"] == 1
    assert stats["dropped"]["currency_mismatch"] == 1 and stats["dropped"]["other_merchant"] == 2


def test_zero_word_overlap_floor_still_applies_when_minilm_is_active():
    products = [_product("ok"), _product("blower", title="Snow blower part")]
    scorer = FakeScorer(default=0.99)

    ranked, stats = filter_and_rank_with_stats(_intent(), products, scorer=scorer)

    assert [r.product.id for r in ranked] == ["ok"]
    assert stats["dropped"]["irrelevant"] == 1
    assert stats["relevance"]["irrelevant_floor_dropped"] == 1
    assert stats["relevance"]["passed_hard_filters"] == 2 and stats["relevance"]["scored"] == 1
    assert scorer.calls[0][1] == ["ok"]


# --- J: size status ----------------------------------------------------------------------

def test_J_unavailable_size_stays_demoted_despite_a_higher_semantic_score():
    intent = _intent(size="UK 8")
    products = [
        _product("nosize", variants=[ProductVariant(options={"size": "9"}, available=True)]),
        _product("has8", variants=[ProductVariant(options={"size": "8"}, available=True)]),
    ]
    scorer = FakeScorer({"nosize": 0.95, "has8": 0.60})

    ranked, _ = filter_and_rank_with_stats(intent, products, scorer=scorer)

    assert [(r.product.id, r.size_status) for r in ranked] == [("has8", "available"), ("nosize", "unavailable")]
    assert decide("p", ranked, intent, "minilm").status == "needs_selection"


def test_J_all_unavailable_is_never_auto_selected_even_at_0_95():
    intent = _intent(size="UK 8")
    products = [_product("a", variants=[ProductVariant(options={"size": "9"}, available=True)])]
    ranked, _ = filter_and_rank_with_stats(intent, products, scorer=FakeScorer(default=0.95))

    decision = decide("p", ranked, intent, "minilm")

    assert ranked[0].size_status == "unavailable"
    assert decision.status == "needs_selection" and decision.selected is None


# --- diagnostics -------------------------------------------------------------------------

def test_report_describes_the_score_distribution_and_clamps_negative_cosines():
    scorer = FakeScorer({"a": 0.9, "b": 0.5, "c": -0.2})
    products = [_product("a"), _product("b"), _product("c")]

    ranked, stats, report = filter_and_rank_with_report(_intent(), products, scorer=scorer)

    assert [r.score for r in ranked] == [0.9, 0.5, 0.0]  # clamped into the schema's 0-1 range
    assert report.scorer == "minilm" and report.minilm_active and not report.lexical_fallback
    assert (report.scored, report.passed_hard_filters, report.evaluated) == (3, 3, 3)
    assert (report.score_min, report.score_max, report.score_mean) == (0.0, 0.9, 0.4667)
    assert (report.raw_min, report.raw_max) == (-0.2, 0.9)
    assert sum(report.histogram) == 3 and report.histogram[9] == 1 and report.histogram[0] == 1
    assert stats["relevance"] == report.to_dict()


def test_empty_candidates_do_not_touch_the_scorer():
    scorer = FakeScorer()
    ranked, stats = filter_and_rank_with_stats(_intent(), [_product(price="9000")], scorer=scorer)
    assert ranked == [] and scorer.calls == []
    assert stats["relevance"]["scored"] == 0 and stats["relevance"]["score_max"] is None


def test_agent_records_scorer_in_events_attempt_log_and_decision():
    tracker = ListTracker()
    provider = RecordingProvider({"running shoes": [_product()]})

    decision = DiscoveryAgent(primary=provider, tracker=tracker, scorer=FakeScorer()).discover(_intent())

    assert decision.status == "selected" and "embedding-based" in decision.message
    assert decision.diagnostics["relevance"]["scorer"] == "minilm"
    assert decision.diagnostics["attempt_log"][0]["relevance"]["minilm_active"] is True
    by_event = {e["event"]: e for e in tracker.events}
    assert by_event["filter_and_rank_completed"]["scorer"] == "minilm"
    assert by_event["filter_and_rank_completed"]["relevance"]["scored"] == 1
    assert by_event["discovery_completed"]["relevance"]["scorer"] == "minilm"


def test_graph_workflow_uses_the_injected_scorer_and_records_diagnostics():
    settings = _settings()
    session = FakeMcpSession(catalog={"running shoes": [{"id": "p1"}]})
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    planner = FakePlanner([_search_plan("running shoes")])
    tracker = ListTracker()
    scorer = FakeScorer(default=0.85)

    state = asyncio.run(arun_discovery_workflow(
        "running shoes", planner=planner, executor=executor, settings=settings,
        tracker=tracker, scorer=scorer))

    decision = state["decision"]
    assert decision.status == "selected" and decision.selected.score == 0.85
    assert scorer.calls and scorer.calls[0][0] == ["running shoes"]
    assert decision.diagnostics["relevance"]["scorer"] == "minilm"
    assert state["attempt_log"][0]["relevance"]["score_max"] == 0.85
    assert tracker.events[-1]["relevance"]["minilm_active"] is True
