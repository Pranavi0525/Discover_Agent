"""A currency is never silently USD.

Regression for the bug where ``UserIntent.currency`` defaulted to "USD": a user
typing "running shoes under 8000" got a USD 8,000 budget sent to a US-market
search, with nothing on screen saying so.  Now "unspecified" is None and is
resolved (explicit words > market default) or asked about, never guessed.
"""
import asyncio
from decimal import Decimal

import pytest
from pydantic import ValidationError

from discovery_agent.budget import budget_currency_missing, reconcile_intent, resolve_currency
from discovery_agent.config import Settings
from discovery_agent.currency import currency_for_country, format_money, normalize_currency_code
from discovery_agent.execution import ExecutorError, ShopifyMcpExecutor
from discovery_agent.graph import arun_discovery_workflow
from discovery_agent.live_check import build_report
from discovery_agent.orchestration import no_match_message
from discovery_agent.planner import GeminiPlanner
from discovery_agent.providers.shopify_ucp import product_arguments, search_arguments
from discovery_agent.ranking import filter_and_rank_with_stats
from discovery_agent.refinement import carry_over_constraints
from discovery_agent.schemas import (
    DiscoveryDecision, ExecutionPlan, PlanStep, Product, ScoredProduct, UserIntent,
)
from ucp_fakes import ScriptedSession, ucp_product


def _settings(monkeypatch, default=None, country=None) -> Settings:
    """Hermetic settings: nothing leaks in from a developer's .env."""
    monkeypatch.delenv("DEFAULT_CURRENCY", raising=False)
    monkeypatch.delenv("SHOPIFY_BUYER_COUNTRY", raising=False)
    settings = Settings()
    settings.google_api_key = "test"
    settings.default_currency = default
    settings.shopify_buyer_country = country
    settings.max_retries_per_tool = 0
    return settings


# --------------------------------------------------------------- schema

def test_unspecified_currency_is_none_not_usd():
    assert UserIntent(query="running shoes").currency is None
    assert UserIntent(query="running shoes", max_price=Decimal("8000")).currency is None


@pytest.mark.parametrize("raw, expected", [("", None), ("  ", None), (" inr ", "INR"), ("usd", "USD"), (None, None)])
def test_currency_is_normalised_and_blank_means_unspecified(raw, expected):
    # Gemini often returns "" for "not stated"; that must not fail the plan.
    assert UserIntent(query="running shoes", currency=raw).currency == expected


def test_a_malformed_currency_is_still_rejected():
    with pytest.raises(ValidationError):
        UserIntent(query="running shoes", currency="RUPEES")


def test_currency_helpers():
    assert currency_for_country("in") == "INR"
    assert currency_for_country("US") == "USD"
    assert currency_for_country("DE") is None  # EUR spans many countries: deliberately not guessed
    assert currency_for_country(None) is None
    assert normalize_currency_code(" eur ") == "EUR"
    assert format_money(Decimal("8000"), "INR") == "INR 8,000.00"
    assert format_money(Decimal("8000"), None) == "8,000.00"  # never "None 8,000.00"


# ------------------------------------------------- minor-unit exponents

@pytest.mark.parametrize("currency, exponent, minor", [
    ("USD", 2, 800000), ("INR", 2, 800000), ("JPY", 0, 8000), ("KRW", 0, 8000), ("KWD", 3, 8000000),
])
def test_budget_is_sent_in_the_currencys_own_minor_units(currency, exponent, minor):
    from discovery_agent.currency import minor_unit_exponent
    from discovery_agent.providers.shopify_ucp import to_minor_units

    assert minor_unit_exponent(currency) == exponent
    assert to_minor_units(Decimal("8000"), currency) == minor
    settings = Settings()
    request = search_arguments(settings, "shoes", 10, Decimal("8000"), currency)["catalog"]
    assert request["filters"]["price"] == {"max": minor}


def test_prices_read_back_use_the_same_scale():
    from discovery_agent.execution import _normalize_product

    def read(index, amount, currency):
        return _normalize_product(ucp_product(index, amount=amount, currency=currency), f"p{index}")

    yen, dinar, rupee = read(1, 3000, "JPY"), read(2, 3000, "KWD"), read(3, 300000, "INR")
    assert (yen.price, dinar.price, rupee.price) == (Decimal("3000"), Decimal("3"), Decimal("3000"))


# ------------------------------------------------------------ settings

def test_default_currency_prefers_explicit_env_then_follows_the_market(monkeypatch):
    monkeypatch.delenv("DEFAULT_CURRENCY", raising=False)
    monkeypatch.delenv("SHOPIFY_BUYER_COUNTRY", raising=False)
    assert Settings().default_currency is None  # nothing configured -> the agent must ask

    monkeypatch.setenv("SHOPIFY_BUYER_COUNTRY", "IN")
    assert Settings().default_currency == "INR"

    monkeypatch.setenv("DEFAULT_CURRENCY", "gbp")
    assert Settings().default_currency == "GBP"  # explicit beats derived

    monkeypatch.setenv("DEFAULT_CURRENCY", "rupees")  # malformed: ignored, not crashed
    assert Settings().default_currency == "INR"


# ----------------------------------------------------------- resolution

def test_resolve_currency_fills_only_what_is_missing_and_reports_a_budget_assumption():
    intent = UserIntent(query="shoes", max_price=Decimal("8000"))
    filled, notes = resolve_currency(intent, "INR")
    assert filled.currency == "INR" and filled.max_price == Decimal("8000")
    assert "assumed INR" in notes[0]

    stated = UserIntent(query="shoes", max_price=Decimal("50"), currency="USD")
    assert resolve_currency(stated, "INR") == (stated, [])  # never overrides a stated currency

    no_budget, notes = resolve_currency(UserIntent(query="shoes"), "INR")
    assert no_budget.currency == "INR" and notes == []  # nothing to warn about without a budget

    assert resolve_currency(intent, None) == (intent, [])  # no default: left unresolved
    assert budget_currency_missing(intent) is True
    assert budget_currency_missing(stated) is False
    assert budget_currency_missing(UserIntent(query="shoes")) is False


# -------------------------------------------------------------- planner

class _FakeModel:
    def __init__(self, **intent):
        self.intent = intent

    def invoke(self, prompt):
        assert "leave `intent.currency` null" in prompt  # Gemini is told not to guess
        return ExecutionPlan(
            intent=self.intent,
            steps=[PlanStep(step_id="1", action="search_catalog", reason="Find products")],
        )


def _planner(settings, **intent) -> GeminiPlanner:
    planner = object.__new__(GeminiPlanner)
    planner.settings = settings
    planner._model = _FakeModel(**intent)
    return planner


def test_planner_uses_the_market_currency_for_a_bare_budget_and_says_so(monkeypatch):
    planner = _planner(_settings(monkeypatch, default="INR"),
                       query="running shoes", max_price=Decimal("8000"))

    plan = planner.plan("running shoes under 8000")

    assert (plan.intent.max_price, plan.intent.currency) == (Decimal("8000"), "INR")
    assert plan.steps[0].action == "search_catalog" and not plan.requires_clarification
    assert any("assumed INR" in note for note in planner.last_corrections)


def test_planner_asks_instead_of_guessing_when_there_is_no_currency_anywhere(monkeypatch):
    planner = _planner(_settings(monkeypatch), query="running shoes", max_price=Decimal("8000"))

    plan = planner.plan("running shoes under 8000")

    assert plan.requires_clarification
    assert plan.steps[0].action == "ask_clarification"
    assert "8,000" in plan.steps[0].reason and "currency" in plan.steps[0].reason
    assert plan.intent.max_price == Decimal("8000")  # the budget itself is kept
    assert plan.intent.currency is None  # and not invented


def test_words_the_user_typed_beat_the_market_default(monkeypatch):
    planner = _planner(_settings(monkeypatch, default="INR"),
                       query="running shoes", max_price=Decimal("50"))

    plan = planner.plan("running shoes under $50")

    assert (plan.intent.max_price, plan.intent.currency) == (Decimal("50"), "USD")


def test_no_budget_means_no_needless_clarification(monkeypatch):
    planner = _planner(_settings(monkeypatch), query="running shoes")

    plan = planner.plan("running shoes")

    assert plan.steps[0].action == "search_catalog" and not plan.requires_clarification
    assert plan.intent.currency is None


def test_reconcile_intent_reports_an_unspecified_currency_readably():
    intent = UserIntent(query="running shoes", max_price=Decimal("400000"))
    fixed, notes = reconcile_intent(intent, "running shoes for 4000 rupees")
    assert fixed.currency == "INR"
    assert "unspecified -> INR" in notes[0]  # not "None -> INR"


# ------------------------------------------------- request builders

def test_search_and_product_requests_never_invent_a_currency():
    settings = Settings()
    settings.shopify_buyer_country = None
    search = search_arguments(settings, "shoes", 10)["catalog"]
    assert search["context"] == {}
    assert "context" not in product_arguments(settings, "gid://shopify/p/1")["catalog"]

    settings.shopify_buyer_country = "IN"  # a configured market localises without a currency
    assert search_arguments(settings, "shoes", 10)["catalog"]["context"] == {"address_country": "IN"}


# -------------------------------------------------------------- ranking

def _product(price, currency):
    return Product(id=f"p-{price}-{currency}", title="Running shoes", url="https://shop.example/p",
                   price=price, currency=currency, description="running shoes", source="test")


def test_a_budget_without_a_currency_fails_closed_under_a_named_reason():
    intent = UserIntent(query="running shoes", max_price=Decimal("8000"))
    ranked, stats = filter_and_rank_with_stats(intent, [_product("100", "USD"), _product("5000", "INR")])
    assert ranked == []
    assert stats["dropped"]["budget_currency_unknown"] == 2  # not a silent guess, not "over budget"


def test_without_a_budget_products_in_any_currency_are_kept():
    intent = UserIntent(query="running shoes")
    ranked, stats = filter_and_rank_with_stats(intent, [_product("100", "USD"), _product("5000", "INR")])
    assert len(ranked) == 2 and stats["dropped"]["currency_mismatch"] == 0


def test_the_empty_result_message_explains_the_missing_currency_without_printing_none():
    intent = UserIntent(query="running shoes", max_price=Decimal("8000"))
    _, stats = filter_and_rank_with_stats(intent, [_product("100", "USD")])
    log = [{"query": "running shoes", "error": None, "fetched": 1,
            "dropped": stats["dropped"], "currencies_seen": stats["currencies_seen"]}]

    message = no_match_message(intent, log)

    assert "no currency was set" in message
    assert "None" not in message
    assert "within 8,000.00" in message


# ------------------------------------------------------------- executor

def test_the_executor_refuses_to_search_with_a_budget_that_has_no_currency():
    def boom():
        raise AssertionError("no MCP session may be opened")

    executor = ShopifyMcpExecutor(session_factory=boom, settings=Settings())
    plan = ExecutionPlan(
        intent=UserIntent(query="shoes", max_price=Decimal("8000")),
        steps=[PlanStep(step_id="s1", action="search_catalog", reason="r")],
    )
    with pytest.raises(ExecutorError, match="no currency"):
        executor.execute_plan(plan)


def test_a_clarification_plan_with_that_same_intent_is_allowed_through_validation():
    executor = ShopifyMcpExecutor(session_factory=lambda: None, settings=Settings())
    plan = ExecutionPlan(
        intent=UserIntent(query="shoes", max_price=Decimal("8000")),
        steps=[PlanStep(step_id="c", action="ask_clarification", reason="Which currency?")],
        requires_clarification=True,
    )
    assert executor._validate_and_bound_plan(plan).requires_clarification


# ----------------------------------------------------------- refinement

def test_a_refinement_replan_keeps_the_unverifiable_constraints_and_an_unspecified_currency():
    original = UserIntent(query="running shoes", delivery_city="Mumbai", return_required=True,
                          unsupported_constraints=["no laces"])
    replanned = UserIntent(query="shoes")  # a re-plan from shortened text lost everything

    restored = carry_over_constraints(replanned, original, 1)

    assert restored.delivery_city == "Mumbai"
    assert restored.return_required is True
    assert restored.unsupported_constraints == ["no laces"]
    assert restored.currency is None and restored.refinement_count == 1


# ------------------------------------------------------------ live_check

def test_live_check_never_calls_a_budget_respected_when_the_currency_is_unknown():
    intent = UserIntent(query="running shoes", max_price=Decimal("8000"))
    plan = ExecutionPlan(intent=intent, steps=[PlanStep(step_id="s", action="search_catalog", reason="r")])
    shown = ScoredProduct(product=_product("100", "USD"), score=0.9)
    decision = DiscoveryDecision(status="clarify", provider="stub", candidates=[shown], message="m")

    assert build_report("q", plan, decision, [])["final"]["budget_respected"] is False


# --------------------------------------------- end to end through the graph

class _Factory:
    def __init__(self, products=()):
        self.sessions, self.products = [], list(products)

    def __call__(self):
        session = ScriptedSession(self.products)
        self.sessions.append(session)
        return session

    @property
    def searches(self):
        return [c[1]["catalog"] for s in self.sessions for c in s.calls if c[0] == "search_catalog"]


def _run(settings, factory, **intent):
    executor = ShopifyMcpExecutor(session_factory=factory, settings=settings)
    return asyncio.run(arun_discovery_workflow(
        "running shoes under 8000", planner=_planner(settings, **intent),
        executor=executor, settings=settings))


def test_workflow_a_bare_budget_is_searched_as_the_market_currency_not_usd(monkeypatch):
    settings = _settings(monkeypatch, default="INR", country="IN")
    factory = _Factory([ucp_product(1, amount=500000)])  # INR 5,000

    state = _run(settings, factory, query="running shoes", max_price=Decimal("8000"))

    sent = factory.searches[0]
    assert sent["context"] == {"currency": "INR", "address_country": "IN"}
    assert sent["filters"]["price"] == {"max": 800000}
    assert state["decision"].candidates
    assert {c.product.currency for c in state["decision"].candidates} == {"INR"}


def test_workflow_without_any_currency_asks_and_never_touches_shopify(monkeypatch):
    settings = _settings(monkeypatch)
    factory = _Factory([ucp_product(1, amount=500000)])

    state = _run(settings, factory, query="running shoes", max_price=Decimal("8000"))

    assert state["decision"].status == "clarify"
    assert "currency" in state["decision"].message
    assert factory.sessions == []  # no MCP session was even opened
