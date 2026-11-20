import asyncio
from decimal import Decimal

import pytest

from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.forms import intent_overrides, normalize_form_inputs, planner_line
from discovery_agent.graph import arun_discovery_workflow
from discovery_agent.schemas import ExecutionPlan, PlanStep, UserIntent
from discovery_agent.storage import SqliteStore

from test_graph import FakeMcpSession, FakePlanner, _search_plan, _settings


# --- pure form helpers ------------------------------------------------------
def test_normalize_cleans_values_and_drops_invalid_ones():
    form = normalize_form_inputs({
        "max_price": "5,000", "currency": " inr ", "preferences": "  lightweight\nblue ",
        "merchant": "", "delivery_city": "mumbia",
    })
    assert form.max_price == Decimal("5000") and form.currency == "INR"
    assert form.preferences == "lightweight blue"  # one line, no injected newlines
    assert form.merchant is None and form.delivery_city == "Mumbai"
    bad = normalize_form_inputs({"max_price": "abc", "currency": "??", "preferences": "  "})
    assert bad == normalize_form_inputs({}) and normalize_form_inputs(None).max_price is None
    assert normalize_form_inputs({"max_price": "-5"}).max_price is None


@pytest.mark.parametrize("value", ["not sure", "Not_Sure", "", "unknown"])
def test_currency_not_sure_means_no_currency(value):
    form = normalize_form_inputs({"max_price": "5000", "currency": value})
    assert form.currency is None
    assert "currency not specified" in planner_line(form)
    assert "currency" not in intent_overrides(form)


def test_planner_line_and_overrides_only_use_intent_fields():
    form = normalize_form_inputs({"max_price": 5000, "currency": "INR", "delivery_city": "Mumbai",
                                  "preferences": "for daily use"})
    assert planner_line(form) == (
        "Form inputs: max budget 5000 INR; delivery city Mumbai; preferences: for daily use"
    )
    assert set(intent_overrides(form)) == {"max_price", "currency", "delivery_city"}
    assert set(intent_overrides(form)) <= set(UserIntent.model_fields)
    assert planner_line(normalize_form_inputs({})) is None


# --- through the graph --------------------------------------------------------
def _run(query, plans, settings, form, session=None, store=None):
    session = session or FakeMcpSession({"running shoes": [{"id": "p1"}]})
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    planner = FakePlanner(plans)
    state = asyncio.run(arun_discovery_workflow(
        query, planner=planner, executor=executor, settings=settings, form_inputs=form, store=store,
    ))
    return state, planner, session


def _plan(max_price=None, currency=None, **intent):
    return ExecutionPlan(
        intent=UserIntent(query="running shoes", max_price=max_price, currency=currency, **intent),
        steps=[PlanStep(step_id="s1", action="search_catalog", reason="find")],
    )


def test_planner_is_told_what_the_form_already_answered():
    state, planner, _ = _run("running shoes", [_plan()], _settings(),
                             {"max_price": "100", "currency": "USD", "delivery_city": "Mumbai"})
    assert planner.calls == ["running shoes\nForm inputs: max budget 100 USD; delivery city Mumbai"]


def test_no_form_leaves_planner_text_untouched():
    _, planner, _ = _run("running shoes", [_plan()], _settings(), None)
    assert planner.calls == ["running shoes"]


def test_form_budget_beats_the_llm_and_leaves_a_correction_note():
    # Gemini read a 10 USD budget (would drop the 29.99 product); the form says 100.
    state, _, _ = _run("running shoes", [_plan(Decimal("10"), "USD")], _settings(),
                       {"max_price": "100", "currency": "USD"})
    assert state["plan"].intent.max_price == Decimal("100")
    assert state["decision"].status == "selected"
    assert "max_price set from the form" in state["decision"].diagnostics["corrections"]


def test_no_note_when_the_form_agrees_with_the_planner():
    state, _, _ = _run("running shoes", [_plan(Decimal("100"), "USD")], _settings(),
                       {"max_price": "100", "currency": "USD"})
    assert state["decision"].diagnostics["corrections"] == []


def test_form_values_survive_a_refinement():
    settings = _settings()
    settings.max_refinements = 1
    settings.max_graph_iterations = 2
    session = FakeMcpSession({"blue running shoes": [{"id": "p1"}]})
    # The re-plan "forgets" the budget entirely, as a shortened query would.
    state, planner, _ = _run(
        "blue running shoes size 9",
        [_plan(Decimal("10"), "USD"), _plan()], settings,
        {"max_price": "100", "currency": "USD", "delivery_city": "Mumbai"}, session=session,
    )
    assert len(planner.calls) == 2 and state["refinement_count"] == 1
    assert state["original_intent"].max_price == Decimal("100")
    assert state["plan"].intent.max_price == Decimal("100")
    assert state["plan"].intent.delivery_city == "Mumbai"


def test_delivery_city_from_form_is_shown_as_unverified():
    state, _, _ = _run("running shoes", [_plan()], _settings(), {"delivery_city": "Mumbai"})
    unverified = state["decision"].diagnostics["unverified"]
    assert any(u["name"] == "delivery_city" for u in unverified)


def test_budget_with_unknown_currency_and_no_market_default_asks_instead_of_searching():
    settings = _settings()
    settings.default_currency = None
    session = FakeMcpSession({"running shoes": [{"id": "p1"}]})
    state, _, _ = _run("running shoes", [_plan()], settings,
                       {"max_price": "5000", "currency": "not sure"}, session=session)
    assert state["decision"].status == "clarify"
    assert "no currency" in state["decision"].message and "5,000" in state["decision"].message
    assert session.calls == []  # never searched


def test_budget_with_unknown_currency_uses_the_market_default_and_says_so():
    settings = _settings()
    settings.default_currency = "USD"
    state, _, _ = _run("running shoes", [_plan()], settings,
                       {"max_price": "100", "currency": "not sure"})
    assert state["plan"].intent.currency == "USD"
    assert state["decision"].status == "selected"
    assert any("assumed USD" in c for c in state["decision"].diagnostics["corrections"])


def test_planner_may_resolve_the_currency_itself():
    settings = _settings()
    settings.default_currency = None
    state, _, _ = _run("running shoes", [_plan(Decimal("1"), "USD")], settings,
                       {"max_price": "100", "currency": "not sure"})
    assert state["plan"].intent.currency == "USD"
    assert state["plan"].intent.max_price == Decimal("100")
    assert state["decision"].status == "selected"


def test_raw_form_values_are_saved_with_the_run(tmp_path):
    store = SqliteStore(tmp_path / "a.db")
    form = {"max_price": "100", "currency": "USD", "preferences": "lightweight"}
    state, _, _ = _run("running shoes", [_plan()], _settings(), form, store=store)
    import json
    assert json.loads(store.get_run(state["run_id"])["form_inputs_json"]) == form
