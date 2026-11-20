"""Delivery-city / return constraints: read from the user's words, shown as unverified."""
from decimal import Decimal

import pytest

from discovery_agent.config import Settings
from discovery_agent.constraints import (
    canonical_city,
    extract_delivery_city,
    extract_return_request,
    reconcile_constraints,
    unverified_constraints,
)
from discovery_agent.planner import GeminiPlanner
from discovery_agent.schemas import ExecutionPlan, PlanStep, UserIntent

QUERY = "I want a running shoes under 8000 rupees and it shoild be deliverable in mumbia and replaceble"


@pytest.mark.parametrize(
    "spelling, city",
    [("mumbai", "Mumbai"), ("Mumbia", "Mumbai"), ("bombay", "Mumbai"), ("bangalore", "Bengaluru"),
     ("Bengalore", "Bengaluru"), ("new delhi", "Delhi"), ("hydrabad", "Hyderabad"), ("xyzzy", None), ("", None)],
)
def test_canonical_city_handles_aliases_and_typos(spelling, city):
    assert canonical_city(spelling) == city


def test_short_words_are_not_fuzzy_matched_to_cities():
    assert canonical_city("pine") is None  # close to "pune" but too short to guess


@pytest.mark.parametrize(
    "text, city",
    [
        (QUERY, "Mumbai"),
        ("shoes that can be shipped to Bangalore", "Bengaluru"),
        ("delivery to new delhi please", "Delhi"),
        ("I live in Mumbai, want running shoes", None),          # no delivery cue
        ("deliver to Mumbai or Pune", None),                      # ambiguous
    ],
)
def test_delivery_city_needs_a_delivery_cue_and_one_city(text, city):
    assert extract_delivery_city(text) == city


@pytest.mark.parametrize(
    "text, required, terms",
    [
        (QUERY, True, ("exchange",)),
        ("returnable running shoes", True, ("return",)),
        ("easy refund and exchange", True, ("exchange", "refund")),
        ("non-returnable clearance shoes", False, ()),
        ("no returns needed", False, ()),
        ("blue running shoes", False, ()),
    ],
)
def test_return_request_detection(text, required, terms):
    request = extract_return_request(text)
    assert (request.required, request.terms) == (required, terms)


def test_reconcile_fills_what_the_llm_dropped():
    llm_intent = UserIntent(query="running shoes", max_price=Decimal("8000"), currency="INR")
    intent, notes = reconcile_constraints(llm_intent, QUERY)
    assert intent.delivery_city == "Mumbai"
    assert intent.return_required is True
    assert intent.max_price == Decimal("8000")  # untouched
    assert len(notes) == 2


def test_reconcile_normalises_an_llm_misspelling_and_keeps_returns_flag():
    llm_intent = UserIntent(query="running shoes", delivery_city="Mumbia", return_required=True)
    intent, _ = reconcile_constraints(llm_intent, "running shoes")
    assert intent.delivery_city == "Mumbai"
    assert intent.return_required is True


def test_reconcile_never_clears_an_llm_constraint():
    llm_intent = UserIntent(query="shoes", return_required=True, delivery_city="Pune")
    intent, notes = reconcile_constraints(llm_intent, "shoes")
    assert (intent.return_required, intent.delivery_city, notes) == (True, "Pune", [])


def test_unverified_list_is_never_empty_when_constraints_exist():
    intent = UserIntent(
        query="running shoes", delivery_city="Mumbai", return_required=True,
        unsupported_constraints=["no leather"],
    )
    names = [item.name for item in unverified_constraints(intent)]
    assert names == ["delivery_city", "return_required", "other"]
    assert unverified_constraints(UserIntent(query="running shoes")) == []


def test_long_or_many_unsupported_constraints_are_trimmed_not_rejected():
    intent = UserIntent(query="shoes", unsupported_constraints=["x" * 500] + [f"c{i}" for i in range(20)])
    assert len(intent.unsupported_constraints) == 8
    assert len(intent.unsupported_constraints[0]) == 200


def test_planner_repairs_a_gemini_plan_that_dropped_the_constraints():
    class DroppingModel:
        def invoke(self, prompt):
            assert "intent.delivery_city" in prompt and "intent.return_required" in prompt
            return ExecutionPlan(
                intent={"query": "running shoes", "max_price": Decimal("8000"), "currency": "INR"},
                steps=[PlanStep(step_id="1", action="search_catalog", reason="Find running shoes")],
            )

    settings = Settings()
    settings.google_api_key = "test"
    planner = object.__new__(GeminiPlanner)
    planner.settings = settings
    planner._model = DroppingModel()

    plan = planner.plan(QUERY)

    assert plan.intent.delivery_city == "Mumbai"
    assert plan.intent.return_required is True
    assert any("delivery_city" in note for note in planner.last_corrections)
