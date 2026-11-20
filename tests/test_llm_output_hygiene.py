"""LLM output hygiene: the word "null" is not a merchant, and a guessed currency is reported."""
from decimal import Decimal

import pytest

from discovery_agent.budget import note_inferred_currency
from discovery_agent.config import Settings
from discovery_agent.constraints import unverified_constraints
from discovery_agent.planner import GeminiPlanner
from discovery_agent.ranking import filter_and_rank_with_stats
from discovery_agent.schemas import ExecutionPlan, PlanStep, Product, UserIntent


@pytest.mark.parametrize("word", ["null", "None", " N/A ", "unknown", "", "-"])
def test_nullish_words_mean_not_stated(word):
    intent = UserIntent(query="running shoes", merchant=word, delivery_city=word, currency=word)

    assert intent.merchant is None
    assert intent.delivery_city is None
    assert intent.currency is None
    assert unverified_constraints(intent) == []  # no phantom "delivery to null"


def test_real_values_are_kept():
    intent = UserIntent(query="running shoes", merchant=" Start Fitness ", delivery_city="Mumbai")

    assert intent.merchant == "Start Fitness"
    assert intent.delivery_city == "Mumbai"


def test_nullish_unsupported_constraints_are_dropped_but_real_ones_stay():
    intent = UserIntent(query="shoes", unsupported_constraints=["N/A", "null", "no laces", " none "])

    assert intent.unsupported_constraints == ["no laces"]


def test_a_null_merchant_does_not_filter_every_product_away():
    product = Product(
        id="p1", title="Running shoe", url="https://shop.example/p1", price=Decimal("10"),
        currency="INR", merchant="Some Shop", description="running shoe", source="test",
    )
    intent = UserIntent(query="running shoe", merchant="null", currency="INR", max_price=Decimal("50"))

    ranked, stats = filter_and_rank_with_stats(intent, [product])

    assert len(ranked) == 1
    assert stats["dropped"]["other_merchant"] == 0


@pytest.mark.parametrize("text", [
    "running shoes under 8000 rupees",
    "running shoes under Rs 8000",
    "running shoes under \u20b98000",
    "running shoes under 8000 in INR",
    "running shoes 8000 yen",
])
def test_no_note_when_the_user_stated_a_currency(text):
    currency = "JPY" if "yen" in text else "INR"
    intent = UserIntent(query="running shoes", max_price=Decimal("8000"), currency=currency)

    assert note_inferred_currency(intent, text) == []


def test_note_when_the_currency_was_only_inferred():
    intent = UserIntent(query="running shoes", max_price=Decimal("8000"), currency="INR")

    notes = note_inferred_currency(intent, "running shoes under 8000")

    assert len(notes) == 1 and "not stated" in notes[0] and "INR" in notes[0]


def test_no_note_without_a_budget_or_without_a_currency():
    assert note_inferred_currency(UserIntent(query="shoes", currency="INR"), "shoes") == []
    assert note_inferred_currency(
        UserIntent(query="shoes", max_price=Decimal("8000")), "shoes under 8000"
    ) == []


def test_planner_reports_a_currency_gemini_filled_in_on_its_own():
    class GuessingModel:
        def invoke(self, prompt):
            return ExecutionPlan(
                intent={"query": "running shoes", "max_price": Decimal("8000"), "currency": "INR"},
                steps=[PlanStep(step_id="1", action="search_catalog", reason="Find shoes")],
            )

    settings = Settings()
    settings.google_api_key = "test"
    settings.default_currency = "INR"
    planner = object.__new__(GeminiPlanner)
    planner.settings = settings
    planner._model = GuessingModel()

    plan = planner.plan("running shoes under 8000")

    assert plan.intent.currency == "INR"  # kept, not silently changed
    assert any("not stated" in note and "INR" in note for note in planner.last_corrections)
