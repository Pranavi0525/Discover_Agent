"""'running shoes for 4000 rupees' means: maximum price INR 4000 (explicit + tested)."""
from decimal import Decimal

import pytest

from discovery_agent.budget import extract_budget, reconcile_intent
from discovery_agent.config import Settings
from discovery_agent.planner import GeminiPlanner
from discovery_agent.ranking import filter_and_rank
from discovery_agent.schemas import ExecutionPlan, PlanStep, Product, UserIntent


@pytest.mark.parametrize(
    "text, amount, currency",
    [
        ("running shoes for 4000 rupees", "4000", "INR"),
        ("I need to try running shoes for 3000 rupees", "3000", "INR"),
        ("running shoes under 8000 rupees", "8000", "INR"),
        ("blue running shoes under 5000 rupees", "5000", "INR"),
        ("shoes under 500 rupees", "500", "INR"),
        ("running shoes under ₹8000 for daily use", "8000", "INR"),
        ("shoes below Rs. 4,000", "4000", "INR"),
        ("shoes within 4k INR", "4000", "INR"),
        ("headphones up to $50", "50", "USD"),
        ("boots max 120 dollars", "120", "USD"),
    ],
)
def test_explicit_budget_is_read_as_a_maximum_in_the_stated_currency(text, amount, currency):
    budget = extract_budget(text)
    assert (budget.amount, budget.currency) == (Decimal(amount), currency)


@pytest.mark.parametrize(
    "text",
    [
        "shoes over 3000 rupees",          # a minimum, never a maximum
        "shoes at least ₹3000",            # a minimum
        "shoes under $50 or 3000 rupees",  # two different amounts: ambiguous
        "shoes size 9",                    # no currency marker at all
        "shoes under 500",                 # amount without a currency
    ],
)
def test_minimums_ambiguous_and_unmarked_amounts_are_left_to_the_planner(text):
    assert extract_budget(text) is None


def test_reconcile_corrects_a_wrong_currency_and_amount_from_the_llm():
    llm = UserIntent(query="running shoes", max_price=Decimal("400000"), currency="USD")
    fixed, notes = reconcile_intent(llm, "running shoes for 4000 rupees")

    assert (fixed.max_price, fixed.currency) == (Decimal("4000"), "INR")
    assert len(notes) == 2


def test_reconcile_fills_in_a_budget_the_llm_forgot():
    llm = UserIntent(query="running shoes", currency="INR")
    fixed, _ = reconcile_intent(llm, "running shoes for 4000 rupees")
    assert fixed.max_price == Decimal("4000")


def test_reconcile_changes_nothing_when_the_llm_already_agrees():
    llm = UserIntent(query="running shoes", max_price=Decimal("4000"), currency="INR")
    fixed, notes = reconcile_intent(llm, "running shoes for 4000 rupees")
    assert fixed == llm and notes == []


def _planner_returning(intent: UserIntent) -> GeminiPlanner:
    class FakeModel:
        def invoke(self, prompt):
            assert "MAXIMUM price" in prompt
            assert "INR" in prompt and "major currency units" in prompt
            return ExecutionPlan(
                intent=intent,
                steps=[PlanStep(step_id="1", action="search_catalog", reason="search")],
            )

    settings = Settings()
    settings.google_api_key = "test"
    planner = object.__new__(GeminiPlanner)
    planner.settings = settings
    planner._model = FakeModel()
    return planner


def test_planner_output_is_reconciled_with_the_users_exact_words():
    # Gemini "forgets" the currency (schema default USD) and the amount.
    planner = _planner_returning(UserIntent(query="running shoes"))

    plan = planner.plan("running shoes for 4000 rupees")

    assert plan.intent.query == "running shoes"
    assert plan.intent.max_price == Decimal("4000")
    assert plan.intent.currency == "INR"
    assert planner.last_corrections  # the UI shows what was corrected


def test_planner_leaves_a_correct_plan_untouched():
    planner = _planner_returning(UserIntent(query="running shoes", max_price=Decimal("8000"), currency="INR"))
    plan = planner.plan("running shoes under ₹8000 for daily use")
    assert (plan.intent.max_price, plan.intent.currency) == (Decimal("8000"), "INR")
    assert planner.last_corrections == []


def _shoe(price, currency="INR"):
    return Product(id="p", title="Running shoes", url="https://shop.example/p", price=price,
                   currency=currency, source="test")


def test_max_price_is_a_hard_inclusive_cap_on_the_final_results():
    intent = UserIntent(query="running shoes", max_price=Decimal("4000"), currency="INR")
    kept = filter_and_rank(intent, [_shoe("4000"), _shoe("3999.99"), _shoe("4000.01"), _shoe("4500")])

    assert sorted(str(s.product.price) for s in kept) == ["3999.99", "4000"]


def test_a_number_in_another_currency_never_satisfies_an_inr_budget():
    intent = UserIntent(query="running shoes", max_price=Decimal("4000"), currency="INR")
    assert filter_and_rank(intent, [_shoe("90", currency="USD")]) == []
