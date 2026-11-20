from decimal import Decimal

import pytest

from discovery_agent.config import Settings
from discovery_agent.planner import GeminiPlanner, PlannerConfigurationError, PlannerResponseError
from discovery_agent.schemas import ExecutionPlan, PlanStep


def test_execution_plan_requires_search_before_product_details():
    with pytest.raises(ValueError):
        ExecutionPlan(
            intent={"query": "black running shorts", "max_price": Decimal("40")},
            steps=[PlanStep(step_id="1", action="get_product", reason="Fetch details")],
        )


def test_execution_plan_accepts_search_then_details():
    plan = ExecutionPlan(
        intent={"query": "black running shorts", "max_price": Decimal("40")},
        steps=[
            PlanStep(step_id="1", action="search_catalog", reason="Find matching products"),
            PlanStep(step_id="2", action="get_product", reason="Enrich returned products", product_ids=["p1"]),
        ],
    )
    assert plan.intent.currency is None  # unspecified stays unspecified; never a silent USD


def test_planner_requires_key_without_constructing_model(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = Settings()
    settings.google_api_key = None
    with pytest.raises(PlannerConfigurationError, match="GOOGLE_API_KEY"):
        GeminiPlanner(settings)


def test_planner_extracts_a_concise_catalog_query(monkeypatch):
    """The planner contract produces a catalog-ready query from natural language."""
    from discovery_agent.planner import GeminiPlanner

    class FakeModel:
        def invoke(self, prompt):
            assert '`intent.query` field is the concise product/category search' in prompt
            assert '"running shoes"' in prompt
            return ExecutionPlan(
                intent={"query": "running shoes", "max_price": Decimal("8000"), "currency": "INR"},
                steps=[PlanStep(step_id="1", action="search_catalog", reason="Find running shoes")],
            )

    settings = Settings()
    settings.google_api_key = "test"
    planner = object.__new__(GeminiPlanner)
    planner.settings = settings
    planner._model = FakeModel()

    plan = planner.plan("running shoes under ₹8000 for daily use")

    assert plan.intent.query == "running shoes"
    assert plan.intent.max_price == Decimal("8000")
    assert plan.intent.currency == "INR"


def test_planner_rejects_empty_structured_response_clearly():
    class EmptyModel:
        def invoke(self, prompt):
            return None

    settings = Settings()
    settings.google_api_key = "test"
    settings.max_planner_retries = 3
    planner = object.__new__(GeminiPlanner)
    planner.settings = settings
    planner._model = EmptyModel()

    with pytest.raises(PlannerResponseError, match="no structured ExecutionPlan"):
        planner.plan("cooking pan under 800 rupees")


def test_planner_does_not_retry_rate_limits():
    class RateLimitedModel:
        calls = 0

        def invoke(self, prompt):
            self.calls += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    settings = Settings()
    settings.google_api_key = "test"
    settings.max_planner_retries = 3
    model = RateLimitedModel()
    planner = object.__new__(GeminiPlanner)
    planner.settings = settings
    planner._model = model

    with pytest.raises(RuntimeError, match="No automatic retry"):
        planner.plan("cooking pan under 800 rupees")

    assert model.calls == 1
