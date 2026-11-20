from decimal import Decimal

import pytest

from discovery_agent.config import Settings
from discovery_agent.execution import ExecutorError, ShopifyMcpExecutor
from discovery_agent.planner import GeminiPlanner
from discovery_agent.schemas import ExecutionPlan, PlanStep, UserIntent


def test_executor_rejects_preinvented_product_ids():
    settings = Settings()
    plan = ExecutionPlan(
        intent=UserIntent(query="running shoes"),
        steps=[
            PlanStep(step_id="search", action="search_catalog", reason="search"),
            PlanStep(
                step_id="detail",
                action="get_product",
                reason="detail",
                product_ids=["invented-id"],
            ),
        ],
    )
    executor = ShopifyMcpExecutor(settings=settings)
    with pytest.raises(ExecutorError, match="product IDs"):
        executor.execute_plan(plan)


def test_executor_rejects_plan_over_python_step_limit():
    settings = Settings()
    settings.max_plan_steps = 1
    plan = ExecutionPlan(
        intent=UserIntent(query="shoes"),
        steps=[
            {"step_id": "s1", "action": "search_catalog", "reason": "search"},
            {"step_id": "s2", "action": "ask_clarification", "reason": "clarify"},
        ],
    )
    executor = ShopifyMcpExecutor(settings=settings)
    with pytest.raises(ExecutorError, match="MAX_PLAN_STEPS"):
        executor.execute_plan(plan)


def test_executor_revalidates_model_dump_before_mcp():
    settings = Settings()
    settings.max_refinements = 0
    plan = ExecutionPlan(
        intent=UserIntent(query="shoes", refinement_count=1),
        steps=[{"step_id": "s1", "action": "search_catalog", "reason": "search"}],
    )
    executor = ShopifyMcpExecutor(settings=settings)
    with pytest.raises(ExecutorError, match="MAX_REFINEMENTS"):
        executor.execute_plan(plan)
