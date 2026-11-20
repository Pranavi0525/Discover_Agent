import pytest

from discovery_agent.config import Settings
from discovery_agent.providers.shopify_live import LiveShopifyMcpProvider
from discovery_agent.schemas import ExecutionPlan, ExecutionReport, Product, StepExecution, UserIntent


class FakeExecutor:
    def __init__(self, report: ExecutionReport) -> None:
        self._report = report

    def execute_plan(self, plan: ExecutionPlan) -> ExecutionReport:
        return self._report


def _plan(intent: UserIntent) -> ExecutionPlan:
    return ExecutionPlan(
        intent=intent,
        steps=[{"step_id": "s1", "action": "search_catalog", "reason": "test"}],
    )


def test_search_returns_products_from_execution_report():
    intent = UserIntent(query="running shoes")
    report = ExecutionReport(
        plan=_plan(intent),
        steps=[StepExecution(step_id="s1", action="search_catalog", status="SUCCEEDED", result_count=1)],
        products=[
            Product(
                id="p1",
                title="Trail Runner",
                url="https://example.com/p1",
                price="49.99",
                currency="USD",
                source="shopify-mcp",
            )
        ],
    )
    provider = LiveShopifyMcpProvider(executor=FakeExecutor(report), settings=Settings())

    result = provider.search(intent)

    assert result.provider == "shopify-mcp"
    assert len(result.products) == 1
    assert result.products[0].id == "p1"


def test_search_raises_when_no_products_and_steps_failed():
    intent = UserIntent(query="running shoes")
    report = ExecutionReport(
        plan=_plan(intent),
        steps=[StepExecution(step_id="s1", action="search_catalog", status="FAILED", error="MCP unreachable")],
        products=[],
    )
    provider = LiveShopifyMcpProvider(executor=FakeExecutor(report), settings=Settings())

    with pytest.raises(RuntimeError, match="MCP unreachable"):
        provider.search(intent)
