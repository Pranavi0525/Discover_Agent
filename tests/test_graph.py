import asyncio
from decimal import Decimal

import pytest

from discovery_agent.config import Settings
from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.graph import WorkflowError, arun_discovery_workflow
from discovery_agent.schemas import ExecutionPlan, PlanStep, UserIntent


def _settings() -> Settings:
    settings = Settings()
    settings.shopify_search_tool = "search_catalog"
    settings.shopify_product_tool = "get_product"
    settings.max_refinements = 2
    settings.max_graph_iterations = 3
    return settings


class FakeMcpSession:
    """Mirrors tests/test_execution.py's fake session so the graph is tested
    against the same executor contract without any real MCP/network calls."""

    def __init__(self, catalog: dict[str, list[dict]]):
        self.catalog = catalog
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == "search_catalog":
            # search_arguments() sends the UCP-shaped payload
            # {"meta": ..., "catalog": {"query": ..., ...}}, not a flat "query" key.
            query = arguments.get("catalog", {}).get("query", "")
            return {"products": self.catalog.get(query, [])}
        await asyncio.sleep(0)
        product_id = arguments["catalog"]["id"]
        return {
            "product": {
                "id": product_id,
                "title": f"Product {product_id}",
                "url": f"https://shop.example/{product_id}",
                "price": "29.99",
                "currency": "USD",
                "available": True,
                "description": "running shoes for everyday training",
            }
        }


class FakePlanner:
    """Returns queued plans in order, one per call, so a test can script a
    refinement loop (first plan finds nothing, second plan finds a match)."""

    def __init__(self, plans: list[ExecutionPlan]):
        self._plans = list(plans)
        self.calls: list[str] = []

    def plan(self, query: str) -> ExecutionPlan:
        self.calls.append(query)
        if not self._plans:
            raise AssertionError("FakePlanner called more times than plans were queued")
        return self._plans.pop(0)


def _search_plan(query: str, max_price=None) -> ExecutionPlan:
    return ExecutionPlan(
        intent=UserIntent(query=query, max_price=max_price),
        steps=[PlanStep(step_id="s1", action="search_catalog", reason="Find matching products")],
    )


def test_workflow_selects_a_high_confidence_product():
    settings = _settings()
    session = FakeMcpSession(catalog={"running shoes": [{"id": "p1"}]})
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    planner = FakePlanner([_search_plan("running shoes")])

    state = asyncio.run(
        arun_discovery_workflow("running shoes", planner=planner, executor=executor, settings=settings)
    )

    assert planner.calls == ["running shoes"]
    assert state["status"] == "decided"
    assert state["decision"].status == "selected"
    assert state["graph_iterations"] == 1
    assert [call[0] for call in session.calls] == ["search_catalog", "get_product"]


def test_workflow_refines_and_terminates_within_bounds_when_nothing_matches():
    settings = _settings()
    settings.max_refinements = 1
    settings.max_graph_iterations = 2
    # Every search returns nothing, however the query is broadened, so the
    # workflow must give up after MAX_REFINEMENTS instead of looping forever.
    session = FakeMcpSession(catalog={})
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    planner = FakePlanner([
        _search_plan("blue running shoes size 9"),
        _search_plan("blue running shoes"),
    ])

    state = asyncio.run(
        arun_discovery_workflow(
            "blue running shoes size 9", planner=planner, executor=executor, settings=settings
        )
    )

    assert planner.calls == ["blue running shoes size 9", "blue running shoes"]
    assert state["decision"].status == "clarify"
    assert state["refinement_count"] == 1
    assert state["graph_iterations"] == 2


def test_workflow_stops_immediately_when_planner_asks_for_clarification():
    settings = _settings()
    executor = ShopifyMcpExecutor(session_factory=lambda: (_ for _ in ()).throw(AssertionError("no MCP call")),
                                   settings=settings)
    plan = ExecutionPlan(
        intent=UserIntent(query="something"),
        steps=[PlanStep(step_id="s1", action="ask_clarification", reason="Too vague to search safely")],
        requires_clarification=True,
    )
    planner = FakePlanner([plan])

    state = asyncio.run(arun_discovery_workflow("something", planner=planner, executor=executor, settings=settings))

    assert state["decision"].status == "clarify"
    assert state["decision"].message == "Too vague to search safely"
    assert state["status"] == "done"


def test_workflow_raises_when_planner_fails():
    settings = _settings()
    executor = ShopifyMcpExecutor(session_factory=lambda: (_ for _ in ()).throw(AssertionError("no MCP call")),
                                   settings=settings)

    class BrokenPlanner:
        def plan(self, query: str) -> ExecutionPlan:
            raise RuntimeError("Gemini quota exceeded")

    with pytest.raises(WorkflowError, match="Gemini quota exceeded"):
        asyncio.run(
            arun_discovery_workflow("anything", planner=BrokenPlanner(), executor=executor, settings=settings)
        )


def test_workflow_rejects_a_plan_that_exceeds_max_plan_steps():
    settings = _settings()
    settings.max_plan_steps = 1
    executor = ShopifyMcpExecutor(session_factory=lambda: (_ for _ in ()).throw(AssertionError("no MCP call")),
                                   settings=settings)
    oversized_plan = ExecutionPlan(
        intent=UserIntent(query="shoes"),
        steps=[
            PlanStep(step_id="s1", action="search_catalog", reason="Find shoes"),
            PlanStep(step_id="s2", action="get_product", reason="Enrich", product_ids=["p1"]),
        ],
    )
    planner = FakePlanner([oversized_plan])

    with pytest.raises(WorkflowError, match="MAX_PLAN_STEPS"):
        asyncio.run(arun_discovery_workflow("shoes", planner=planner, executor=executor, settings=settings))


def test_graph_iteration_ceiling_is_enforced_even_with_unlimited_refinements():
    settings = _settings()
    settings.max_refinements = 100
    settings.max_graph_iterations = 2
    session = FakeMcpSession(catalog={})
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    planner = FakePlanner([
        _search_plan("a b c"),
        _search_plan("a b"),
    ])

    state = asyncio.run(arun_discovery_workflow("a b c", planner=planner, executor=executor, settings=settings))

    # Bounded by MAX_GRAPH_ITERATIONS even though MAX_REFINEMENTS would allow more.
    assert state["graph_iterations"] == 2
    assert len(planner.calls) == 2