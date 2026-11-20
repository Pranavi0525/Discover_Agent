"""The workflow result must say what was assumed and what was NOT verified."""
import asyncio
from decimal import Decimal

from discovery_agent.config import Settings
from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.graph import arun_discovery_workflow
from discovery_agent.schemas import ExecutionPlan, PlanStep, UserIntent
from ucp_fakes import ScriptedSession, ucp_product


def _settings() -> Settings:
    settings = Settings()
    settings.max_retries_per_tool = 0
    settings.max_refinements = 2
    settings.max_graph_iterations = 3
    settings.shopify_buyer_country = None
    return settings


class FixedPlanner:
    """Returns one fixed intent, like Gemini + the deterministic reconcilers."""

    def __init__(self, notes=(), **intent_fields):
        self.last_corrections = list(notes)
        self._intent = UserIntent(
            query="running shoes", currency="INR", max_price=Decimal("4000"), **intent_fields
        )

    def plan(self, text):
        return ExecutionPlan(
            intent=self._intent,
            steps=[PlanStep(step_id="s1", action="search_catalog", reason="search")],
        )


def _run(planner):
    settings = _settings()
    session = ScriptedSession([ucp_product(1, amount=300000)])  # INR 3000, in budget
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    return asyncio.run(
        arun_discovery_workflow("running shoes", planner=planner, executor=executor, settings=settings)
    )


def test_delivery_city_and_returns_are_reported_as_unverified():
    state = _run(FixedPlanner(delivery_city="Mumbai", return_required=True))

    decision = state["decision"]
    assert decision.status == "selected"
    names = [item["name"] for item in decision.diagnostics["unverified"]]
    assert names == ["delivery_city", "return_required"]
    assert "Not verified against catalog data" in decision.message
    assert "delivery to Mumbai" in decision.message


def test_planner_assumptions_are_reported():
    note = "currency not stated; assumed INR (store market default)"
    state = _run(FixedPlanner(notes=[note]))

    decision = state["decision"]
    assert decision.diagnostics["corrections"] == [note]
    assert "Assumptions made" in decision.message
    assert "assumed INR" in decision.message


def test_a_plain_request_gets_no_extra_notes():
    state = _run(FixedPlanner())

    decision = state["decision"]
    assert decision.diagnostics["unverified"] == []
    assert decision.diagnostics["corrections"] == []
    assert "Not verified" not in decision.message
    assert "Assumptions" not in decision.message


def test_selected_message_does_not_claim_semantic_confidence():
    state = _run(FixedPlanner())

    assert "confidence" in state["decision"].message  # says it is NOT semantic confidence
    assert "high-confidence" not in state["decision"].message.lower()
