"""The LangGraph workflow must keep the user's budget through every refinement."""
import asyncio
import re
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


class TextOnlyPlanner:
    """Behaves like Gemini re-planning from shortened text: it only knows what
    the text says, so a refined query carries no budget at all."""

    def __init__(self):
        self.calls = []

    def plan(self, text):
        self.calls.append(text)
        match = re.search(r"(\d+)\s*rupees", text)
        query = re.sub(r"\s*(for|under)?\s*\d+\s*rupees", "", text).strip()
        intent = UserIntent(query=query, currency="INR",
                            max_price=Decimal(match.group(1)) if match else None)
        return ExecutionPlan(intent=intent, steps=[PlanStep(step_id="s1", action="search_catalog", reason="r")])


class SessionFactory:
    def __init__(self, products=()):
        self.sessions = []
        self.products = list(products)

    def __call__(self):
        session = ScriptedSession(self.products)
        self.sessions.append(session)
        return session

    @property
    def searches(self):
        return [c[1]["catalog"] for s in self.sessions for c in s.calls if c[0] == "search_catalog"]

    @property
    def total_calls(self):
        return sum(len(s.calls) for s in self.sessions)


def _run(query, factory, settings=None, planner=None):
    settings = settings or _settings()
    executor = ShopifyMcpExecutor(session_factory=factory, settings=settings)
    return asyncio.run(arun_discovery_workflow(
        query, planner=planner or TextOnlyPlanner(), executor=executor, settings=settings))


def test_refinements_replanned_from_text_still_send_the_original_price_filter():
    factory = SessionFactory()  # Shopify has nothing, so the workflow refines

    state = _run("running shoes for 4000 rupees", factory)

    searches = factory.searches
    assert [s["query"] for s in searches] == ["running shoes", "shoes", "running sneakers"]
    # Before the fix the second search had NO price filter at all.
    assert all(s["filters"]["price"] == {"max": 400000} for s in searches)
    assert all(s["context"]["currency"] == "INR" for s in searches)
    assert state["decision"].status == "clarify"
    assert state["refinement_count"] == 2
    assert "Your budget was not changed" in state["decision"].message
    assert state["decision"].attempts == 3


def test_the_workflow_never_exceeds_the_configured_iterations_or_refinements():
    factory = SessionFactory()
    settings = _settings()
    state = _run("blue running shoes for 4000 rupees", factory, settings)

    assert len(factory.searches) <= 1 + settings.max_refinements
    assert state["graph_iterations"] <= settings.max_graph_iterations
    assert state["refinement_count"] <= settings.max_refinements


def test_weak_but_real_matches_are_shown_not_silently_replaced_by_a_broader_search():
    # Shares exactly one word ("shoes") with the 3-word search: score 1/3, a weak
    # but real match.  (A 2-word search can only score 0, 0.5 or 1.0.)
    weak = ucp_product(1, amount=300000, title="Garden shoes")
    weak["description"] = {"plain": "shoes for the garden"}
    factory = SessionFactory([weak])

    state = _run("blue running shoes for 4000 rupees", factory)

    assert len(factory.searches) == 1
    assert state["decision"].status == "clarify"
    assert state["decision"].candidates  # the weak match is displayed
    assert 0 < state["decision"].candidates[0].score < 0.5


def test_tool_call_budget_is_shared_by_all_graph_attempts():
    settings = _settings()
    settings.max_tool_calls = 4
    # over-budget products keep every attempt "empty" so the graph refines
    factory = SessionFactory([ucp_product(i, amount=900000) for i in range(1, 4)])

    state = _run("blue running shoes for 4000 rupees", factory, settings)

    assert factory.total_calls <= 4
    assert state["tool_calls_used"] <= 4
    assert all(s["filters"]["price"] == {"max": 400000} for s in factory.searches)
def test_products_sharing_no_word_with_the_search_are_never_shown_and_the_budget_is_kept():
    junk = ucp_product(1, amount=300000, title="Garden hose")
    junk["description"] = {"plain": "hose for the garden"}
    factory = SessionFactory([junk])

    state = _run("running shoes for 4000 rupees", factory)

    searches = factory.searches
    assert [s["query"] for s in searches] == ["running shoes", "shoes", "running sneakers"]
    assert all(s["filters"]["price"] == {"max": 400000} for s in searches)
    decision = state["decision"]
    assert decision.status == "clarify"
    assert decision.candidates == []
    assert state["refinement_count"] == 2
    assert "shared no word with your search" in decision.message
    assert "Your budget was not changed" in decision.message