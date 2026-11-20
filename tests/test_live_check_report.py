from decimal import Decimal

from discovery_agent.live_check import build_report
from discovery_agent.orchestration import DiscoveryAgent
from discovery_agent.schemas import ExecutionPlan, PlanStep, Product, SearchResult, UserIntent


class Provider:
    name = "stub"

    def search(self, intent):
        products = [Product(id="p", title="Running shoes", url="https://shop.example/p", price="3500",
                            currency="INR", description="running shoes", source="stub")]
        diag = {"search_request": {"filters": {"price": {"max": 400000}}, "context": {"currency": "INR"}},
                "search_returned": 4, "details_requested": 4, "details_ok": 4, "details_failed": 0,
                "tool_calls": 5}
        return SearchResult(provider="stub", products=products, query=intent.query, diagnostics=diag)


def test_report_states_what_was_sent_returned_and_whether_the_budget_held():
    intent = UserIntent(query="running shoes", max_price=Decimal("4000"), currency="INR")
    plan = ExecutionPlan(intent=intent, steps=[PlanStep(step_id="s", action="search_catalog", reason="r")])
    decision = DiscoveryAgent(primary=Provider()).discover(intent)

    report = build_report("running shoes for 4000 rupees", plan, decision, [])

    assert report["gemini"] == {"query": "running shoes", "max_price": "4000", "currency": "INR",
                                "merchant": None, "corrected_from_users_words": []}
    attempt = report["attempts"][0]
    assert attempt["mcp_price_filter"] == {"max": 400000}
    assert attempt["shopify_products_returned"] == 4
    assert attempt["product_detail_calls"] == 4
    assert report["final"]["budget_respected"] is True
    assert report["final"]["total_tool_calls"] == 5
    assert report["final"]["refinements_used"] == 0
