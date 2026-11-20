"""Live verification harness: real Gemini planner + real Shopify UCP catalog.

    python -m discovery_agent.live_check
    python -m discovery_agent.live_check "running shoes for 4000 rupees" "shoes under 500 rupees"

It runs exactly the objects the Streamlit UI uses (GeminiPlanner ->
DiscoveryAgent -> ShopifyUcpCatalogProvider) and prints, per query, what
Gemini understood, what was sent to Shopify, what came back and whether the
budget was respected.  It never prints API keys.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from typing import Any

from .config import get_settings
from .orchestration import DiscoveryAgent
from .planner import GeminiPlanner
from .providers.shopify_ucp import ShopifyUcpCatalogProvider
from .schemas import DiscoveryDecision, ExecutionPlan

DEFAULT_QUERIES = [
    "running shoes for 4000 rupees",
    "running shoes under 8000 rupees",
    "blue running shoes under 5000 rupees",
    "shoes under 500 rupees",
]


def build_report(
    query: str, plan: ExecutionPlan, decision: DiscoveryDecision, corrections: list[str]
) -> dict[str, Any]:
    """Summarize one run.  Pure function: no network, easy to unit test."""
    intent = plan.intent
    log = decision.diagnostics.get("attempt_log", [])
    searches = [entry for entry in log if entry.get("error") is None]
    shopify = [entry.get("shopify", {}) for entry in searches]
    shown = decision.candidates

    def within_budget(scored) -> bool:
        product = scored.product
        if intent.currency is not None and product.currency != intent.currency:
            return False
        if intent.max_price is None:
            return True
        # A budget with no currency cannot be verified, so it is never "respected".
        return (
            intent.currency is not None
            and product.price is not None
            and product.price <= intent.max_price
        )

    return {
        "user_query": query,
        "gemini": {
            "query": intent.query,
            "max_price": str(intent.max_price) if intent.max_price is not None else None,
            "currency": intent.currency,
            "merchant": intent.merchant,
            "corrected_from_users_words": corrections,
        },
        "attempts": [
            {
                "query_sent": entry.get("query"),
                "mcp_price_filter": (s.get("search_request", {}).get("filters", {}) or {}).get("price"),
                "mcp_context": s.get("search_request", {}).get("context"),
                "shopify_products_returned": s.get("search_returned"),
                "product_detail_calls": s.get("details_requested"),
                "details_ok": s.get("details_ok"),
                "details_failed": s.get("details_failed"),
                "products_after_normalization": entry.get("fetched"),
                "products_after_hard_filters": entry.get("kept"),
                "dropped": entry.get("dropped"),
                "currencies_seen": entry.get("currencies_seen"),
                "shopify_messages": s.get("shopify_messages"),
                "tool_calls": s.get("tool_calls"),
            }
            for entry, s in zip(searches, shopify)
        ],
        "errors": [entry["error"] for entry in log if entry.get("error")],
        "final": {
            "status": decision.status,
            "products_shown": len(shown),
            "titles": [c.product.title for c in shown],
            "prices": [f"{c.product.currency} {c.product.price}" for c in shown],
            "budget_respected": all(within_budget(c) for c in shown),
            "graph_iterations_or_search_attempts": decision.attempts,
            "total_tool_calls": sum((s.get("tool_calls") or 0) for s in shopify),
            "refinements_used": max((entry.get("refinement", 0) for entry in searches), default=0),
            "message": decision.message,
        },
    }


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    queries = list(argv if argv is not None else sys.argv[1:]) or DEFAULT_QUERIES
    settings = get_settings()
    planner = GeminiPlanner(settings)
    print(f"model={settings.gemini_model}  endpoint={settings.shopify_ucp_mcp_url}")
    for query in queries:
        plan = planner.plan(query)
        if plan.requires_clarification or plan.steps[0].action == "ask_clarification":
            print("\n" + "=" * 78)
            print(f"{query!r} needs clarification before searching: {plan.steps[0].reason}")
            continue
        agent = DiscoveryAgent(
            primary=ShopifyUcpCatalogProvider(settings),
            max_refinements=settings.max_refinements,
            max_graph_iterations=settings.max_graph_iterations,
        )
        decision = agent.discover(plan.intent)
        report = build_report(query, plan, decision, getattr(planner, "last_corrections", []))
        print("\n" + "=" * 78)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
