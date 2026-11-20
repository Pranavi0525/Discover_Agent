from __future__ import annotations

from ..config import Settings, get_settings
from ..execution import ShopifyMcpExecutor
from ..schemas import ExecutionPlan, PlanStep, SearchResult, UserIntent


class LiveShopifyMcpProvider:
    """DiscoveryProvider adapter backed by the existing ShopifyMcpExecutor.

    `ShopifyProvider` (providers/shopify.py) is a seam waiting for Akanksha's
    search callable and normalizes raw mappings through the Tavily-shaped
    normalizer. That doesn't fit `ShopifyMcpExecutor`, which already performs
    the MCP calls itself and returns fully normalized `Product` records.

    This adapter wraps `ShopifyMcpExecutor` behind the `DiscoveryProvider`
    protocol (`ports.py`) directly, so `DiscoveryAgent.discover()` can call a
    real Shopify MCP server today without waiting on the separate callable.
    It builds a single-step `search_catalog` plan for every request; get_product
    detail calls are already handled inside the executor.
    """

    name = "shopify-mcp"

    def __init__(
        self,
        executor: ShopifyMcpExecutor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.executor = executor or ShopifyMcpExecutor(settings=self.settings)

    def search(self, intent: UserIntent) -> SearchResult:
        plan = ExecutionPlan(
            intent=intent,
            steps=[
                PlanStep(
                    step_id="search_catalog",
                    action="search_catalog",
                    reason="Direct UI-triggered catalog search",
                )
            ],
        )
        report = self.executor.execute_plan(plan)
        failed_search_steps = [
            step
            for step in report.steps
            if step.action == "search_catalog" and step.status in ("FAILED", "TIMED_OUT")
        ]
        error = failed_search_steps[0].error if failed_search_steps else None
        if error and not report.products:
            # Raise so DiscoveryAgent.discover() records this provider's failure
            # in its error list instead of silently falling through to fallback
            # with no explanation.
            raise RuntimeError(f"Shopify MCP search_catalog failed: {error}")
        return SearchResult(provider=self.name, products=report.products, query=intent.query, error=error)
