import argparse
import json

from .config import get_settings
from .execution import ExecutorError, ShopifyMcpExecutor
from .schemas import ExecutionPlan, UserIntent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Shopify MCP product discovery without an LLM.")
    parser.add_argument("query", help="Catalog search query")
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--currency", default=None,
                        help="ISO code; defaults to DEFAULT_CURRENCY / the buyer market")
    args = parser.parse_args()
    currency = args.currency or get_settings().default_currency
    if args.max_price is not None and not currency:
        parser.error("--max-price needs --currency (or set DEFAULT_CURRENCY)")
    intent = UserIntent(query=args.query, max_price=args.max_price, currency=currency)
    try:
        report = ShopifyMcpExecutor().execute_plan(
            ExecutionPlan(
                intent=intent,
                steps=[{
                    "step_id": "step_1",
                    "action": "search_catalog",
                    "reason": "Direct structured catalog request",
                }],
            )
        )
    except ExecutorError as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(f"Shopify MCP startup failed: {exc}")
    print(json.dumps(report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()