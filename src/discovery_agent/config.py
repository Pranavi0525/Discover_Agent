from dotenv import load_dotenv
import os
import shlex
from pathlib import Path

from .currency import currency_for_country, normalize_currency_code


# Load .env before Settings reads any environment variables.
load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

        # Keep the model configurable because Gemini model availability changes.
        self.gemini_model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite",
        )

        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

        self.ledger_path = Path(
            os.getenv(
                "DISCOVERY_LEDGER_PATH",
                "data/discovery-runs.jsonl",
            )
        )

        # Hard Python-side safety limits.
        # The model cannot raise these values.
        self.max_refinements = int(
            os.getenv("MAX_REFINEMENTS", "2")
        )
        self.max_plan_steps = int(
            os.getenv("MAX_PLAN_STEPS", "8")
        )
        self.max_product_details = int(
            os.getenv("MAX_PRODUCT_DETAILS", "10")
        )
        self.max_products = int(
            os.getenv("MAX_PRODUCTS", "10")
        )
        # Products whose get_product failed are shown as unverified candidates,
        # at most this many per result.
        self.max_unverified_shown = int(
            os.getenv("MAX_UNVERIFIED_SHOWN", "3")
        )
        self.max_tool_calls = int(
            os.getenv("MAX_TOOL_CALLS", "30")
        )
        self.max_retries_per_tool = int(
            os.getenv("MAX_RETRIES_PER_TOOL", "2")
        )
        self.max_planner_retries = int(
            os.getenv("MAX_PLANNER_RETRIES", "1")
        )
        self.max_graph_iterations = int(
            os.getenv("MAX_GRAPH_ITERATIONS", "3")
        )
        self.tool_timeout_seconds = float(
            os.getenv("TOOL_TIMEOUT_SECONDS", "20")
        )
        # Longest Retry-After (seconds) the executor will wait out before its
        # single retry of search_catalog after an HTTP 429; longer -> fail fast.
        self.max_retry_after_seconds = float(
            os.getenv("MAX_RETRY_AFTER_SECONDS", "10")
        )

        # Shopify UCP MCP over Streamable HTTP.
        self.shopify_ucp_mcp_url = os.getenv(
            "SHOPIFY_UCP_MCP_URL",
            "https://catalog.shopify.com/api/ucp/mcp",
        )

        self.shopify_ucp_agent_profile = os.getenv(
            "SHOPIFY_UCP_AGENT_PROFILE",
            "https://shopify.dev/ucp/agent-profiles/2026-08-25/valid-with-capabilities.json",
        )

        self.shopify_catalog_id = os.getenv(
            "SHOPIFY_CATALOG_ID"
        )

        # Read each store's refund/shipping policy pages to check "returnable"
        # and "deliverable in <city>".  Off by default (it makes web requests).
        self.policy_check = os.getenv("POLICY_CHECK", "0").strip().lower() in {"1", "true", "yes", "on"}

        # Buyer country sent as catalog.context.address_country so the Global
        # Catalog localizes availability and pricing.  When unset it is derived
        # from the requested currency (INR -> IN, USD -> US, ...).
        self.shopify_buyer_country = (
            os.getenv("SHOPIFY_BUYER_COUNTRY") or None
        )

        # Currency assumed ONLY when the user types a budget without one.
        # DEFAULT_CURRENCY wins; otherwise it follows the buyer market
        # (SHOPIFY_BUYER_COUNTRY=IN -> INR).  None means "ask the user".
        configured_currency = normalize_currency_code(os.getenv("DEFAULT_CURRENCY"))
        if not (
            isinstance(configured_currency, str)
            and len(configured_currency) == 3
            and configured_currency.isalpha()
        ):
            configured_currency = None
        self.default_currency = configured_currency or currency_for_country(
            self.shopify_buyer_country
        )

        # Legacy/local Shopify MCP settings retained for compatibility.
        self.shopify_mcp_command = os.getenv(
            "SHOPIFY_MCP_COMMAND",
            "npx",
        )

        self.shopify_mcp_args = shlex.split(
            os.getenv(
                "SHOPIFY_MCP_ARGS",
                "-y shopify-mcp",
            )
        )

        self.shopify_search_tool = os.getenv(
            "SHOPIFY_SEARCH_TOOL",
            "search_catalog",
        )

        self.shopify_product_tool = os.getenv(
            "SHOPIFY_PRODUCT_TOOL",
            "get_product",
        )

        self.shopify_shop_domain = (
            os.getenv("SHOPIFY_SHOP_DOMAIN")
            or os.getenv("MYSHOPIFY_DOMAIN")
        )

        self.shopify_storefront_access_token = (
            os.getenv("SHOPIFY_STOREFRONT_ACCESS_TOKEN")
            or os.getenv("SHOPIFY_ACCESS_TOKEN")
        )

        # Phase 2 / M2.1 input guard.  Rules are always on when enabled; the
        # classifier is optional (none | promptguard-22m | promptguard-86m |
        # deberta | any org/model Hugging Face id).
        _on = {"1", "true", "yes", "on"}
        self.guard_enabled = os.getenv("GUARD_ENABLED", "1").strip().lower() in _on
        self.guard_classifier = os.getenv("GUARD_CLASSIFIER", "none").strip()
        self.guard_max_chars = int(os.getenv("GUARD_MAX_CHARS", "1000"))
        self.guard_block_threshold = float(os.getenv("GUARD_BLOCK_THRESHOLD", "0.90"))
        self.guard_review_threshold = float(os.getenv("GUARD_REVIEW_THRESHOLD", "0.50"))
        self.guard_fail_closed = os.getenv("GUARD_FAIL_CLOSED", "0").strip().lower() in _on

        # Phase 2 / M2.2.  An identical request (same session, query and form
        # values) inside the window returns the existing run instead of a new
        # one; 0 disables.  The rate limit caps runs per session per window;
        # 0 disables.  Both protect the Gemini free-tier quota.
        self.dedup_window_seconds = int(os.getenv("DEDUP_WINDOW_SECONDS", "120"))
        self.rate_limit_max_runs = int(os.getenv("RATE_LIMIT_MAX_RUNS", "5"))
        self.rate_limit_window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

        # Phase 2 / M2.2 abuse limits, enforced atomically in SQLite (storage.reserve_slot).
        # These two defaults are conservative starting points, not measured against your
        # Gemini quota; tune them to it.  0 switches a limit off, and the app then has no
        # protection of that kind.
        #   RATE_LIMIT_GLOBAL_MAX_RUNS: runs across ALL sessions per window.  A new session
        #     gets a fresh per-session quota, so only a global cap bounds total Gemini
        #     and Shopify calls.  Default 30 = six sessions at full per-session rate.
        #   MAX_CONCURRENT_RUNS: runs executing at once.  Each run can make several Gemini
        #     and MCP calls, so this bounds simultaneous load.  Default 4.
        #   RATE_LIMIT_SLOT_TTL_SECONDS: a run that never reports finishing (process killed)
        #     stops counting as running after this long.  Default 300; keep it above the
        #     slowest legitimate run.
        self.rate_limit_global_max_runs = int(os.getenv("RATE_LIMIT_GLOBAL_MAX_RUNS", "30"))
        self.max_concurrent_runs = int(os.getenv("MAX_CONCURRENT_RUNS", "4"))
        self.rate_limit_slot_ttl_seconds = int(os.getenv("RATE_LIMIT_SLOT_TTL_SECONDS", "300"))


def get_settings() -> Settings:
    return Settings()