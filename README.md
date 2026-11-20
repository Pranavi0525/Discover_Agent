# Discovery Agent

Modular Python foundation for the trust-based shopping discovery agent described in
[`docs/hld.md`](docs/hld.md) and [`docs/roadmap.md`](docs/roadmap.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

The package is intentionally offline by default. It does not contact Tavily or
Shopify during import or tests.

## Approved architecture

The approved flow is a hosted-model planner followed by a deterministic
executor:

```text
User -> Gemini Flash planner -> validated plan -> LangGraph executor
	-> Shopify MCP search_catalog -> parallel get_product calls
	-> filters/ranking -> Streamlit result and SQLite step state
```

An MCP server can be called without an LLM. In this project the executor owns
MCP calls; Gemini only interprets the request, creates a bounded plan, asks for
clarification, or summarizes validated results. Python guardrails own tool
permissions, budgets, inventory checks, retries, timeouts, and loop limits.

The project does not host a model. The approved initial option is Gemini Flash
through Google AI Studio's free API tier. Free-tier quotas and model names are
provider-controlled and must be checked when configuring the environment.

```bash
export GOOGLE_API_KEY="your-local-key"
export GEMINI_MODEL="the-current-supported-flash-model"
```

The implementation will add `langchain-google-genai` for the hosted model,
LangGraph for workflow state, and SQLite for sessions, conversations,
checkpoints, step events, and reviewed mistake records. Do not put keys in
source control, prompts, or audit logs.

## Run the planner only

The planner is currently independent from the executor. It extracts intent and
prints a validated bounded plan; it does not call Shopify or any other MCP tool.
Keep the real `.env` local and run:

```bash
python -m pip install -e .
PYTHONPATH=src python -m discovery_agent.planner_cli \
	"Find black running shorts under 40 dollars from Shopify stores"
```

Expected first step:

```json
{
	"action": "search_catalog",
	"product_ids": []
}
```

The direct executor path is now available for known catalog discovery. It skips
Gemini, calls the configured Shopify MCP server directly, and creates parallel
`get_product` calls from returned product IDs:

```bash
PYTHONPATH=src python -m discovery_agent.executor_cli \
	"black running shorts" --max-price 40
```

Configure the MCP command, tool names, shop domain, and storefront token in
your local `.env`. The default package is the published `shopify-mcp` server,
which uses `get-products` and `get-product-by-id`. Private stores require
client credentials or an access token; a token may be omitted only when the
selected server/store supports public access. The executor retries failed MCP calls within the configured
limit, times out slow calls, caps product enrichment, and returns partial
success when individual product calls fail. `.env` is ignored and must never
be committed.

## Provider wiring

All providers implement the `DiscoveryProvider` protocol and return normalized
`Product` records. Akanksha's Shopify MCP or Storefront GraphQL function can be
plugged into `ShopifyProvider`:

```python
from discovery_agent.orchestration import DiscoveryAgent
from discovery_agent.providers.shopify import ShopifyProvider
from discovery_agent.providers.tavily import TavilyProvider

agent = DiscoveryAgent(
	primary=ShopifyProvider(search=akanksha_shopify_search),
	fallback=TavilyProvider(api_key="..."),
)
decision = agent.discover(UserIntent(query="black running shorts", max_price=40))
```

The Shopify callable accepts `UserIntent` and returns mappings containing at
least `title` and `url`; optional fields are `id`, `price`, `currency`,
`available`, `merchant`, `description`, and `image_url`. Tavily uses the same
compact shape and never exposes raw page HTML to the rest of the agent.

Set `TAVILY_API_KEY` in the environment for live fallback use. Do not commit
keys. `JsonlRunTracker` can write an append-only audit ledger via
`DISCOVERY_LEDGER_PATH`; it records run metadata and product IDs, not secrets.

## Current boundary

The repository now contains the planner and deterministic Shopify MCP executor
foundation. The executor can run a structured catalog request without Gemini,
or consume a validated planner output later. SQLite persistence, Streamlit UI,
checkout, cryptographic mandates, and payment remain later phases.