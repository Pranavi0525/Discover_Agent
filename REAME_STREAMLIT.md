## Run the Streamlit UI

The UI is a thin presentation layer over `DiscoveryAgent` — it builds a
`UserIntent` from a form, calls `DiscoveryAgent.discover()`, and renders the
resulting decision. It does not add any business logic of its own; ranking,
confidence tiers, and provider fallback all come from the modules described
above.

```bash
python -m pip install -e ".[dev,ui]"
streamlit run src/discovery_agent/ui/streamlit_app.py
```

This works whether or not the package is pip-installed, since the app adds
`src/` to `sys.path` itself. The primary provider is
`LiveShopifyMcpProvider`, which drives the existing `ShopifyMcpExecutor`
directly (no separate wiring needed); `TavilyProvider` is used as fallback
automatically whenever `TAVILY_API_KEY` is set.

Before running a live search, configure the same environment variables used
by the executor CLI, either exported directly or via a local `.env`:

```bash
SHOPIFY_MCP_COMMAND=npx
SHOPIFY_MCP_ARGS=-y shopify-mcp
SHOPIFY_SHOP_DOMAIN=your-store.myshopify.com
SHOPIFY_STOREFRONT_ACCESS_TOKEN=your-token   # required for private stores
TAVILY_API_KEY=your-tavily-key               # optional, enables fallback
```

The sidebar shows which of these are currently configured, so a search
that comes back empty is self-explanatory rather than a silent failure.

### What the UI handles

- **`selected`** — a high-confidence match (score ≥ 0.80): shown as a product
  card, with any other close matches in an expander.
- **`needs_selection`** — a mid-confidence match (0.50–0.80): the UI lists
  candidates and requires the user to explicitly confirm one, matching the
  documented confidence-tier policy.
- **`clarify`** — low-confidence or no strong match: shown as a warning, with
  any weak matches available to inspect.
- **`failed`** — both providers errored or returned nothing usable: shown as
  an error with the underlying message.

### Current UI scope

- No LangGraph wrapping yet — each search is a single synchronous
  `discover()` call.
- No SQLite persistence yet — state lives in `st.session_state` for the
  current browser session only; a page refresh loses the last result.
- No Gemini natural-language parsing yet — the form fields (query, max
  price, currency, merchant) map directly onto `UserIntent`. Wiring
  `GeminiPlanner` in as an optional "parse this for me" step is a natural
  follow-up once SQLite persistence lands.
## Current boundary

The repository now contains the planner, the deterministic Shopify MCP
executor foundation, and a Streamlit UI over `DiscoveryAgent`. The executor
can run a structured catalog request without Gemini, or consume a validated
planner output later. SQLite persistence, LangGraph wrapping, checkout,
cryptographic mandates, and payment remain later phases.
