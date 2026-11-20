# Skills and Capabilities Needed

This is the working skill matrix for the Discovery Agent. It separates capabilities that belong in code from human or operational skills needed to integrate the project safely.

## Code capabilities

- **Contract design**: Pydantic v2 models for intents, products, search results, decisions, and audit events.
- **Planner design**: produce a structured `ExecutionPlan` with bounded steps; never allow the model to execute tools directly.
- **Executor design**: validate plans, allowlist MCP tools, enforce limits, handle retries/timeouts, and run independent product calls in parallel.
- **Provider adapters**: implement the `DiscoveryProvider` protocol for Shopify MCP/GraphQL, Tavily, and future merchants.
- **Normalization**: reduce every external response to the compact `Product` contract; discard raw HTML and unknown provider payloads.
- **Routing**: prefer Shopify, fall back to Tavily, and keep provider failures isolated from the orchestration layer.
- **Constraint filtering**: enforce availability, currency, merchant, and max-price rules before ranking.
- **Semantic ranking**: use `all-MiniLM-L6-v2` in the approved implementation; keep deterministic filters ahead of semantic scoring.
- **Decision policy**: implement the documented confidence tiers and bounded refinement attempts.
- **Run tracking**: emit an append-only run ledger with run IDs, provider attempts, counts, decisions, and error summaries without secrets or PII.
- **Testability**: use injected callables and fake providers so tests never require network access or credentials.
- **State modeling**: represent plan and tool-step statuses explicitly so partial success and recovery are observable.
- **Persistence**: use SQLite for sessions, conversations, checkpoints, step events, and reviewed feedback.

## Integration skills

- **Hosted Gemini**: create a Google AI Studio key, set `GOOGLE_API_KEY` and `GEMINI_MODEL`, respect free-tier quotas, and never log credentials.
- **LangChain**: use `langchain-google-genai` for the hosted model, structured output, and MCP integration helpers.
- **LangGraph**: model the planner/executor graph, conditional branches, bounded loops, parallel detail calls, and resume points.
- **Shopify MCP / Storefront GraphQL**: map Akanksha's existing tool response into the provider callable shape documented in the README.
- **Tavily API**: configure `TAVILY_API_KEY`, tune search depth and result count, and preserve compact extraction to control token usage.
- **LangGraph integration**: wrap `DiscoveryAgent.discover` as a graph node after the Phase 1 behavior is stable.
- **Streamlit integration**: render `DiscoveryDecision.candidates`, expose explicit user selection for ambiguous matches, and show the run ID in the audit panel.
- **Evaluation and observability**: measure provider latency, candidate counts, score distributions, fallback rate, and clarification rate before adding autonomous actions.
- **Reviewed learning**: store confirmed mistakes and corrections, retrieve only reviewed records, and present them as bounded planner warnings rather than automatic retraining.

## Security and production skills for later phases

- Prompt-injection detection and adversarial test design.
- Idempotency keys, rate limiting, and replay-safe state transitions.
- PII minimization and secret management.
- ECDSA P-256 signing and UIM/MCM reconciliation.
- Multi-merchant cart isolation and payment-agent boundaries.

## Integration checklist

- [ ] Confirm the Shopify callable accepts `UserIntent` and returns mappings with `title` and `url`.
- [ ] Confirm `search_catalog` returns stable product identifiers for parallel `get_product` calls.
- [ ] Confirm the executor can continue when one detail call fails or times out.
- [ ] Confirm planner output cannot select tools outside the allowlist or exceed limits.
- [ ] Add a fake Shopify response test before enabling live MCP transport.
- [ ] Configure Tavily only through environment variables or a secret manager.
- [ ] Confirm no run ledger record contains API keys, raw HTML, shipping data, or billing data.
- [ ] Add LangGraph and Streamlit only after the provider and ranking contracts are stable.
