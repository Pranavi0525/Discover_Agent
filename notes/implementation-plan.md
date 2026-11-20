# Discovery Agent Implementation Plan

## Approval status

**Approved by the project owner on September 17, 2026.** This document is the
implementation contract for the next build phase. Implementation must follow
this plan; changes to the flow or model choice require a new approval.

## Goal
Build a modular Python shopping discovery agent using a **planner/executor**
architecture. It accepts a shopping request, uses a hosted free-tier LLM to
extract intent and create a bounded plan, executes approved MCP/provider tools,
normalizes products, applies deterministic and semantic ranking, and records
every run for auditability. Shopify MCP is the primary discovery route across
many stores; Tavily is the fallback for unsupported merchants.

## Approved model decision

The project will not host a model locally and will not run Ollama, vLLM, or a
self-managed inference server. The initial model provider is **Google Gemini
Flash through Google AI Studio's free API tier**, subject to Google's current
quota, model availability, and account terms. The exact supported Flash model
identifier will be configured in an environment variable and verified during
implementation rather than hard-coded in business logic.

The model is used for natural-language interpretation, plan generation,
clarification questions, and result summarization. It is never the authority
for price, stock, tool permissions, step limits, retries, or security policy.

### Hosted free-model setup process

1. Create a Google AI Studio API key using the project's approved account.
2. Store it locally as `GOOGLE_API_KEY`; never commit it or place it in run logs.
3. Configure the model name, for example `GEMINI_MODEL`, from the currently
  available Gemini Flash free-tier model.
4. Install the Google Gemini LangChain integration when implementation starts:
  `langchain-google-genai`.
5. Send only the minimum user request and safe catalog context to the model.
6. Validate every structured response with Pydantic before it reaches the
  executor.
7. Track model request status, latency, and token metadata where available,
  but never store the API key or unnecessary personal data.

The application must fail clearly when the key is missing. Tests must use
recorded planner responses and fake MCP responses, so the test suite does not
consume the free quota.

## Constraints and decisions from the docs
- Use Python with Pydantic v2 for contracts and LangGraph state boundaries.
- Use LangChain for Gemini integration, structured output, and MCP tool wrappers.
- Use LangGraph for planner/executor workflow state, conditional routing,
  bounded loops, retries, parallel work, and resumable execution.
- Keep provider integrations behind a common discovery port so Akanksha's Shopify or Tavily work can be plugged in without changing ranking or UI code.
- Prefer live discovery over a pre-indexed catalog because stock and pricing change.
- Enforce `max_price` and availability before ranking.
- Use the documented confidence tiers: `>= 0.80` auto-select, `0.50-<0.80` ask the user to choose, `< 0.50` ask for clarification.
- Cap automated refinement at two attempts and enforce executor limits on every
  plan, regardless of what the LLM requests.
- Keep Tavily extraction narrow: title, price, currency, availability, merchant, image, and direct URL; never pass raw pages through the agent.
- Do not implement checkout, payment credentials, cart creation,
  cryptographic mandates, or autonomous purchase authorization in this first
  discovery slice. Those are later phases.

## Approved planner/executor flow

```text
Streamlit request
  -> planner LLM: UserIntent + bounded ExecutionPlan
  -> Pydantic plan validation and deterministic guardrails
  -> executor: search_catalog through Shopify MCP
  -> executor: parallel get_product calls for selected product IDs
  -> normalize and validate Product records
  -> deterministic stock/budget/currency filters
  -> semantic ranking
  -> decision: select, ask user to choose, or clarify
  -> persist step states, conversation, errors, and learning feedback
```

The planner cannot call MCP tools directly. The executor owns tool calls and
can call an MCP server without an LLM. The executor must use an allowlisted
tool registry, validate arguments, deduplicate product IDs, and continue with
successful product-detail calls when another call fails or times out.

### Execution limits

Initial defaults, configurable by environment:

| Limit | Default |
|---|---:|
| Maximum plan steps | 8 |
| Maximum tool calls per run | 30 |
| Maximum product details to fetch | 10 |
| Maximum retries per tool | 2 |
| Maximum graph iterations | 3 |
| Tool timeout | 20 seconds |
| Automatic query refinements | 2 |

Repeated identical failures, unknown tools, invalid arguments, exceeded limits,
and oversized results terminate the relevant step safely. No LLM response can
raise these limits.

## Step state and persistence decision

Yes, every meaningful step needs state. Parallel calls, retries, timeouts,
partial success, user selection, and future resume/recovery all require an
explicit state record.

Each step records `run_id`, `step_id`, `step_type`, `status`, sanitized input
summary, sanitized output summary, timestamps, retry count, and error category.
Allowed statuses are `PLANNED`, `RUNNING`, `SUCCEEDED`, `RETRYING`, `FAILED`,
`TIMED_OUT`, `SKIPPED`, and `WAITING_FOR_USER`.

SQLite is approved for the initial persistence layer. It stores user sessions,
conversation messages, LangGraph checkpoints, step events, tool-call metadata,
and reviewed mistake/correction records. PostgreSQL is a later deployment
option; it is not required for the prototype.

## Learning from mistakes

The first learning mechanism is reviewed memory, not automatic model training.
For each confirmed mistake, store the original request, plan, failed action,
correction, final outcome, provider, and review status. Before planning a new
request, retrieve relevant reviewed corrections and present them as bounded
warnings to the planner. Unreviewed failures must not become instructions.

This creates an auditable improvement loop:

```text
run -> failure or user correction -> reviewed mistake record
     -> retrieve similar reviewed records -> planner warning -> guarded run
```

## Target module boundaries
```text
src/discovery_agent/
  config.py             # environment-backed settings
  schemas.py             # UserIntent, Product, SearchResult, run/audit contracts
  ports.py               # DiscoveryProvider protocol and RunTracker protocol
  providers/
    tavily.py           # Tavily search adapter and compact result normalization
    shopify.py          # Shopify adapter seam; MCP/GraphQL implementation can be added here
  ranking.py            # deterministic filtering and similarity scoring
  planning.py           # Gemini planner and structured ExecutionPlan
  graph.py              # LangGraph planner/executor state machine
  execution.py          # guardrails, MCP calls, retries, and parallel details
  persistence.py         # SQLite sessions, checkpoints, events, and feedback
  ui/streamlit_app.py   # modular Streamlit interface
  tracking.py           # audit events and safe run summaries
```

## Delivery steps
1. **Foundation and dependencies**: add LangChain, LangGraph, Gemini, MCP,
  SQLite, and Streamlit dependencies; document environment variables.
2. **Contracts and state**: add `ExecutionPlan`, `PlanStep`, `AgentState`,
  `StepState`, and strict Pydantic validation.
3. **Planner**: call hosted Gemini Flash and produce only structured plans;
  reject malformed or unsafe plans.
4. **Executor**: implement allowlisted MCP calls, limits, timeouts, retries,
  loop detection, and parallel `get_product` execution after catalog search.
5. **Providers**: connect real Shopify MCP for discovery and retain Tavily as
  a fallback adapter for non-Shopify merchants.
6. **Ranking and decisions**: add the approved deterministic filters and
  embedding scorer, then expose selection/clarification decisions.
7. **Persistence and learning**: add SQLite sessions, step events,
  checkpoints, and reviewed mistake records.
8. **Streamlit**: keep the UI modular and show progress, partial results,
  user selection, clarification, and run audit information.
9. **Tests and evaluation**: test planner parsing, guardrails, parallel calls,
  MCP failures, persistence, recorded model responses, and end-to-end fake MCP.

## Definition of done
- `pytest` passes without network access or provider credentials.
- Importing the package does not call Gemini, Tavily, or Shopify.
- The planner produces a validated bounded plan and the executor alone calls MCP.
- Search results trigger parallel product-detail calls with partial-failure handling.
- A discovery run produces a traceable run ID and persistent step states.
- A provider can be replaced by implementing the provider/MCP adapter contract.
- Tavily output is compact and normalized; raw HTML is not part of the internal contract.
- Missing Tavily credentials produce a clear configuration error only when the Tavily provider is used.
- The README explains setup, test commands, provider wiring, and the current Phase 1 boundary.

## Follow-up phases
- Phase 2: prompt-injection gatekeeper, idempotency/rate limiting, PII minimization, and adversarial evaluation harness.
- Phase 3: signed UIM/MCM reconciliation, nonce replay protection, and payment-agent boundaries.
- Phase 4: multi-merchant federation, LangSmith tracing, reviewed learning memory, and production deployment.

## Suggested handoff checklist
- [ ] Point `ShopifyProvider` at Akanksha's MCP or Storefront GraphQL callable.
- [ ] Confirm the callable returns product dictionaries that map to `Product`.
- [ ] Add a Tavily API key only in local environment configuration, never in source control.
- [ ] Run the offline test suite before enabling live providers.
- [ ] Review the JSONL run ledger for secret or PII leakage before sharing traces.
- [x] Add direct no-LLM execution for structured catalog discovery.
