System Specification: Trust-Based Autonomous Shopping Agent (Shopify MCP)
1. Executive Summary & Project Purpose
This specification outlines the architecture, data contracts, security mandates, and implementation blueprint for building an autonomous, trust-based e-commerce shopping agent.

The system enables users to express open-ended shopping intent (e.g., "Find black running shorts under $40") and autonomously executes product discovery, semantic filtering, cart creation, and cryptographically verified checkout mandates via the Model Context Protocol (MCP) using the Shopify MCP Server and Shopify Storefront GraphQL API.

The project follows a phased delivery model:
- Phase 1 (Proof of Concept / Prototype): Autonomous Discovery Agent using LangGraph, Streamlit UI, Shopify MCP & GraphQL Storefront API, vector similarity matching (all-MiniLM-L6-v2), and structured Pydantic schemas.
- Phase 2 (Security & Verification): Cryptographic mandates (ECDSA P-256), mandate reconciliation (Intent Mandate vs. Cart Mandate), single-use nonces for replay protection, request idempotency/rate limiting, and prompt injection gatekeeper guardrails.
- Phase 3 (Full Autonomous Commerce & Fallbacks): Level 3 autonomous transactions, bank mandate authorization, and hybrid web browsing fallback using Tavily for unsupported merchants.



2. High-Level System Architecture & Flow
The system comprises 6 sequential layers:
1. User / Streamlit FrontEnd (Receives natural language prompt)
2. Planner LLM (Hosted Gemini Flash creates a structured, bounded execution plan)
3. Guardrail Validator (Pydantic plan validation, allowlisted tools, step and loop limits)
4. Executor State Machine (LangGraph controller owns MCP calls, retries, parallelism, and step state)
5. Merchant Layer (Primary: Shopify MCP Server / Storefront GraphQL; Fallback: Tavily Scraper Agent)
6. Product Filtering & Matching (Deterministic budget/stock validation, all-MiniLM-L6-v2 semantic embeddings, similarity thresholding)
7. Cryptographic Mandate Core (Future phase: User Intent Mandates, Merchant Cart Mandates, reconciliation, ECDSA P-256 signatures)

### Approved Phase 1 flow

```text
Streamlit
   -> hosted Gemini Flash planner
   -> validated ExecutionPlan
   -> LangGraph executor
   -> Shopify MCP search_catalog
   -> parallel Shopify MCP get_product calls
   -> normalized Product records
   -> deterministic filters and semantic ranking
   -> Streamlit selection or clarification
   -> SQLite session, step states, and audit events
```

The MCP server can be called without an LLM. The executor is a normal Python
component and calls MCP tools directly after the planner has produced and the
guardrail layer has approved a plan. The LLM must never be the authority for
tool permissions, budgets, stock, retries, or execution limits.



3. Core Architectural Principles & Decisions
3.1 Three-Level Agent Commerce Framework
- Level 1 (Basic Handshake): Simple URL generation and basic redirection to web storefronts.
- Level 2 (Personalized Context): Contextual session data exchange using JSON-LD and JWTs.
- Level 3 (Autonomous API-Driven Commerce - Target): Full machine-to-machine mandate exchange where autonomous agents execute discovery, cart reservation, and tokenized authorization without exposed human credentials.

3.2 Protocol Standardization
- Adherence to Universal Commerce Protocol (UCP) for catalog and merchant discovery.
- Adherence to Agentic Commerce Protocol (ACP) for cart construction, pricing verification, and checkout fields.
- Trust identification headers: Outgoing agent HTTP/MCP requests attach Web-Bot-Auth headers containing verified agent public keys and cryptographic signatures.

3.3 Hybrid Discovery Strategy
1. Primary Route (Shopify Stores): Route catalog queries to the Shopify MCP Server or the Shopify Mock Shop GraphQL API for zero-latency, zero-scraping, direct structured JSON responses.
2. Fallback Route (Non-Shopify Stores): Route queries to Tavily Search API with HTML extraction, constrained to extract only product cards, titles, prices, and direct links to avoid 1KB+ HTML bloat.

3.4 Real-Time Querying vs. Pre-Indexing
Product discovery queries Shopify live in real time. E-commerce catalogs have fluctuating inventory and dynamic pricing; pre-indexing creates stale state that invalidates cart mandates.

3.5 Planner/Executor Decision
- The planner converts a user request into `UserIntent` and a bounded `ExecutionPlan`.
- The executor validates each step and calls only allowlisted tools.
- `search_catalog` runs first. Its product IDs are deduplicated and capped.
- `get_product` runs in parallel for the capped product IDs.
- Partial success is valid: failed or timed-out details are recorded and do not discard successful products.
- The executor stops on repeated failures, unknown tools, invalid arguments, or exceeded limits.

3.6 Hosted Model Decision
- The project does not host an inference model.
- The initial model is Gemini Flash through Google AI Studio's free API tier, subject to the provider's current quota and availability.
- `GOOGLE_API_KEY` and `GEMINI_MODEL` are environment-only settings.
- Tests use recorded model responses and fake MCP calls; tests do not consume hosted-model quota.

3.7 State and Persistence Decision
Every meaningful plan and tool step has a state record. Required statuses are
`PLANNED`, `RUNNING`, `SUCCEEDED`, `RETRYING`, `FAILED`, `TIMED_OUT`,
`SKIPPED`, and `WAITING_FOR_USER`. SQLite stores sessions, conversations,
LangGraph checkpoints, step events, tool-call metadata, and reviewed
mistake/correction records. PostgreSQL is a later production option.

3.8 Guardrail Defaults
- Maximum plan steps: 8
- Maximum MCP/tool calls per run: 30
- Maximum product details: 10
- Maximum retries per tool: 2
- Maximum graph iterations: 3
- Tool timeout: 20 seconds
- Automatic query refinements: 2

These limits are enforced in Python and cannot be increased by model output.

3.9 Reviewed Learning Loop
The agent does not automatically retrain itself. Reviewed mistake records
capture the request, plan, failure, correction, and final outcome. Similar
reviewed records may be supplied to a future planner as bounded warnings.
Unreviewed failures are never treated as instructions.



4. Technical Stack & Dependencies
- Orchestration & State: LangGraph (>=0.2.0), LangChain (>=0.3.0)
- Data Validation: Pydantic (v2)
- User Interface: Streamlit (>=1.38.0)
- Protocol Integration: Model Context Protocol (MCP) SDK, httpx
- Embeddings & Search: sentence-transformers (all-MiniLM-L6-v2)
- Vector Similarity: scikit-learn / numpy
- Cryptography: cryptography (hazmat) for ECDSA with NIST P-256 curve (secp256r1)
- Fallback Scraper: tavily-python
- LLM Inference: Hosted Gemini Flash through Google AI Studio's free API tier; no locally hosted model in the approved Phase 1 design
- LLM Integration: `langchain-google-genai`
- Workflow: LangGraph planner/executor state machine with LangChain model and MCP integrations
- Persistence: SQLite for sessions, conversation history, checkpoints, step events, and reviewed feedback



5. Data Contracts & Pydantic Schemas
Includes Pydantic schemas for ProductVariant, Product, UserIntent, CartItem, Cart, UserIntentMandate, and MerchantCartMandate.



6. Model Context Protocol (MCP) & Shopify Integration
- Shopify MCP Server Configuration: via @modelcontextprotocol/server-shopify with SHOPIFY_SHOP_DOMAIN and SHOPIFY_STOREFRONT_ACCESS_TOKEN.
- Mock Testing: Shopify Mock Shop GraphQL endpoint (https://mock.shop/api).
- Exposed MCP Tools: shopify_search_products, shopify_get_product_details, shopify_create_cart, shopify_add_to_cart.
- Storefront GraphQL queries for live inventory and variant prices.



7. Matching, Ranking & Decision Engine
1. Deterministic Pre-Filtering: Drop out-of-stock items, drop items exceeding budget, filter hard category mismatches.
2. Semantic Similarity Scoring: Cosine similarity between user query and product title/description using all-MiniLM-L6-v2.
3. Threshold Gatekeeping:
   - High Confidence (>= 0.80): Automatically select top candidate, create cart, proceed to mandate generation.
   - Ambiguous Match (0.50 <= S < 0.80): Present top 3 candidates on Streamlit for user selection.
   - Low Match (< 0.50): Trigger clarification prompt back to user.



8. Cryptographic Security, Mandates & Guardrails (Phase 2 Specification)
- Mandate Reconciliation: Strict check ensuring Cart.total_price <= Intent.max_price, Cart.merchant_id == Intent.allowed_merchant, and currency matches.
- Nonce Replay Protection: Single-use 128-bit cryptographic nonces with 24-hour expiration window.
- Idempotency & Rate Limiting: 2-minute deduplication window with composite SHA-256 hashes.
- Prompt Injection Gatekeeper: Lightweight classifier filtering adversarial prompts before execution.



9. Streamlit User Interface Specification
Query input bar with budget filters, responsive product gallery with images, prices, and match badges, Add to Cart buttons, and a collapsible mandate audit panel.



10. Phased Implementation Roadmap
- Phase 1: Prototype / Proof of Concept (Discovery, MCP/GraphQL, Embeddings, Streamlit UI)
- Phase 2: Trust & Cryptographic Mandates (ECDSA P-256, UIM/MCM reconciliation, Nonces, Idempotency)
- Phase 3: Autonomous Payments & Universal Fallback (Bank Agent, Tavily fallback scraper)



11. Historical Discussion Log & Running Notes
Meeting 1: September 13, 2026
- Participants: S N, Akanksha B, Pranavi D
- Core Topics: Cryptographic Mandates & Three-Agent Workflow
- Key Outcomes & Decisions: Established 3-agent topology (Shopping, Merchant, Bank); UIM -> MCM -> PM sequence; ECDSA P-256 keys; Universal Commerce Protocol (UCP) & Agentic Commerce Protocol (ACP) alignment; gatekeeper prompt injection model.
- Open Questions: How bank agents verify transactions without line-item basket details; verifiable agent identity standards across platforms.
Meeting 2: September 14, 2026
- Participants: S N, Akanksha B, Pranavi D
- Core Topics: Agent Trust Levels, Shopify MCP Strategy & Privacy
- Key Outcomes & Decisions: 3-tier trust framework (Level 1, Level 2, Level 3 target); Shopify MCP server adopted over browser HTML parsing; LangGraph/LangChain/LangSmith selected; Streamlit chosen for PoC; Phase 1 prioritized.
- Open Questions: PII protection for shipping/billing addresses in schemata; token exposure and cardinality risks.
Meeting 3: September 15, 2026
- Participants: S N, Akanksha B, Pranavi D
- Core Topics: Cryptographic Tokens, Mandate Tallying & Idempotency
- Key Outcomes & Decisions: Mandate encryption mechanics; Shopping Agent tallies UIM against MCM; bank single-use Nonces prevent replay attacks; rate limiting & 2-minute request deduplication window (SHA-256 hash); Pydantic schemas and Mermaid diagram adopted.
- Open Questions: Strictness of mandate tallying for fuzzy intent (e.g. multi-item bundles); handling out-of-stock size/color variants.
Meeting 4: September 16, 2026
- Participants: S N, Akanksha B, Pranavi D
- Core Topics: Repository Setup, Real-Time Search & Matching Thresholds
- Key Outcomes & Decisions: Repository named Discovery Agent; real-time Storefront querying over static indexing; two-way filtering (budget/stock filters + all-MiniLM-L6-v2 vector embeddings); open-source models (Qwen, Mistral) on Hugging Face; parallel thread pools for product queries.
- Open Questions: Preventing infinite loops during iterative search refinement; open-source vs hosted inference trade-offs.
Meeting 5: September 17, 2026
- Participants: S N, Akanksha B, Pranavi D
- Core Topics: Tool Binding Blocker, Hybrid Browsing Strategy & UI Decision
- Key Outcomes & Decisions:
  - Hugging Face tool binding connection failures addressed by providing exposed API keys / alternative providers.
  - Hybrid browsing strategy adopted: primary Shopify MCP server + Tavily web scraper fallback.
  - DuckDuckGo dismissed due to slow speed and poor structured extraction.
  - Streamlit finalized over WhatsApp due to WhatsApp's inability to show multi-item image cards without a business merchant account.
  - Action Items: S N to connect Shopify MCP and optimize Tavily token usage; Pranavi to verify Storefront GraphQL endpoint; Akanksha to refactor codebase.
