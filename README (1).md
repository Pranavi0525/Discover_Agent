# Discovery Agent: Trust-Based Shopping Discovery Agent

A natural-language product discovery agent for Shopify stores. A user types a request such as *"men cotton casual shirt under 1500 INR"*; a hosted **Gemini Flash** planner turns it into a bounded plan, and a **deterministic LangGraph executor** carries the plan out against a real Shopify catalog through the Model Context Protocol (MCP). Results are filtered, ranked and shown in a Streamlit UI.

The design principle: **the LLM plans, Python enforces.** Every limit, filter and safety check lives in deterministic code that the model cannot override.

The agent **finds products only**. It does not create carts or buy anything; the user follows a "View on store" link.

| Phase | Scope | Status |
|---|---|---|
| 1 | Discovery prototype (planner, executor, Shopify, filtering, UI) | Done |
| 2 | Attack mitigation and security guardrails | Done |
| 3 | Signed mandates, autonomous commerce | Planned (not implemented) |
| 4 | Multi-merchant federation, fallbacks, production scale | Planned |

## Contents
[How it works](#how-it-works) · [Features](#features) · [Tech stack](#tech-stack) · [Getting started](#getting-started) · [Configuration](#configuration) · [Usage](#usage) · [How to verify](#how-to-verify) · [Safety limits](#safety-limits) · [Security model](#security-model) · [Known limitations](#known-limitations) · [Roadmap](#roadmap) · [Team](#team)

## How it works

```
User query
   |
   v
Input guard (rules + optional Prompt Guard classifier) --block--> recorded as a step event
   |
   v
Dedup + rate limiting (per session, 120 s window)
   |
   v
Gemini Flash planner (structured output -> ExecutionPlan)
   |
   v
Pydantic validation (plan checked BEFORE any tool call)
   |
   v
LangGraph executor: search_catalog -> get_product (parallel), bounded retries/timeouts
   |
   v
Hard filters -> MiniLM ranking -> decision tiers
   |
   v
Streamlit UI + SQLite state (step statuses, partial results, failures, run IDs)
```

### Hard filters

Applied in code, in a fixed order, to the offer a product is judged on: availability, currency, budget and merchant.

- A product is judged on **one** available variant priced in the requested currency. Price, seller and URL come from that same variant. There is no currency conversion.
- Merchant identity is proven by the **URL host**, not by a seller name alone.
- If `get_product` fails, the product goes to a separate **unverified** lane. It is labelled, not checked against the budget, ranked after verified products and never auto-selected.

### Decision tiers

| Score | Outcome |
|---|---|
| >= 0.80 | Strong match (auto-selected only if the product is verified and any requested size is available) |
| 0.50 to 0.80 | Top candidates shown for the user to choose (`needs_selection`) |
| < 0.50 | Query refinement or a clarification prompt |

Automatic query reformulation is capped at 2 iterations, and the budget is never silently widened. The 0.80 / 0.50 thresholds come from the project roadmap and have **not** been tuned on a labelled set.

## Features

### Phase 1: Discovery prototype
- **Planner/executor split.** Gemini Flash (Google AI Studio free tier) produces an `ExecutionPlan`; LangGraph runs it with state, conditional routing, bounded loops, retries and parallel detail fetches.
- **Shopify integration** via the MCP tools `search_catalog` and `get_product` (Shopify UCP).
- **Strict schemas.** Pydantic v2 models for `Product`, `ProductVariant`, `UserIntent` and `ExecutionPlan`.
- **Ranking.** MiniLM (`all-MiniLM-L6-v2`) when the `semantic` extra is installed; otherwise an explicit lexical fallback. The UI shows a percentage only for MiniLM and labels the fallback "keyword match".
- **Live queries**, not pre-indexing, because stock and prices change constantly.
- **Persistence and tracing.** SQLite stores sessions, step events and feedback; each run records a per-attempt call ledger and a per-product trail explaining why each product was kept or dropped.
- **Streamlit UI** with search, budget and preference inputs, a product grid, step statuses, failures and run IDs.

### Phase 2: Security guardrails
- **Prompt-injection gatekeeper.** A fast rule tier plus an optional Llama Prompt Guard 2 classifier in front of the planner; fails closed.
- **Adversarial harness** reporting rule-tier recall, paraphrase-tier recall and false-positive rate.
- **Idempotency and rate limiting.** An identical request (same session, normalised query and form values) inside the 120 s sliding window returns the existing run; in-flight duplicates are caught; per-session and global rate limits apply.
- **PII and secret handling.** Phone numbers, card numbers, street addresses, SSN/passport/date of birth (cue-based) and API keys are masked before the planner and in stored/logged text. The delivery city is kept because search needs it; in the run ledger it is replaced by a keyed hash (set `PII_HASH_SALT`).
- **Bounded fetching.** MCP responses and fetched policy pages are size-capped.
- **Executor limits** enforced in Python (see [Safety limits](#safety-limits)).
- **Plan-validation decision.** `validator_benchmark` compares Pydantic validation of planner output with an optional classifier. On its bundled case set, deterministic validation caught 12/12 malformed or hostile plans and rejected 0/3 valid ones. That is a small sample, and the classifier comparison is optional.

## Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.11+ |
| Planner LLM | Gemini Flash via Google AI Studio |
| Orchestration | LangChain and LangGraph |
| Commerce backend | Shopify UCP over MCP |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, optional extra) |
| Injection classifier | Llama Prompt Guard 2 (optional extra) |
| Validation | Pydantic v2 |
| State | SQLite |
| UI | Streamlit |
| Tests | pytest |

## Getting started

Prerequisites: Python 3.11+, a Google AI Studio key (`GOOGLE_API_KEY`), and, for the classifier, a Hugging Face account with the Prompt Guard license accepted.

```bash
git clone https://github.com/Pranavi0525/Discover_Agent.git
cd Discover_Agent
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,semantic,guard,ui]"
```

| Extra | Adds |
|---|---|
| `dev` | pytest |
| `semantic` | sentence-transformers (enables MiniLM; otherwise lexical fallback) |
| `guard` | transformers and torch (Prompt Guard classifier only) |
| `ui` | Streamlit |

Prompt Guard models are gated: accept the license on the model page, then run `huggingface-cli login`.

## Configuration

Create `.env` in the project root (never commit it):

```env
GOOGLE_API_KEY=your-google-ai-studio-key
PII_HASH_SALT=a-long-random-secret
GUARD_CLASSIFIER=promptguard-86m   # optional; omit for rules only
```

## Usage

```bash
streamlit run src/discovery_agent/ui/streamlit_app.py
python -m discovery_agent.live_check "men cotton casual shirt under 1500 INR"
```

## How to verify

Run from the project root with the virtual environment active. Steps 1-3 and 6 are offline. Step 5 needs `GOOGLE_API_KEY`.

1. **Tests:** `python -m pytest -q`. All tests should pass; use the count your own run prints.
2. **Planner eval, replaying cached Gemini responses:** `python -m discovery_agent.eval --mode replay`. This needs `eval_cache/gemini_responses.json`, which must be committed (it is currently listed in `.gitignore`). Without it, every case shows as pending and the result is `INCOMPLETE`.
3. **Guard harness, rules only:** `python -m discovery_agent.eval.adversarial`. On the bundled cases the rule tier gives recall 1.0, paraphrase-tier recall 0.5 and a false-positive rate of 0.0. The paraphrase figure measures generalisation to wording the rules were not written for.
4. **Guard with the classifier:** `python -m discovery_agent.eval.adversarial --classifier promptguard-86m`. Record the numbers from your own run here.
5. **Live check** (uses API quota): `python -m discovery_agent.live_check "men cotton casual shirt under 1500 INR"`. Expect real INR-priced products, none over budget, and the scorer name `minilm`. Results vary with the live catalog.
6. **UI:** `streamlit run src/discovery_agent/ui/streamlit_app.py`.

## Safety limits

Defaults below live in `config.py`. An operator can override them through environment variables; the planner cannot.

| Setting | Default |
|---|---|
| `MAX_PLAN_STEPS` | 8 |
| `MAX_PRODUCT_DETAILS` | 10 |
| `MAX_TOOL_CALLS` | 30 |
| `MAX_RETRIES_PER_TOOL` | 2 |
| `MAX_GRAPH_ITERATIONS` | 3 |
| `TOOL_TIMEOUT_SECONDS` | 20 |
| `MAX_REFINEMENTS` | 2 |
| `MAX_PRODUCTS` | 10 |

## Security model

| Threat | Mitigation |
|---|---|
| Prompt injection, jailbreaks, prompt extraction | Rule tier plus optional classifier before the planner |
| Price or discount override | Prices come only from Shopify; filters run in Python, not in the LLM |
| Runaway loops, token abuse | Hard caps on steps, calls, retries, iterations, refinements and timeouts |
| Duplicate or burst requests | 120 s dedup window and rate limiting |
| Unsafe tool calls from the planner | `ExecutionPlan` validated by Pydantic before any MCP call |
| PII and secret leakage | Masking before the planner and in stored/logged text |

The guard is defense in depth, not a guarantee.

## Known limitations

- MiniLM scoring needs the `semantic` extra; otherwise ranking is lexical.
- Decision thresholds are untuned; there is no labelled ranking-quality evaluation yet.
- PII masking is pattern-based and misses cases such as a bare passport number or birth date with no cue word.
- Dedup and rate limiting use SQLite on one node, and the identity is an anonymous session ID, not an authenticated user.
- Merchant matching is strict by design: a brand whose domain differs from its name can be rejected.
- Prices and stock are a live snapshot and can change before the user reaches the store.
- The agent relies on Shopify's catalog endpoint and the Gemini free tier, so it inherits their limits.

## Roadmap

### Phase 3 (planned, not implemented)
ECDSA P-256 signing; User Intent, Merchant Cart and Payment mandates with single-use nonces; UCP/ACP alignment; Web-Bot-Auth headers; privacy-preserving bank verification.

### Phase 4 (planned)
Tavily fallback for non-MCP stores, intent-based merchant routing, concurrent multi-merchant fetching, LangSmith tracing, reviewed learning memory, multi-store carts.

### Remaining hardening ideas
Tokenised customer identity for merchant-facing tools; short-lived checkout tokens bound to merchant domain and session; a labelled ranking evaluation and threshold tuning.

## Team

Built by a student team. Contributors: Pranavi Devulapalli, Akanksha B, S N.

## License

Add a license (for example MIT) before making the repository public.
