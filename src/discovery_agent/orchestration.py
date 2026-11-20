from collections import Counter
from uuid import UUID, uuid4

from .currency import format_money
from .ports import DiscoveryProvider, RunTracker
from .query_refinement import (
    STRATEGY_CORE,
    STRATEGY_SYNONYM,
    QueryStep,
    build_query_ladder,
    refinement_note,
)
from .ranking import filter_and_rank_with_stats, unverified_from_stats
from .schemas import DiscoveryDecision, ScoredProduct, UnverifiedCandidate, UserIntent
from .semantic import RelevanceScorer, RunScorer, get_default_scorer

# Confidence tiers approved in notes/implementation-plan.md and docs/roadmap.md
# (M1.5): >=0.80 auto-select, 0.50-<0.80 ask the user to choose, <0.50 clarify.
AUTO_SELECT_THRESHOLD = 0.80
NEEDS_SELECTION_THRESHOLD = 0.50


def _selected_message(scorer_name: str) -> str:
    """Only a real embedding scorer may be described as embedding-based."""
    if scorer_name == "minilm":
        return "Best match selected (embedding-based similarity score)."
    return "Best keyword match selected (a keyword-overlap score, not semantic confidence)."


def decide(
    provider: str,
    ranked: list[ScoredProduct],
    intent: UserIntent,
    scorer_name: str = "lexical",
) -> DiscoveryDecision:
    """Turn ranked candidates into a bounded decision.

    Shared by :class:`DiscoveryAgent` (provider-based flow) and the
    LangGraph planner/executor workflow in ``graph.py`` so the confidence
    tiers only live in one place.  ``scorer_name`` decides how a selection is
    worded; a requested size that no candidate offers is never auto-selected.
    """
    # Tier by RELEVANCE: `ranked` is ordered size-first, so ranked[0] is not
    # necessarily the highest-scoring product.
    best = max(ranked, key=lambda item: item.score)
    shown = ranked[:3] if best in ranked[:3] else [best, *ranked[:2]]
    ranked = shown + [r for r in ranked if r not in shown]
    if best.score >= AUTO_SELECT_THRESHOLD:
        if best.size_status == "unavailable":
            return DiscoveryDecision(
                status="needs_selection", provider=provider, candidates=ranked[:3],
                message=(
                    f"These products match your search, but size {intent.size} was not "
                    "found among their available sizes. Choose one to continue or "
                    "change the size."
                ),
            )
        return DiscoveryDecision(
            status="selected", provider=provider, candidates=ranked[:3], selected=best,
            message=_selected_message(scorer_name),
        )
    if best.score >= NEEDS_SELECTION_THRESHOLD:
        message = (
            "Several products may match. Choose one to continue."
            if len(ranked) > 1
            else "One possible match was found (partial keyword overlap). "
                 "Check that it fits before continuing."
        )
        return DiscoveryDecision(
            status="needs_selection", provider=provider, candidates=ranked[:3],
            message=message,
        )
    return DiscoveryDecision(
        status="clarify", provider=provider, candidates=ranked[:3],
        message="The results are weak matches. Please add a category, brand, or style.",
    )


def unverified_note(candidates: list[UnverifiedCandidate]) -> str:
    """One sentence saying how many products are listed separately, and why."""
    if not candidates:
        return ""
    reasons = Counter(c.reason_text or c.unverified_reason for c in candidates)
    why = ", ".join(f"{n} {word}" for word, n in reasons.items())
    count = len(candidates)
    return (
        f"{count} more product{'s' if count != 1 else ''} matched your search but "
        f"{'their' if count != 1 else 'its'} details could not be loaded ({why}). "
        f"{'They are' if count != 1 else 'It is'} listed separately as unverified: the final "
        "offer, price and budget compliance were not checked."
    )


def attach_unverified(
    decision: DiscoveryDecision, candidates: list[UnverifiedCandidate]
) -> DiscoveryDecision:
    """Keep unverified candidates on the decision (never in ``candidates``)."""
    decision.unverified_candidates = list(candidates)
    note = unverified_note(decision.unverified_candidates)
    if note and decision.status != "failed":
        decision.message = f"{decision.message} {note}"
    return decision


def no_match_message(intent: UserIntent, attempt_log: list[dict]) -> str:
    """Truthful explanation of an empty result (never a changed budget)."""
    budget = (
        f" within {format_money(intent.max_price, intent.currency)}"
        if intent.max_price is not None
        else ""
    )
    searches = [entry for entry in attempt_log if entry.get("error") is None]
    stopped = [entry for entry in attempt_log if entry.get("stopped")]
    queries = ", ".join(f"\u201c{entry['query']}\u201d" for entry in searches)
    fetched = sum(entry.get("fetched", 0) for entry in searches)

    parts = [f"No matching products were found{budget}."]
    if searches:
        parts.append(f"Searched {len(searches)} time(s): {queries}.")
    if fetched == 0:
        parts.append("Shopify returned no products for these searches.")
    else:
        dropped: Counter[str] = Counter()
        currencies: Counter[str] = Counter()
        for entry in searches:
            dropped.update(entry.get("dropped", {}))
            currencies.update(entry.get("currencies_seen", {}))
        reasons = []
        if dropped["over_budget"]:
            reasons.append(f"{dropped['over_budget']} over your budget")
        if dropped["currency_mismatch"]:
            other = ", ".join(
                c for c in currencies if c not in (intent.currency, "unknown")
            ) or "another currency"
            reasons.append(
                f"{dropped['currency_mismatch']} priced in {other} instead of {intent.currency}"
            )
        if dropped["currency_unknown"]:
            reasons.append(
                f"{dropped['currency_unknown']} had no stated currency, so their price "
                "could not be verified"
            )
        if dropped["budget_currency_unknown"]:
            reasons.append(
                f"{dropped['budget_currency_unknown']} could not be checked against your "
                "budget because no currency was set"
            )
        if dropped["unavailable"]:
            reasons.append(f"{dropped['unavailable']} unavailable")
        if dropped["no_price"]:
            reasons.append(f"{dropped['no_price']} without a price")
        if dropped["irrelevant"]:
            reasons.append(f"{dropped['irrelevant']} shared no word with your search")
        if dropped["other_merchant"]:
            reasons.append(f"{dropped['other_merchant']} from another or unidentified merchant")
        if dropped["merchant_unverified"]:
            reasons.append(
                f"{dropped['merchant_unverified']} could not be verified as sold by "
                f"{intent.merchant or 'the requested merchant'} (only the seller's own "
                "name matched)"
            )
        if dropped["availability_unknown"]:
            reasons.append(f"{dropped['availability_unknown']} had no confirmed availability")
        parts.append(
            f"Shopify returned {fetched} product(s), but none qualified"
            + (f" ({'; '.join(reasons)})." if reasons else ".")
        )
    if stopped:
        parts.append(f"Refinement stopped early: {stopped[0]['error']}.")
    if intent.max_price is not None:
        parts.append("Your budget was not changed.")
    parts.append("Please clarify your request.")
    return " ".join(parts)


class DiscoveryAgent:
    """Provider orchestration with bounded, budget-preserving refinement.

    The first provider gets ``1 + max_refinements`` searches (capped by
    ``max_graph_iterations``).  Each refinement changes only the query text;
    max_price / currency / merchant are carried over unchanged, and ranking is
    always scored against the ORIGINAL user query.

    ``scorer`` is optional: tests inject a fake, production uses MiniLM when it
    is installed and an explicit lexical fallback otherwise.  One scorer is
    pinned for the whole run so keyword and embedding scores never mix.
    """

    def __init__(self, primary: DiscoveryProvider, fallback: DiscoveryProvider | None = None,
                 tracker: RunTracker | None = None, max_refinements: int = 2,
                 max_graph_iterations: int = 3,
                 scorer: RelevanceScorer | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.tracker = tracker
        self.max_refinements = max_refinements
        self.max_graph_iterations = max_graph_iterations
        self.scorer = scorer

    def _run_scorer(self) -> RunScorer:
        base = self.scorer or get_default_scorer()
        return base if isinstance(base, RunScorer) else RunScorer(base)

    def discover(self, intent: UserIntent) -> DiscoveryDecision:
        run_id = uuid4()
        run_scorer = self._run_scorer()
        providers = [self.primary] + ([self.fallback] if self.fallback else [])
        errors: list[str] = []
        attempt_log: list[dict] = []
        attempts = 0
        last_fetched = 0
        last_relevance: dict | None = None
        last_unverified: list[UnverifiedCandidate] = []
        self._record(run_id, "discovery_started", provider_count=len(providers))
        # Python-enforced ceiling: 1 search + MAX_REFINEMENTS, never above MAX_GRAPH_ITERATIONS.
        primary_searches = max(1, min(1 + self.max_refinements, self.max_graph_iterations))

        for index, provider in enumerate(providers):
            begin_run = getattr(provider, "begin_run", None)
            if callable(begin_run):
                begin_run()
            # The query ladder (specific, core, broaden, synonym) decides which
            # catalog queries are tried; the budget and every other constraint
            # stay on the intent unchanged.
            if index == 0:
                rungs = build_query_ladder(intent, self.max_refinements)[:primary_searches]
            else:
                rungs = [QueryStep(intent.query, STRATEGY_CORE)]
            for position, step in enumerate(rungs):
                current = intent.model_copy(
                    update={"query": step.query, "refinement_count": min(position, 2)}
                )
                attempts += 1
                self._record(
                    run_id,
                    "provider_search_started",
                    provider=provider.name,
                    attempt=attempts,
                    query=current.query,
                    refinement_count=current.refinement_count,
                )
                try:
                    result = provider.search(current)
                except Exception as exc:
                    errors.append(f"{provider.name}: {exc}")
                    self._record(run_id, "provider_failed", provider=provider.name,
                                 attempt=attempts, error=str(exc))
                    attempt_log.append({
                        "attempt": attempts, "provider": provider.name, "query": current.query,
                        "error": str(exc)[:300],
                    })
                    break

                self._record(
                    run_id,
                    "provider_search_completed",
                    provider=provider.name,
                    attempt=attempts,
                    raw_product_count=len(result.products),
                    provider_error=result.error,
                )

                if result.error and not result.products:
                    # The provider declined to search (e.g. the per-run MAX_TOOL_CALLS
                    # budget is spent).  That is a stop, not a search: it is logged as
                    # such, is not counted in ``attempts`` and is never reported as
                    # "searched" in the no-result message.
                    attempt_log.append({
                        "attempt": attempts, "provider": provider.name, "query": current.query,
                        "refinement": current.refinement_count, "fetched": 0,
                        "error": result.error, "stopped": True, "shopify": result.diagnostics,
                    })
                    attempts -= 1
                    break

                # Scored against the ORIGINAL query (``intent``), never the broadened
                # one; a synonym rung adds its rewrite as a second query.
                extra = [current.query] if step.strategy == STRATEGY_SYNONYM else None
                ranked, stats = filter_and_rank_with_stats(
                    intent, result.products, scorer=run_scorer, extra_queries=extra
                )
                relevance = stats["relevance"]
                last_relevance = relevance
                last_unverified = unverified_from_stats(stats)
                last_fetched = len(result.products)
                self._record(
                    run_id,
                    "filter_and_rank_completed",
                    provider=provider.name,
                    attempt=attempts,
                    raw_product_count=len(result.products),
                    ranked_product_count=len(ranked),
                    top_scores=[item.score for item in ranked[:3]],
                    filter_stats=stats,
                    scorer=relevance["scorer"],
                    relevance=relevance,
                )
                attempt_log.append({
                    "attempt": attempts,
                    "provider": provider.name,
                    "query": current.query,
                    "refinement": current.refinement_count,
                    "max_price": str(intent.max_price) if intent.max_price is not None else None,
                    "currency": intent.currency,
                    "fetched": last_fetched,
                    "kept": stats["kept"],
                    "unverified": stats.get("unverified", 0),
                    "dropped": stats["dropped"],
                    "currencies_seen": stats["currencies_seen"],
                    "products": stats.get("products", []),
                    "relevance": relevance,
                    "error": None,
                    "shopify": result.diagnostics,
                })
                if ranked:
                    decision = self._decision(provider.name, ranked, intent, run_scorer.name)
                    if position > 0 and current.query != intent.query:
                        decision.message += refinement_note(intent.query, step)
                    decision.attempts = attempts
                    decision.diagnostics = {"attempt_log": attempt_log, "relevance": relevance}
                    decision.run_id = run_id
                    attach_unverified(decision, last_unverified)
                    self._record(
                        run_id,
                        "decision_created",
                        provider=provider.name,
                        status=decision.status,
                        selected_product_id=decision.selected.product.id if decision.selected else None,
                        candidate_count=len(decision.candidates),
                    )
                    self._track(decision, last_fetched, current)
                    return decision
                if result.error:
                    break  # e.g. MAX_TOOL_CALLS exhausted: do not keep searching

        diagnostics: dict = {"attempt_log": attempt_log}
        if last_relevance is not None:
            diagnostics["relevance"] = last_relevance
        decision = DiscoveryDecision(
            run_id=run_id,
            status="failed" if errors else "clarify",
            provider=providers[-1].name,
            message="; ".join(errors) if errors else no_match_message(intent, attempt_log),
            attempts=max(1, attempts),
            diagnostics=diagnostics,
        )
        attach_unverified(decision, last_unverified)
        self._record(run_id, "decision_created", provider=decision.provider, status=decision.status,
                 selected_product_id=None, candidate_count=0)
        self._track(decision, last_fetched, intent)
        return decision

    def _decision(self, provider: str, ranked: list[ScoredProduct], intent: UserIntent,
                  scorer_name: str = "lexical") -> DiscoveryDecision:
        return decide(provider, ranked, intent, scorer_name)

    def _track(self, decision: DiscoveryDecision, raw_count: int, searched: UserIntent | None = None) -> None:
        if self.tracker:
            self._record(
                decision.run_id,
                "discovery_completed",
                status=decision.status,
                provider=decision.provider,
                raw_candidate_count=raw_count,
                ranked_candidate_count=len(decision.candidates),
                selected_product_id=decision.selected.product.id if decision.selected else None,
                attempts=decision.attempts,
                final_query=searched.query if searched else None,
                refinement_count=searched.refinement_count if searched else 0,
                attempt_log=decision.diagnostics.get("attempt_log", []),
                relevance=decision.diagnostics.get("relevance"),
            )

    def _record(self, run_id: UUID, event: str, **details: object) -> None:
        if self.tracker:
            self.tracker.record({"run_id": str(run_id), "event": event, **details})
