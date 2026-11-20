"""Hard filters -> relevance floor -> scorer -> ranking (M1.4 / M1.5).

Order matters:
  1. hard filters (availability, currency, budget, merchant) run first, so a
     similarity score can never bring a filtered product back;
  2. a product that shares no word with any search query is dropped;
  3. only the survivors are scored (MiniLM when active, keyword overlap otherwise);
  4. scores are clamped into 0-1 and products are ranked, size-available first.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from .merchant_identity import merchant_match
from .schemas import (
    PayloadOffer,
    Product,
    ScoredProduct,
    UnverifiedCandidate,
    UserIntent,
    unverified_reason_text,
)
from .semantic import (  # noqa: F401  (re-exported for older imports)
    LexicalScorer,
    RelevanceScorer as ProductScorer,
    RunScorer,
    _tokens,
    get_default_scorer,
    lexical_similarity,
)

# A product with no word in common with any query is a hard drop.
RELEVANCE_FLOOR = 0.0

# At most this many unverified candidates are shown per result (settings override).
DEFAULT_MAX_UNVERIFIED = 3
_MAX_PAYLOAD_OFFERS = 5

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Units and labels that are not part of the size itself ("UK 9" == "9").
_SIZE_NOISE = frozenset({"uk", "us", "eu", "in", "india", "size", "cm", "mm"})


def _stem(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _floor_overlap(query: str, product: Product) -> float:
    """Used ONLY by the relevance floor (scoring still uses the scorer).

    Plurals ('shirts' ~ 'shirt') and joined/hyphenated forms ('tshirt' ~ 'T-Shirt')
    count as shared words, so the floor drops only clearly unrelated products.
    """
    query_words = {_stem(w) for w in _TOKEN_RE.findall(query.lower())}
    if not query_words:
        return 0.0
    text = f"{product.title} {product.description}".lower()
    words = {_stem(w) for w in _TOKEN_RE.findall(text)}
    compact = "".join(_TOKEN_RE.findall(text))
    hits = sum(1 for w in query_words if w in words or (len(w) >= 5 and w in compact))
    return hits / len(query_words)


def _default_scorer() -> ProductScorer:
    """MiniLM when installed, else an explicit lexical fallback; one per run."""
    return RunScorer(get_default_scorer())


# ---------------------------------------------------------------- size status
_SIZE_TOKEN = re.compile(
    r"\b(?:(uk|us|eu|in)\s*)?(\d{1,2}(?:\.5)?|2xl|3xl|xxxl|xxl|xl|xxs|xs|s|m|l)\b",
    re.IGNORECASE,
)


# Spelled-out sizes.  Longest forms first so "Extra Large" is one size (xl), never
# "extra" + "large" (l).
_SIZE_WORD = re.compile(
    r"\b(?:(?:2x|xx|extra[\s-]*extra)[\s-]*(small|large)"
    r"|(?:x|extra)[\s-]*(small|large)"
    r"|(small|medium|large))\b",
    re.IGNORECASE,
)


def _size_words_to_letters(text: str) -> str:
    """'Extra Large' -> 'xl', 'Medium (M)' -> 'm (M)'; text without such words is unchanged."""

    def letter(m: re.Match[str]) -> str:
        if m.group(1):
            return "xx" + m.group(1)[0].lower()
        if m.group(2):
            return "x" + m.group(2)[0].lower()
        return m.group(3)[0].lower()

    return _SIZE_WORD.sub(letter, text)


def _size_readings(value: object) -> list[tuple[str | None, str]]:
    """Every (system, value) in a size label: '9 (44)' -> [(None,'9'), (None,'44')]."""
    return [
        ((m.group(1) or "").lower() or None, m.group(2).lower())
        for m in _SIZE_TOKEN.finditer(_size_words_to_letters(str(value)))
    ]


def _readings_match(requested, label) -> tuple[bool, bool]:
    """(same size, same number under a conflicting system such as UK 9 vs US 9)."""
    same = clash = False
    for req_sys, req_val in requested:
        for lab_sys, lab_val in label:
            if req_val != lab_val:
                continue
            if req_sys is None or lab_sys is None or req_sys == lab_sys:
                same = True
            else:
                clash = True
    return same, clash


def size_status(requested_size: str | None, product: Product) -> str | None:
    """'available', 'unavailable' or 'unknown' for the requested size.

    None when no size was requested.  Units such as UK/US/EU are not compared,
    only the size value itself.
    """
    if not requested_size or requested_size.strip().lower() == "any":
        return None

    requested = _size_readings(requested_size)
    if not requested:
        return "unknown"
    variants = getattr(product, "variants", None) or []
    if not variants:
        return "unknown"

    saw_size_option = False
    saw_conflict = False
    for variant in variants:
        options = getattr(variant, "options", None) or {}
        variant_size = None
        for key, value in options.items():
            if str(key).strip().lower() == "size":
                variant_size = _size_readings(value)
                break
        if variant_size is None:
            continue
        saw_size_option = True
        same, clash = _readings_match(requested, variant_size)
        saw_conflict = saw_conflict or clash
        if same and getattr(variant, "available", False):
            return "available"

    return "unavailable" if saw_size_option and not saw_conflict else "unknown"


# ------------------------------------------------------------------- report
@dataclass(frozen=True)
class RelevanceReport:
    """M1.5 scoring diagnostics for one ranking call."""

    scorer: str
    minilm_active: bool
    lexical_fallback: bool
    fallback_reason: str | None
    queries: tuple[str, ...]

    evaluated: int
    passed_hard_filters: int
    scored: int

    score_min: float | None
    score_max: float | None
    score_mean: float | None
    raw_min: float | None
    raw_max: float | None

    histogram: tuple[int, ...]
    irrelevant_floor_dropped: int

    def to_dict(self) -> dict[str, object]:
        return {
            "scorer": self.scorer,
            "minilm_active": self.minilm_active,
            "lexical_fallback": self.lexical_fallback,
            "fallback_reason": self.fallback_reason,
            "queries": list(self.queries),
            "evaluated": self.evaluated,
            "passed_hard_filters": self.passed_hard_filters,
            "scored": self.scored,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_mean": self.score_mean,
            "raw_min": self.raw_min,
            "raw_max": self.raw_max,
            "histogram": list(self.histogram),
            "irrelevant_floor_dropped": self.irrelevant_floor_dropped,
        }


def _histogram(scores: list[float]) -> tuple[int, ...]:
    buckets = [0] * 10
    for score in scores:
        buckets[0 if score <= 0 else min(9, int(score * 10))] += 1
    return tuple(buckets)


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _clamp(score: float) -> float:
    return min(1.0, max(0.0, float(score)))


def _query_list(original: str, extra: Sequence[str] | None) -> list[str]:
    """The ORIGINAL query first, then same-meaning rewrites, without duplicates."""
    queries = [original]
    for query in extra or ():
        if query and query not in queries:
            queries.append(query)
    return queries


def _score_products(
    scorer: ProductScorer, queries: list[str], products: list[Product]
) -> list[float]:
    raw = list(scorer.score(queries, products))
    if len(raw) != len(products):
        raise ValueError("scorer returned a different number of scores than products")
    return [float(score) for score in raw]


# ------------------------------------------------------------ offer selection
def _offer_for_currency(product: Product, currency: str) -> tuple[Product | None, str | None]:
    """The product as ONE verifiable offer in ``currency``.

    Returns (product, None) to keep going, or (None, reason) with reason one of
    the existing drop reasons.  Only a variant priced in ``currency`` and
    available is ever used: no conversion, no relabelling, no foreign price.
    Price, seller and URL come from that same variant (a Global Catalog
    product can mix sellers).  A product whose variants carry no price is
    returned unchanged for the existing checks.  Ties go to the first variant.
    """
    priced = [v for v in product.variants if v.price is not None and v.currency]
    if not priced:
        return product, None

    in_currency = [v for v in priced if v.currency == currency]
    if not in_currency:
        return None, "currency_mismatch"
    # Only an offer the catalog explicitly says is available is buyable.  An offer
    # with no stated availability is "availability_unknown", never assumed in stock.
    buyable = [v for v in in_currency if v.available and v.availability_known]
    if not buyable:
        unknown = any(not v.availability_known for v in in_currency)
        return None, "availability_unknown" if unknown else "unavailable"

    for chosen in sorted(buyable, key=lambda v: v.price):  # stable: first wins a tie
        data = product.model_dump()
        data.update(
            price=chosen.price,
            currency=currency,
            price_note=None,
            availability_known=True,  # the chosen variant is explicitly available
            selected_variant_id=chosen.id,
            merchant=chosen.seller or product.merchant,
            url=chosen.url or product.url,
            # Only this seller's variants in this currency (unpriced ones of the
            # same seller stay), so a size that exists only in another offer is
            # never reported as available.
            variants=[
                v.model_dump()
                for v in product.variants
                if v.currency in (None, currency) and v.seller == chosen.seller
            ],
        )
        try:
            return Product.model_validate(data), None
        except ValidationError:
            continue  # e.g. a malformed variant URL: never fall back to another seller's

    # Every matching offer was malformed: keep the product, but unverified.
    return product.model_copy(
        update={
            "price": None,
            "currency": None,
            "price_note": f"No usable {currency} offer could be verified.",
        }
    ), None


# ------------------------------------------------------------ hard filters
def _hard_filter_one(intent: UserIntent, product: Product) -> tuple[Product, str | None]:
    """The hard filters for ONE product, in their fixed order.

    Returns (product, reason): ``reason`` is the drop reason, or None when the
    product passes.  The returned product is the selected offer when offer
    selection ran, so a dropped product still carries the price it was judged on.
    """
    if intent.currency is not None:
        offer, reason = _offer_for_currency(product, intent.currency)
        if offer is None:
            return product, reason
        product = offer

    if not product.availability_known:
        return product, "availability_unknown"
    if not product.available:
        return product, "unavailable"

    # A price with no stated currency cannot be checked against a currency
    # or a budget, so it is never shown as if it had been.
    if (
        product.currency is None
        and product.price is not None
        and (intent.currency is not None or intent.max_price is not None)
    ):
        return product, "currency_unknown"

    if (
        intent.currency is not None
        and product.currency is not None
        and product.currency != intent.currency
    ):
        return product, "currency_mismatch"

    if intent.max_price is not None:
        if intent.currency is None:
            return product, "budget_currency_unknown"
        if product.price is None:
            return product, "no_price"
        if product.price > intent.max_price:
            return product, "over_budget"

    if intent.merchant:
        # The seller's displayed name is self-asserted.  Only the host of the URL
        # the buyer would be sent to proves who the merchant is.  After offer
        # selection, product.url is the chosen offer's own URL.
        verdict = merchant_match(intent.merchant, product.merchant, [product.url])
        if verdict == "mismatch":
            return product, "other_merchant"
        if verdict == "unverified":
            return product, "merchant_unverified"

    return product, None


def _trail_entry(
    intent: UserIntent, original: Product, shown: Product, outcome: str
) -> dict[str, object]:
    """One log line about one fetched product (small, JSON-serialisable)."""
    chosen = next(
        (
            v
            for v in original.variants
            if v.id and v.id == shown.selected_variant_id
        ),
        None,
    )
    span = None
    if intent.currency is not None:
        prices = [
            v.price
            for v in original.variants
            if v.price is not None and v.currency == intent.currency and v.available
        ]
        if prices:
            span = [str(min(prices)), str(max(prices))]
    return {
        "id": original.id[-12:],
        "title": original.title[:50],
        "outcome": outcome,
        "listed_currency": original.currency,
        "variant_currencies": sorted(
            {v.currency for v in original.variants if v.price is not None and v.currency}
        ),
        "price": str(shown.price) if shown.price is not None else None,
        "currency": shown.currency,
        "merchant": shown.merchant,
        "variant": shown.selected_variant_id[-12:] if shown.selected_variant_id else None,
        "variant_size": chosen.options.get("size") if chosen else None,
        "in_currency_span": span,
        "detail_source": original.detail_source,
    }


# ------------------------------------------------------------------ pipeline
# ------------------------------------------------------ unverified candidates
def _payload_offers(product: Product) -> list[PayloadOffer]:
    return [
        PayloadOffer(price=v.price, currency=v.currency, seller=v.seller, available=v.available)
        for v in product.variants[:_MAX_PAYLOAD_OFFERS]
    ]


# Order in which the hard checks run; when every offer of a search-payload product
# is contradicted, the reason reported is the one that got furthest.
_PAYLOAD_REASON_ORDER = ("currency_mismatch", "unavailable", "over_budget", "other_merchant")


def _payload_screen(
    intent: UserIntent, product: Product
) -> tuple[Product, str | None, str]:
    """Apply the hard constraints to what a search payload ALREADY says.

    A candidate whose detail call failed keeps only the offers that nothing in the
    payload contradicts.  An offer is contradicted by: a price in a different
    currency than requested, an explicit "unavailable", a price above the budget in
    the requested currency, or a merchant evidence points away from.  Missing data
    is not a contradiction (the candidate stays unverified and labelled), but known
    violations are never shown.

    Returns (product narrowed to its surviving offers, drop reason or None,
    merchant_status).
    """
    if product.variants:
        offers = [
            (v.price, v.currency, v.available if v.availability_known else None,
             v.seller or product.merchant, v.url or product.url)
            for v in product.variants
        ]
    else:
        offers = [(
            product.price, product.currency,
            product.available if product.availability_known else None,
            product.merchant, product.url,
        )]

    reasons: list[str] = []
    survivors: list[tuple[int, str]] = []  # (index, merchant verdict)
    for index, (price, currency, available, seller, url) in enumerate(offers):
        reason = None
        if intent.currency and currency and currency != intent.currency:
            reason = "currency_mismatch"
        elif available is False:
            reason = "unavailable"
        elif (
            intent.max_price is not None and intent.currency and price is not None
            and currency == intent.currency and price > intent.max_price
        ):
            reason = "over_budget"
        verdict = "verified"
        if reason is None and intent.merchant:
            verdict = merchant_match(intent.merchant, seller, [url])
            if verdict == "mismatch":
                reason = "other_merchant"
        if reason is not None:
            reasons.append(reason)
        else:
            survivors.append((index, verdict))

    if not survivors:
        reason = max(reasons, key=_PAYLOAD_REASON_ORDER.index) if reasons else "unavailable"
        return product, reason, "not_requested"

    status = "not_requested"
    if intent.merchant:
        verified = [item for item in survivors if item[1] == "verified"]
        if verified:
            survivors, status = verified, "verified"
        else:
            status = "unverified"
    if product.variants:
        keep = {index for index, _ in survivors}
        product = product.model_copy(
            update={"variants": [v for i, v in enumerate(product.variants) if i in keep]}
        )
    return product, None, status


def _unverified_candidate(
    intent: UserIntent,
    product: Product,
    merchant_status: str = "not_requested",
    listed_from: Product | None = None,
) -> UnverifiedCandidate:
    """A search-payload product as a display-only candidate.

    Nothing here is converted, relabelled or compared with the budget.  A payload
    that lists only other currencies is NOT reported as unavailable in the
    requested currency: that offer's status is simply unknown.
    """
    priced = [v for v in product.variants if v.price is not None and v.currency]
    price = currency = merchant = None
    if priced:
        listed = {v.currency for v in priced}
        if len(listed) == 1:
            cheapest = min(priced, key=lambda v: v.price)  # first wins a tie
            price, currency, merchant = cheapest.price, cheapest.currency, cheapest.seller
        merchant = merchant or product.merchant
    else:
        listed = {product.currency} if product.currency else set()
        price, currency, merchant = product.price, product.currency, product.merchant
    kind = product.detail_failure_kind
    lists_requested = None
    if intent.currency is not None and listed:
        lists_requested = intent.currency in listed
    return UnverifiedCandidate(
        product=product,
        verification_status=(
            "unverified_detail_skipped" if kind == "budget_skipped" else "unverified_detail_failed"
        ),
        unverified_reason=kind or "unknown",
        reason_text=unverified_reason_text(kind),
        candidate_price=price,
        candidate_currency=currency,
        candidate_merchant=merchant,
        # The search listing is kept as the search returned it (display only); the
        # price/merchant above come only from offers nothing contradicts.
        listed_offers=_payload_offers(listed_from or product),
        payload_lists_requested_currency=lists_requested,
        merchant_status=merchant_status,  # type: ignore[arg-type]
    )


def _unverified_trail_entry(
    original: Product, candidate: UnverifiedCandidate, outcome: str
) -> dict[str, object]:
    """Trail line for a search-payload product.  ``price``/``currency`` stay None on
    purpose: only the candidate_* fields carry the unverified payload values."""
    return {
        "id": original.id[-12:],
        "title": original.title[:50],
        "outcome": outcome,
        "listed_currency": original.currency,
        "variant_currencies": sorted(
            {v.currency for v in original.variants if v.price is not None and v.currency}
        ),
        "price": None,
        "currency": None,
        "merchant": None,
        "variant": None,
        "variant_size": None,
        "in_currency_span": None,
        "detail_source": "search_payload",
        "verification_status": candidate.verification_status,
        "unverified_reason": candidate.unverified_reason,
        "candidate_price": (
            str(candidate.candidate_price) if candidate.candidate_price is not None else None
        ),
        "candidate_currency": candidate.candidate_currency,
        "candidate_merchant": candidate.candidate_merchant,
        "payload_lists_requested_currency": candidate.payload_lists_requested_currency,
        "n_listed_offers": len(candidate.listed_offers),
    }


def unverified_from_stats(stats: dict[str, object]) -> list[UnverifiedCandidate]:
    """Rebuild the candidates ``filter_and_rank_with_report`` put in ``stats``."""
    out: list[UnverifiedCandidate] = []
    for item in stats.get("unverified_candidates") or []:  # type: ignore[union-attr]
        try:
            out.append(UnverifiedCandidate.model_validate(item))
        except ValidationError:
            continue
    return out


def filter_and_rank_with_report(
    intent: UserIntent,
    products: list[Product],
    scorer: ProductScorer | None = None,
    extra_queries: Sequence[str] | None = None,
    max_unverified: int = DEFAULT_MAX_UNVERIFIED,
) -> tuple[list[ScoredProduct], dict[str, object], RelevanceReport]:
    """Hard-filter, drop zero-overlap products, score the rest, rank them.

    Delivery city, returns, size and gender are never turned into catalog
    facts here; size only affects ordering through ``size_status``.
    """
    scorer = scorer or _default_scorer()
    queries = _query_list(intent.query, extra_queries)

    currencies: Counter[str] = Counter()
    dropped = {
        "unavailable": 0,
        "currency_mismatch": 0,
        "currency_unknown": 0,
        "no_price": 0,
        "budget_currency_unknown": 0,
        "over_budget": 0,
        "other_merchant": 0,
        "merchant_unverified": 0,
        "availability_unknown": 0,
        "irrelevant": 0,
    }

    hard_filtered: list[Product] = []
    trail: list[dict[str, object]] = []  # one entry per fetched product, input order
    entry_of: dict[int, dict[str, object]] = {}  # id(kept product) -> its entry
    unverified_pool: list[tuple[UnverifiedCandidate, dict[str, object]]] = []
    for original in products:
        currencies[original.currency or "unknown"] += 1

        if original.detail_source == "search_payload":
            # get_product failed or was skipped: the price was never confirmed, so
            # it can only be a display-only candidate.  But the hard constraints are
            # still applied to whatever the payload already says: a candidate whose
            # own payload shows a wrong currency, an over-budget price, an explicit
            # "unavailable" or a different merchant is dropped, never shown.
            screened, screen_reason, merchant_status = _payload_screen(intent, original)
            candidate = _unverified_candidate(
                intent, original if screen_reason else screened, merchant_status,
                listed_from=original,
            )
            entry = _unverified_trail_entry(original, candidate, "unverified_candidate")
            trail.append(entry)
            if screen_reason is not None:
                dropped[screen_reason] += 1
                entry["outcome"] = screen_reason
                continue
            original = screened
            if max(_floor_overlap(query, original) for query in queries) <= RELEVANCE_FLOOR:
                dropped["irrelevant"] += 1
                entry["outcome"] = "irrelevant"
            else:
                unverified_pool.append((candidate, entry))
            continue

        product, reason = _hard_filter_one(intent, original)
        entry = _trail_entry(intent, original, product, reason or "kept")
        trail.append(entry)
        if reason is not None:
            dropped[reason] += 1
            continue

        entry_of[id(product)] = entry
        hard_filtered.append(product)

    # Relevance floor: no word in common with any query -> never shown.
    survivors: list[Product] = []
    floor_dropped = 0
    for product in hard_filtered:
        overlap = max(_floor_overlap(query, product) for query in queries)
        if overlap <= RELEVANCE_FLOOR:
            dropped["irrelevant"] += 1
            floor_dropped += 1
            entry_of[id(product)]["outcome"] = "irrelevant"
            continue
        survivors.append(product)

    raw_scores = _score_products(scorer, queries, survivors) if survivors else []
    scores = [_clamp(score) for score in raw_scores]

    ranked = [
        ScoredProduct(
            product=product,
            score=score,
            size_status=size_status(intent.size, product) or "unknown",
        )
        for product, score in zip(survivors, scores)
    ]
    entry_of_item: dict[int, dict[str, object]] = {}
    for product, score, item in zip(survivors, scores, ranked):
        entry = entry_of[id(product)]
        entry["score"] = _round(score)
        entry["size_status"] = item.size_status
        entry_of_item[id(item)] = entry

    ranked.sort(
        key=lambda item: (
            0 if item.size_status == "available" else 1 if item.size_status == "unknown" else 2,
            -item.score,
        )
    )
    for position, item in enumerate(ranked, start=1):
        entry_of_item[id(item)]["rank"] = position

    # Unverified candidates: scored for ORDER only, never ranked with verified ones.
    shown_unverified: list[UnverifiedCandidate] = []
    if unverified_pool:
        u_scores = [
            _clamp(s)
            for s in _score_products(scorer, queries, [c.product for c, _ in unverified_pool])
        ]
        by_relevance = sorted(zip(unverified_pool, u_scores), key=lambda pair: -pair[1])
        limit = max(0, max_unverified)
        for (candidate, entry), score in by_relevance:
            entry["score"] = _round(score)
            if len(shown_unverified) < limit:
                shown_unverified.append(candidate.model_copy(update={"relevance": score}))
                entry["unverified_rank"] = len(shown_unverified)
            else:
                entry["outcome"] = "unverified_not_shown"

    # Read the name AFTER scoring: a RunScorer may have degraded to the fallback.
    name = getattr(scorer, "name", "unknown")
    report = RelevanceReport(
        scorer=name,
        minilm_active=name == "minilm",
        lexical_fallback=name == "lexical_fallback",
        fallback_reason=getattr(scorer, "fallback_reason", None),
        queries=tuple(queries),
        evaluated=len(products),
        passed_hard_filters=len(hard_filtered),
        scored=len(scores),
        score_min=_round(min(scores)) if scores else None,
        score_max=_round(max(scores)) if scores else None,
        score_mean=_round(sum(scores) / len(scores)) if scores else None,
        raw_min=_round(min(raw_scores)) if raw_scores else None,
        raw_max=_round(max(raw_scores)) if raw_scores else None,
        histogram=_histogram(scores),
        irrelevant_floor_dropped=floor_dropped,
    )

    stats: dict[str, object] = {
        "evaluated": len(products),
        "kept": len(ranked),
        "dropped": dropped,
        "currencies_seen": dict(currencies),
        "relevance": report.to_dict(),
        "products": trail,
        "unverified": len(shown_unverified),
        "unverified_not_shown": len(unverified_pool) - len(shown_unverified),
        "unverified_candidates": [c.model_dump(mode="json") for c in shown_unverified],
    }
    return ranked, stats, report


def filter_and_rank_with_stats(
    intent: UserIntent,
    products: list[Product],
    scorer: ProductScorer | None = None,
    extra_queries: Sequence[str] | None = None,
    max_unverified: int = DEFAULT_MAX_UNVERIFIED,
) -> tuple[list[ScoredProduct], dict[str, object]]:
    """M1.4-compatible wrapper around the M1.5 ranking pipeline."""
    ranked, stats, _ = filter_and_rank_with_report(
        intent, products, scorer=scorer, extra_queries=extra_queries,
        max_unverified=max_unverified,
    )
    return ranked, stats


def filter_and_rank(intent: UserIntent, products: list[Product]) -> list[ScoredProduct]:
    return filter_and_rank_with_stats(intent, products)[0]
