"""Deterministic reading of constraints the product catalog cannot answer.

Gemini proposes a UserIntent, but "deliverable in Mumbai" and "replaceable"
must not depend on the LLM keeping them (or spelling them right).  As with
``budget.py``, plain Python reads the user's own words and fills or corrects
the intent.

Every constraint has a KIND, which decides what the executor may do with it:

* search        - goes into the catalog query text
* hard_filter   - checked against product data (price, currency, merchant)
* verify_later  - the catalog cannot confirm it; shown as UNVERIFIED

Honesty rule: a verify_later constraint is never reported as satisfied.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .attributes import is_any
from .schemas import UserIntent

CONSTRAINT_KINDS: dict[str, str] = {
    "query": "search",
    "max_price": "hard_filter",
    "currency": "hard_filter",
    "merchant": "hard_filter",
    "delivery_city": "verify_later",
    "return_required": "verify_later",
}

# canonical city -> lower-case aliases and common misspellings
_CITY_ALIASES: dict[str, tuple[str, ...]] = {
    "Mumbai": ("mumbai", "bombay", "mumbay", "mumbia", "mumabi"),
    "Delhi": ("delhi", "new delhi", "dilli"),
    "Bengaluru": ("bengaluru", "bangalore", "bangaluru", "bengalooru"),
    "Chennai": ("chennai", "madras", "chenai"),
    "Hyderabad": ("hyderabad", "hydrabad", "hyderbad"),
    "Kolkata": ("kolkata", "calcutta", "kolkatta"),
    "Pune": ("pune", "poona"),
    "Ahmedabad": ("ahmedabad", "ahmadabad"),
    "Jaipur": ("jaipur",),
    "Lucknow": ("lucknow",),
    "Kochi": ("kochi", "cochin"),
    "Chandigarh": ("chandigarh",),
    "Gurugram": ("gurugram", "gurgaon"),
    "Noida": ("noida",),
    "Surat": ("surat",),
    "Indore": ("indore",),
    "Bhopal": ("bhopal",),
    "Nagpur": ("nagpur",),
    "Visakhapatnam": ("visakhapatnam", "vizag"),
    "Coimbatore": ("coimbatore",),
}
_ALIAS_TO_CITY = {a: city for city, aliases in _CITY_ALIASES.items() for a in aliases}
# Every city in the table above is in India (ISO 3166 alpha-2).  Add a country
# here when you add a city outside India.
_CITY_COUNTRY: dict[str, str] = {city: "IN" for city in _CITY_ALIASES}


def city_country(city: str | None) -> str | None:
    """ISO country code of a known city, or None."""
    canonical = canonical_city(city) if city else None
    return _CITY_COUNTRY.get(canonical) if canonical else None


def city_market_note(city: str | None, market_country: str | None) -> str | None:
    """Warn when the delivery city is in a different country than the search market."""
    home = city_country(city)
    if not home or not market_country or home == market_country.upper():
        return None
    return (
        f"{canonical_city(city)} is in {home}, but this search is localised to "
        f"{market_country.upper()}; stores may not deliver there."
    )
_SINGLE_ALIASES = [a for a in _ALIAS_TO_CITY if " " not in a]
_WORD = re.compile(r"[a-z]+")

# A city only counts as a DELIVERY city when the text talks about delivery.
_DELIVERY_CUE = re.compile(r"\b(del[ie]v\w*|ship\w*|dispatch\w*|send\w*|reach\w*)", re.IGNORECASE)

_RETURN_TERMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("exchange", re.compile(r"\b(replac\w*|exchang\w*)", re.IGNORECASE)),
    ("return", re.compile(r"\breturn(?:s|able)?\b", re.IGNORECASE)),
    ("refund", re.compile(r"\brefund\w*|\bmoney[- ]back\b", re.IGNORECASE)),
)
_NEGATION = re.compile(r"(non[- ]?|no |not |without |un)$", re.IGNORECASE)


def canonical_city(value: str | None) -> str | None:
    """Map a city spelling (incl. common misspellings) to its canonical name."""
    if not value:
        return None
    text = " ".join(value.lower().split())
    if text in _ALIAS_TO_CITY:
        return _ALIAS_TO_CITY[text]
    if " " not in text and len(text) >= 5:
        close = difflib.get_close_matches(text, _SINGLE_ALIASES, n=1, cutoff=0.82)
        if close:
            return _ALIAS_TO_CITY[close[0]]
    return None


def extract_delivery_city(text: str) -> str | None:
    """The single delivery city named in ``text``, or None (none / ambiguous)."""
    if not _DELIVERY_CUE.search(text):
        return None
    lowered = text.lower()
    found: set[str] = set()
    for alias, city in _ALIAS_TO_CITY.items():
        if " " in alias and alias in lowered:
            found.add(city)
    for word in _WORD.findall(lowered):
        city = canonical_city(word)
        if city:
            found.add(city)
    # "new delhi" also yields the single word "delhi": both map to Delhi.
    return next(iter(found)) if len(found) == 1 else None


@dataclass(frozen=True)
class ReturnPolicyRequest:
    required: bool
    terms: tuple[str, ...] = ()  # e.g. ("exchange",): what the user actually said


def extract_return_request(text: str) -> ReturnPolicyRequest:
    """Did the user ask for returns / replacement / refund?  Negations are skipped."""
    kinds: list[str] = []
    for kind, pattern in _RETURN_TERMS:
        for match in pattern.finditer(text):
            if _NEGATION.search(text[: match.start()][-6:]):
                continue
            if kind not in kinds:
                kinds.append(kind)
    return ReturnPolicyRequest(required=bool(kinds), terms=tuple(kinds))


@dataclass(frozen=True)
class UnverifiedConstraint:
    name: str
    value: str
    reason: str


def unverified_constraints(intent: UserIntent) -> list[UnverifiedConstraint]:
    """What the UI must show as UNVERIFIED (never as satisfied)."""
    items: list[UnverifiedConstraint] = []
    if intent.delivery_city:
        items.append(
            UnverifiedConstraint(
                "delivery_city",
                intent.delivery_city,
                "Catalog search only localises by country; delivery to a city is "
                "confirmed at cart/checkout.",
            )
        )
    if intent.return_required:
        items.append(
            UnverifiedConstraint(
                "return_required",
                "returns / exchange",
                "Return and exchange terms are store-level policy, not product data.",
            )
        )
    if intent.size and not is_any(intent.size):
        items.append(
            UnverifiedConstraint(
                "size",
                f"size {intent.size}",
                "Sizes are only confirmed per variant; a store may not list them.",
            )
        )
    audience = [v for v in (intent.age_group, intent.gender) if v and not is_any(v)]
    if audience:
        items.append(
            UnverifiedConstraint(
                "audience",
                "for " + " / ".join(audience),
                "Catalog data has no reliable men/women/kids field.",
            )
        )
    if intent.min_rating is not None:
        items.append(
            UnverifiedConstraint(
                "min_rating",
                f"rating {intent.min_rating:g}+",
                "Catalog search results carry no customer ratings.",
            )
        )
    for attribute in intent.attributes:
        if isinstance(attribute, dict):
            name, value = attribute.get("name"), attribute.get("value")
        else:
            name, value = getattr(attribute, "name", None), getattr(attribute, "value", None)
        if name and value and not is_any(str(value)):
            items.append(
                UnverifiedConstraint("attribute", f"{name}: {value}", "Not checked against catalog data.")
            )
    for text in intent.unsupported_constraints:
        items.append(UnverifiedConstraint("other", text, "Not supported by catalog data."))
    return items
def reconcile_constraints(intent: UserIntent, text: str) -> tuple[UserIntent, list[str]]:
    """Make the intent agree with the delivery/return words the user typed.

    * city: an unambiguous city in a delivery sentence overrides the LLM value;
      an LLM-provided spelling is normalised ("Mumbia" -> "Mumbai").
    * returns: fills ``return_required`` when the LLM missed it; never clears it.
    * return-policy terms are removed from ``unsupported_constraints``.
    """
    updates: dict[str, object] = {}
    notes: list[str] = []

    typed_city = extract_delivery_city(text)
    normalised_llm_city = canonical_city(intent.delivery_city) or intent.delivery_city
    city = typed_city or normalised_llm_city
    if city != intent.delivery_city:
        updates["delivery_city"] = city
        notes.append(f"delivery_city {intent.delivery_city} -> {city}")

    request = extract_return_request(text)
    if request.required:
        if not intent.return_required:
            updates["return_required"] = True
            notes.append(
                f"return_required -> True (you said: {', '.join(request.terms)})"
            )

        return_kinds = set(request.terms)

        cleaned_unsupported = [
            value
            for value in intent.unsupported_constraints
            if not any(
                kind in return_kinds and pattern.search(value)
                for kind, pattern in _RETURN_TERMS
            )
        ]

        if cleaned_unsupported != intent.unsupported_constraints:
            updates["unsupported_constraints"] = cleaned_unsupported
            notes.append(
                "removed return-policy terms from unsupported_constraints: "
                + ", ".join(sorted(return_kinds))
            )

    if not updates:
        return intent, []

    return intent.model_copy(update=updates), notes
def reconcile_all(
    intent: "UserIntent",
    text: str,
    *,
    default_currency: str | None = None,
) -> tuple["UserIntent", list[str]]:
    """Run all deterministic reconciliation passes over the user's typed words."""
    from .attributes import reconcile_attributes
    from .budget import reconcile_intent, resolve_currency
    from .categories import detect_category

    notes: list[str] = []
    current = intent

    # 1. Explicit budget/currency typed by the user wins over the LLM.
    current, budget_notes = reconcile_intent(current, text)
    notes.extend(budget_notes)

    # 2. Fill/correct size, gender, age group and rating from typed words.
    current, attribute_notes = reconcile_attributes(current, text)
    notes.extend(attribute_notes)

    # 3. Fill/correct delivery city and return requirement.
    current, constraint_notes = reconcile_constraints(current, text)
    notes.extend(constraint_notes)

    # 4. Detect category from the actual query when Gemini missed it.
    if not current.category:
        category = detect_category(current.query)
        if category:
            current = current.model_copy(update={"category": category})
            notes.append(f"category set to {category}")

    # 5. Resolve a missing currency only when the caller supplies a market
    # default. Never invent one inside this function.
    current, currency_notes = resolve_currency(current, default_currency)
    notes.extend(currency_notes)

    return current, notes
