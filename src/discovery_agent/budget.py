"""Deterministic budget/currency extraction from the user's own words.

Gemini proposes an intent, but the hard constraint the user literally typed
("for 4000 rupees") must not depend on the LLM getting a number, a unit or a
currency right.  This module reads only amounts that carry an explicit
currency marker (symbol or word) and returns them as a MAXIMUM price.

Semantics (documented and tested):

* "for 4000 rupees", "under Rs. 4000", "below ₹4,000", "within 4k INR",
  "up to $50", "max 50 dollars"  ->  max_price = that amount (major units).
* "above/over/at least/more than 3000 rupees" is a MINIMUM, so it is ignored
  here (never turned into a maximum).
* More than one different amount in the text is ambiguous -> nothing is
  overridden and the planner's value stands.
* A budget with NO currency marker ("under 8000") is not silently read as USD:
  ``resolve_currency`` fills the configured market currency (and says so), and
  when there is none the planner asks the user which currency they mean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .schemas import UserIntent

_NUM = r"\d[\d,]*(?:\.\d+)?"
_PREFIX = r"(?P<pre>₹|\$|£|€|(?<![a-z])rs\.?|(?<![a-z])inr|(?<![a-z])usd|(?<![a-z])gbp|(?<![a-z])eur)"
_SUFFIX = r"(?P<post>rupees?|rs\.?|inr|dollars?|usd|pounds?|gbp|euros?|eur)(?![a-z])"

_PRE_RE = re.compile(rf"{_PREFIX}\s*(?P<num>{_NUM})\s*(?P<k>k(?![a-z]))?", re.IGNORECASE)
_POST_RE = re.compile(rf"(?<![\d.,])(?P<num>{_NUM})\s*(?P<k>k(?![a-z]))?\s*{_SUFFIX}", re.IGNORECASE)
_MIN_WORDS = re.compile(
    r"(above|over|more than|greater than|at least|minimum|min|starting at|starting from|from)\s*$",
    re.IGNORECASE,
)

_MARKER_TO_CURRENCY = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "rupee": "INR", "rupees": "INR",
    "$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
}


@dataclass(frozen=True)
class ExplicitBudget:
    amount: Decimal
    currency: str


def _amount(number: str, k: str | None) -> Decimal | None:
    try:
        value = Decimal(number.replace(",", ""))
    except InvalidOperation:
        return None
    if k:
        value *= 1000
    return value if value > 0 else None


def extract_budget(text: str) -> ExplicitBudget | None:
    """Return the single explicit maximum budget in ``text``, or None."""
    found: set[tuple[Decimal, str]] = set()
    for regex, group in ((_PRE_RE, "pre"), (_POST_RE, "post")):
        for match in regex.finditer(text):
            if _MIN_WORDS.search(text[: match.start()][-20:]):
                continue
            currency = _MARKER_TO_CURRENCY.get(match.group(group).lower())
            amount = _amount(match.group("num"), match.group("k"))
            if currency and amount is not None:
                found.add((amount, currency))
    if len(found) != 1:
        return None
    amount, currency = next(iter(found))
    return ExplicitBudget(amount=amount, currency=currency)


def budget_currency_missing(intent: UserIntent) -> bool:
    """True when a budget exists but nothing says which currency it is in.

    Such a budget cannot be enforced (a bare number says nothing about price),
    so every layer refuses to act on it instead of guessing.
    """
    return intent.max_price is not None and intent.currency is None


_ANY_CURRENCY_MARKER = re.compile(
    r"[₹$£€¥]|(?<![a-z])(?:rs\.?|inr|usd|gbp|eur|cad|aud|nzd|sgd|jpy|aed|"
    r"rupees?|dollars?|pounds?|euros?|yen|dirhams?)(?![a-z])",
    re.IGNORECASE,
)


def note_inferred_currency(intent: UserIntent, text: str) -> list[str]:
    """Say so when the LLM supplied a currency the user never wrote.

    The prompt tells Gemini to leave the currency null unless the request states
    one, but it sometimes fills one in anyway.  The value is kept (it may well be
    right), but the result must not present a guess as something the user said.
    """
    if intent.max_price is None or intent.currency is None:
        return []
    if _ANY_CURRENCY_MARKER.search(text) or intent.currency.lower() in text.lower():
        return []
    return [f"currency not stated in your request; {intent.currency} was inferred"]


def resolve_currency(
    intent: UserIntent, default_currency: str | None
) -> tuple[UserIntent, list[str]]:
    """Fill a missing currency from the configured market, never over a stated one."""
    if intent.currency is not None or not default_currency:
        return intent, []
    notes: list[str] = []
    if intent.max_price is not None:
        # Only a budget makes the assumption matter, so only then is it reported.
        notes.append(
            f"currency not stated; assumed {default_currency} (store market default)"
        )
    return intent.model_copy(update={"currency": default_currency}), notes


def reconcile_intent(intent: UserIntent, text: str) -> tuple[UserIntent, list[str]]:
    """Make the intent agree with an explicit budget typed by the user."""
    budget = extract_budget(text)
    if budget is None:
        return intent, []

    updates: dict[str, object] = {}
    notes: list[str] = []
    if intent.currency != budget.currency:
        updates["currency"] = budget.currency
        was = intent.currency or "unspecified"
        notes.append(f"currency {was} -> {budget.currency} (stated in the request)")
    if intent.max_price != budget.amount:
        updates["max_price"] = budget.amount
        notes.append(f"max_price {intent.max_price} -> {budget.amount} (stated in the request)")
    if not updates:
        return intent, []
    return intent.model_copy(update=updates), notes
