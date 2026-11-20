"""M1.6 - turn the search-form fields into planner text and intent overrides.

Only fields that exist on ``UserIntent`` become overrides.

Pure functions, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, NamedTuple

from .attributes import ANY, normalize_gender
from .constraints import canonical_city
from .currency import is_nullish, normalize_currency_code

MAX_PREFERENCES_CHARS = 200

NOT_SURE_VALUES = {
    "",
    "not sure",
    "not_sure",
    "unknown",
    "none",
    "n/a",
}

FORM_LINE_PREFIX = "Form inputs:"


class FormInputs(NamedTuple):
    max_price: Decimal | None = None
    currency: str | None = None
    preferences: str | None = None
    merchant: str | None = None
    delivery_city: str | None = None

    # M1.4 search-form fields
    size: str | None = None
    gender: str | None = None
    min_rating: float | None = None


def _text(value: Any, limit: int) -> str | None:
    if value is None:
        return None

    text = " ".join(str(value).split())

    if not text or is_nullish(text):
        return None

    if text.lower() in NOT_SURE_VALUES:
        return None

    return text[:limit]


def _normalize_size(value: Any) -> str | None:
    text = _text(value, 40)

    if not text:
        return None

    return text


def _normalize_gender(value: Any) -> str | None:
    """men / women / unisex, or None.

    'any' and 'either' mean the form gives no gender (None).  A word that is
    not a gender, such as 'kids', is not turned into one.
    """
    text = _text(value, 30)

    if not text:
        return None

    result = normalize_gender(text)

    if result is None or result == ANY:
        return None

    return result


def _normalize_rating(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        rating = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if rating <= 0:
        return None

    # Ratings are normally 0-5. Treat impossible values as invalid form input.
    if rating > 5:
        return None

    return rating


def normalize_form_inputs(raw: Mapping[str, Any] | None) -> FormInputs:
    """Clean raw form values; invalid values become 'not given'."""

    raw = raw or {}

    # -------------------------
    # Maximum price
    # -------------------------
    max_price: Decimal | None = None

    value = raw.get("max_price")

    if value not in (None, ""):
        try:
            candidate = Decimal(str(value).replace(",", "").strip())

            if candidate.is_finite() and candidate > 0:
                max_price = candidate

        except (InvalidOperation, ValueError):
            pass

    # -------------------------
    # Currency
    # -------------------------
    currency: str | None = None

    raw_currency = raw.get("currency")

    if (
        isinstance(raw_currency, str)
        and raw_currency.strip().lower() not in NOT_SURE_VALUES
    ):
        code = normalize_currency_code(raw_currency)

        if isinstance(code, str) and len(code) == 3 and code.isalpha():
            currency = code.upper()

    # -------------------------
    # Delivery city
    # -------------------------
    city = _text(raw.get("delivery_city"), 80)

    if city:
        city = canonical_city(city) or city

    # -------------------------
    # M1.4 fields
    # -------------------------
    size = _normalize_size(raw.get("size"))
    gender = _normalize_gender(raw.get("gender"))
    min_rating = _normalize_rating(raw.get("min_rating"))

    return FormInputs(
        max_price=max_price,
        currency=currency,
        preferences=_text(raw.get("preferences"), MAX_PREFERENCES_CHARS),
        merchant=_text(raw.get("merchant"), 200),
        delivery_city=city,
        size=size,
        gender=gender,
        min_rating=min_rating,
    )


def _plain(amount: Decimal) -> str:
    return f"{amount.normalize():f}"


def planner_line(form: FormInputs) -> str | None:
    """One deterministic line supplied to the planner.

    Anything already supplied through the form should not be asked again.
    """

    parts: list[str] = []

    if form.max_price is not None:
        if form.currency:
            parts.append(
                f"max budget {_plain(form.max_price)} {form.currency}"
            )
        else:
            parts.append(
                f"max budget {_plain(form.max_price)} "
                f"(currency not specified)"
            )

    if form.delivery_city:
        parts.append(f"delivery city {form.delivery_city}")

    if form.merchant:
        parts.append(f"merchant {form.merchant}")

    if form.size:
        parts.append(f"size {form.size}")

    if form.gender:
        parts.append(f"gender {form.gender}")

    if form.min_rating is not None:
        parts.append(f"minimum rating {form.min_rating:g}")

    if form.preferences:
        parts.append(f"preferences: {form.preferences}")

    if not parts:
        return None

    return f"{FORM_LINE_PREFIX} " + "; ".join(parts)


def intent_overrides(form: FormInputs) -> dict[str, Any]:
    """Values Python deterministically imposes on ``plan.intent``."""

    overrides: dict[str, Any] = {}

    if form.max_price is not None:
        overrides["max_price"] = form.max_price

    if form.currency:
        overrides["currency"] = form.currency

    if form.merchant:
        overrides["merchant"] = form.merchant

    if form.delivery_city:
        overrides["delivery_city"] = form.delivery_city

    # M1.4 typed fields
    if form.size:
        overrides["size"] = form.size

    if form.gender:
        overrides["gender"] = form.gender

    if form.min_rating is not None:
        overrides["min_rating"] = form.min_rating

    return overrides
