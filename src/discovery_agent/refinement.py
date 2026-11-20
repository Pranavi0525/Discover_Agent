"""Bounded, deterministic query refinement.

Refinement may only change the *query text*.  The user's constraints
(``max_price``, ``currency``, ``merchant``) are immutable: a refinement that
relaxed the budget to avoid an empty result would be a silent budget increase.
The same goes for the constraints the catalog cannot verify (delivery city,
returns, size, gender, rating, other unsupported ones): a re-plan from
shortened text no longer contains them, so they are restored from the
original intent.
"""

from __future__ import annotations

from .schemas import UserIntent

# UserIntent.refinement_count is schema-capped at 2 (MAX_REFINEMENTS default).
_SCHEMA_REFINEMENT_CAP = 2


def broaden_query(query: str) -> str:
    """Make a query broader while keeping its head noun.

    English product queries put the noun last ("blue running shoes"), so the
    broader query drops modifiers from the FRONT, and a trailing qualifier such
    as "size 9" from the END:

        "blue running shoes size 9" -> "blue running shoes"
        "blue running shoes"        -> "running shoes"
        "running shoes"             -> "shoes"
        "shoes"                     -> "shoes"   (nothing left to drop)
    """
    tokens = query.strip().split()
    if len(tokens) <= 1:
        return query.strip()
    if tokens[-1].isdigit() and len(tokens) > 2:
        broadened = tokens[:-2]
    else:
        broadened = tokens[1:]
    if not any(any(ch.isalpha() for ch in token) for token in broadened):
        return query.strip()
    return " ".join(broadened)


def carry_over_constraints(
    planned: UserIntent, original: UserIntent, refinement_count: int
) -> UserIntent:
    """Return ``planned`` with the ORIGINAL user's hard constraints restored."""
    return planned.model_copy(
        update={
            "max_price": original.max_price,
            "currency": original.currency,
            "merchant": original.merchant,
            "delivery_city": original.delivery_city,
            "return_required": original.return_required,
            "unsupported_constraints": list(original.unsupported_constraints),
            "size": original.size,
            "gender": original.gender,
            "age_group": original.age_group,
            "min_rating": original.min_rating,
            "category": original.category,
            "attributes": list(original.attributes),
            "refinement_count": min(refinement_count, _SCHEMA_REFINEMENT_CAP),
        }
    )


def refine_intent(intent: UserIntent, max_refinements: int) -> UserIntent | None:
    """Next bounded refinement of ``intent``, or None when none is allowed/useful."""
    if intent.refinement_count >= max_refinements:
        return None
    if intent.refinement_count >= _SCHEMA_REFINEMENT_CAP:
        return None
    broadened = broaden_query(intent.query)
    if broadened == intent.query.strip():
        return None
    return intent.model_copy(
        update={"query": broadened, "refinement_count": intent.refinement_count + 1}
    )
