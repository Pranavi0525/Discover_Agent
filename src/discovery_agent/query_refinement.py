"""M1.4 query refinement layer.

One place decides which catalog queries are tried and in what order.

Refinement changes query text only. All user constraints remain attached to
the original UserIntent and are never silently relaxed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .attributes import ANY, strip_size_phrase
from .refinement import _SCHEMA_REFINEMENT_CAP, broaden_query
from .schemas import UserIntent

STRATEGY_SPECIFIC = "specific"
STRATEGY_CORE = "core"
STRATEGY_BROADEN = "broaden"
STRATEGY_SYNONYM = "synonym"

SYNONYMS: dict[str, tuple[str, ...]] = {
    "shoes": ("sneakers", "trainers"),
    "shoe": ("sneaker", "trainer"),
    "sneakers": ("shoes", "trainers"),
    "trainers": ("sneakers", "shoes"),
    "sandals": ("slides", "flip flops"),
    "earbuds": ("earphones", "in-ear headphones"),
    "earphones": ("earbuds", "headphones"),
    "headphones": ("earphones", "headset"),
    "laptop": ("notebook",),
    "phone": ("smartphone", "mobile phone"),
    "smartphone": ("phone", "mobile phone"),
    "tshirt": ("t-shirt", "tee"),
    "t-shirt": ("tshirt", "tee"),
    "trousers": ("pants",),
    "pants": ("trousers",),
    "sofa": ("couch",),
    "couch": ("sofa",),
    "backpack": ("rucksack",),
}

_AUDIENCE_WORDS = {
    "men",
    "man",
    "mens",
    "male",
    "women",
    "woman",
    "womens",
    "female",
    "ladies",
    "unisex",
    "kid",
    "kids",
    "child",
    "children",
    "boys",
    "girls",
    "boy",
    "girl",
    "baby",
    "infant",
    "toddler",
}

_EDGE_FILLER = {
    "for",
    "and",
    "with",
    "in",
    "a",
    "an",
    "the",
    "of",
    "to",
    "i",
    "need",
    "want",
    "me",
    "my",
}

_SEARCH_ATTRIBUTES = {
    "color",
    "colour",
    "brand",
    "material",
    "style",
}

_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class QueryStep:
    query: str
    strategy: str


def clean_query(query: str) -> str:
    """Return the core product query.

    Removes size and audience phrases but keeps the product itself.
    """

    text = strip_size_phrase(query)

    words = [
        w.strip(",.;:!?")
        for w in _WORD.findall(text)
    ]

    words = [
        w
        for w in words
        if w and w.lower().removesuffix("'s") not in _AUDIENCE_WORDS
    ]

    while words and words[0].lower() in _EDGE_FILLER:
        words.pop(0)

    while words and words[-1].lower() in _EDGE_FILLER:
        words.pop()

    cleaned = " ".join(words)

    if len(cleaned) >= 2:
        return cleaned

    return " ".join(query.split())


def _audience_word(intent: UserIntent) -> str | None:
    if intent.age_group in {"kids", "infant"}:
        return intent.age_group

    if intent.gender in {"men", "women"}:
        return intent.gender

    return None


def _attribute_pair(attribute: Any) -> tuple[str, str]:
    """(name, value) of an Attribute object or of a plain dict."""
    if isinstance(attribute, dict):
        return (
            str(attribute.get("name", "")).strip().lower(),
            str(attribute.get("value", "")).strip(),
        )

    return (
        str(getattr(attribute, "name", "")).strip().lower(),
        str(getattr(attribute, "value", "")).strip(),
    )


def _specific_query(intent: UserIntent) -> str:
    extras: list[str] = []

    audience = _audience_word(intent)

    if audience:
        extras.append(audience)

    for attribute in intent.attributes:
        name, value = _attribute_pair(attribute)

        if not name or not value:
            continue

        if name in _SEARCH_ATTRIBUTES and value.lower() != ANY:
            if value.lower() not in intent.query.lower():
                extras.append(value)

        if len(extras) >= 3:
            break

    if not extras:
        return intent.query

    # Example:
    # extras = ["men", "blue"]
    # query  = "running shoes"
    # result = "blue men running shoes"
    modifiers = " ".join(reversed(extras))

    return f"{modifiers} {intent.query}".strip()


def synonym_queries(query: str) -> list[str]:
    tokens = query.split()

    if not tokens:
        return []

    head = tokens[-1].lower()

    return [
        " ".join([*tokens[:-1], alt])
        for alt in SYNONYMS.get(head, ())
    ]


def _dedupe(steps: Iterable[QueryStep]) -> list[QueryStep]:
    seen: set[str] = set()
    out: list[QueryStep] = []

    for step in steps:
        key = " ".join(step.query.lower().split())

        if len(key) < 2 or key in seen:
            continue

        seen.add(key)
        out.append(step)

    return out


def build_query_ladder(
    intent: UserIntent,
    max_refinements: int,
) -> list[QueryStep]:
    """Build deterministic search queries.

    Order:

        specific
        core
        broaden
        synonym
        broaden
        synonym
    """

    core = intent.query.strip()

    steps: list[QueryStep] = []

    specific = _specific_query(intent)

    if specific != core:
        steps.append(
            QueryStep(
                specific,
                STRATEGY_SPECIFIC,
            )
        )

    steps.append(
        QueryStep(
            core,
            STRATEGY_CORE,
        )
    )

    broader: list[QueryStep] = []

    current = core

    while True:
        nxt = broaden_query(current)

        if nxt == current.strip():
            break

        broader.append(
            QueryStep(
                nxt,
                STRATEGY_BROADEN,
            )
        )

        current = nxt

    synonyms = [
        QueryStep(
            query,
            STRATEGY_SYNONYM,
        )
        for query in synonym_queries(core)
    ]

    ordered = (
        steps
        + broader[:1]
        + synonyms[:1]
        + broader[1:]
        + synonyms[1:]
    )

    return _dedupe(ordered)[
        : 1
        + max(
            0,
            min(
                max_refinements,
                _SCHEMA_REFINEMENT_CAP,
            ),
        )
    ]


def refinement_note(
    original_query: str,
    step: QueryStep,
) -> str:
    """Explain why a refinement was attempted."""

    left, right = "\u201c", "\u201d"

    how = {
        STRATEGY_BROADEN: (
            f"the search was broadened to {left}{step.query}{right}"
        ),
        STRATEGY_SYNONYM: (
            f"the search was retried with a similar term, "
            f"{left}{step.query}{right}"
        ),
        STRATEGY_CORE: (
            f"the search was retried without the extra details, "
            f"as {left}{step.query}{right}"
        ),
    }.get(
        step.strategy,
        f"the search was retried as {left}{step.query}{right}",
    )

    return (
        f" No products matched {left}{original_query}{right}, so {how}; "
        f"your budget was not changed."
    )


def ladder_to_state(
    ladder: list[QueryStep],
) -> list[dict[str, str]]:
    return [
        {
            "query": step.query,
            "strategy": step.strategy,
        }
        for step in ladder
    ]


def ladder_from_state(
    raw: Any,
) -> list[QueryStep]:
    return [
        QueryStep(
            item["query"],
            item["strategy"],
        )
        for item in (raw or [])
    ]
