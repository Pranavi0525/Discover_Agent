"""Category registry and the clarify-before-search rule.

A category says which details a shopper must give before a search is worth
running (footwear and clothing need a size and a gender; a phone case does
not).  Gemini may label the category, but plain Python decides what is
missing, using the registry and the words of the query.

Asking is bounded: ONE question, once.  After the user has answered (or
declined) the search goes ahead and whatever is still missing is reported as
"not specified" instead of blocking the run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .attributes import ANY
from .schemas import ExecutionPlan, PlanStep, UserIntent


@dataclass(frozen=True)
class Category:
    key: str
    keywords: frozenset[str]
    required: tuple[str, ...] = ()  # subset of: size, gender


CATEGORIES: tuple[Category, ...] = (
    Category(
        "footwear",
        frozenset({
            "shoe", "shoes", "sneaker", "sneakers", "trainer", "trainers", "sandal", "sandals",
            "boot", "boots", "slipper", "slippers", "loafer", "loafers", "heels", "flipflops",
            "flip-flops", "footwear", "crocs", "sliders",
        }),
        ("size", "gender"),
    ),
    Category(
        "clothing",
        frozenset({
            "shirt", "shirts", "tshirt", "tshirts", "t-shirt", "t-shirts", "tee", "jeans", "trousers",
            "pants", "shorts", "dress", "dresses", "jacket", "jackets", "hoodie", "hoodies",
            "sweater", "sweatshirt", "kurta", "kurti", "top", "tops", "skirt", "leggings",
            "blazer", "coat", "track", "tracksuit",
        }),
        ("size", "gender"),
    ),
    Category("electronics", frozenset({
        "phone", "smartphone", "laptop", "earbuds", "earphones", "headphones", "charger",
        "speaker", "tablet", "camera", "watch", "smartwatch",
    })),
    Category("home", frozenset({"sofa", "couch", "lamp", "mattress", "pillow", "curtain", "table", "chair"})),
)
_BY_KEY = {c.key: c for c in CATEGORIES}
_WORD = re.compile(r"[a-z0-9\-]+")


def category_keys() -> list[str]:
    return [c.key for c in CATEGORIES]


def prompt_vocabulary() -> str:
    """Text for the planner prompt so Gemini uses the same category labels."""
    return ", ".join(category_keys())


def detect_category(query: str) -> str | None:
    """Category of the first registry keyword found in the query, or None."""
    words = _WORD.findall(query.lower())
    for category in CATEGORIES:
        if any(word in category.keywords for word in words):
            return category.key
    return None


def resolve_category(intent: UserIntent) -> str | None:
    """The LLM's label if it is a known one, else the one read from the query."""
    label = (intent.category or "").strip().lower()
    return label if label in _BY_KEY else detect_category(intent.query)


def _given(intent: UserIntent, name: str) -> bool:
    if name == "size":
        value = intent.size
    elif name == "gender":
        # a kids/infant search is not asked for a gender
        if intent.age_group in {"kids", "infant"}:
            return True
        value = intent.gender
    else:  # pragma: no cover - registry only uses size/gender today
        value = None
    if value:
        return True
    return any(a.name.strip().lower() == name and a.value.strip() for a in intent.attributes)


def missing_required(intent: UserIntent) -> list[str]:
    """Required details for this category that the intent does not have yet."""
    key = resolve_category(intent)
    category = _BY_KEY.get(key) if key else None
    if category is None:
        return []
    return [name for name in category.required if not _given(intent, name)]


def _question(intent: UserIntent, missing: list[str]) -> str:
    subject = intent.query
    parts = []
    if "size" in missing:
        parts.append("your size")
    if "gender" in missing:
        parts.append("who it is for (men, women or unisex)")
    ask = " and ".join(parts)
    return (
        f"To find the right {subject}, tell me {ask}. "
        f"For example: \u201cmen size 9\u201d. If it does not matter, reply \u201c{ANY}\u201d."
    )


def clarify_missing_attributes(
    plan: ExecutionPlan, *, clarify_round: int = 0, enabled: bool = True
) -> ExecutionPlan | None:
    """A clarification plan when a required detail is missing, else None.

    Only ever asks on the first pass (``clarify_round == 0``): once the user
    has answered a question, the search runs with whatever they gave.
    """
    if not enabled or clarify_round > 0:
        return None
    if plan.requires_clarification or plan.steps[0].action != "search_catalog":
        return None
    missing = missing_required(plan.intent)
    if not missing:
        return None
    return ExecutionPlan(
        intent=plan.intent,
        steps=[PlanStep(step_id="clarify_attributes", action="ask_clarification",
                        reason=_question(plan.intent, missing))],
        requires_clarification=True,
    )
