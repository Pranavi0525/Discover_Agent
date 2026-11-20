"""Field-level scoring of one planned intent against its expectation."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ..schemas import ExecutionPlan

FIELDS = (
    "query",
    "max_price",
    "currency",
    "merchant",
    "delivery_city",
    "return_required",
    "unsupported",
    "surfaced",
    "clarify",
)


@dataclass
class FieldResult:
    name: str
    ok: bool
    expected: Any
    got: Any


@dataclass
class CaseScore:
    case_id: str
    fields: list[FieldResult] = field(default_factory=list)

    @property
    def correct(self) -> int:
        return sum(f.ok for f in self.fields)

    @property
    def total(self) -> int:
        return len(self.fields)

    @property
    def perfect(self) -> bool:
        return all(f.ok for f in self.fields)

    def failures(self) -> list[FieldResult]:
        return [f for f in self.fields if not f.ok]


def _has(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def _match_unsupported(groups: list[list[str]], produced: list[str]) -> bool:
    """Every group matched by a distinct item, and no produced item left over."""
    if len(groups) != len(produced):
        return False
    remaining = list(produced)
    for group in groups:
        hit = next((p for p in remaining if any(_has(p, kw) for kw in group)), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return not remaining


def _surfaced_texts(intent) -> list[str]:
    """What the user is shown as UNVERIFIED for this intent (the real contract)."""
    from ..constraints import unverified_constraints

    return [f"{u.name} {u.value}" for u in unverified_constraints(intent)]


def _groups_surfaced(groups: list[list[str]], texts: list[str]) -> bool:
    """Recall only: every group is matched by at least one surfaced item."""
    return all(any(_has(t, kw) for t in texts for kw in group) for group in groups)


def score_case(case_id: str, expect: dict[str, Any], plan: ExecutionPlan) -> CaseScore:
    intent = plan.intent
    score = CaseScore(case_id)

    def add(name: str, ok: bool, expected: Any, got: Any) -> None:
        score.fields.append(FieldResult(name, bool(ok), expected, got))

    if "query_include" in expect or "query_exclude" in expect:
        inc = expect.get("query_include", [])
        exc = expect.get("query_exclude", [])
        ok = all(_has(intent.query, w) for w in inc) and not any(_has(intent.query, w) for w in exc)
        add("query", ok, {"include": inc, "exclude": exc}, intent.query)

    if "max_price" in expect:
        want = None if expect["max_price"] is None else Decimal(str(expect["max_price"]))
        add("max_price", intent.max_price == want, want, intent.max_price)

    if "currency" in expect:
        add("currency", intent.currency == expect["currency"], expect["currency"], intent.currency)

    if "merchant" in expect:
        want = expect["merchant"]
        got = intent.merchant
        ok = got is None if want is None else (got is not None and _has(got, want))
        add("merchant", ok, want, got)

    if "delivery_city" in expect:
        add("delivery_city", intent.delivery_city == expect["delivery_city"],
            expect["delivery_city"], intent.delivery_city)

    if "return_required" in expect:
        add("return_required", intent.return_required == expect["return_required"],
            expect["return_required"], intent.return_required)

    if "unsupported" in expect:
        add("unsupported", _match_unsupported(expect["unsupported"], intent.unsupported_constraints),
            expect["unsupported"], intent.unsupported_constraints)

    if "surfaced" in expect:
        texts = _surfaced_texts(intent)
        add("surfaced", _groups_surfaced(expect["surfaced"], texts), expect["surfaced"], texts)

    if "clarify" in expect:
        add("clarify", plan.requires_clarification == expect["clarify"],
            expect["clarify"], plan.requires_clarification)

    return score
