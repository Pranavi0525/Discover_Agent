from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from .attributes import (
    normalize_age_group,
    normalize_gender,
    normalize_rating,
    normalize_size,
)
from .currency import is_nullish, normalize_currency_code


# ---------------------------------------------------------------------------
# User intent
# ---------------------------------------------------------------------------

class Attribute(BaseModel):
    """One extra requirement from the user, e.g. color=blue."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=50)
    value: str = Field(min_length=1, max_length=100)


class UserIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=300)

    # Budget
    max_price: Decimal | None = Field(default=None, ge=0)

    # None means the user did not explicitly state a currency.
    currency: str | None = Field(default=None, min_length=3, max_length=3)

    # Merchant constraint
    merchant: str | None = Field(default=None, max_length=200)

    # Refinement control
    refinement_count: int = Field(default=0, ge=0, le=2)

    # -----------------------------------------------------------------------
    # M1.4 typed attributes
    # -----------------------------------------------------------------------

    size: str | None = Field(default=None, max_length=50)

    gender: str | None = Field(default=None, max_length=30)

    age_group: str | None = Field(default=None, max_length=30)

    min_rating: float | None = Field(default=None, ge=0, le=5)

    category: str | None = Field(default=None, max_length=50)

    attributes: list[Attribute] = Field(default_factory=list)

    # -----------------------------------------------------------------------
    # Existing trust / constraint fields
    # -----------------------------------------------------------------------

    delivery_city: str | None = Field(default=None, max_length=80)

    return_required: bool = False

    unsupported_constraints: list[str] = Field(default_factory=list)

    # -----------------------------------------------------------------------
    # Currency normalization
    # -----------------------------------------------------------------------

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: Any) -> Any:
        return normalize_currency_code(value)

    # -----------------------------------------------------------------------
    # Text normalization
    # -----------------------------------------------------------------------

    @field_validator("merchant", "delivery_city", mode="before")
    @classmethod
    def nullish_text_is_none(cls, value: Any) -> Any:
        if isinstance(value, str):
            return None if is_nullish(value) else value.strip()
        return value

    @field_validator("max_price", mode="before")
    @classmethod
    def nullish_budget_is_none(cls, value: Any) -> Any:
        """The word "null"/"none"/"" for a budget means no budget was stated, not a parse error."""
        if isinstance(value, str) and is_nullish(value):
            return None
        return value

    # Size, gender, age group and rating are read by attributes.py, so messy
    # LLM output ("Men's", "Children", rating 9) becomes a clean value or None
    # instead of failing the whole plan.
    @field_validator("size", mode="before")
    @classmethod
    def _read_size(cls, value: Any) -> Any:
        return normalize_size(value)

    @field_validator("gender", mode="before")
    @classmethod
    def _read_gender(cls, value: Any) -> Any:
        return normalize_gender(value)

    @field_validator("age_group", mode="before")
    @classmethod
    def _read_age_group(cls, value: Any) -> Any:
        return normalize_age_group(value)

    @field_validator("min_rating", mode="before")
    @classmethod
    def _read_rating(cls, value: Any) -> Any:
        return normalize_rating(value)

    @field_validator("category", mode="before")
    @classmethod
    def normalize_optional_category(cls, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, str):
            value = value.strip()

            if not value or is_nullish(value):
                return None

            return value

        return value

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip().lower()

        if not value:
            return None

        return value

    # -----------------------------------------------------------------------
    # Attributes normalization
    # -----------------------------------------------------------------------

    @field_validator("attributes", mode="before")
    @classmethod
    def normalize_attributes(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return []

        cleaned: list[dict[str, str]] = []

        for item in value:
            if isinstance(item, Attribute):
                name, attr_value = item.name, item.value
            elif isinstance(item, dict):
                name, attr_value = item.get("name"), item.get("value")
            else:
                continue

            if name is None or attr_value is None:
                continue

            name = str(name).strip()[:50]
            attr_value = str(attr_value).strip()[:100]

            if not name or not attr_value:
                continue

            if is_nullish(attr_value):
                continue

            cleaned.append({"name": name, "value": attr_value})

        return cleaned[:20]

    # -----------------------------------------------------------------------
    # Delivery city
    # -----------------------------------------------------------------------

    @field_validator("delivery_city")
    @classmethod
    def blank_city_is_none(cls, value: str | None) -> str | None:
        value = value.strip() if value else value
        return value or None

    # -----------------------------------------------------------------------
    # Unsupported constraints
    # -----------------------------------------------------------------------

    @field_validator("unsupported_constraints", mode="before")
    @classmethod
    def bound_unsupported(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [
                str(item).strip()[:200]
                for item in value
                if str(item).strip()
                and not is_nullish(str(item))
            ][:8]

        return []

    # -----------------------------------------------------------------------
    # Normalize query
    # -----------------------------------------------------------------------

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return value.strip()


# ---------------------------------------------------------------------------
# Planner step
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1, max_length=80)

    action: Literal[
        "search_catalog",
        "get_product",
        "ask_clarification",
    ]

    reason: str = Field(min_length=1, max_length=500)

    # Product IDs are intentionally not accepted from the LLM planner.
    # They are populated by the executor only after search.
    product_ids: list[str] = Field(
        default_factory=list,
        max_length=10,
    )

    @model_validator(mode="after")
    def no_pre_search_product_ids(self) -> "PlanStep":
        if self.action != "get_product" and self.product_ids:
            raise ValueError(
                "product_ids are only valid on executor-generated "
                "get_product steps"
            )

        return self


# ---------------------------------------------------------------------------
# Execution plan
# ---------------------------------------------------------------------------

class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: UserIntent

    steps: list[PlanStep] = Field(
        min_length=1,
        max_length=8,
    )

    requires_clarification: bool = False

    @field_validator("steps")
    @classmethod
    def validate_step_order(
        cls,
        steps: list[PlanStep],
    ) -> list[PlanStep]:

        ids = [step.step_id for step in steps]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "step_id values must be unique"
            )

        if steps[0].action not in {
            "search_catalog",
            "ask_clarification",
        }:
            raise ValueError(
                "the first planner step must be "
                "search_catalog or ask_clarification"
            )

        if any(
            step.action == "get_product"
            for step in steps
        ):
            search_positions = [
                i
                for i, step in enumerate(steps)
                if step.action == "search_catalog"
            ]

            if not search_positions:
                raise ValueError(
                    "get_product steps require a search_catalog step"
                )

        if (
            steps[0].action == "ask_clarification"
            and not any(
                step.action == "ask_clarification"
                for step in steps
            )
        ):
            raise ValueError(
                "clarification plan must contain ask_clarification"
            )

        return steps


# ---------------------------------------------------------------------------
# Product variants
# ---------------------------------------------------------------------------

class ProductVariant(BaseModel):
    model_config = ConfigDict(extra="ignore")

    options: dict[str, str] = Field(
        default_factory=dict
    )

    available: bool = True
    # False = the catalog did not state availability for this offer.  An unknown
    # offer is never recommended (ranking treats it as not verified available).
    # Default True keeps products built outside the executor, which pass
    # ``available`` explicitly, unchanged.
    availability_known: bool = True

    # Price in the variant's OWN currency (major units).
    price: Decimal | None = None
    currency: str | None = None

    # Which offer this variant belongs to.  The Global Catalog groups several
    # sellers under one product id, so price, seller and URL only make sense
    # together.  All optional: older payloads do not carry them.
    id: str | None = None
    title: str | None = None
    seller: str | None = None
    url: str | None = None

    @field_validator("options", mode="before")
    @classmethod
    def normalize_options(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return {}

        cleaned: dict[str, str] = {}

        for key, val in value.items():
            if key is None or val is None:
                continue

            key = str(key).strip().lower()
            val = str(val).strip()

            if key and val:
                cleaned[key] = val

        return cleaned


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

# Shown on a product whose details could not be loaded: its price comes from the
# search results only and never counts as a verified budget check.
PRICE_UNVERIFIED_NOTE = (
    "Price not verified: product details could not be loaded, "
    "so this comes from the search results only."
)

class Product(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str

    title: str = Field(
        min_length=1
    )

    url: HttpUrl

    price: Decimal | None = Field(
        default=None,
        ge=0,
    )

    # None = the catalog did not state a currency.  Never defaulted: an amount
    # with no currency cannot be compared with a budget.
    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    available: bool = True
    # False = the catalog did not state availability (see ProductVariant).  The
    # executor then also sets ``available`` False, so any consumer that only
    # reads ``available`` fails closed too.
    availability_known: bool = True

    merchant: str | None = None

    description: str = ""

    image_url: HttpUrl | None = None

    source: str

    variants: list[ProductVariant] = Field(
        default_factory=list
    )

    # Why the price is missing or uncertain, shown to the user.
    price_note: str | None = None

    # Set only by offer selection in ranking: the variant whose price, seller
    # and URL this product now carries.
    selected_variant_id: str | None = None

    # Where this product's data came from: the get_product call ("detail") or,
    # when that call failed or was skipped, the search results ("search_payload").
    # None = not stamped (products built outside the executor).
    detail_source: Literal["detail", "search_payload"] | None = None

    # Why get_product failed or was skipped (set only with detail_source ==
    # "search_payload"): rate_limited | timeout | service_error | tool_error |
    # invalid_result | budget_skipped.  None on runs saved before this field existed.
    detail_failure_kind: str | None = None

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        return value.upper() if value is not None else None


# ---------------------------------------------------------------------------
# Unverified candidates (products whose get_product call failed or was skipped)
# ---------------------------------------------------------------------------

UNVERIFIED_REASON_TEXT = {
    "rate_limited": "rate limited",
    "timeout": "timed out",
    "service_error": "service error",
    "tool_error": "rejected by the catalog",
    "invalid_result": "no usable details returned",
    "budget_skipped": "lookup skipped (tool-call limit)",
}


def unverified_reason_text(kind: str | None) -> str:
    """Short, human wording for a ``Product.detail_failure_kind``."""
    if not kind:
        return "reason not recorded"
    return UNVERIFIED_REASON_TEXT.get(kind, kind.replace("_", " "))


UNVERIFIED_CANDIDATE_NOTE = (
    "Product details could not be loaded, so the final offer, price and budget "
    "compliance were not verified. Price, currency and merchant come from the "
    "search results only."
)


class PayloadOffer(BaseModel):
    """One offer exactly as the search results listed it (never converted)."""

    price: Decimal | None = None
    currency: str | None = None
    seller: str | None = None
    available: bool | None = None


class UnverifiedCandidate(BaseModel):
    """A product kept for display only: it is never ranked with verified offers,
    never checked against the budget, and never counted as available in a currency."""

    product: Product

    verification_status: Literal["unverified_detail_failed", "unverified_detail_skipped"]
    unverified_reason: str
    reason_text: str = ""

    # Search-payload values, copied as listed.  candidate_price/currency are set only
    # when every priced payload offer shares one currency (or the payload carries a
    # single product-level price); otherwise both are None and listed_offers has them.
    candidate_price: Decimal | None = None
    candidate_currency: str | None = None
    candidate_merchant: str | None = None
    listed_offers: list[PayloadOffer] = Field(default_factory=list)

    # Whether the shown offer is proven to come from the requested merchant:
    # "not_requested" (no merchant filter), "verified" (an offer URL host belongs to
    # the merchant) or "unverified" (nothing contradicts it, nothing proves it; the
    # seller's own name never counts).  A contradicted merchant is dropped, not shown.
    merchant_status: Literal["not_requested", "verified", "unverified"] = "not_requested"

    # Always "unknown": an offer in the requested currency was not verified either way.
    requested_currency_status: Literal["unknown"] = "unknown"
    # True: a payload offer is in the requested currency.  False: the payload lists
    # other currencies only (NOT a claim that no such offer exists).  None: no priced
    # payload offers, or no currency was requested.
    payload_lists_requested_currency: bool | None = None

    budget_status: Literal["unverified"] = "unverified"
    size_status: Literal["unverified"] = "unverified"

    relevance: float = Field(default=0.0, ge=0, le=1)
    note: str = UNVERIFIED_CANDIDATE_NOTE


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

class StepExecution(BaseModel):
    step_id: str

    action: str

    status: Literal[
        "PLANNED",
        "RUNNING",
        "SUCCEEDED",
        "RETRYING",
        "FAILED",
        "TIMED_OUT",
        "SKIPPED",
        "WAITING_FOR_USER",
    ]

    error: str | None = None

    result_count: int = 0

    retries: int = 0


# ---------------------------------------------------------------------------
# Execution report
# ---------------------------------------------------------------------------

class ExecutionReport(BaseModel):
    plan: ExecutionPlan

    steps: list[StepExecution] = Field(
        default_factory=list
    )

    products: list[Product] = Field(
        default_factory=list
    )

    graph_iterations: int = 0

    tool_calls: int = 0

    diagnostics: dict[str, Any] = Field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Scored product
# ---------------------------------------------------------------------------

class ScoredProduct(BaseModel):
    product: Product

    score: float = Field(
        ge=0,
        le=1,
    )

    rejection_reason: str | None = None

    # M1.4 size handling:
    # available / unavailable / unknown
    size_status: Literal[
        "available",
        "unavailable",
        "unknown",
    ] = "unknown"


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    provider: str

    products: list[Product] = Field(
        default_factory=list
    )

    query: str

    error: str | None = None

    diagnostics: dict[str, Any] = Field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Discovery decision
# ---------------------------------------------------------------------------

class DiscoveryDecision(BaseModel):
    run_id: UUID = Field(
        default_factory=uuid4
    )

    status: Literal[
        "selected",
        "needs_selection",
        "clarify",
        "failed",
    ]

    provider: str

    candidates: list[ScoredProduct] = Field(
        default_factory=list
    )

    selected: ScoredProduct | None = None

    message: str

    attempts: int = Field(
        default=1,
        ge=1,
    )

    diagnostics: dict[str, Any] = Field(
        default_factory=dict
    )

    # Products whose details could not be loaded: shown separately, never ranked,
    # never a verified price or budget check.  Absent in runs saved before this field.
    unverified_candidates: list[UnverifiedCandidate] = Field(
        default_factory=list
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
