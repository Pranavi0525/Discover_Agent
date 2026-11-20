

"""The M1.4 eval set: 30 messy shopping requests and what the intent must be.

How to read ``expect``
----------------------
A key that is ABSENT is not scored for that case.  A key that is present is
scored strictly, so ``"unsupported": []`` means "Gemini must not invent any
extra constraint".

* ``query_include`` / ``query_exclude``  words that must / must not appear in
  ``intent.query`` (case-insensitive substring).  This checks that constraints
  moved out of the query without freezing the exact wording Gemini picks.
* ``max_price``       Decimal-comparable number or None.
* ``currency``        ISO code or None.
* ``merchant``        None, or a substring the merchant must contain.
* ``delivery_city``   canonical city name or None.
* ``return_required`` bool.
* ``unsupported``     list of keyword groups.  Every group must be matched by
  a distinct produced item (any keyword, case-insensitive substring) and no
  produced item may be left over.
* ``clarify``         bool, ``plan.requires_clarification``.

``settings`` overrides the eval defaults (market currency INR) per case.
``critical`` cases must be fully correct for the run to pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    expect: dict[str, Any]
    settings: dict[str, Any] = field(default_factory=dict)
    critical: bool = False
    note: str = ""


def _surfaced(**extra: Any) -> dict[str, Any]:
    """Like _plain, but the constraint may live in attributes/size/unsupported:
    it must be SHOWN to the user as unverified (never silently dropped)."""
    base = _plain(**extra)
    base.pop("unsupported", None)
    return base


def _plain(**extra: Any) -> dict[str, Any]:
    """Expectation for a request with no delivery, return or extra constraints."""
    base: dict[str, Any] = {
        "merchant": None,
        "delivery_city": None,
        "return_required": False,
        "unsupported": [],
        "clarify": False,
    }
    base.update(extra)
    return base


CASES: tuple[EvalCase, ...] = (
    # --- clean baselines -------------------------------------------------
    EvalCase("basic_no_budget", "wireless earbuds",
             _plain(query_include=["earbuds"], max_price=None)),
    EvalCase("symbol_rupee", "running shoes under ₹8000",
             _plain(query_include=["running", "shoes"], query_exclude=["8000", "under", "₹"],
                    max_price=8000, currency="INR")),
    EvalCase("rs_dot_comma", "cotton kurta below Rs. 1,500",
             _plain(query_include=["kurta"], query_exclude=["1,500", "below", "rs"],
                    max_price=1500, currency="INR")),
    EvalCase("k_suffix", "gaming mouse within 2k INR",
             _plain(query_include=["mouse"], query_exclude=["2k", "within"],
                    max_price=2000, currency="INR")),
    EvalCase("dollars_symbol", "yoga mat up to $40",
             _plain(query_include=["yoga", "mat"], query_exclude=["$", "up to"],
                    max_price=40, currency="USD")),
    EvalCase("pounds_word", "leather wallet max 60 pounds",
             _plain(query_include=["wallet"], query_exclude=["60", "pounds"],
                    max_price=60, currency="GBP")),
    EvalCase("euro_symbol_for", "ceramic coffee mug for €15",
             _plain(query_include=["mug"], query_exclude=["€", "15"],
                    max_price=15, currency="EUR")),
    EvalCase("rupees_word_for", "steel water bottle for 500 rupees",
             _plain(query_include=["bottle"], query_exclude=["500", "rupees"],
                    max_price=500, currency="INR")),
    # --- currency handling ----------------------------------------------
    EvalCase("no_currency_uses_market", "bluetooth speaker under 3000",
             _plain(query_include=["speaker"], query_exclude=["3000"],
                    max_price=3000, currency="INR"),
             note="Bare number: the market default (INR) is applied, never USD."),
    EvalCase("no_currency_no_market_asks", "laptop stand under 1200",
             _plain(query_include=["laptop", "stand"], max_price=1200, currency=None,
                    clarify=True),
             settings={"default_currency": None},
             note="No stated currency and no market default: must ask, not guess."),
    EvalCase("minimum_is_not_maximum", "office chair above 5000 rupees",
             _plain(query_include=["chair"], query_exclude=["5000"], max_price=None,
                    unsupported=[["5000", "above", "minimum", "more than"]]),
             note="'above' is a minimum. It must not become max_price and must not vanish."),
    # --- delivery city ---------------------------------------------------
    EvalCase("mumbai_typo_full", "replaceable phone case under 500 rupees deliverable in mumbia",
             _plain(query_include=["phone", "case"],
                    query_exclude=["mumbia", "mumbai", "replaceable", "deliverable"],
                    max_price=500, currency="INR", delivery_city="Mumbai",
                    return_required=True),
             critical=True,
             note="The roadmap's Mumbai query: every constraint lands in its own field."),
    EvalCase("city_bangalore_alias", "table lamp, deliver to Bangalore",
             _plain(query_include=["lamp"], query_exclude=["bangalore", "deliver"],
                    delivery_city="Bengaluru")),
    EvalCase("city_ship_to_pune", "hiking backpack ship to Pune under $80",
             _plain(query_include=["backpack"], query_exclude=["pune", "ship"],
                    max_price=80, currency="USD", delivery_city="Pune")),
    EvalCase("city_new_delhi", "study desk deliver to new delhi",
             _plain(query_include=["desk"], query_exclude=["delhi", "deliver"],
                    delivery_city="Delhi")),
    EvalCase("city_not_delivery_madras", "madras check cotton shirt under 1200 rupees",
             _plain(query_include=["shirt"], max_price=1200, currency="INR",
                    delivery_city=None),
             note="'madras' is a fabric here, not a delivery city (alias of Chennai)."),
    # --- returns ---------------------------------------------------------
    EvalCase("returnable", "returnable sneakers under 4000 rupees",
             _plain(query_include=["sneakers"], query_exclude=["returnable"],
                    max_price=4000, currency="INR", return_required=True)),
    EvalCase("exchange", "formal shirt with easy exchange",
             _plain(query_include=["shirt"], query_exclude=["exchange"],
                    return_required=True)),
    EvalCase("money_back", "noise cancelling headphones with money back guarantee under $150",
             _plain(query_include=["headphones"], query_exclude=["money back", "guarantee"],
                    max_price=150, currency="USD", return_required=True)),
    EvalCase("negated_return", "non returnable clearance jacket",
             {"query_include": ["jacket"], "delivery_city": None, "return_required": False,
              "clarify": False},
             note="'non returnable' must not switch return_required on."),
    # --- constraints the catalog cannot answer --------------------------
    EvalCase("warranty", "air fryer with 2 year warranty under 6000 rupees",
             _surfaced(query_include=["air", "fryer"], query_exclude=["warranty"],
                       max_price=6000, currency="INR", surfaced=[["warranty"]])),
    EvalCase("brand_exclusion", "white sneakers, no nike, under 3500 rupees",
             _plain(query_include=["sneakers"], query_exclude=["nike"],
                    max_price=3500, currency="INR", unsupported=[["nike"]])),
    EvalCase("two_unsupported", "organic cotton bedsheet, machine washable, made in India, under Rs 2500",
             _surfaced(query_include=["bedsheet"], query_exclude=["washable", "made in"],
                       max_price=2500, currency="INR",
                       surfaced=[["washable"], ["india", "made in"]])),
    # --- merchant --------------------------------------------------------
    EvalCase("merchant_from", "sony headphones from Amazon",
             _plain(query_include=["headphones"], query_exclude=["amazon"],
                    merchant="amazon")),
    EvalCase("merchant_and_price", "nike air max from nike.com under $120",
             _plain(query_include=["air", "max"], max_price=120, currency="USD",
                    merchant="nike")),
    # --- vague / messy ---------------------------------------------------
    EvalCase("vague_gift_sister", "something nice for my sister",
             {"clarify": True, "max_price": None, "delivery_city": None,
              "return_required": False}),
    EvalCase("vague_gift", "I need a gift",
             {"clarify": True, "max_price": None, "delivery_city": None,
              "return_required": False}),
    EvalCase("typos_k_rs", "cheep bluetooth speeker undr 2k rs",
             {"query_include": ["bluetooth"], "query_exclude": ["undr", "2k"],
              "max_price": 2000, "currency": "INR", "delivery_city": None,
              "return_required": False, "clarify": False}),
    EvalCase("long_rambling", "hey so im looking for a pair of running shoes for my dad, hes 60, size 9, "
                              "needs to be returnable... actually deliver to chennai, under 5000 rupees pls",
             _surfaced(query_include=["running", "shoes"], query_exclude=["chennai", "returnable", "5000"],
                       max_price=5000, currency="INR", delivery_city="Chennai",
                       return_required=True, surfaced=[["size 9"]]),
             note="Age is not a constraint; shoe size is, and has no field."),
    EvalCase("all_four", "cotton saree deliver to Kolkata under 3000 rupees returnable",
             _plain(query_include=["saree"], query_exclude=["kolkata", "returnable", "3000"],
                    max_price=3000, currency="INR", delivery_city="Kolkata",
                    return_required=True)),
)

assert len(CASES) == 30, len(CASES)
assert len({c.id for c in CASES}) == len(CASES)
