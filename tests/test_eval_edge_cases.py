"""Edge cases for the eval scoring and the deterministic readers (offline, no Gemini)."""
from decimal import Decimal

import pytest

from discovery_agent.attributes import extract_min_rating, extract_size
from discovery_agent.budget import extract_budget
from discovery_agent.constraints import reconcile_all
from discovery_agent.eval.dataset import CASES
from discovery_agent.eval.scoring import score_case
from discovery_agent.merchant_identity import merchant_match
from discovery_agent.schemas import ExecutionPlan, PlanStep, UserIntent


def _plan(**intent) -> ExecutionPlan:
    intent.setdefault("query", "air fryer")
    return ExecutionPlan(
        intent=intent,
        steps=[PlanStep(step_id="1", action="search_catalog", reason="search")],
    )


def _case(case_id):
    return next(c for c in CASES if c.id == case_id)


# ---- the three formerly failing eval cases ---------------------------------
@pytest.mark.parametrize("where", [
    {"attributes": [{"name": "warranty", "value": "2 year"}]},
    {"unsupported_constraints": ["2 year warranty"]},
])
def test_warranty_is_accepted_wherever_the_planner_files_it(where):
    plan = _plan(max_price=Decimal(6000), currency="INR", **where)
    assert score_case("warranty", _case("warranty").expect, plan).perfect


def test_warranty_silently_dropped_still_fails():
    plan = _plan(max_price=Decimal(6000), currency="INR")
    assert not score_case("warranty", _case("warranty").expect, plan).perfect


def test_two_unsupported_needs_both_constraints_shown():
    expect = _case("two_unsupported").expect
    both = _plan(query="organic cotton bedsheet", max_price=Decimal(2500), currency="INR",
                 attributes=[{"name": "care", "value": "machine washable"}],
                 unsupported_constraints=["made in India"])
    assert score_case("two", expect, both).perfect
    only_one = _plan(query="organic cotton bedsheet", max_price=Decimal(2500), currency="INR",
                     unsupported_constraints=["made in India"])
    assert not score_case("two", expect, only_one).perfect


def test_long_rambling_size_9_is_read_into_the_size_field_and_shown():
    text = ("hey so im looking for a pair of running shoes for my dad, hes 60, size 9, "
            "needs to be returnable... actually deliver to chennai, under 5000 rupees pls")
    intent, _ = reconcile_all(UserIntent(query="running shoes"), text)
    assert intent.size == "9" and intent.return_required and intent.delivery_city == "Chennai"
    assert intent.max_price == Decimal(5000) and intent.currency == "INR"
    plan = ExecutionPlan(intent=intent,
                         steps=[PlanStep(step_id="1", action="search_catalog", reason="s")])
    assert score_case("long", _case("long_rambling").expect, plan).perfect


def test_other_cases_still_reject_invented_constraints():
    assert not score_case("brand", _case("basic_no_budget").expect,
                          _plan(query="phone case", unsupported_constraints=["gold"])).perfect


# ---- size ---------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("size 9", "9"), ("size uk 9", "UK 9"), ("uk9", "UK 9"), ("size M", "M"),
    ("size: xl", "XL"), ("size 10.5", "10.5"), ("any size", "any"), ("no size", "any"),
    ("m", None), ("size 100", None), ("size matters", None), ("", None),
])
def test_size_edge_cases(text, expected):
    assert extract_size(text) == expected


# ---- budget: never guess a currency, never accept zero or ranges ---------------
@pytest.mark.parametrize("text,amount,currency", [
    ("under 1,500 rupees", 1500, "INR"), ("under Rs 2500", 2500, "INR"),
    ("under $120", 120, "USD"), ("under rs. 1,50,000", 150000, "INR"),
    ("under 1500 INR", 1500, "INR"),
])
def test_budget_reads_explicit_amounts(text, amount, currency):
    b = extract_budget(text)
    assert b and b.amount == Decimal(amount) and b.currency == currency


@pytest.mark.parametrize("text", [
    "no budget", "under 0 rupees", "between 500 and 800", "5k budget", "for 20k", "",
])
def test_budget_refuses_to_guess(text):
    assert extract_budget(text) is None


# ---- rating ----------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("4 stars", 4.0), ("rated 4.5", 4.5), ("rating above 3", 3.0),
    ("rating 7", None), ("0 stars", None), ("no rating", None),
])
def test_rating_edge_cases(text, expected):
    assert extract_min_rating(text) == expected


# ---- merchant identity: the URL host is the only proof --------------------------
@pytest.mark.parametrize("requested,seller,urls,verdict", [
    ("nike.com", "x", ["https://www.nike.com/p"], "verified"),
    ("nike.com", "x", ["https://in.nike.com/p"], "verified"),
    ("nike.com", "Nike", ["https://nike.com.evil.com/p"], "mismatch"),
    ("Nike", "Nike", ["https://evilnike.com/p"], "unverified"),
    ("Nike", "Nike", ["https://xn--nke-9la.com/p"], "unverified"),  # name alone proves nothing
    ("Nike", "Other", ["https://xn--nke-9la.com/p"], "mismatch"),
    ("Nike", "Nike", ["ftp://nike.com/p"], "unverified"),
    ("Nike", "Nike", [], "unverified"),
    ("M", "M", ["https://m.com/p"], "unverified"),
    ("H&M", "H&M", ["https://www2.hm.com/p"], "verified"),
    ("Marks & Spencer", "x", ["https://marksandspencer.com/p"], "verified"),
    ("", None, [], "verified"),
])
def test_merchant_identity_edge_cases(requested, seller, urls, verdict):
    assert merchant_match(requested, seller, urls) == verdict
