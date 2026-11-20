"""A budget of "null" from the LLM means no budget; real and invalid budgets behave as before."""
from decimal import Decimal

import pytest
from pydantic import ValidationError

from discovery_agent.planner import _LenientIntent
from discovery_agent.schemas import UserIntent


@pytest.mark.parametrize("word", ["null", "None", " N/A ", "unknown", "", "-"])
def test_nullish_budget_means_not_stated(word):
    assert UserIntent(query="shoes", max_price=word).max_price is None
    assert _LenientIntent(query="shoes", max_price=word).max_price is None


def test_real_budgets_are_kept():
    assert UserIntent(query="shoes", max_price="1500").max_price == Decimal("1500")
    assert UserIntent(query="shoes", max_price=Decimal("499.50")).max_price == Decimal("499.50")


@pytest.mark.parametrize("bad", ["-5", "cheap", -1])
def test_invalid_budgets_are_still_rejected(bad):
    with pytest.raises(ValidationError):
        UserIntent(query="shoes", max_price=bad)
