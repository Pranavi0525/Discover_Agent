import pytest
from pydantic import ValidationError

from discovery_agent.schemas import UserIntent


def test_intent_normalizes_currency_and_rejects_extra_fields():
    intent = UserIntent(query="running shoes", currency="usd")
    assert intent.currency == "USD"
    with pytest.raises(ValidationError):
        UserIntent(query="running shoes", unexpected="value")
