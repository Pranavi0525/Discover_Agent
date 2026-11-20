"""Test defaults.

The category clarification (ask for size/gender before searching shoes or
clothes) is ON in the app.  Many older graph/UI tests use shoe queries to
exercise other behaviour, so it is OFF here unless a test turns it on
(``settings.clarify_missing_attributes = True``, see ``tests/test_m14_attributes.py``).
"""

import pytest

@pytest.fixture(autouse=True)
def _no_real_guard_model(monkeypatch):
    monkeypatch.setenv("GUARD_CLASSIFIER", "none")

@pytest.fixture(autouse=True)
def _no_attribute_clarification_by_default(monkeypatch):
    monkeypatch.setenv("CLARIFY_MISSING_ATTRIBUTES", "0")

@pytest.fixture(autouse=True)
def _lexical_relevance_scorer_by_default(monkeypatch):
 monkeypatch.setenv("DISCOVERY_RELEVANCE_SCORER", "lexical")
