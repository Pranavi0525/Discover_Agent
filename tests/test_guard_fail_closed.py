from types import SimpleNamespace

from discovery_agent import guard as guard_module
from discovery_agent.guard import build_guard


def _settings(**over):
    base = dict(
        guard_enabled=True, guard_classifier="promptguard-86m", guard_max_chars=1000,
        guard_block_threshold=0.90, guard_review_threshold=0.50, guard_fail_closed=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_missing_classifier_blocks_when_fail_closed(monkeypatch):
    monkeypatch.setattr(guard_module, "build_classifier", lambda choice: None)
    result = build_guard(_settings()).check("blue shirts under 2000 rupees")
    assert result.blocked
    assert "classifier_unavailable" in result.reasons


def test_missing_classifier_still_allows_when_fail_open(monkeypatch):
    monkeypatch.setattr(guard_module, "build_classifier", lambda choice: None)
    result = build_guard(_settings(guard_fail_closed=False)).check("blue shirts under 2000 rupees")
    assert result.verdict == "allow"


def test_classifier_none_is_a_deliberate_choice_and_is_not_blocked():
    result = build_guard(_settings(guard_classifier="none")).check("blue shirts under 2000 rupees")
    assert result.verdict == "allow"
