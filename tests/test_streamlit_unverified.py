"""Streamlit page: verified results and unverified candidates are separate sections.

Same approach as test_streamlit_app.py (AppTest, no network); Shopify is the repo's
ScriptedSession and the planner is a local fake.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from ucp_fakes import ScriptedSession, ucp_product  # noqa: E402
from discovery_agent import service  # noqa: E402
from discovery_agent.config import Settings  # noqa: E402
from discovery_agent.execution import ShopifyMcpExecutor  # noqa: E402
from discovery_agent.schemas import ExecutionPlan, PlanStep, UserIntent  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "discovery_agent" / "ui" / "streamlit_app.py")


class _Planner:
    def plan(self, text):
        intent = UserIntent(query="running shoes", currency="INR", max_price=Decimal("4000"))
        return ExecutionPlan(intent=intent, steps=[PlanStep(step_id="s1", action="search_catalog", reason="r")])


class _MixedSession(ScriptedSession):
    async def call_tool(self, name, arguments):
        if name == "get_product" and arguments["catalog"]["id"].endswith(("prod2", "prod3")):
            raise RuntimeError("upstream 500")
        return await super().call_tool(name, arguments)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DISCOVERY_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setattr(service, "planner_configured", lambda: True)
    settings = Settings()
    settings.ledger_path = tmp_path / "ledger.jsonl"
    settings.max_retries_per_tool = 0
    settings.shopify_buyer_country = None
    return settings


def _search(monkeypatch, settings, session):
    real = service.discover

    def fake(query, session_id, form_inputs=None, **kw):
        executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
        return real(query, session_id, form_inputs, settings=settings, planner=_Planner(), executor=executor)

    monkeypatch.setattr(service, "discover", fake)
    at = st_testing.AppTest.from_file(APP, default_timeout=30).run()
    at.text_input(key="query").set_value("running shoes")
    at.button(key="submit").click().run()
    assert not at.exception
    return at


def _subheaders(at):
    return [s.value for s in at.subheader]


def _warning(at):
    return "\n".join(w.value for w in at.warning)


def test_unverified_only_result_has_its_own_section_and_no_verified_section(env, monkeypatch):
    products = [ucp_product(i, amount=300000) for i in range(1, 6)]
    at = _search(monkeypatch, env, ScriptedSession(products, fail_details=True))

    heads = _subheaders(at)
    assert "Unverified candidates (3)" in heads  # capped at three
    assert not any(h.startswith("Results (") for h in heads)
    assert len([c for c in at.caption if c.value.startswith("🟠 Not verified")]) == 3
    assert not any(b.key and b.key.startswith(("up_", "down_")) for b in at.button)  # no feedback on them
    assert any("not verified" in i.value for i in at.info)


def test_partial_banner_counts_what_is_displayed_not_what_was_fetched(env, monkeypatch):
    products = [ucp_product(i, amount=300000) for i in range(1, 6)]
    at = _search(monkeypatch, env, ScriptedSession(products, fail_details=True))

    banner = _warning(at)
    assert "could not be loaded" in banner
    assert "Showing 0 verified results and 3 unverified candidates (listed separately)." in banner
    assert "2 other products with search-result data only were not shown" in banner  # 5 fetched - 3 shown


def test_verified_and_unverified_sections_are_separate_and_the_banner_matches(env, monkeypatch):
    products = [ucp_product(i, amount=300000) for i in range(1, 4)]
    at = _search(monkeypatch, env, _MixedSession(products))

    heads = _subheaders(at)
    assert "Results (1)" in heads and "Unverified candidates (2)" in heads
    assert heads.index("Results (1)") < heads.index("Unverified candidates (2)")
    assert "Showing 1 verified result and 2 unverified candidates (listed separately)." in _warning(at)
    assert any(b.key and b.key.startswith("up_") for b in at.button)  # feedback only on the verified card
