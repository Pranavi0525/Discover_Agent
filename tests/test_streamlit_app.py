"""Smoke tests for the M1.6 Streamlit page (streamlit.testing.v1.AppTest, no network)."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from discovery_agent import service  # noqa: E402
from discovery_agent.execution import ShopifyMcpExecutor  # noqa: E402
from discovery_agent.schemas import ExecutionPlan, PlanStep, UserIntent  # noqa: E402
from discovery_agent.storage import SqliteStore  # noqa: E402

from test_graph import FakeMcpSession, FakePlanner, _search_plan, _settings  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "discovery_agent" / "ui" / "streamlit_app.py")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DISCOVERY_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setattr(service, "planner_configured", lambda: True)
    settings = _settings()
    settings.ledger_path = tmp_path / "ledger.jsonl"
    return settings


def _patch_discover(monkeypatch, settings, plans, session=None):
    real = service.discover
    session = session or FakeMcpSession({"running shoes": [{"id": "p1"}, {"id": "p2"}]})
    planner = FakePlanner(plans)
    calls = []

    def fake(query, session_id, form_inputs=None, **kw):
        calls.append((query, dict(form_inputs or {})))
        executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
        return real(query, session_id, form_inputs, settings=settings, planner=planner, executor=executor)

    monkeypatch.setattr(service, "discover", fake)
    return calls


def _app(sid=None, run=None):
    at = st_testing.AppTest.from_file(APP, default_timeout=30)
    if sid:
        at.query_params["sid"] = sid
    if run:
        at.query_params["run"] = run
    return at


def _text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.error] + [w.value for w in at.warning]
    parts += [s.value for s in at.success] + [i.value for i in at.info]
    return "\n".join(parts)


def test_empty_state_renders_without_exceptions(env):
    at = _app().run()
    assert not at.exception
    assert any("Nothing here yet" in i.value for i in at.info)


def test_missing_api_key_shows_plain_message_and_disables_search(env, monkeypatch):
    monkeypatch.setattr(service, "planner_configured", lambda: False)
    at = _app().run()
    assert not at.exception
    assert any("GOOGLE_API_KEY" in e.value for e in at.error)
    assert at.button(key="submit").disabled


def test_search_creates_run_shows_products_steps_and_history(env, monkeypatch):
    calls = _patch_discover(monkeypatch, env, [_search_plan("running shoes")])
    at = _app().run()
    at.text_input(key="query").set_value("running shoes")
    at.number_input(key="max_price").set_value(100.0)
    at.selectbox(key="currency").set_value("USD")
    at.button(key="submit").click().run()

    assert not at.exception
    assert calls == [("running shoes", {"max_price": "100.0", "currency": "USD"})]
    assert at.session_state["run_id"] and at.session_state["session_id"]
    text = _text(at)
    assert "Run ID" in text and "selected" in text
    assert "Product p1" in text and "Product p2" in text
    assert any(b.key and b.key.startswith("hist_") for b in at.sidebar.button)
    assert any(b.key.startswith("up_") for b in at.button if b.key)
    assert any(x.label.startswith("Steps") for x in at.expander)

    sid = at.session_state["session_id"]
    store = SqliteStore(Path(env.ledger_path).parent / "agent.db")
    (row,) = store.list_runs(sid)
    assert row["status"] == "selected"
    assert store.get_step_events(row["run_id"])


def test_thumbs_up_writes_feedback(env, monkeypatch):
    _patch_discover(monkeypatch, env, [_search_plan("running shoes")])
    at = _app().run()
    at.text_input(key="query").set_value("running shoes")
    at.button(key="submit").click().run()
    run_id = at.session_state["run_id"]
    at.button(key=f"up_{run_id}_p1").click().run()
    assert [(f["product_id"], f["rating"]) for f in service.get_feedback(run_id)] == [("p1", 1)]


def test_history_reload_after_restart_makes_no_discover_call(env, monkeypatch):
    _patch_discover(monkeypatch, env, [_search_plan("running shoes")])
    at = _app().run()
    at.text_input(key="query").set_value("running shoes")
    at.button(key="submit").click().run()
    sid, run_id = at.session_state["session_id"], at.session_state["run_id"]

    def boom(*a, **k):
        raise AssertionError("history reload must not search")

    monkeypatch.setattr(service, "discover", boom)
    fresh = _app(sid=sid).run()  # a brand-new browser session, same sid in the URL
    assert not fresh.exception and fresh.session_state["run_id"] is None
    fresh.sidebar.button(key=f"hist_{run_id}").click().run()
    assert not fresh.exception
    assert fresh.session_state["run_id"] == run_id
    assert "Product p1" in _text(fresh)


class PartialSession(FakeMcpSession):
    async def call_tool(self, name, arguments):
        if name == "get_product" and arguments["catalog"]["id"] == "p2":
            raise RuntimeError("upstream 500")
        return await super().call_tool(name, arguments)


def test_partial_failure_banner(env, monkeypatch):
    env.max_retries_per_tool = 0
    _patch_discover(monkeypatch, env, [_search_plan("running shoes")],
                    session=PartialSession({"running shoes": [{"id": "p1"}, {"id": "p2"}]}))
    at = _app().run()
    at.text_input(key="query").set_value("running shoes")
    at.button(key="submit").click().run()
    assert not at.exception
    banner = "\n".join(w.value for w in at.warning)
    assert "could not be loaded" in banner and "upstream 500" in banner
    assert "Product p1" in _text(at) and "Product p2" not in _text(at)


def test_unverified_badge_and_assumptions(env, monkeypatch):
    plan = ExecutionPlan(
        intent=UserIntent(query="running shoes", delivery_city="Mumbai", return_required=True),
        steps=[PlanStep(step_id="s1", action="search_catalog", reason="find")],
    )
    _patch_discover(monkeypatch, env, [plan])
    at = _app().run()
    at.text_input(key="query").set_value("running shoes")
    at.button(key="submit").click().run()
    caps = [c for c in at.caption if c.value.startswith("🟠 Unverified")]
    assert any("delivery to Mumbai" in c.value for c in caps)
    assert all(c.help == service.__dict__.get("UNVERIFIED_HELP", c.help) for c in caps)
    assert "Not checkable in Phase 1" in caps[0].help


def test_clarification_flow_and_cap(env, monkeypatch):
    env.max_refinements = 1
    monkeypatch.setattr(service, "max_clarifications", lambda: 1)
    clarify = ExecutionPlan(
        intent=UserIntent(query="something"),
        steps=[PlanStep(step_id="s1", action="ask_clarification", reason="Which product?")],
        requires_clarification=True,
    )
    calls = _patch_discover(monkeypatch, env, [clarify, clarify])
    at = _app().run()
    at.text_input(key="query").set_value("something")
    at.button(key="submit").click().run()
    run_id = at.session_state["run_id"]
    assert any("Which product?" in w.value for w in at.warning)

    at.text_input(key=f"clarify_answer_{run_id}").set_value("shoes")
    at.button(key=f"clarify_submit_{run_id}").click().run()
    assert calls[1][0] == "something shoes" and calls[1][1]["clarify_round"] == 1
    assert at.session_state["run_id"] != run_id
    # cap reached: no more input box, just a plain message
    assert any("limit for follow-up" in i.value for i in at.info)


def test_error_run_shows_plain_message(env, monkeypatch):
    class Broken:
        def plan(self, query):
            raise RuntimeError("Gemini quota exceeded")

    real = service.discover
    monkeypatch.setattr(service, "discover", lambda q, s, f=None: real(
        q, s, f, settings=env, planner=Broken(),
        executor=ShopifyMcpExecutor(session_factory=lambda: FakeMcpSession({}), settings=env)))
    at = _app().run()
    at.text_input(key="query").set_value("running shoes")
    at.button(key="submit").click().run()
    assert not at.exception
    assert any("Gemini quota exceeded" in e.value for e in at.error)
    assert "Traceback" not in _text(at)
