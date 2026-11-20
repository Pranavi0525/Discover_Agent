import pytest

from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.planner import PlannerConfigurationError
from discovery_agent.storage import SqliteStore
from discovery_agent import service

from test_graph import FakeMcpSession, FakePlanner, _search_plan, _settings


@pytest.fixture()
def env(tmp_path):
    settings = _settings()
    settings.ledger_path = tmp_path / "ledger.jsonl"
    store = SqliteStore(tmp_path / "agent.db")
    return settings, store, store.create_session()


def _executor(session, settings):
    return ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)


def test_discover_success_then_load_run_matches(env):
    settings, store, sid = env
    session = FakeMcpSession({"running shoes": [{"id": "p1"}]})
    result = service.discover(
        "running shoes", sid, {"max_price": "5000", "currency": "USD"}, store=store, settings=settings,
        planner=FakePlanner([_search_plan("running shoes")]), executor=_executor(session, settings),
    )
    assert result.status == "selected" and result.run_id and result.error is None
    assert settings.ledger_path.exists()  # JSONL tracker still runs alongside SQLite

    loaded = service.load_run(result.run_id, store=store)
    assert loaded.status == "selected"
    assert loaded.decision.model_dump() == result.decision.model_dump()
    assert service.load_run_meta(result.run_id, store=store)["form_inputs"] == {"max_price": "5000", "currency": "USD"}
    assert [r["run_id"] for r in service.list_history(sid, store=store)] == [result.run_id]
    assert any(e["step"] == "decide" for e in service.get_steps(result.run_id, store=store))


def test_load_run_makes_no_network_calls(env):
    settings, store, sid = env
    result = service.discover(
        "running shoes", sid, store=store, settings=settings,
        planner=FakePlanner([_search_plan("running shoes")]),
        executor=_executor(FakeMcpSession({"running shoes": [{"id": "p1"}]}), settings),
    )
    # A fresh SqliteStore = an app restart; nothing else is available to load_run.
    loaded = service.load_run(result.run_id, store=SqliteStore(store.path))
    assert loaded.decision.selected.product.id == "p1"


def test_load_run_unknown_id_is_none(env):
    assert service.load_run("nope", store=env[1]) is None


def test_planner_exception_returns_error_result_and_saves_it(env):
    settings, store, sid = env

    class Broken:
        def plan(self, query):
            raise RuntimeError("Gemini quota exceeded")

    result = service.discover("running shoes", sid, store=store, settings=settings,
                              planner=Broken(), executor=_executor(FakeMcpSession({}), settings))
    assert result.status == "error" and "Gemini quota exceeded" in result.error
    assert "Traceback" not in result.error
    saved = service.load_run(result.run_id, store=store)
    assert saved.status == "error" and "Gemini quota exceeded" in saved.error


def test_missing_api_key_gives_plain_message_and_a_run_row(env, monkeypatch):
    settings, store, sid = env
    settings.google_api_key = None

    def boom(self, s=None):
        raise PlannerConfigurationError("GOOGLE_API_KEY is not set")

    monkeypatch.setattr("discovery_agent.graph.GeminiPlanner.__init__", boom)
    result = service.discover("running shoes", sid, store=store, settings=settings)
    assert result.status == "error" and "GOOGLE_API_KEY" in result.error
    assert service.load_run(result.run_id, store=store).status == "error"


def test_blank_query_is_an_error_result_not_an_exception(env):
    settings, store, sid = env
    result = service.discover("  ", sid, store=store, settings=settings)
    assert result.status == "error" and result.run_id is None


def test_unverified_constraints_are_surfaced(env):
    settings, store, sid = env
    from discovery_agent.schemas import ExecutionPlan, PlanStep, UserIntent

    plan = ExecutionPlan(
        intent=UserIntent(query="running shoes", delivery_city="Mumbai", return_required=True),
        steps=[PlanStep(step_id="s1", action="search_catalog", reason="find")],
    )
    result = service.discover(
        "running shoes", sid, store=store, settings=settings, planner=FakePlanner([plan]),
        executor=_executor(FakeMcpSession({"running shoes": [{"id": "p1"}]}), settings),
    )
    assert {u["name"] for u in result.unverified} >= {"delivery_city"}
    assert service.load_run(result.run_id, store=store).unverified == result.unverified


def test_feedback_wrappers(env):
    settings, store, sid = env
    run_id = store.create_run(sid, "q")
    service.record_feedback(run_id, "p1", 1, store=store)
    assert service.get_feedback(run_id, store=store)[0]["rating"] == 1
