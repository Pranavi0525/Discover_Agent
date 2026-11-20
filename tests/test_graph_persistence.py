import asyncio
import json

import pytest

from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.graph import WorkflowError, arun_discovery_workflow, compute_partial
from discovery_agent.schemas import DiscoveryDecision, ExecutionPlan, PlanStep, UserIntent
from discovery_agent.storage import SqliteStore

from test_graph import FakeMcpSession, FakePlanner, _search_plan, _settings


@pytest.fixture()
def store(tmp_path):
    return SqliteStore(tmp_path / "agent.db")


def _run(query, planner, session, settings, store, **kwargs):
    executor = ShopifyMcpExecutor(session_factory=lambda: session, settings=settings)
    return asyncio.run(
        arun_discovery_workflow(
            query, planner=planner, executor=executor, settings=settings, store=store, **kwargs
        )
    )


def test_success_persists_run_events_and_round_trips_decision(store):
    settings = _settings()
    session = FakeMcpSession({"running shoes": [{"id": "p1"}]})
    sid = store.create_session()
    state = _run("running shoes", FakePlanner([_search_plan("running shoes")]), session, settings,
                 store, session_id=sid, form_inputs={"max_price": "5000", "currency": "USD"})

    run = store.get_run(state["run_id"])
    assert run["status"] == "selected" == state["decision"].status
    assert run["partial"] is False
    assert run["session_id"] == sid
    assert json.loads(run["form_inputs_json"]) == {"max_price": "5000", "currency": "USD"}
    assert run["plan_json"] and run["intent_json"]
    # decision.run_id is the persisted run id (JSONL and SQLite share one ID)
    decision = DiscoveryDecision.model_validate_json(run["decision_json"])
    assert str(decision.run_id) == state["run_id"]
    assert decision.selected.product.id == "p1"

    events = store.get_step_events(state["run_id"])
    steps = [(e["attempt"], e["step"]) for e in events]
    for node in ("plan", "validate", "execute", "decide"):
        assert (1, node) in steps
    detail = [e for e in events if e["step"].startswith("get_product")]
    assert detail and detail[0]["latency_ms"] is None
    assert json.loads(detail[0]["payload_json"])["result_count"] == 1
    assert all(e["latency_ms"] is not None for e in events if e["step"] in {"plan", "execute"})


def test_no_store_leaves_state_free_of_run_ids():
    settings = _settings()
    session = FakeMcpSession({"running shoes": [{"id": "p1"}]})
    state = _run("running shoes", FakePlanner([_search_plan("running shoes")]), session, settings, None)
    assert "run_id" not in state and state["decision"].status == "selected"


def test_refinement_logs_events_for_attempts_one_and_two(store):
    settings = _settings()
    settings.max_refinements = 1
    settings.max_graph_iterations = 2
    session = FakeMcpSession({"blue running shoes": [{"id": "p1"}]})
    planner = FakePlanner([_search_plan("blue running shoes size 9"), _search_plan("blue running shoes")])
    state = _run("blue running shoes size 9", planner, session, settings, store)

    events = store.get_step_events(state["run_id"])
    assert {e["attempt"] for e in events} == {1, 2}
    assert ("refine" in [e["step"] for e in events if e["attempt"] == 1])
    assert [e["step"] for e in events if e["attempt"] == 2][:2] == ["plan", "validate"]
    assert store.get_run(state["run_id"])["status"] == state["decision"].status


class PartialFailSession(FakeMcpSession):
    async def call_tool(self, name, arguments):
        if name == "get_product" and arguments["catalog"]["id"] == "p2":
            raise RuntimeError("upstream 500")
        return await super().call_tool(name, arguments)


def test_partial_failure_keeps_surviving_products(store):
    settings = _settings()
    settings.max_retries_per_tool = 0
    session = PartialFailSession({"running shoes": [{"id": "p1"}, {"id": "p2"}]})
    state = _run("running shoes", FakePlanner([_search_plan("running shoes")]), session, settings, store)

    run = store.get_run(state["run_id"])
    assert run["partial"] is True
    assert compute_partial(state["report"]) is True
    decision = DiscoveryDecision.model_validate_json(run["decision_json"])
    assert {c.product.id for c in decision.candidates} == {"p1"}
    failed = [e for e in store.get_step_events(state["run_id"])
              if e["status"] in {"FAILED", "TIMED_OUT"}]
    assert failed and "upstream 500" in (failed[0]["error"] or "")


def test_planner_error_is_saved_before_the_exception_is_raised(store):
    settings = _settings()

    class Broken:
        def plan(self, query):
            raise RuntimeError("Gemini quota exceeded")

    sid = store.create_session()
    with pytest.raises(WorkflowError, match="Gemini quota exceeded"):
        _run("running shoes", Broken(), FakeMcpSession({}), settings, store, session_id=sid)

    (row,) = store.list_runs(sid)
    run = store.get_run(row["run_id"])
    assert run["status"] == "error" and "Gemini quota exceeded" in run["error"]
    events = store.get_step_events(row["run_id"])
    assert [(e["step"], e["status"]) for e in events] == [("guard", "ok"), ("plan", "error")]


def test_clarification_run_is_persisted(store):
    settings = _settings()
    plan = ExecutionPlan(
        intent=UserIntent(query="something"),
        steps=[PlanStep(step_id="s1", action="ask_clarification", reason="Too vague")],
        requires_clarification=True,
    )
    state = _run("something", FakePlanner([plan]), FakeMcpSession({}), settings, store)
    run = store.get_run(state["run_id"])
    assert run["status"] == "clarify"
    assert DiscoveryDecision.model_validate_json(run["decision_json"]).message == "Too vague"


def test_a_broken_database_does_not_break_the_run(store, monkeypatch):
    settings = _settings()
    monkeypatch.setattr(store, "log_step", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    session = FakeMcpSession({"running shoes": [{"id": "p1"}]})
    state = _run("running shoes", FakePlanner([_search_plan("running shoes")]), session, settings, store)
    assert state["decision"].status == "selected"
