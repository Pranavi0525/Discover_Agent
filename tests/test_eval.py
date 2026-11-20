from decimal import Decimal

import pytest

from discovery_agent.eval.cache import BatchLimitReached, CacheMiss, CachingModel, ResponseCache, cache_key
from discovery_agent.eval.dataset import CASES, EvalCase
from discovery_agent.eval.runner import run_eval
from discovery_agent.eval.scoring import _match_unsupported, score_case
from discovery_agent.planner import _LenientExecutionPlan
from discovery_agent.schemas import ExecutionPlan, PlanStep


def _plan(**intent) -> ExecutionPlan:
    intent.setdefault("query", "phone case")
    return ExecutionPlan(
        intent=intent,
        steps=[PlanStep(step_id="1", action="search_catalog", reason="search")],
    )


# ---- dataset --------------------------------------------------------------
def test_dataset_has_30_unique_cases_and_the_mumbai_case_is_critical():
    assert len(CASES) == 30
    assert len({c.id for c in CASES}) == 30
    mumbai = next(c for c in CASES if c.id == "mumbai_typo_full")
    assert mumbai.critical and mumbai.expect["delivery_city"] == "Mumbai"


# ---- scoring --------------------------------------------------------------
def test_query_must_include_and_exclude():
    expect = {"query_include": ["case"], "query_exclude": ["mumbai"]}
    assert score_case("x", expect, _plan(query="Phone Case")).perfect
    assert not score_case("x", expect, _plan(query="phone case mumbai")).perfect
    assert not score_case("x", expect, _plan(query="phone cover")).perfect


def test_absent_keys_are_not_scored_present_keys_are_strict():
    assert score_case("x", {}, _plan()).total == 0
    got = score_case("x", {"unsupported": []}, _plan(unsupported_constraints=["warranty"]))
    assert not got.perfect  # an invented constraint is a failure


def test_price_currency_city_return_clarify():
    plan = _plan(max_price=Decimal("500"), currency="INR", delivery_city="Mumbai", return_required=True)
    expect = {"max_price": 500, "currency": "INR", "delivery_city": "Mumbai",
              "return_required": True, "clarify": False}
    assert score_case("x", expect, plan).perfect
    assert not score_case("x", {**expect, "max_price": 5000}, plan).perfect


def test_merchant_none_and_substring():
    assert score_case("x", {"merchant": None}, _plan()).perfect
    assert not score_case("x", {"merchant": None}, _plan(merchant="Amazon")).perfect
    assert score_case("x", {"merchant": "amazon"}, _plan(merchant="Amazon India")).perfect


def test_unsupported_needs_distinct_match_per_group_and_no_leftovers():
    groups = [["washable"], ["india", "made in"]]
    assert _match_unsupported(groups, ["machine washable", "Made in India"])
    assert not _match_unsupported(groups, ["machine washable"])
    assert not _match_unsupported(groups, ["machine washable", "made in india", "extra"])
    assert not _match_unsupported([["a"], ["a"]], ["a b", "c"])  # one item can't satisfy two groups
    assert _match_unsupported([], [])


# ---- cache ----------------------------------------------------------------
class _FakeInner:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return _LenientExecutionPlan.model_validate(
            {"intent": {"query": "phone case", "max_price": "500", "currency": "INR"},
             "steps": [{"step_id": "1", "action": "search_catalog", "reason": "r"}]}
        )


def _model(tmp_path, inner, **kw):
    cache = ResponseCache(tmp_path / "c.json")
    return CachingModel(inner, cache, "m", _LenientExecutionPlan.model_validate, **kw)


def test_cache_records_then_replays_without_calling_the_model(tmp_path):
    inner = _FakeInner()
    _model(tmp_path, inner).invoke("prompt A")
    assert inner.calls == 1

    replay = _model(tmp_path, None)  # fresh object, reads the file
    out = replay.invoke("prompt A")
    assert out.intent.query == "phone case" and out.intent.max_price == Decimal("500")
    assert replay.cache_hits == 1 and inner.calls == 1
    with pytest.raises(CacheMiss):
        replay.invoke("prompt B")


def test_cache_key_changes_with_prompt_and_model():
    assert cache_key("m", "a") != cache_key("m", "b")
    assert cache_key("m1", "a") != cache_key("m2", "a")


def test_batch_limit_stops_real_calls(tmp_path):
    inner = _FakeInner()
    m = _model(tmp_path, inner, max_real_calls=1)
    m.invoke("a")
    with pytest.raises(BatchLimitReached):
        m.invoke("b")
    assert inner.calls == 1


def test_delay_between_real_calls_only(tmp_path):
    slept = []
    m = _model(tmp_path, _FakeInner(), delay=7, sleep=slept.append)
    m.invoke("a"); m.invoke("b"); m.invoke("a")  # third is a cache hit
    assert slept == [7]


# ---- runner ---------------------------------------------------------------
def test_baseline_shows_python_guards_alone_fix_city_returns_and_budget():
    report = run_eval(mode="baseline")
    by_id = {o.case.id: o for o in report.outcomes}
    assert report.complete
    mumbai = {f.name: f.ok for f in by_id["mumbai_typo_full"].score.fields}
    assert mumbai["delivery_city"] and mumbai["return_required"]
    assert mumbai["max_price"] and mumbai["currency"]
    assert not mumbai["query"]  # the stub cannot clean the query: that is Gemini's job
    assert report.per_field()["delivery_city"] == (30, 30)


def test_replay_with_empty_cache_is_pending_not_failed(tmp_path):
    report = run_eval(mode="replay", cache_path=tmp_path / "none.json")
    assert len(report.pending) == 30 and not report.scored and not report.errors
    assert not report.passed(0.9)


class _Ideal:
    """Answers each case as the expected intent would look, to test the runner."""

    def __init__(self, fail_after=None, message="RESOURCE_EXHAUSTED 429"):
        self.calls, self.fail_after, self.message = 0, fail_after, message

    def invoke(self, prompt):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError(self.message)
        return _LenientExecutionPlan.model_validate(
            {"intent": {"query": "thing"},
             "steps": [{"step_id": "1", "action": "search_catalog", "reason": "r"}]}
        )


def test_record_stops_cleanly_on_429_and_keeps_what_it_cached(tmp_path):
    cases = CASES[:4]
    path = tmp_path / "cache.json"
    report = run_eval(cases, mode="record", cache_path=path, inner_model=_Ideal(fail_after=2),
                      delay=0, sleep=lambda s: None)
    assert [o.status for o in report.outcomes] == ["ok", "ok", "pending", "pending"]
    assert "429" in report.stopped_reason
    assert len(ResponseCache(path).entries) == 2  # nothing lost

    # Next run: the 2 cached cases replay for free, the rest are recorded.
    inner = _Ideal()
    report2 = run_eval(cases, mode="record", cache_path=path, inner_model=inner,
                       delay=0, sleep=lambda s: None)
    assert report2.complete and inner.calls == 2 and report2.cache_hits == 2


def test_record_respects_batch_limit(tmp_path):
    inner = _Ideal()
    report = run_eval(CASES[:5], mode="record", cache_path=tmp_path / "c.json",
                      inner_model=inner, limit=2, delay=0, sleep=lambda s: None)
    assert inner.calls == 2
    assert len(report.scored) == 2 and len(report.pending) == 3
    assert "batch limit" in report.stopped_reason


def test_non_rate_limit_failure_is_an_error_not_a_stop(tmp_path):
    report = run_eval(CASES[:3], mode="record", cache_path=tmp_path / "c.json",
                      inner_model=_Ideal(fail_after=0, message="boom"),
                      delay=0, sleep=lambda s: None)
    assert len(report.errors) == 3 and not report.stopped_reason


def test_per_case_settings_override_reaches_the_planner(tmp_path):
    case = next(c for c in CASES if c.id == "no_currency_no_market_asks")
    assert case.settings == {"default_currency": None}
    # Gemini returns a budget with no currency; with no market default the planner must ask.
    class Budget:
        def invoke(self, prompt):
            return _LenientExecutionPlan.model_validate(
                {"intent": {"query": "laptop stand", "max_price": "1200"},
                 "steps": [{"step_id": "1", "action": "search_catalog", "reason": "r"}]})
    report = run_eval([case], mode="record", cache_path=tmp_path / "c.json", inner_model=Budget(),
                      delay=0, sleep=lambda s: None)
    assert report.scored[0].score.perfect
