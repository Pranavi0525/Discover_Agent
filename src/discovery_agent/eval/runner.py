"""Run the eval cases through the REAL planner pipeline.

Only the model call is swapped (for the cache, or a naive stub).  The prompt,
limit checks and the deterministic guards (budget, currency, city, returns)
are exactly what the app runs, so a green eval means the app behaves.

Modes
-----
replay    read the cache only; no network.  Missing cases are "pending".
record    call Gemini for cache misses (bounded by ``limit``/``delay``), cache
          every answer, stop cleanly on a 429.
baseline  a stub "model" that returns the raw request as the query.  Shows what
          the Python guards alone achieve, i.e. the floor Gemini must beat.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..planner import GeminiPlanner, _LenientExecutionPlan
from ..schemas import ExecutionPlan
from .cache import BatchLimitReached, CacheMiss, CachingModel, ResponseCache
from .dataset import CASES, EvalCase
from .scoring import CaseScore, FIELDS, score_case

DEFAULT_CACHE = Path("eval_cache/gemini_responses.json")
EVAL_MARKET_CURRENCY = "INR"

_REQUEST = re.compile(r"shopping request:\s*\n\n(.*?)\n\nRules:", re.DOTALL)


class _NaiveModel:
    """Baseline stub: no understanding at all, the whole request is the query."""

    def invoke(self, prompt: str) -> _LenientExecutionPlan:
        match = _REQUEST.search(prompt)
        text = (match.group(1) if match else prompt).strip()[:300]
        return _LenientExecutionPlan.model_validate(
            {
                "intent": {"query": text},
                "steps": [{"step_id": "1", "action": "search_catalog", "reason": "search"}],
            }
        )


@dataclass
class CaseOutcome:
    case: EvalCase
    status: str  # ok | pending | error
    score: CaseScore | None = None
    plan: ExecutionPlan | None = None
    corrections: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class EvalReport:
    mode: str
    outcomes: list[CaseOutcome]
    real_calls: int = 0
    cache_hits: int = 0
    stopped_reason: str = ""

    @property
    def scored(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if o.status == "ok" and o.score]

    @property
    def pending(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if o.status == "pending"]

    @property
    def errors(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if o.status == "error"]

    @property
    def complete(self) -> bool:
        return not self.pending and not self.errors

    def field_accuracy(self) -> float:
        total = sum(o.score.total for o in self.scored)
        return sum(o.score.correct for o in self.scored) / total if total else 0.0

    def per_field(self) -> dict[str, tuple[int, int]]:
        table: dict[str, list[int]] = {name: [0, 0] for name in FIELDS}
        for o in self.scored:
            for f in o.score.fields:
                table[f.name][1] += 1
                table[f.name][0] += int(f.ok)
        return {k: (v[0], v[1]) for k, v in table.items() if v[1]}

    def critical_failures(self) -> list[str]:
        return [
            o.case.id
            for o in self.outcomes
            if o.case.critical and not (o.status == "ok" and o.score and o.score.perfect)
        ]

    def perfect_rate(self) -> float:
        return sum(1 for o in self.scored if o.score.perfect) / len(self.scored) if self.scored else 0.0

    def passed(self, threshold: float, min_perfect: float = 0.0) -> bool:
        return (
            self.complete
            and self.field_accuracy() >= threshold
            and self.perfect_rate() >= min_perfect
            and not self.critical_failures()
        )


def _settings_for(case: EvalCase, base: Settings) -> Settings:
    settings = base
    settings.default_currency = EVAL_MARKET_CURRENCY
    for key, value in case.settings.items():
        setattr(settings, key, value)
    return settings


def _shim_planner(settings: Settings, model: Any) -> GeminiPlanner:
    """A GeminiPlanner that runs the real ``plan()`` around our model."""
    planner = object.__new__(GeminiPlanner)
    planner.settings = settings
    planner._model = model
    planner._build_model = lambda: None  # a retry must keep using the cached model
    return planner


def run_eval(
    cases: tuple[EvalCase, ...] | list[EvalCase] = CASES,
    *,
    mode: str = "replay",
    cache_path: Path = DEFAULT_CACHE,
    limit: int | None = None,
    delay: float = 5.0,
    inner_model: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> EvalReport:
    if mode not in {"replay", "record", "baseline"}:
        raise ValueError(f"unknown mode {mode!r}")

    base = Settings()
    base.google_api_key = base.google_api_key or "eval-offline"
    base.max_planner_retries = 1

    cache = ResponseCache(cache_path)
    caching: CachingModel | None = None
    model: Any
    if mode == "record":
        if inner_model is None:
            if base.google_api_key == "eval-offline":
                raise SystemExit("record mode needs GOOGLE_API_KEY in .env or the environment")
            inner_model = GeminiPlanner(base)._model
        caching = CachingModel(
            inner_model, cache, base.gemini_model, _LenientExecutionPlan.model_validate,
            max_real_calls=limit, delay=delay, sleep=sleep,
        )
        model = caching
    elif mode == "replay":
        caching = CachingModel(
            None, cache, base.gemini_model, _LenientExecutionPlan.model_validate
        )
        model = caching
    else:
        model = _NaiveModel()

    report = EvalReport(mode=mode, outcomes=[])
    halted = ""
    for case in cases:
        if halted:
            report.outcomes.append(CaseOutcome(case, "pending", detail=halted))
            continue
        planner = _shim_planner(_settings_for(case, base), model)
        try:
            plan = planner.plan(case.query)
        except Exception as exc:  # noqa: BLE001 - classify, never crash the whole run
            cause = exc.__cause__ or exc
            text = str(exc)
            if isinstance(cause, BatchLimitReached):
                halted = "batch limit reached; run again to record the rest"
                report.stopped_reason = halted
                report.outcomes.append(CaseOutcome(case, "pending", detail=halted))
            elif isinstance(cause, CacheMiss):
                report.outcomes.append(CaseOutcome(case, "pending", detail="not in cache"))
            elif "429" in text or "RESOURCE_EXHAUSTED" in text.upper():
                halted = "Gemini rate limit (429); cache saved, wait for quota and re-run"
                report.stopped_reason = halted
                report.outcomes.append(CaseOutcome(case, "pending", detail=halted))
            else:
                report.outcomes.append(CaseOutcome(case, "error", detail=text[:300]))
            continue
        score = score_case(case.id, case.expect, plan)
        report.outcomes.append(
            CaseOutcome(case, "ok", score, plan, list(getattr(planner, "last_corrections", [])))
        )

    if caching:
        report.real_calls, report.cache_hits = caching.real_calls, caching.cache_hits
    return report
