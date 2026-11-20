from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import CASES
from .runner import DEFAULT_CACHE, EvalReport, run_eval


def _fmt(value: object) -> str:
    return str(value)


def render(report: EvalReport, threshold: float, min_perfect: float = 0.0) -> str:
    lines: list[str] = []
    n = len(report.outcomes)
    lines.append(f"mode={report.mode}  cases scored {len(report.scored)}/{n}"
                 f"  gemini calls={report.real_calls}  cache hits={report.cache_hits}")
    if report.stopped_reason:
        lines.append(f"stopped: {report.stopped_reason}")
    lines.append("")
    for name, (ok, total) in report.per_field().items():
        lines.append(f"  {name:<16}{ok:>3}/{total:<3} {ok / total:>6.0%}")
    lines.append("")
    perfect = sum(1 for o in report.scored if o.score.perfect)
    lines.append(f"field accuracy: {report.field_accuracy():.1%}   (target {threshold:.0%})")
    lines.append(f"cases fully correct: {perfect}/{len(report.scored)}"
                 f" ({report.perfect_rate():.0%})"
                 + (f"   (target {min_perfect:.0%})" if min_perfect else ""))

    for o in report.scored:
        bad = o.score.failures()
        if bad:
            lines.append("")
            lines.append(f"FAIL {o.case.id}: {o.case.query}")
            for f in bad:
                lines.append(f"   {f.name}: expected {_fmt(f.expected)} | got {_fmt(f.got)}")
    for o in report.errors:
        lines.append(f"\nERROR {o.case.id}: {o.detail}")
    if report.pending:
        lines.append(f"\n{len(report.pending)} case(s) pending (not scored): "
                     + ", ".join(o.case.id for o in report.pending[:8])
                     + (" ..." if len(report.pending) > 8 else ""))
    crit = report.critical_failures()
    lines.append("")
    if crit:
        lines.append(f"critical cases not fully correct: {', '.join(crit)}")
    if report.passed(threshold, min_perfect):
        lines.append("RESULT: PASS")
    elif not report.complete:
        lines.append("RESULT: INCOMPLETE (score covers only the cases above)")
    else:
        lines.append("RESULT: FAIL")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m discovery_agent.eval", description=__doc__)
    p.add_argument("--mode", choices=["replay", "record", "baseline"], default="replay")
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--limit", type=int, default=None,
                   help="record mode: max real Gemini calls this run (batch size)")
    p.add_argument("--delay", type=float, default=5.0,
                   help="record mode: seconds between real calls")
    p.add_argument("--cases", nargs="*", help="only these case ids")
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--min-perfect", type=float, default=0.0,
                   help="also require this share of cases to be fully correct (e.g. 0.8); "
                        "field accuracy alone is inflated by fields that are trivially None/False")
    p.add_argument("--json", type=Path, help="write a machine-readable report here")
    args = p.parse_args(argv)

    cases = [c for c in CASES if not args.cases or c.id in args.cases]
    if not cases:
        p.error("no matching case ids")
    report = run_eval(cases, mode=args.mode, cache_path=args.cache,
                      limit=args.limit, delay=args.delay)
    print(render(report, args.threshold, args.min_perfect))

    if args.json:
        args.json.write_text(json.dumps({
            "mode": report.mode,
            "field_accuracy": report.field_accuracy(),
            "per_field": report.per_field(),
            "complete": report.complete,
            "cases": [{
                "id": o.case.id, "status": o.status, "detail": o.detail,
                "fields": [{"name": f.name, "ok": f.ok, "expected": _fmt(f.expected),
                            "got": _fmt(f.got)} for f in o.score.fields] if o.score else [],
                "corrections": o.corrections,
            } for o in report.outcomes],
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    if report.passed(args.threshold, args.min_perfect):
        return 0
    return 2 if not report.complete else 1


if __name__ == "__main__":
    sys.exit(main())
