"""Offline-first evaluation of intent extraction (roadmap M1.4).

Run ``python -m discovery_agent.eval --help``.
"""

from .dataset import CASES, EvalCase
from .scoring import CaseScore, score_case

__all__ = ["CASES", "EvalCase", "CaseScore", "score_case"]
