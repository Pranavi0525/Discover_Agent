"""Relevance scoring behind an injectable interface (M1.5).

Hard filters (availability, currency, budget, merchant) run BEFORE any scorer
and a scorer can only rank the survivors, so similarity can never bring a
filtered product back.  Nothing here imports a model at import time:
``sentence_transformers`` is imported (and the model downloaded/loaded) only on
the first real scoring call, once per process.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import re
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Protocol

from .schemas import Product

logger = logging.getLogger(__name__)

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SCORER_ENV_VAR = "DISCOVERY_RELEVANCE_SCORER"  # auto (default) | minilm | lexical

NAME_MINILM = "minilm"
NAME_LEXICAL = "lexical"  # chosen on purpose (DISCOVERY_RELEVANCE_SCORER=lexical)
NAME_LEXICAL_FALLBACK = "lexical_fallback"  # MiniLM wanted but unavailable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(value: str) -> Counter[str]:
    return Counter(_TOKEN_RE.findall(value.lower()))


def lexical_similarity(query: str, product: Product) -> float:
    """Fraction of query words found in title + description (deterministic)."""
    query_tokens = set(_tokens(query))
    product_tokens = set(_tokens(f"{product.title} {product.description}"))
    if not query_tokens or not product_tokens:
        return 0.0
    return len(query_tokens & product_tokens) / len(query_tokens)


def product_text(product: Product) -> str:
    return f"{product.title} {product.description}".strip()


class RelevanceScorer(Protocol):
    """Scores products against one or more query texts.

    ``queries[0]`` is always the ORIGINAL user query; later entries are
    same-meaning synonym rewrites.  Return one raw score per product (the best
    over the queries), in product order.  ``semantic`` is True only for a real
    embedding scorer; it controls whether messages may say "semantic".
    """

    name: str
    semantic: bool

    def score(self, queries: Sequence[str], products: Sequence[Product]) -> list[float]: ...


class LexicalScorer:
    """The M1.4 keyword-overlap scorer, also the explicit fallback."""

    semantic = False

    def __init__(self, name: str = NAME_LEXICAL, fallback_reason: str | None = None) -> None:
        self.name = name
        self.fallback_reason = fallback_reason

    def score(self, queries: Sequence[str], products: Sequence[Product]) -> list[float]:
        return [max(lexical_similarity(query, product) for query in queries) for product in products]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class EmbeddingScorer:
    """Cosine similarity over any ``embed(texts) -> vectors`` callable.

    Tests inject a fake deterministic ``embed``; production uses
    :class:`MiniLMScorer`.  Query embeddings are cached so the original query
    is embedded once per scorer, however many rungs score against it.
    """

    semantic = True

    def __init__(self, embed: Callable[[list[str]], Sequence[Sequence[float]]],
                 name: str = NAME_MINILM) -> None:
        self._embed = embed
        self.name = name
        self._query_cache: dict[str, Sequence[float]] = {}

    def score(self, queries: Sequence[str], products: Sequence[Product]) -> list[float]:
        if not products:
            return []
        missing = [q for q in dict.fromkeys(queries) if q not in self._query_cache]
        texts = [product_text(p) for p in products]
        vectors = list(self._embed(missing + texts))
        if len(vectors) != len(missing) + len(texts):
            raise ValueError("embedder returned the wrong number of vectors")
        for query, vector in zip(missing, vectors):
            if len(self._query_cache) >= 64:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[query] = vector
        product_vectors = vectors[len(missing):]
        query_vectors = [self._query_cache[q] for q in queries]
        return [max(_cosine(qv, pv) for qv in query_vectors) for pv in product_vectors]


class MiniLMScorer(EmbeddingScorer):
    """sentence-transformers/all-MiniLM-L6-v2, loaded lazily and only once."""

    def __init__(self, model_name: str = MINILM_MODEL_NAME) -> None:
        super().__init__(self._encode, NAME_MINILM)
        self.model_name = model_name
        self.load_count = 0
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer  # lazy: heavy import

                self._model = SentenceTransformer(self.model_name)
                self.load_count += 1
            return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()


class RunScorer:
    """Pins ONE scorer for a whole run and makes any fallback explicit.

    If the primary scorer raises (model missing, download failed, bad output)
    the run switches to the lexical scorer from then on and records why, so a
    run never mixes embedding and keyword scores against the same tiers.
    """

    def __init__(self, primary: RelevanceScorer) -> None:
        self._primary = primary
        self._fallback: LexicalScorer | None = None
        self.fallback_reason: str | None = getattr(primary, "fallback_reason", None)

    @property
    def _active(self) -> RelevanceScorer:
        return self._fallback or self._primary

    @property
    def name(self) -> str:
        return self._active.name

    @property
    def semantic(self) -> bool:
        return bool(self._active.semantic)

    def score(self, queries: Sequence[str], products: Sequence[Product]) -> list[float]:
        if self._fallback is None:
            try:
                scores = list(self._primary.score(queries, products))
                if len(scores) != len(products):
                    raise ValueError("scorer returned the wrong number of scores")
                return scores
            except Exception as exc:  # noqa: BLE001 - any scorer failure degrades explicitly
                self.fallback_reason = f"{type(exc).__name__}: {str(exc)[:160]}"
                logger.warning("relevance scorer %s failed; using lexical fallback (%s)",
                               self._primary.name, self.fallback_reason)
                self._fallback = LexicalScorer(NAME_LEXICAL_FALLBACK, self.fallback_reason)
        return self._fallback.score(queries, products)


def get_default_scorer() -> RelevanceScorer:
    """MiniLM when ``sentence_transformers`` is installed, else explicit fallback.

    Uses ``find_spec`` so nothing heavy is imported here.  Set
    ``DISCOVERY_RELEVANCE_SCORER=lexical`` to choose keyword scoring on purpose.
    """
    choice = os.getenv(SCORER_ENV_VAR, "auto").strip().lower()
    if choice == "lexical":
        return LexicalScorer()
    if importlib.util.find_spec("sentence_transformers") is None:
        return LexicalScorer(NAME_LEXICAL_FALLBACK, "sentence-transformers is not installed")
    return _shared_minilm()


_MINILM: MiniLMScorer | None = None
_MINILM_LOCK = threading.Lock()


def _shared_minilm() -> MiniLMScorer:
    """One scorer per process, so the model is loaded at most once and reused."""
    global _MINILM
    with _MINILM_LOCK:
        if _MINILM is None:
            _MINILM = MiniLMScorer()  # cheap: the model itself loads on first score
        return _MINILM
