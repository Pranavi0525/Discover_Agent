"""Response cache so the eval never spends Gemini free-tier quota twice.

The cache key is a hash of (model name, full prompt).  Editing the planner
prompt or switching model therefore re-records automatically, while an
unchanged prompt replays for free.  The file is written after EVERY real call,
so a 429 half-way through a batch loses nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel


class CacheMiss(RuntimeError):
    """The response is not cached and a real call is not allowed right now."""


class BatchLimitReached(CacheMiss):
    """The per-run limit on real Gemini calls was reached."""


def cache_key(model_name: str, prompt: str) -> str:
    return hashlib.sha256(f"{model_name}\n---\n{prompt}".encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, Any] = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> Any | None:
        return self.entries.get(key)

    def put(self, key: str, value: Any) -> None:
        self.entries[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.entries, fh, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(tmp, self.path)  # atomic: never leaves a half-written cache


class CachingModel:
    """Drop-in for the planner's structured-output model (``.invoke(prompt)``).

    ``inner`` is the real Gemini model, or None for replay-only.  ``parse``
    rebuilds the model's typed response from the cached JSON.
    """

    def __init__(
        self,
        inner: Any | None,
        cache: ResponseCache,
        model_name: str,
        parse: Callable[[Any], Any],
        *,
        max_real_calls: int | None = None,
        delay: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.model_name = model_name
        self.parse = parse
        self.max_real_calls = max_real_calls
        self.delay = delay
        self.sleep = sleep
        self.real_calls = 0
        self.cache_hits = 0

    def invoke(self, prompt: str) -> Any:
        key = cache_key(self.model_name, prompt)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return self.parse(cached)
        if self.inner is None:
            raise CacheMiss("not in cache (replay mode makes no Gemini calls)")
        if self.max_real_calls is not None and self.real_calls >= self.max_real_calls:
            raise BatchLimitReached(f"batch limit of {self.max_real_calls} real calls reached")
        if self.real_calls and self.delay:
            self.sleep(self.delay)  # stay under the free-tier requests/minute
        response = self.inner.invoke(prompt)
        self.real_calls += 1
        if response is not None:
            payload = response.model_dump(mode="json") if isinstance(response, BaseModel) else response
            self.cache.put(key, payload)
        return response
