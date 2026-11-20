from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from .schemas import Product, SearchResult, UserIntent


class DiscoveryProvider(Protocol):
    name: str

    def search(self, intent: UserIntent) -> SearchResult:
        """Return normalized products without exposing provider-specific payloads."""


class RunTracker(Protocol):
    def record(self, event: Mapping[str, object]) -> None:
        """Persist one audit event."""


ProviderCallable = Callable[[UserIntent], list[Mapping[str, object]]]
ProductMapper = Callable[[Mapping[str, object]], Product]
