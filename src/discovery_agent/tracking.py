import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from . import pii


class JsonlRunTracker:
    """Append-only tracker that stores operational metadata, not credentials or raw pages.

    Events pass through ``pii.mask_event_for_ledger`` first: contact details are masked
    and the delivery city is replaced by a keyed hash (M2.4).
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, event: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"recorded_at": datetime.now(timezone.utc).isoformat(), **pii.mask_event_for_ledger(event)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
