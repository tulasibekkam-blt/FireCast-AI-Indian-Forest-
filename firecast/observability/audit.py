from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLog:
    """Append-only JSON Lines audit log for edge inference events."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, payload: dict[str, Any], observed_at: datetime | None = None) -> None:
        if not event_type or not isinstance(payload, dict):
            raise ValueError("event_type and payload are required")
        event = {
            "event_type": event_type,
            "observed_at": (observed_at or datetime.now(timezone.utc)).isoformat(),
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
