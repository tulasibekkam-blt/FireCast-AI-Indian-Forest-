from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from firecast.ingest.iot import SensorObservation


class ObservationBuffer:
    """Durable SQLite queue for disconnected edge nodes."""

    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        self.connection.commit()

    def enqueue(self, observation: SensorObservation) -> None:
        payload = json.dumps({
            "sensor_id": observation.sensor_id,
            "observed_at": observation.observed_at.isoformat(),
            "values": observation.values,
        })
        self.connection.execute("INSERT INTO observations(payload) VALUES (?)", (payload,))
        self.connection.commit()

    def drain(self, limit: int = 100) -> list[SensorObservation]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self.connection.execute("SELECT id, payload FROM observations ORDER BY id LIMIT ?", (limit,)).fetchall()
        observations = []
        for row_id, payload in rows:
            document = json.loads(payload)
            observations.append(_from_document(document))
            self.connection.execute("DELETE FROM observations WHERE id = ?", (row_id,))
        self.connection.commit()
        return observations

    def close(self) -> None:
        self.connection.close()


def _from_document(document: dict) -> SensorObservation:
    from datetime import datetime
    return SensorObservation(document["sensor_id"], datetime.fromisoformat(document["observed_at"]), document["values"])
