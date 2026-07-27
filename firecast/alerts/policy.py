from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass(frozen=True)
class AlertDecision:
    state: str
    probability: float
    reason: str


class HysteresisPolicy:
    """Stateful alert policy with separate raise/clear thresholds and cooldown."""

    def __init__(self, raise_threshold: float = 0.8, clear_threshold: float = 0.55,
                 cooldown_seconds: int = 300):
        if not 0 <= clear_threshold < raise_threshold <= 1:
            raise ValueError("Require 0 <= clear_threshold < raise_threshold <= 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        self.raise_threshold = raise_threshold
        self.clear_threshold = clear_threshold
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self.active = False
        self.last_alert: datetime | None = None

    def update(self, probability: float, observed_at: datetime) -> AlertDecision:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be between 0 and 1")
        if self.active:
            if probability <= self.clear_threshold:
                self.active = False
                return AlertDecision("clear", probability, "probability crossed clear threshold")
            return AlertDecision("active", probability, "alert remains active within hysteresis band")
        cooldown_elapsed = self.last_alert is None or observed_at - self.last_alert >= self.cooldown
        if probability >= self.raise_threshold and cooldown_elapsed:
            self.active = True
            self.last_alert = observed_at
            return AlertDecision("raise", probability, "probability crossed raise threshold")
        return AlertDecision("normal", probability, "below raise threshold or cooldown active")

    def state_dict(self) -> dict[str, Any]:
        return {"active": self.active, "last_alert": self.last_alert.isoformat() if self.last_alert else None}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or not isinstance(state.get("active"), bool):
            raise ValueError("Invalid alert policy state")
        self.active = state["active"]
        timestamp = state.get("last_alert")
        self.last_alert = datetime.fromisoformat(timestamp) if timestamp else None
