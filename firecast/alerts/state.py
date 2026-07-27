from __future__ import annotations

import json
from pathlib import Path

from firecast.alerts.policy import HysteresisPolicy


def save_policy_state(policy: HysteresisPolicy, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(policy.state_dict(), indent=2), encoding="utf-8")


def restore_policy_state(policy: HysteresisPolicy, path: str | Path) -> None:
    source = Path(path)
    if not source.exists():
        return
    policy.load_state_dict(json.loads(source.read_text(encoding="utf-8")))
