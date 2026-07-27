from datetime import datetime, timezone

import numpy as np

from firecast.alerts.policy import HysteresisPolicy
from firecast.observability.audit import AuditLog
from firecast.runtime.engine import RiskEngine


class Model:
    def predict_proba(self, _features):
        return np.array([[0.1, 0.9]])


def test_engine_emits_alert_and_audit_event(tmp_path):
    engine = RiskEngine(Model(), HysteresisPolicy(0.8, 0.5), AuditLog(tmp_path / "audit.jsonl"))
    decision = engine.evaluate(None, datetime.now(timezone.utc))
    assert decision.state == "raise"
