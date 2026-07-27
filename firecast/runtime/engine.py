from __future__ import annotations

from datetime import datetime
from typing import Any

from firecast.alerts.policy import AlertDecision, HysteresisPolicy
from firecast.observability.audit import AuditLog
from firecast.artifacts import load_model_artifact
from firecast.fusion.decision import aggregate_risk
from firecast.alerts.state import restore_policy_state, save_policy_state


class RiskEngine:
    """Run a persisted probabilistic model with operational alert policy and audit logging."""

    def __init__(self, model: Any, policy: HysteresisPolicy, audit: AuditLog, state_path: str | None = None):
        self.model = model
        self.policy = policy
        self.audit = audit
        self.state_path = state_path
        if state_path:
            restore_policy_state(policy, state_path)

    def evaluate(self, features: Any, observed_at: datetime, context: dict[str, Any] | None = None) -> AlertDecision:
        probability = float(self.model.predict_proba(features)[:, 1][0])
        decision = self.policy.update(probability, observed_at)
        if self.state_path:
            save_policy_state(self.policy, self.state_path)
        self.audit.write("risk_decision", {
            "probability": probability, "state": decision.state,
            "reason": decision.reason, "context": context or {},
        }, observed_at)
        return decision

    def evaluate_modalities(self, probabilities: dict[str, float], availability: dict[str, bool],
                            observed_at: datetime, weights: dict[str, float] | None = None,
                            context: dict[str, Any] | None = None) -> AlertDecision:
        fused_probability = aggregate_risk(probabilities, availability, weights)
        decision = self.policy.update(fused_probability, observed_at)
        if self.state_path:
            save_policy_state(self.policy, self.state_path)
        self.audit.write("multimodal_decision", {
            "probabilities": probabilities, "availability": availability,
            "fused_probability": fused_probability, "state": decision.state,
            "reason": decision.reason, "context": context or {},
        }, observed_at)
        return decision

    @classmethod
    def from_artifact(cls, artifact: str, audit: AuditLog) -> "RiskEngine":
        model, metadata = load_model_artifact(artifact)
        policy = HysteresisPolicy(
            raise_threshold=float(metadata.get("alert_raise_threshold", metadata["threshold"])),
            clear_threshold=float(metadata.get("alert_clear_threshold", metadata["threshold"] * 0.7)),
            cooldown_seconds=int(metadata.get("cooldown_seconds", 300)),
        )
        return cls(model, policy, audit)
