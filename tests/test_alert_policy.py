from datetime import datetime, timedelta, timezone

from firecast.alerts.policy import HysteresisPolicy


def test_policy_hysteresis_prevents_flapping():
    policy = HysteresisPolicy(0.8, 0.5, 60)
    now = datetime.now(timezone.utc)
    assert policy.update(0.9, now).state == "raise"
    assert policy.update(0.7, now + timedelta(seconds=1)).state == "active"
    assert policy.update(0.4, now + timedelta(seconds=2)).state == "clear"
