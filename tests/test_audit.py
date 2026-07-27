import json

from firecast.observability.audit import AuditLog


def test_audit_log_writes_jsonl(tmp_path):
    path = tmp_path / "audit" / "events.jsonl"
    AuditLog(path).write("prediction", {"risk": 0.9})
    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["event_type"] == "prediction"
    assert event["payload"]["risk"] == 0.9
