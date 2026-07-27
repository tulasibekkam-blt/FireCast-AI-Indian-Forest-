from firecast.vision.detector import Detection
from firecast.vision.risk import detection_risk


def test_fire_evidence_dominates_smoke_evidence():
    detections = [Detection(0, "smoke", 0.9, (0, 0, 1, 1)), Detection(1, "fire", 0.7, (0, 0, 1, 1))]
    assert detection_risk(detections) == 0.7
