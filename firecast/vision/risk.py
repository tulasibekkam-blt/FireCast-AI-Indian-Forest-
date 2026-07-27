from __future__ import annotations

from collections.abc import Iterable

from firecast.vision.detector import Detection


def detection_risk(detections: Iterable[Detection], fire_labels: set[str] | None = None,
                  smoke_labels: set[str] | None = None) -> float:
    """Convert fire/smoke detections into a bounded evidence score."""
    fire_labels = fire_labels or {"fire", "flame"}
    smoke_labels = smoke_labels or {"smoke"}
    score = 0.0
    for detection in detections:
        label = detection.label.strip().lower()
        if label in fire_labels:
            score = max(score, detection.confidence)
        elif label in smoke_labels:
            score = max(score, detection.confidence * 0.8)
    return float(min(max(score, 0.0), 1.0))
