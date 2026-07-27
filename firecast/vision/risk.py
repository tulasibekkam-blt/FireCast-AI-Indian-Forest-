from __future__ import annotations

from collections.abc import Iterable

from firecast.vision.detector import Detection


def detection_risk(
    detections: Iterable[Detection],
    fire_labels: set[str] | None = None,
    smoke_labels: set[str] | None = None,
) -> float:
    """
    Convert fire/smoke detections into a bounded evidence score.
    """

    fire_labels = fire_labels or {
        "fire",
        "flame",
    }

    smoke_labels = smoke_labels or {
        "smoke",
    }

    fire_score = 0.0
    smoke_score = 0.0

    for detection in detections:
        label = detection.label.strip().lower()

        if label in fire_labels:
            fire_score = max(
                fire_score,
                detection.confidence,
            )

        elif label in smoke_labels:
            smoke_score = max(
                smoke_score,
                detection.confidence * 0.8,
            )

    if fire_score > 0.0:
        return float(
            min(
                max(fire_score, 0.0),
                1.0,
            )
        )

    return float(
        min(
            max(smoke_score, 0.0),
            1.0,
        )
    )