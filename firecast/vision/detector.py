from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class InferenceRecord:
    source: str
    detections: tuple[Detection, ...]
    latency_ms: float


class UltralyticsDetector:
    """Inference-only adapter for trained Ultralytics YOLO checkpoints.

    Training and evaluation datasets remain explicit external inputs; this adapter never
    downloads weights or silently substitutes a pretrained model.
    """

    def __init__(self, weights: str | Path, confidence: float = 0.25, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Install the vision extra to use Ultralytics inference") from error
        checkpoint = Path(weights)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Detector checkpoint does not exist: {checkpoint}")
        self.model = YOLO(str(checkpoint))
        self.confidence = confidence
        self.device = device

    def predict(self, source: str | Path, stream: bool = False) -> Iterator[InferenceRecord]:
        results = self.model.predict(
            source=str(source), conf=self.confidence, device=self.device, stream=stream, verbose=False,
        )
        if not stream:
            results = iter(results)
        for result in results:
            detections = []
            boxes = result.boxes
            names = result.names
            for box, confidence, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                index = int(class_id)
                detections.append(Detection(index, str(names[index]), float(confidence), tuple(map(float, box))))
            speed = getattr(result, "speed", {}) or {}
            latency_ms = float(sum(speed.values())) if speed else 0.0
            if latency_ms <= 0.0:
                latency_ms = 0.001
            yield InferenceRecord(str(getattr(result, "path", source)), tuple(detections), latency_ms)

    def benchmark(self, source: str | Path, iterations: int = 20) -> dict[str, float]:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        records = []
        for index, record in enumerate(self.predict(source, stream=False)):
            if index >= iterations:
                break
            records.append(record.latency_ms)
        if not records:
            raise ValueError("No frames were produced by the inference source")
        mean_latency = sum(records) / len(records)
        checkpoint = Path(self.model.ckpt_path) if getattr(self.model, "ckpt_path", None) else None
        model_size_mb = checkpoint.stat().st_size / (1024 * 1024) if checkpoint and checkpoint.exists() else float("nan")
        return {
            "frames": float(len(records)), "mean_latency_ms": mean_latency,
            "p50_latency_ms": float(np.percentile(records, 50)),
            "p95_latency_ms": float(np.percentile(records, 95)),
            "fps": 1000.0 / mean_latency, "model_size_mb": float(model_size_mb),
        }

    def run_stream(self, source: str | int, on_record: Callable[[InferenceRecord], Any], max_frames: int | None = None) -> int:
        """Process a live source with an application callback and optional frame bound."""
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive when provided")
        count = 0
        for record in self.predict(source, stream=True):
            on_record(record)
            count += 1
            if max_frames is not None and count >= max_frames:
                break
        return count
