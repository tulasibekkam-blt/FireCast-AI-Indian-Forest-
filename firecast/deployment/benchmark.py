from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def benchmark_onnx(model: str | Path, input_shape: tuple[int, int, int], iterations: int = 100,
                   warmup: int = 10, providers: list[str] | None = None) -> dict[str, float | str]:
    """Benchmark an ONNX model with the installed ONNX Runtime execution provider."""
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Install the deployment extra to benchmark ONNX") from error
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")
    source = Path(model)
    if not source.exists():
        raise FileNotFoundError(f"ONNX model does not exist: {source}")
    session = ort.InferenceSession(str(source), providers=providers or ort.get_available_providers())
    input_name = session.get_inputs()[0].name
    values = np.zeros(input_shape, dtype=np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: values})
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {input_name: values})
        durations.append((time.perf_counter() - start) * 1000)
    mean_latency = float(np.mean(durations))
    return {
        "provider": str(session.get_providers()[0]), "iterations": float(iterations),
        "mean_latency_ms": mean_latency, "p50_latency_ms": float(np.percentile(durations, 50)),
        "p95_latency_ms": float(np.percentile(durations, 95)), "fps": 1000.0 / mean_latency,
    }
