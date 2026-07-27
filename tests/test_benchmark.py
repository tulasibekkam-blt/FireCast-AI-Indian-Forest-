import pytest

from firecast.deployment.benchmark import benchmark_onnx


def test_benchmark_rejects_invalid_iterations():
    with pytest.raises(ValueError, match="iterations"):
        benchmark_onnx("missing.onnx", (1, 2, 3), iterations=0)
