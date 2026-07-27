from __future__ import annotations

import numpy as np


def perturb_image(image: np.ndarray, condition: str, seed: int = 42) -> np.ndarray:
    """Apply a deterministic surveillance-condition perturbation to an image."""
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError("Expected image with shape [height, width, channels]")
    rng = np.random.default_rng(seed)
    values = image.astype(np.float32)
    if condition == "low_light":
        values *= 0.35
    elif condition == "motion_blur":
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Install OpenCV for vision robustness testing") from error
        kernel = np.zeros((1, 11), dtype=np.float32)
        kernel[0] = 1 / 11
        values = cv2.filter2D(values, -1, kernel)
    elif condition == "sensor_noise":
        values += rng.normal(0, 12, values.shape)
    elif condition == "fog":
        values = values * 0.65 + 255 * 0.35
    else:
        raise ValueError(f"Unknown image condition: {condition}")
    return np.clip(values, 0, 255).astype(image.dtype)


def robustness_suite(image: np.ndarray, conditions: list[str], seed: int = 42) -> dict[str, np.ndarray]:
    return {condition: perturb_image(image, condition, seed) for condition in conditions}
