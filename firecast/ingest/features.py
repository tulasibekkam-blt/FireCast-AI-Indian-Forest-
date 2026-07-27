from __future__ import annotations

from typing import Mapping

import pandas as pd

from firecast.ingest.iot import SensorObservation
from firecast.ingest.weather import weather_features


def merge_observations(weather: Mapping[str, float | str], sensor: SensorObservation) -> pd.DataFrame:
    """Create a single-row feature frame from live weather and IoT observations."""
    features = weather_features(dict(weather))
    overlap = set(features).intersection(sensor.to_features())
    if overlap:
        raise ValueError(f"Weather and sensor feature names overlap: {sorted(overlap)}")
    features.update(sensor.to_features())
    return pd.DataFrame([features])
