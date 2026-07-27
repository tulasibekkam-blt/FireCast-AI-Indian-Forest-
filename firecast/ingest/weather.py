from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def fetch_open_meteo(latitude: float, longitude: float, timeout_seconds: float = 5.0) -> dict[str, float | str]:
    """Fetch current weather from Open-Meteo without requiring an API key."""
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Invalid latitude or longitude")
    query = urlencode({
        "latitude": latitude, "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
        "timezone": "UTC",
    })
    request = Request(f"https://api.open-meteo.com/v1/forecast?{query}", headers={"User-Agent": "FireCast-AI/0.1"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except OSError as error:
        raise ConnectionError("Weather API unavailable; use the last valid observation offline") from error
    current = payload.get("current")
    if not isinstance(current, dict):
        raise ValueError("Weather API response did not contain current observations")
    required = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation"]
    if any(key not in current for key in required):
        raise ValueError("Weather API response is missing required observations")
    return {"observed_at": str(current.get("time", datetime.now(timezone.utc).isoformat())), **{key: float(current[key]) for key in required}}


def weather_features(observation: dict[str, float | str]) -> dict[str, float]:
    """Convert normalized weather observations into model-ready numeric features."""
    required = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation"]
    missing = [key for key in required if key not in observation]
    if missing:
        raise ValueError(f"Weather observation is missing fields: {missing}")
    return {key: float(observation[key]) for key in required}
