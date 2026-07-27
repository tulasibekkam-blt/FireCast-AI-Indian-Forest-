from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


@dataclass(frozen=True)
class SensorObservation:
    sensor_id: str
    observed_at: datetime
    values: dict[str, float]

    def to_features(self) -> dict[str, float]:
        """Return namespaced numeric features suitable for multimodal fusion."""
        return {f"sensor_{key}": value for key, value in self.values.items()}


def parse_sensor_message(payload: str | bytes) -> SensorObservation:
    """Validate a JSON telemetry message before it enters the model pipeline."""
    try:
        document = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Sensor payload must be valid JSON") from error
    if not isinstance(document, dict) or not isinstance(document.get("sensor_id"), str):
        raise ValueError("Sensor payload requires a string sensor_id")
    try:
        observed_at = datetime.fromisoformat(str(document["observed_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as error:
        raise ValueError("Sensor payload requires an ISO-8601 observed_at") from error
    values = document.get("values")
    if not isinstance(values, dict) or not values:
        raise ValueError("Sensor payload requires non-empty numeric values")
    try:
        normalized = {str(key): float(value) for key, value in values.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("All sensor values must be numeric") from error
    return SensorObservation(document["sensor_id"], observed_at, normalized)


def subscribe_mqtt(host: str, topic: str, on_observation: Callable[[SensorObservation], Any],
                   port: int = 1883, timeout_seconds: int = 10) -> None:
    """Subscribe to MQTT telemetry; requires the optional paho-mqtt package."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError as error:
        raise RuntimeError("Install paho-mqtt to subscribe to IoT telemetry") from error
    if not host or not topic:
        raise ValueError("host and topic are required")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(host, port, timeout_seconds)
    client.subscribe(topic)
    client.on_message = lambda _client, _userdata, message: on_observation(parse_sensor_message(message.payload))
    client.loop_forever()
