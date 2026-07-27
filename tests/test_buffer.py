from datetime import datetime, timezone

from firecast.ingest.buffer import ObservationBuffer
from firecast.ingest.iot import SensorObservation


def test_observation_buffer_survives_reopen(tmp_path):
    path = tmp_path / "observations.sqlite"
    observation = SensorObservation("node-1", datetime.now(timezone.utc), {"temperature": 31.5})
    first = ObservationBuffer(path)
    first.enqueue(observation)
    first.close()
    second = ObservationBuffer(path)
    drained = second.drain()
    second.close()
    assert drained[0].sensor_id == "node-1"
    assert drained[0].values["temperature"] == 31.5
