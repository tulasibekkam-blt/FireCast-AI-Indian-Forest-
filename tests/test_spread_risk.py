import numpy as np

from firecast.spread.risk import forecast_risk


def test_forecast_risk_is_bounded():
    assert forecast_risk(np.array([2.0, 8.0]), 4.0) == 1.0
