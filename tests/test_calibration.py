import numpy as np

from firecast.evaluation import expected_calibration_error


def test_perfectly_calibrated_extreme_predictions_have_zero_error():
    assert expected_calibration_error(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0
