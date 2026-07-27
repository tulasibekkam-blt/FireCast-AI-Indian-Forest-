import numpy as np
import pandas as pd
import pytest

from firecast.robustness.tabular import evaluate_sensor_noise


def test_noise_evaluation_rejects_unknown_feature():
    with pytest.raises(ValueError, match="Unknown"):
        evaluate_sensor_noise(None, pd.DataFrame({"x": [1]}), np.array([0]), ["missing"])
