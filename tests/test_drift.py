import pandas as pd

from firecast.observability.drift import population_stability_index


def test_identical_distributions_have_zero_drift():
    values = pd.Series([1, 2, 3, 4, 5])
    assert population_stability_index(values, values) == 0.0
