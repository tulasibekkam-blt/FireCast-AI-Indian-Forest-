import pandas as pd

from firecast.data.splits import chronological_split, geographic_split


def test_chronological_split_keeps_future_in_validation():
    frame = pd.DataFrame({"time": [3, 1, 2, 4], "x": [0, 1, 2, 3]})
    train, validation = chronological_split(frame, "time", 0.5)
    assert train["time"].tolist() == [1, 2]
    assert validation["time"].tolist() == [3, 4]


def test_geographic_split_disjoint_groups():
    frame = pd.DataFrame({"region": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})
    train, validation = geographic_split(frame, "region", 0.5)
    assert set(train.region).isdisjoint(set(validation.region))
