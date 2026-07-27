import pandas as pd

from firecast.spread.data import make_windows


def test_make_windows_preserves_temporal_order():
    frame = pd.DataFrame({"wind": range(6), "spread": range(10, 16)})
    inputs, labels = make_windows(frame, ["wind"], "spread", lookback=3, horizon=2)
    assert inputs.shape == (2, 3, 1)
    assert labels.tolist() == [[13, 14], [14, 15]]
