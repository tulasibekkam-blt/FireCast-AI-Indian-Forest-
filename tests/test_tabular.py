import pandas as pd
import pytest

from firecast.data.tabular import load_tabular


def test_loader_rejects_non_binary_target(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"temperature": [20, 21], "ignition": [0, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary"):
        load_tabular(path, "ignition")
