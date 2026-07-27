import pandas as pd

from firecast.data.quality import quality_report


def test_quality_report_is_serializable_and_counts_duplicates():
    frame = pd.DataFrame({"temperature": [1.0, 1.0, None], "ignition": [0, 0, 1]})
    report = quality_report(frame, "ignition")
    assert report["rows"] == 3
    assert report["duplicate_rows"] == 1
    assert report["missing_values"]["temperature"] == 1
