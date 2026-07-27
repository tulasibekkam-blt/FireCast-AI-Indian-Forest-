from firecast.observability.health import readiness_report


def test_readiness_requires_artifacts(tmp_path):
    report = readiness_report(tmp_path, ["json"])
    assert report["ready"] is False
    assert report["modules_available"]["json"] is True
