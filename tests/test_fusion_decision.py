from firecast.fusion.decision import aggregate_risk


def test_aggregate_risk_ignores_unavailable_modalities():
    score = aggregate_risk({"sensor": 0.8, "vision": 0.1}, {"sensor": True, "vision": False})
    assert score == 0.8
