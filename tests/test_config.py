from firecast.config import load_config


def test_default_config_has_required_sections():
    config = load_config("configs/default.yaml")
    assert {"data", "model", "evaluation", "deployment"}.issubset(config)
