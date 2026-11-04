"""Import paths and isolated configuration environments for continuous tests."""
import importlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def legacy_beta_environment():
    """Run historical b2-3 assertions without changing the surrounding grid state."""
    import step6x_config as cfg
    import step6x_config_v4 as cfg4

    original = os.environ.get("SCENA_BETA_ANCHOR")
    os.environ["SCENA_BETA_ANCHOR"] = "2.0,3.0"
    importlib.reload(cfg)
    importlib.reload(cfg4)
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("SCENA_BETA_ANCHOR", None)
        else:
            os.environ["SCENA_BETA_ANCHOR"] = original
        importlib.reload(cfg)
        importlib.reload(cfg4)
