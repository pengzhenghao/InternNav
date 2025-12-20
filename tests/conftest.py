import sys
from pathlib import Path
import pytest


# Ensure the repo root is on PYTHONPATH so `import internnav` works in tests
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def tmp_cfg(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("hello: world\n")
    return p


# global hook: skip mark
def pytest_runtest_setup(item):
    if "gpu" in item.keywords:
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("No CUDA for gpu-marked test")
        except Exception:
            pytest.skip("Torch not available")
    if "ray" in item.keywords:
        try:
            import ray

            ray.init()
            assert ray.is_initialized()
        except Exception:
            pytest.skip("ray not available")
