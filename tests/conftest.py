import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synth import make_image, make_jpeg  # noqa: E402


@pytest.fixture(scope="session")
def image():
    return make_image()


@pytest.fixture(scope="session")
def healthy_jpeg(image) -> bytes:
    return make_jpeg(image)


@pytest.fixture(scope="session")
def healthy_jpeg_rst(image) -> bytes:
    return make_jpeg(image, restart_rows=1)


@pytest.fixture(scope="session")
def healthy_progressive(image) -> bytes:
    return make_jpeg(image, progressive=True)
