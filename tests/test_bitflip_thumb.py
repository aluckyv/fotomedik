"""A bit flip that makes another photo's sectors decode 'cleanly' must not be kept:
the thumbnail shows the colours after it are wrong (see r5_bitflip.thumbnail_rejects)."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench import damage, exif_with_thumbnail  # noqa: E402

from fotomedik.cli import repair_bytes  # noqa: E402
from synth import make_image  # noqa: E402


def _photo(seed: int) -> tuple[bytes, np.ndarray]:
    img = make_image(640, 480, seed=seed)
    b = io.BytesIO()
    img.save(b, "JPEG", quality=90, exif=exif_with_thumbnail(img))
    return b.getvalue(), np.asarray(img).astype(int)


@pytest.mark.parametrize("seed", [35, 1, 21, 29])  # 35: a flip made foreign data decode
def test_damaged_photo_keeps_its_colours(seed):
    data, truth = _photo(3)
    other, _ = _photo(40 + seed % 6)
    bad, _what = damage(data, other, seed=7000 + seed)
    ctx, _rep = repair_bytes(bad, "x.jpg", {"time_budget": 1.0})
    got = np.asarray(Image.open(io.BytesIO(ctx.data)).convert("RGB")).astype(int)
    assert got.shape == truth.shape
    # colour per 32x32 tile: blur and detail may be lost, a colour cast must not remain
    t = truth.reshape(15, 32, 20, 32, 3).mean(axis=(1, 3))
    g = got.reshape(15, 32, 20, 32, 3).mean(axis=(1, 3))
    off = np.abs(t - g).max(axis=-1)
    assert np.mean(off > 40) < 0.1
