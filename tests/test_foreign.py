import io

import numpy as np
from PIL import Image

from fotomedik.cli import repair_bytes
from fotomedik.jpeg import markers as M
from fotomedik.recipes.r4_foreign import SECTOR, foreign_runs, sector_kind


def test_sector_kinds():
    rng = np.random.default_rng(0)
    assert sector_kind(bytes(SECTOR)) == "zeros"
    assert sector_kind(b"\xff" * SECTOR) == "fill FF"
    assert sector_kind(b"\xff\xd8\xff\xe1" + rng.bytes(SECTOR - 4)) == "other file"
    assert sector_kind((b"The quick brown fox jumps over the lazy dog. " * 20)[:SECTOR]) == "text"
    assert sector_kind(rng.bytes(SECTOR)) is None
    # flat areas of a JPEG repeat a short bit pattern: low entropy, but not foreign
    assert sector_kind((bytes.fromhex("0028a0028a") * 120)[:SECTOR]) is None


def test_zero_sectors_in_scan_are_blanked_and_repaired(healthy_jpeg_rst):
    data = healthy_jpeg_rst
    res = M.parse_segments(data)
    a = -(-(res.scan_start + 2000) // SECTOR) * SECTOR
    assert a + 2 * SECTOR < res.eoi_offset
    broken = data[:a] + bytes(2 * SECTOR) + data[a + 2 * SECTOR :]
    assert foreign_runs(broken, res.scan_start, len(broken)) == [(a, a + 2 * SECTOR, "zeros")]
    ctx, rep = repair_bytes(broken, "x.jpg", {"time_budget": 1.0})
    assert "foreign_sectors" in ctx.applied
    Image.open(io.BytesIO(ctx.data)).load()
    # the image outside the damaged stretch is the original
    good = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=float)
    got = np.asarray(Image.open(io.BytesIO(ctx.data)).convert("L"), dtype=float)
    assert np.abs(good[:16] - got[:16]).mean() < 1
    assert rep.recovered_pct is not None and rep.recovered_pct > 50


def test_healthy_photo_has_no_foreign_sectors(healthy_jpeg, healthy_jpeg_rst):
    for d in (healthy_jpeg, healthy_jpeg_rst):
        res = M.parse_segments(d)
        assert foreign_runs(d, res.scan_start, res.eoi_offset) == []
