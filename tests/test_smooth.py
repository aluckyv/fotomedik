import io

import numpy as np
from PIL import Image

from fotomedik.cli import repair_bytes
from fotomedik.jpeg import decoder as D
from fotomedik.jpeg import markers as M
from fotomedik.jpeg.tables import FrameTables
from fotomedik.recipes.r12_smooth import smooth_holes, surface
from synth import flip_bits
from test_decoder import breaking_flip


def _pixels(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=float)


def test_surface_passes_through_block_averages():
    rng = np.random.default_rng(1)
    mean = rng.normal(0, 30, (6, 9))
    p = surface(mean)
    assert p.shape == (48, 72)
    assert np.allclose(p.reshape(6, 8, 9, 8).mean(axis=(1, 3)), mean, atol=1.5)
    flat = surface(np.full((4, 4), 17.0))
    assert np.allclose(flat, 17.0)


def test_smooth_holes_keeps_real_blocks_and_softens_holes(healthy_jpeg_rst):
    res = M.parse_segments(healthy_jpeg_rst)
    ft = FrameTables.from_parse(res)
    sr = D.decode_scan(healthy_jpeg_rst, res, ft)
    total = sr.total_mcus
    holes = np.zeros(total, dtype=bool)
    holes[total // 3 : total // 3 + 20] = True
    new = smooth_holes(healthy_jpeg_rst, holes)
    assert new is not None and new != healthy_jpeg_rst
    a, b = _pixels(healthy_jpeg_rst), _pixels(new)
    _comps, per_row, _r = D._scan_components(ft)
    mcu_px = a.shape[1] // per_row
    first_row = (total // 3) // per_row
    # rows of MCUs well above the hole are untouched
    top = max(0, (first_row - 1) * mcu_px)
    assert np.array_equal(a[:top], b[:top])
    sr2 = D.decode_scan(new, M.parse_segments(new), FrameTables.from_parse(M.parse_segments(new)))
    for cid in sr.dc:
        assert np.array_equal(sr.dc[cid], sr2.dc[cid])


def test_pipeline_smooths_filled_holes(healthy_jpeg_rst):
    res = M.parse_segments(healthy_jpeg_rst)
    off, mask = breaking_flip(healthy_jpeg_rst, res.scan_start + 300, res.scan_start + 900)
    broken = flip_bits(healthy_jpeg_rst, off, mask)
    ctx, rep = repair_bytes(broken, "x.jpg", {"time_budget": 2.0})
    if ctx.holes:
        assert "smooth_holes" in ctx.applied
        Image.open(io.BytesIO(ctx.data)).load()
    ctx2, _ = repair_bytes(broken, "x.jpg", {"time_budget": 2.0, "mark_holes": True})
    assert "smooth_holes" not in ctx2.applied


def test_block_local_coefficients_match_the_full_surface():
    from fotomedik.recipes.r12_smooth import DCT, KEEP, ZIGZAG, hole_coefficients

    rng = np.random.default_rng(1)
    dc = rng.normal(0, 20, (12, 15))
    q = np.full(64, 2.0)
    a = hole_coefficients(dc, np.ones_like(dc, bool), q)
    plane = surface(dc * (q[0] / 8))
    b = (DCT @ plane.reshape(12, 8, 15, 8).transpose(0, 2, 1, 3) @ DCT.T).reshape(12, 15, 64) / 2
    b = np.rint(b[..., ZIGZAG])
    b[..., 0] = 0
    b[..., KEEP:] = 0
    assert np.abs(a - b).max() <= 1
