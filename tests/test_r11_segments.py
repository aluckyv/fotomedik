import io

import numpy as np
from PIL import Image

from fotomedik.jpeg import decoder as D
from fotomedik.jpeg import markers as M
from fotomedik.jpeg.tables import FrameTables
from fotomedik.recipes.base import Context
from fotomedik.recipes.r9_holes import FillHoles
from fotomedik.recipes.r10_tails import TailRecovery
from fotomedik.recipes.r11_segments import SegmentPlacement, intact_segments, merge_blocks
from fotomedik.recipes.registry import run_pipeline
from synth import make_image, make_jpeg
from test_dcfield import _with_thumbnail


def _pixels(data):
    im = Image.open(io.BytesIO(data))
    im.load()
    return np.asarray(im.convert("RGB")).astype(int)


def _scan(data):
    res = M.parse_segments(data)
    return res, FrameTables.from_parse(res)


def test_intact_segments_split_at_broken_ff():
    data = bytes(100) + b"\xff\x12" + bytes(2000) + b"\xff\x00" + bytes(1000) + b"\xff\x34" + bytes(10)
    segs = intact_segments(data, 0, len(data), min_len=512)
    assert segs == [(102, 102 + 2000 + 2 + 1000)]


def test_merge_blocks_joins_contiguous_runs():
    assert merge_blocks([(0, 10, 32), (32, 42, 32), (64, 80, 32)]) == [(0, 10, 64), (64, 80, 32)]


def test_stretch_after_damage_without_restart_markers_is_placed():
    """No restart markers: the decoder stops at the first rotted stretch, R10 recovers the
    data after the last one; what lies between them only R11 can place (thumbnail match)."""
    img = make_image(512, 384)
    healthy = _with_thumbnail(make_jpeg(img), img, size=(160, 120))
    res, ft = _scan(healthy)
    span = res.eoi_offset - res.scan_start
    buf = bytearray(healthy)
    rng = np.random.default_rng(2)
    for a in (res.scan_start + span // 4, res.scan_start + 3 * span // 4):  # two rotted stretches
        for o in range(a, a + 600):
            if rng.random() < 0.3:
                buf[o] ^= 1 << int(rng.integers(0, 8))
    broken = bytes(buf)
    base = run_pipeline(Context(data=broken, name="b.jpg"), recipes=[TailRecovery(), FillHoles()])
    ctx = run_pipeline(Context(data=broken, name="s.jpg"), recipes=[TailRecovery(), SegmentPlacement(), FillHoles()])
    assert ctx.memo.get("segments_done")
    assert ctx.recovered_fraction > base.recovered_fraction + 0.2
    sr = D.decode_scan(ctx.data, *_scan(ctx.data))
    assert sr.error is None
    # where real data was put back, it matches the original picture
    got, want = _pixels(ctx.data), _pixels(healthy)
    err = np.abs(got - want).mean(axis=2)
    placed_rows = [r for r in range(0, 384, 16) if err[r : r + 16].mean() < 3]
    assert len(placed_rows) >= 10


def test_stretch_is_placed_without_any_reference():
    """No thumbnail, nothing clean next to the stretch: the chain's own seam (the end of
    every picture row beside the start of the next) tells its column; the row comes from
    the byte position, so it may be off by a row or two."""
    img = make_image(512, 384)
    healthy = make_jpeg(img)  # no thumbnail, no restart markers
    res, ft = _scan(healthy)
    span = res.eoi_offset - res.scan_start
    buf = bytearray(healthy)
    rng = np.random.default_rng(2)
    for a in (res.scan_start + span // 4, res.scan_start + 3 * span // 4):
        for o in range(a, a + 600):
            if rng.random() < 0.3:
                buf[o] ^= 1 << int(rng.integers(0, 8))
    broken = bytes(buf)
    base = run_pipeline(Context(data=broken, name="b.jpg"), recipes=[TailRecovery(), FillHoles()])
    ctx = run_pipeline(Context(data=broken, name="s.jpg"), recipes=[TailRecovery(), SegmentPlacement(), FillHoles()])
    assert ctx.recovered_fraction > base.recovered_fraction + 0.2
    got, want = _pixels(ctx.data), _pixels(healthy)
    mh = 16
    # rows of the output that match a row of the original at most 3 MCU rows away: right column
    matched = 0
    for r in range(0, 384 - mh + 1, mh):
        best = min(np.abs(got[r : r + mh] - want[q : q + mh]).mean() for q in range(max(0, r - 3 * mh), min(384 - mh, r + 3 * mh) + 1, mh))
        matched += best < 4
    assert matched >= 12, matched
