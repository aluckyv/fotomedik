import io

import numpy as np
from PIL import Image

from fotomedik.jpeg import decoder as D
from fotomedik.jpeg import markers as M
from fotomedik.jpeg.tables import FrameTables
from fotomedik.recipes.base import Context
from fotomedik.recipes.r9_holes import FillHoles
from fotomedik.recipes.r10_tails import TailRecovery
from fotomedik.recipes.registry import run_pipeline
from synth import make_image, make_jpeg


def _decode(data):
    res = M.parse_segments(data)
    ft = FrameTables.from_parse(res)
    return D.decode_scan(data, res, ft), res, ft


def rot(data: bytes, lo: int, hi: int, rate: float = 0.05, seed: int = 3) -> bytes:
    """Dense bit-rot: each byte in [lo, hi) gets a random bit flipped with probability rate*8."""
    rng = np.random.default_rng(seed)
    buf = bytearray(data)
    for o in range(lo, hi):
        if rng.random() < rate * 8:
            buf[o] ^= 1 << int(rng.integers(0, 8))
    return bytes(buf)


def _pixels(data):
    im = Image.open(io.BytesIO(data))
    im.load()
    return np.asarray(im.convert("RGB")).astype(int)


def test_tail_after_rot_is_recovered_and_matches_original():
    healthy = make_jpeg(make_image(512, 256), restart_rows=1)
    ref, res, ft = _decode(healthy)
    iv = ref.intervals[7]  # one MCU row (32 MCUs)
    a = iv.byte_start + (iv.byte_end - iv.byte_start) // 4
    b = iv.byte_start + (iv.byte_end - iv.byte_start) // 2
    broken = rot(healthy, a, b)
    ctx = run_pipeline(Context(data=broken, name="r.jpg"), recipes=[TailRecovery(), FillHoles()])
    assert "fill_holes" in ctx.applied
    spliced = ctx.log[-1].extra["spliced_mcus"]
    assert spliced >= 10  # most of the second half of the row came back
    sr, _, _ = _decode(ctx.data)
    assert sr.error is None and all(i.ok for i in sr.intervals)
    # the recovered right part of row 7 is pixel-identical to the healthy image (same coefficients)
    got, want = _pixels(ctx.data), _pixels(healthy)
    rows = slice(7 * 16, 8 * 16)
    right = slice(512 - (spliced - 1) * 16, 512)  # skip one MCU: smoothing across the splice
    assert np.abs(got[rows, right] - want[rows, right]).mean() < 2.0
    # untouched rows identical (the last pixel rows above row 7 are excluded: libjpeg's
    # fancy chroma upsampling blends chroma across the MCU-row boundary)
    assert np.abs(got[: 7 * 16 - 2] - want[: 7 * 16 - 2]).max() <= 1


def test_rot_across_marker_recovers_following_interval_fully():
    """Rot covers the end of row 4, its RST marker and the start of row 5... no: only row 4's end
    and the marker. Row 5 must come back entirely and row 4's tail-less end is filled."""
    healthy = make_jpeg(make_image(512, 256), restart_rows=1)
    ref, res, ft = _decode(healthy)
    iv4, iv5 = ref.intervals[4], ref.intervals[5]
    a = iv4.byte_start + (iv4.byte_end - iv4.byte_start) // 2
    broken = bytearray(rot(healthy, a, iv4.byte_end + 2, rate=0.06, seed=9))
    broken[iv4.byte_end] = 0x7F  # make sure the marker is unrecognisable (>2 bits off)
    broken[iv4.byte_end + 1] = 0x55
    broken = bytes(broken)
    sr0, _, _ = _decode(broken)
    assert not sr0.intervals[4].ok
    ctx = run_pipeline(Context(data=broken, name="m.jpg"), recipes=[TailRecovery(), FillHoles()])
    sr, _, _ = _decode(ctx.data)
    assert sr.error is None and all(i.ok for i in sr.intervals)
    got, want = _pixels(ctx.data), _pixels(healthy)
    assert np.abs(got[5 * 16 : 6 * 16] - want[5 * 16 : 6 * 16]).mean() < 1.0  # row 5 intact
    assert ctx.recovered_fraction > 1 - 1.0 / 16  # at most half a row of 16 rows lost


def test_hole_fill_interpolates_between_above_and_below():
    healthy = make_jpeg(make_image(512, 256), restart_rows=1)
    ref, res, ft = _decode(healthy)
    iv = ref.intervals[8]
    broken = rot(healthy, iv.byte_start + 2, iv.byte_end - 2, rate=0.1, seed=4)
    ctx = run_pipeline(Context(data=broken, name="i.jpg"), recipes=[FillHoles()])
    new, _, _ = _decode(ctx.data)
    assert any(h.mcu_first // 32 == 8 for h in ctx.holes)
    # DC of the filled band vs the truth: interpolating between the rows above and below
    # beats the old fill (copy of the block row directly above) for every component
    for cid, rows in ((1, [16, 17]), (2, [8]), (3, [8])):
        truth = ref.dc[cid][rows]
        err_new = np.abs(new.dc[cid][rows] - truth).mean()
        err_old = np.abs(ref.dc[cid][rows[0] - 1] - truth).mean()
        assert err_new < err_old, (cid, err_new, err_old)
