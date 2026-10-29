import io

import numpy as np
from PIL import Image, ImageDraw

from fotomedik.jpeg import decoder as D
from fotomedik.jpeg import markers as M
from fotomedik.jpeg.tables import FrameTables
from fotomedik.recipes.base import Context
from fotomedik.recipes.dcfield import build_field, thumbnail_base, vertical_reference
from fotomedik.recipes.r9_holes import FillHoles
from fotomedik.recipes.registry import run_pipeline
from synth import make_image, make_jpeg


def _with_thumbnail(data: bytes, img: Image.Image, size=(128, 64)) -> bytes:
    t = img.copy()
    t.thumbnail(size)
    tb = io.BytesIO()
    t.save(tb, format="JPEG", quality=85)
    seg = b"\xff\xe1" + (len(tb.getvalue()) + 2 + 6).to_bytes(2, "big") + b"Exif\x00\x00" + tb.getvalue()
    return data[:2] + seg + data[2:]


def _decode(data):
    res = M.parse_segments(data)
    ft = FrameTables.from_parse(res)
    return D.decode_scan(data, res, ft), res, ft


def _striped():
    """A red band that lives only in MCU rows 7-9: interpolating from rows 6 and 10 cannot know it."""
    img = make_image(512, 256)
    ImageDraw.Draw(img).rectangle((0, 7 * 16 + 4, 511, 10 * 16 - 4), fill=(220, 30, 30))
    return img


def _kill_rows(data, sr, rows):
    buf = bytearray(data)
    for r in rows:
        iv = sr.intervals[r]
        for o in range(iv.byte_start + 1, iv.byte_end - 1):
            buf[o] = 0x5A  # garbage: decodes to nothing sensible
    return bytes(buf)


def test_vertical_reference_interpolates_and_reports_distance():
    arr = np.array([[0.0], [np.nan], [np.nan], [30.0]])
    okm = np.array([[True], [False], [False], [True]])
    val, dist = vertical_reference(arr, okm)
    assert np.allclose(val[:, 0], [0, 10, 20, 30])
    assert list(dist[:, 0]) == [0, 1, 1, 0]


def test_field_with_base_follows_base_far_from_clean_rows():
    arr = np.zeros((20, 1))
    okm = np.zeros((20, 1), bool)
    okm[0] = True
    arr[0] = 5.0
    base = np.full((20, 1), 50.0)
    f = build_field(arr, okm, 8, base, 2.0)
    assert f.val[0, 0] == 5.0  # clean block: itself
    assert abs(f.val[1, 0] - 5.0) < 1e-6  # next to it: base + full correction (-45)
    assert abs(f.val[19, 0] - 50.0) < 1.0  # far away: the thumbnail's value


def test_thumbnail_guided_fill_restores_band_in_lost_rows():
    img = _striped()
    healthy = make_jpeg(img, restart_rows=1)
    with_thumb = _with_thumbnail(healthy, img)
    ref, _, _ = _decode(with_thumb)
    broken = _kill_rows(with_thumb, ref, [7, 8, 9])
    no_thumb = _kill_rows(healthy, _decode(healthy)[0], [7, 8, 9])

    sr0, res0, ft0 = _decode(broken)
    clean = sr0.mcu_ok.copy()
    for iv in sr0.intervals:
        if not iv.ok:
            clean[iv.mcu_first : iv.mcu_first + iv.mcu_count] = False
    tb = thumbnail_base(broken, ft0, sr0, clean)
    assert tb is not None and tb.r > 0.9

    got = run_pipeline(Context(data=broken, name="t.jpg"), recipes=[FillHoles()])
    base = run_pipeline(Context(data=no_thumb, name="n.jpg"), recipes=[FillHoles()])
    new, _, _ = _decode(got.data)
    old, _, _ = _decode(base.data)
    for cid in (1, 3):  # luma and Cr carry the red band
        rows = slice(7 * (2 if cid == 1 else 1), 10 * (2 if cid == 1 else 1))
        truth = ref.dc[cid][rows]
        err_new = np.abs(new.dc[cid][rows] - truth).mean()
        err_old = np.abs(old.dc[cid][rows] - truth).mean()
        assert err_new < 0.5 * err_old, (cid, err_new, err_old)
