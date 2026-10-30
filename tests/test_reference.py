"""A reference picture guiding the repair: pairing, alignment, calibration, and the
holes carrying its pixels while decoded MCUs stay as they were."""

import numpy as np
from PIL import Image

from fotomedik import compare as C
from fotomedik import reference as R
from fotomedik.cli import repair_bytes
from fotomedik.jpeg import markers as M
from synth import make_image, make_jpeg


def _textured(w=640, h=480, seed=4) -> Image.Image:
    """A picture with texture everywhere (the thumbnail cannot fake it): gradients plus
    a dense pattern of blobs and lines."""
    rng = np.random.default_rng(seed)
    base = np.asarray(make_image(w, h, seed=seed).convert("RGB")).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    tex = 25 * np.sin(xx / 3.1 + yy / 7.3) * np.cos(yy / 4.7 - xx / 11.0)
    noise = rng.normal(0, 6, (h, w))
    out = base + (tex + noise)[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _warped(img: Image.Image, dy_top=-3.0, dy_bottom=4.0, dx=2.0, gain=1.12, offset=10.0) -> Image.Image:
    """What an editor or an image model gives back: shifted by a smoothly varying amount,
    brighter and with more contrast."""
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dy = dy_top + (dy_bottom - dy_top) * yy / h
    sy = np.clip(np.rint(yy - dy).astype(int), 0, h - 1)
    sx = np.clip(np.rint(xx - dx).astype(int), 0, w - 1)
    return Image.fromarray(np.clip(arr[sy, sx] * gain + offset, 0, 255).astype(np.uint8))


def _damaged(healthy: bytes, seed=1, stretches=6, length=1500) -> bytes:
    res = M.parse_segments(healthy)
    rng = np.random.default_rng(seed)
    b = bytearray(healthy)
    for _ in range(stretches):
        s0 = res.scan_start + int(rng.integers(0, len(healthy) - res.scan_start - 2 * length))
        b[s0 : s0 + length] = bytes(length)
    return bytes(b)


def test_shift_field_recovers_a_smooth_warp():
    img = _textured()
    real = np.asarray(img.convert("L"), np.float32) - 128
    ref = np.asarray(_warped(img).convert("L"), np.float32) - 128
    trusted = np.ones(real.shape, bool)
    trusted[150:230] = False  # a band of holes
    dy, dx = R.shift_field(ref, real, trusted)
    # the reference sits where the truth was moved to: dy -3 at the top .. +4 at the bottom, dx +2
    assert abs(dy[20, 320] - (-3)) <= 1.0 and abs(dy[460, 320] - 4) <= 1.2
    assert np.abs(dx - 2).max() <= 1.0
    al = R.warp(ref, dy, dx)
    a, b = R.calibrate(al, real, trusted)
    assert 0.85 <= a <= 0.95  # 1 / 1.12
    assert np.abs(a * al + b - real)[~trusted].mean() < 6


def test_pairing_by_name_and_by_content(tmp_path):
    img = _textured(320, 240, seed=7)
    other = make_image(320, 240, seed=9)
    photos = []
    for name, im in (("IMG_0001.JPG", img), ("IMG_0002.JPG", other)):
        p = tmp_path / name
        p.write_bytes(make_jpeg(im, restart_rows=1))
        photos.append(p)
    refs = tmp_path / "refs"
    refs.mkdir()
    _warped(other).save(refs / "IMG_0002_gemini.jpg")  # by name
    _warped(img).save(refs / "restored copy.jpg")  # only by content
    from PIL import ImageOps

    ImageOps.invert(make_image(320, 240, seed=3).convert("RGB")).transpose(Image.FLIP_LEFT_RIGHT).save(refs / "unrelated.jpg")
    got = R.pair(photos, refs)
    assert got[photos[1]].name == "IMG_0002_gemini.jpg"
    assert got[photos[0]].name == "restored copy.jpg"


def test_guided_repair_fills_holes_from_the_reference(tmp_path):
    img = _textured()
    healthy = make_jpeg(img, restart_rows=1)
    broken = _damaged(healthy)
    truth = tmp_path / "truth.png"
    img.save(truth)
    ref = tmp_path / "ref.jpg"
    _warped(img).save(ref, quality=92)
    plain, rep0 = repair_bytes(broken, "x.jpg", {"time_budget": 0.5})
    guided, rep1 = repair_bytes(broken, "x.jpg", {"time_budget": 0.5, "reference": str(ref)})
    assert rep0.holes and rep1.holes and rep1.reference == "ref.jpg"
    (tmp_path / "plain.jpg").write_bytes(plain.data)
    (tmp_path / "guided.jpg").write_bytes(guided.data)
    c0 = C.compare(tmp_path / "plain.jpg", truth, {"holes": rep0.holes}).summary()
    c1 = C.compare(tmp_path / "guided.jpg", truth, {"holes": rep1.holes}).summary()
    assert c1["hole_error"] < 0.5 * c0["hole_error"]  # the holes carry the picture, not a blur
    assert c1["trusted_error"] <= c0["trusted_error"] + 0.5  # decoded parts as good as before
    # decoded MCUs are the same bits: their luma agrees exactly in rows no hole touches
    a = np.asarray(Image.open(tmp_path / "plain.jpg").convert("YCbCr"), np.int16)[..., 0]
    b = np.asarray(Image.open(tmp_path / "guided.jpg").convert("YCbCr"), np.int16)[..., 0]
    per_row = 640 // 16
    hole_rows = {h["mcu_first"] // per_row + k for rep in (rep0, rep1) for h in rep.holes
                 for k in range(h["mcu_count"] // per_row + 2)}
    clean = [r for r in range(480 // 16) if r not in hole_rows and r - 1 not in hole_rows and r + 1 not in hole_rows]
    assert clean
    for r in clean:
        assert np.array_equal(a[r * 16 : (r + 1) * 16], b[r * 16 : (r + 1) * 16])
