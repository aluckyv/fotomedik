"""A repair measured against an intact copy of the picture."""

import numpy as np
from PIL import Image

from fotomedik import compare as C
from fotomedik.cli import repair_bytes
from fotomedik.jpeg import markers as M
from synth import make_image, make_jpeg


def _damaged(healthy: bytes, start: int, n: int) -> bytes:
    b = bytearray(healthy)
    b[start : start + n] = bytes(n)
    return bytes(b)


def test_compare_tells_right_parts_from_filled_holes(tmp_path):
    img = make_image(640, 480, seed=11)
    healthy = make_jpeg(img, restart_rows=1)
    res = M.parse_segments(healthy)
    broken = _damaged(healthy, res.scan_start + 2000, 1500)
    ctx, rep = repair_bytes(broken, "x.jpg", {"time_budget": 0.5})
    assert rep.holes
    (tmp_path / "x.jpg").write_bytes(ctx.data)
    truth = tmp_path / "truth.jpg"
    img.save(truth, quality=92)
    cmp = C.compare(tmp_path / "x.jpg", truth, {"holes": rep.holes})
    s = cmp.summary()
    assert s["trusted_mcus"] + s["hole_mcus"] == s["mcus"]
    assert s["trusted_error"] < 4 and s["right"] > 0.9 * s["trusted_mcus"] and s["wrong"] == 0
    assert s["hole_error"] > s["trusted_error"]
    assert "right" in cmp.format_text()
    pic = C.side_by_side(tmp_path / "x.jpg", truth, cmp)
    assert pic.size[0] == 3 * pic.size[1] * 640 // 480 or pic.size[0] <= 2400


def test_compare_survives_a_shifted_rotated_copy(tmp_path):
    img = make_image(320, 240, seed=5)
    healthy = make_jpeg(img, restart_rows=1)
    (tmp_path / "a.jpg").write_bytes(healthy)
    # the "intact copy" is two pixels lower and turned on its side, as a re-saved copy may be
    arr = np.asarray(img.convert("RGB"))
    shifted = np.roll(arr, 2, axis=0)
    Image.fromarray(shifted).transpose(Image.ROTATE_90).save(tmp_path / "t.png")
    cmp = C.compare(tmp_path / "a.jpg", tmp_path / "t.png", None)
    assert cmp.holes is None and cmp.err.mean() < 4


def test_compare_follows_a_drifting_copy(tmp_path):
    img = make_image(640, 480, seed=3)
    healthy = make_jpeg(img, restart_rows=1)
    (tmp_path / "a.jpg").write_bytes(healthy)
    # an AI-restored copy: the picture slides by up to ten pixels from one side to the other
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    H, W = arr.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    sy = np.clip(np.rint(yy + 10 * xx / W).astype(int), 0, H - 1)
    sx = np.clip(np.rint(xx - 6 * yy / H).astype(int), 0, W - 1)
    Image.fromarray(arr[sy, sx].astype(np.uint8)).save(tmp_path / "t.png")
    cmp = C.compare(tmp_path / "a.jpg", tmp_path / "t.png", None)
    assert cmp.drift > 6 and cmp.err.mean() < 6, (cmp.drift, cmp.err.mean())
    assert "drifted" in cmp.format_text()
