"""A neighbouring shot of the same scene colours the holes of a photo without a thumbnail."""

import io
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fotomedik import twins as T
from fotomedik.cli import repair_bytes
from fotomedik.jpeg import markers as M
from synth import make_image, make_jpeg
from test_dcfield import _with_thumbnail


def _scene(dx: int = 0, seed: int = 4) -> Image.Image:
    img = make_image(512, 384, seed=seed)
    d = ImageDraw.Draw(img)
    d.ellipse((180 + dx, 150, 330 + dx, 230), fill=(200, 30, 40))  # lies wholly inside the hole band
    return img


def _holed(healthy: bytes) -> bytes:
    res = M.parse_segments(healthy)
    span = res.eoi_offset - res.scan_start
    b = bytearray(healthy)
    a = res.scan_start + int(span * 0.36)
    n = int(span * 0.24)
    b[a : a + n] = bytes(n)
    return bytes(b)


def _pixels(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(float)


def test_twin_colours_the_holes(tmp_path):
    photo = _scene()
    broken = _holed(make_jpeg(photo, restart_rows=1))  # no thumbnail of its own
    twin = _scene(dx=6)  # the next shot: a little to the side
    tp = tmp_path / "twin.jpg"
    tp.write_bytes(_with_thumbnail(make_jpeg(twin, restart_rows=1), twin, size=(160, 120)))

    plain, _ = repair_bytes(broken, "a.jpg", {"time_budget": 0.5})
    ctx, rep = repair_bytes(broken, "a.jpg", {"time_budget": 0.5, "twin": str(tp)})
    assert rep.twin == "twin.jpg" and "twin.jpg" in rep.format_text("en")

    truth = np.asarray(photo).astype(float)
    hole = np.abs(_pixels(plain.data) - _pixels(ctx.data)).mean(axis=2) > 0  # where the twin changed anything
    band = slice(150, 230)
    e_plain = np.abs(_pixels(plain.data) - truth)[band].mean()
    e_twin = np.abs(_pixels(ctx.data) - truth)[band].mean()
    assert hole.any()
    assert e_twin < 0.8 * e_plain, (e_plain, e_twin)
    # the decoded parts are untouched
    top = slice(0, 96)
    assert np.abs(_pixels(ctx.data)[top] - _pixels(plain.data)[top]).mean() < 0.5


def test_twin_of_another_scene_is_not_used(tmp_path):
    photo = _scene()
    broken = _holed(make_jpeg(photo, restart_rows=1))
    other = Image.fromarray(255 - np.asarray(make_image(512, 384, seed=9)))  # nothing like it
    tp = tmp_path / "other.jpg"
    tp.write_bytes(_with_thumbnail(make_jpeg(other, restart_rows=1), other, size=(160, 120)))
    _ctx, rep = repair_bytes(broken, "a.jpg", {"time_budget": 0.5, "twin": str(tp)})
    assert rep.twin is None


def test_photo_with_its_own_thumbnail_keeps_it(tmp_path):
    photo = _scene()
    broken = _holed(_with_thumbnail(make_jpeg(photo, restart_rows=1), photo, size=(160, 120)))
    twin = _scene(dx=6)
    tp = tmp_path / "twin.jpg"
    tp.write_bytes(_with_thumbnail(make_jpeg(twin, restart_rows=1), twin, size=(160, 120)))
    _ctx, rep = repair_bytes(broken, "a.jpg", {"time_budget": 0.5, "twin": str(tp)})
    assert rep.twin is None


def test_find_twins_pairs_shots_seconds_apart(tmp_path):
    data = make_jpeg(make_image(64, 48))
    names = ["2005-08-11 11.29.17.jpg", "2005-08-11 11.29.21.jpg", "2005-08-11 11.35.00.jpg"]
    for n in names:
        (tmp_path / n).write_bytes(data)
    # a rotted header no longer parses, but its date is still in the first bytes
    rotted = b"\xff\xd8\x00\x01junk 2005:08:11 11:35:20 more" + bytes(200)
    (tmp_path / "IMG_0001.JPG").write_bytes(rotted)
    paths = sorted(tmp_path.iterdir())
    tw = T.find_twins(paths)
    p = {q.name: q for q in paths}
    assert tw[p[names[0]]] == p[names[1]] and tw[p[names[1]]] == p[names[0]]
    assert tw[p[names[2]]] == p["IMG_0001.JPG"]
    assert T.taken_at(p["IMG_0001.JPG"]) == datetime(2005, 8, 11, 11, 35, 20)


def test_plan_hands_each_photo_its_twin(tmp_path):
    from fotomedik.runner import plan

    data = make_jpeg(make_image(64, 48))
    for n in ("2005-08-11 11.29.17.jpg", "2005-08-11 11.29.21.jpg", "2005-08-11 12.00.00.jpg"):
        (tmp_path / n).write_bytes(data)
    files = [(p, Path(p.name)) for p in sorted(tmp_path.iterdir())]
    items = plan(files, {"twins": True})
    got = {p.name: o.get("twin") for p, _r, o in items}
    assert got["2005-08-11 11.29.17.jpg"].endswith("11.29.21.jpg")
    assert got["2005-08-11 12.00.00.jpg"] is None
    assert all(o.get("twin") is None for _p, _r, o in plan(files, {}))
