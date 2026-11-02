"""The Exif thumbnail: found, judged, renewed in place when it rotted."""

import io
import struct

import numpy as np
from PIL import Image

from fotomedik.jpeg import exifthumb as T
from fotomedik.jpeg import markers as M
from fotomedik.overview import Overview, PhotoFacts
from fotomedik.recipes.base import Context
from fotomedik.recipes.r15_thumbnail import ThumbnailRenew
from fotomedik.recipes.registry import run_pipeline
from fotomedik.scan.scanner import scan_bytes
from fotomedik.train.damage import exif_with_thumbnail
from synth import make_image, make_jpeg


def _with_thumbnail(img: Image.Image, quality: int = 85) -> bytes:
    """A camera-like file: SOI, APP1 with an IFD1 thumbnail, then the picture's segments."""
    data = make_jpeg(img, quality=quality)
    payload = exif_with_thumbnail(img)
    return data[:2] + b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload + data[2:]


def _rot_thumbnail(data: bytes, seed: int = 3) -> bytes:
    slot = T.find_slot(data)
    rng = np.random.default_rng(seed)
    buf = bytearray(data)
    scan = data.find(b"\xff\xda", slot.start) + 40
    for i in range(scan, slot.blob_end - 2, 7):
        buf[i] ^= int(rng.integers(1, 255))
    return bytes(buf)


def _thumb_pixels(data: bytes):
    slot = T.find_slot(data)
    im = Image.open(io.BytesIO(data[slot.start : slot.blob_end]))
    im.load()
    return np.asarray(im.convert("RGB")).astype(int)


def test_slot_found_and_whole():
    data = _with_thumbnail(make_image(320, 240))
    slot = T.find_slot(data)
    assert slot is not None and data[slot.start : slot.start + 2] == b"\xff\xd8"
    assert T.thumbnail_damaged(data) is False
    assert T.find_slot(make_jpeg(make_image(64, 48))) is None
    assert T.thumbnail_damaged(make_jpeg(make_image(64, 48))) is None


def test_rotted_thumbnail_is_renewed_from_the_picture():
    img = make_image(640, 480, seed=11)
    data = _with_thumbnail(img)
    rotted = _rot_thumbnail(data)
    assert T.thumbnail_damaged(rotted) is True
    assert scan_bytes(rotted, "t.jpg").has("thumbnail_damaged")
    assert not scan_bytes(data, "t.jpg").has("thumbnail_damaged")
    ctx = run_pipeline(Context(data=rotted, name="t.jpg"), recipes=[ThumbnailRenew()])
    assert "thumbnail_renew" in ctx.applied
    out = ctx.data
    # the picture's bytes are untouched, the preview is whole again and shows the picture
    assert out[M.parse_segments(out).scan_start :] == data[M.parse_segments(data).scan_start :]
    assert T.thumbnail_damaged(out) is False
    got = _thumb_pixels(out)
    want = np.asarray(img.resize((got.shape[1], got.shape[0]), Image.LANCZOS)).astype(int)
    assert np.abs(got - want).mean() < 8
    im = Image.open(io.BytesIO(out))
    im.load()
    assert im.size == (640, 480)
    # the Exif block still parses: the same tags, only the preview changed
    assert Image.open(io.BytesIO(out)).getexif().get(0x0112) == Image.open(io.BytesIO(data)).getexif().get(0x0112)


def test_bigger_thumbnail_grows_the_block():
    img = make_image(640, 480, seed=12)
    data = _with_thumbnail(img)
    slot = T.find_slot(data)
    # shrink the stored preview to a tiny one so the new one cannot fit in its place
    tiny = T.make_thumbnail(img, (16, 12))
    buf = bytearray(data)
    buf[slot.start : slot.blob_end] = tiny
    struct.pack_into("<I", buf, slot.len_field, len(tiny))
    struct.pack_into(">H", buf, slot.app1 + 2, slot.app1_end - slot.app1 - 2 - (slot.length - len(tiny)))
    small = bytes(buf)
    assert T.find_slot(small).length == len(tiny)
    out = T.renew(small, img)
    assert out is not None
    assert T.thumbnail_damaged(out) is False
    assert T.find_slot(out).length > len(tiny)
    assert out[M.parse_segments(out).scan_start :] == data[M.parse_segments(data).scan_start :]


def test_whole_thumbnail_is_left_alone():
    data = _with_thumbnail(make_image(320, 240))
    ctx = run_pipeline(Context(data=data, name="w.jpg"), recipes=[ThumbnailRenew()])
    assert ctx.data == data and "thumbnail_renew" not in ctx.applied


def test_overview_names_damaged_previews():
    ov = Overview([PhotoFacts("a.jpg", state="healthy", recovered=1.0, thumb="part", thumb_damaged=True),
                   PhotoFacts("b.jpg", state="healthy", recovered=1.0, thumb="whole")])
    preset, why = ov.suggest()
    assert preset == "faithful" and "1 with a damaged preview" in why[0]
