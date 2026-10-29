import numpy as np

from fotomedik.jpeg import markers as M
from fotomedik.learn.profile import CameraProfile
from fotomedik.recipes.base import Context
from fotomedik.recipes.dcfield import thumbnail_image
from fotomedik.recipes.r7_reference import ReferenceHeader, find_scan_start
from fotomedik.recipes.registry import run_pipeline
from synth import make_image, make_jpeg
from test_dcfield import _with_thumbnail


def _rot(data: bytes, lo: int, hi: int, seed: int = 1, rate: float = 0.3) -> bytes:
    rng = np.random.default_rng(seed)
    buf = bytearray(data)
    for o in range(lo, hi):
        if rng.random() < rate:
            buf[o] ^= 1 << int(rng.integers(0, 8))
    return bytes(buf)


def test_eoi_of_the_thumbnail_is_not_the_image_end_and_a_damaged_eoi_counts():
    img = make_image(256, 192)
    data = _with_thumbnail(make_jpeg(img), img)
    damaged_end = data[:-1] + b"\x59"  # FF D9 -> FF 59
    res = M.parse_segments(damaged_end)
    assert res.scan_start is not None
    assert res.eoi_offset == len(damaged_end) - 2  # not the thumbnail's FF D9 inside APP1


def test_scan_start_is_the_main_images_sos_not_the_thumbnails():
    img = make_image(256, 192)
    data = _with_thumbnail(make_jpeg(img), img)
    res = M.parse_segments(data)
    prof = CameraProfile.from_jpeg(data)
    assert find_scan_start(data, prof.sos_bytes) == res.scan_start


def test_destroyed_header_rebuilt_from_a_photo_of_the_set_in_the_right_orientation():
    land = make_jpeg(make_image(320, 240, seed=3))
    port = make_jpeg(make_image(240, 320, seed=4))
    profs = [CameraProfile.from_jpeg(land, source="land"), CameraProfile.from_jpeg(port, source="port")]
    target = make_jpeg(make_image(240, 320, seed=5))  # portrait, same "camera" tables
    scan = M.parse_segments(target).scan_start
    broken = _rot(target, 2, scan - 4, rate=0.5)  # header unreadable, scan intact
    assert not M.parse_segments(broken).ok
    ctx = run_pipeline(Context(data=broken, name="p.jpg", options={"ref_candidates": profs}),
                       recipes=[ReferenceHeader()])
    assert "reference_header" in ctx.applied
    res = M.parse_segments(ctx.data)
    from fotomedik.jpeg.tables import parse_sof

    fi = parse_sof(res.sof.marker, res.sof.payload)
    assert (fi.width, fi.height) == (240, 320)


def test_damaged_thumbnail_is_rescued_with_a_healthy_thumbnail_header():
    img = make_image(320, 240, seed=6)
    good = _with_thumbnail(make_jpeg(img), img, size=(160, 120))
    thumb_start = good.find(b"\xff\xd8", 2)
    thumb_ref = CameraProfile.from_jpeg(good[thumb_start:good.find(b"\xff\xd9", thumb_start) + 2])
    img2 = make_image(320, 240, seed=7)
    data = _with_thumbnail(make_jpeg(img2), img2, size=(160, 120))
    t0 = data.find(b"\xff\xd8", 2)
    broken = _rot(data, t0 + 4, t0 + 300, rate=0.4)  # the thumbnail's own header rots
    assert thumbnail_image(broken) is None
    got = thumbnail_image(broken, [thumb_ref])
    assert got is not None
    pixels, mask = got
    assert pixels.shape[:2] == (120, 160)
