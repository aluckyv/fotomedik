import io
import json

from PIL import Image

from fotomedik.jpeg import markers as M
from fotomedik.learn.profile import CameraProfile, exif_model
from fotomedik.recipes.base import Context
from fotomedik.recipes.r2_header import HeaderRepair
from fotomedik.recipes.r7_reference import ReferenceHeader
from fotomedik.recipes.r8_thumbnail import ThumbnailFallback
from fotomedik.recipes.r9_holes import FillHoles
from fotomedik.recipes.registry import run_pipeline
from synth import make_image, make_jpeg


def _with_exif(img, model=b"TestCam 1", thumb=True):
    exif = Image.Exif()
    exif[0x010F] = "Acme"
    exif[0x0110] = model.decode() if isinstance(model, bytes) else model
    data = make_jpeg(img, exif=exif.tobytes())
    if not thumb:
        return data
    # Pillow does not write IFD1 thumbnails; append one by hand into a second APP1 is
    # not standard, so instead embed the thumbnail JPEG bytes inside an APP2 segment —
    # the extractor searches APP segments for an embedded FFD8..FFD9 stream.
    t = img.copy()
    t.thumbnail((64, 48))
    tb = io.BytesIO()
    t.save(tb, format="JPEG", quality=70)
    seg = b"\xff\xe2" + (len(tb.getvalue()) + 2 + 6).to_bytes(2, "big") + b"THUMB\x00" + tb.getvalue()
    res = M.parse_segments(data)
    first = res.segments[0]
    return data[: first.end] + seg + data[first.end :]


def test_profile_roundtrip(healthy_jpeg_rst):
    p = CameraProfile.from_jpeg(healthy_jpeg_rst)
    assert p.width == 256 and p.height == 192
    assert p.restart_interval == 16
    assert p.dqt and p.dht and p.sof and p.sos
    j = p.to_json()
    q = CameraProfile.from_json(j)
    assert q == p
    assert json.loads(j)["width"] == 256


def test_profile_reads_exif_model(image):
    data = _with_exif(image, thumb=False)
    assert exif_model(data) == ("Acme", "TestCam 1")
    p = CameraProfile.from_jpeg(data)
    assert p.model == "TestCam 1"


def test_header_bytes_rebuild_opens(healthy_jpeg_rst):
    p = CameraProfile.from_jpeg(healthy_jpeg_rst)
    res = M.parse_segments(healthy_jpeg_rst)
    rebuilt = p.header_bytes(256, 192) + healthy_jpeg_rst[res.scan_start :]
    im = Image.open(io.BytesIO(rebuilt))
    im.load()
    assert im.size == (256, 192)


def test_r7_transplants_header(healthy_jpeg_rst):
    p = CameraProfile.from_jpeg(healthy_jpeg_rst)
    res = M.parse_segments(healthy_jpeg_rst)
    # wreck DQT and DHT contents (not just markers)
    buf = bytearray(healthy_jpeg_rst)
    for seg in res.all(M.DQT) + res.all(M.DHT):
        for i in range(seg.offset + 4, seg.end, 3):
            buf[i] ^= 0x5A
    broken = bytes(buf)
    ctx = run_pipeline(Context(data=broken, name="w.jpg", ref=p), recipes=[HeaderRepair(), ReferenceHeader(), FillHoles()])
    assert ctx.applied[-1] != "fill_holes" or ctx.holes == []
    assert ctx.data[M.parse_segments(ctx.data).scan_start :] == healthy_jpeg_rst[res.scan_start :]
    im = Image.open(io.BytesIO(ctx.data))
    im.load()
    assert im.size == (256, 192)
    assert "reference_header" in ctx.applied


def test_r7_finds_scan_start_when_sos_destroyed(healthy_jpeg_rst):
    p = CameraProfile.from_jpeg(healthy_jpeg_rst)
    res = M.parse_segments(healthy_jpeg_rst)
    sos = res.first(M.SOS)
    buf = bytearray(healthy_jpeg_rst)
    buf[sos.offset] ^= 0x0F  # kill the SOS marker so parsing cannot find the scan
    buf[sos.offset + 1] ^= 0x33
    ctx = run_pipeline(Context(data=bytes(buf), name="s.jpg", ref=p), recipes=[ReferenceHeader()])
    assert M.parse_segments(ctx.data).scan_start is not None
    assert ctx.data[M.parse_segments(ctx.data).scan_start :] == healthy_jpeg_rst[res.scan_start :]


def test_r7_does_nothing_without_reference(healthy_jpeg_rst):
    ctx = run_pipeline(Context(data=healthy_jpeg_rst, name="n.jpg"), recipes=[ReferenceHeader()])
    assert ctx.log == []


def test_r7_picks_matching_candidate(image, healthy_jpeg_rst):
    data = _with_exif(image, model="TestCam 1", thumb=False)
    other = CameraProfile.from_jpeg(make_jpeg(make_image(64, 64), quality=30))
    other.model = "OtherCam"
    good = CameraProfile.from_jpeg(data)
    res = M.parse_segments(data)
    buf = bytearray(data)
    for seg in res.all(M.DHT):
        for i in range(seg.offset + 4, seg.end, 2):
            buf[i] ^= 0x5A
    ctx = Context(data=bytes(buf), name="c.jpg", options={"ref_candidates": [other, good]})
    run_pipeline(ctx, recipes=[HeaderRepair(), ReferenceHeader()])
    assert "reference_header" in ctx.applied
    assert ctx.data[M.parse_segments(ctx.data).scan_start :] == data[res.scan_start :]
    assert "TestCam 1" in ctx.log[-1].message


def test_r8_thumbnail_fallback(image):
    data = _with_exif(image)
    res = M.parse_segments(data)
    # destroy the whole scan
    wrecked = data[: res.scan_start] + bytes((i * 37) & 0xFF for i in range(len(data) - res.scan_start))
    ctx = run_pipeline(Context(data=wrecked, name="t.jpg"), recipes=[ThumbnailFallback()])
    assert ctx.fallback == "thumbnail"
    im = Image.open(io.BytesIO(ctx.data))
    im.load()
    assert im.size == (64, 48)
    assert ctx.suggested_name == "t.thumbnail.jpg"


def test_r8_not_used_when_image_is_mostly_fine(image):
    data = _with_exif(image)
    ctx = run_pipeline(Context(data=data, name="t.jpg"), recipes=[ThumbnailFallback()])
    assert ctx.fallback is None and ctx.data == data
