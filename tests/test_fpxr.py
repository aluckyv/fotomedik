"""The FlashPix preview (Kodak 640x480) found, joined and used for the holes."""

import io

import numpy as np
from PIL import Image
from synth import make_image, make_jpeg

from fotomedik.ai.enhance import preview_rgb
from fotomedik.jpeg.fpxr import screennail


def _app2(index: int, offset: int, chunk: bytes) -> bytes:
    payload = b"FPXR\x00\x00\x02" + index.to_bytes(2, "big") + offset.to_bytes(4, "big") + chunk
    return b"\xff\xe2" + (len(payload) + 2).to_bytes(2, "big") + payload


def _with_preview(photo: bytes, preview: bytes) -> bytes:
    """The photo with the preview in a FlashPix stream split over two APP2 segments (an
    OLE prefix before the JPEG, as Kodak writes it), written second part first."""
    stream = b"\x00" * 40 + preview
    half = len(stream) // 2
    segs = _app2(1, half, stream[half:]) + _app2(1, 0, stream[:half])
    return photo[:2] + b"\xff\xe2\x00\x10FPXR\x00\x00\x01" + b"\x00" * 6 + segs + photo[2:]


def _preview_bytes(w=640, h=480) -> bytes:
    return make_jpeg(make_image().resize((w, h)), quality=85)


def test_screennail_joins_the_stream_in_order():
    pv = _preview_bytes()
    data = _with_preview(make_jpeg(make_image()), pv)
    assert screennail(data) == pv
    assert screennail(make_jpeg(make_image())) is None


def test_whole_preview_replaces_a_smaller_thumbnail():
    img = make_image()
    pv = _preview_bytes(img.width // 2, img.height // 2)
    data = _with_preview(make_jpeg(img), pv)
    small = np.asarray(img.resize((80, 60)).convert("RGB"))
    got, rescued, note = preview_rgb(data, small, None, (img.height, img.width))
    assert got.shape[:2] == (img.height // 2, img.width // 2) and rescued is None
    assert "preview used instead" in note
    ref = np.asarray(Image.open(io.BytesIO(pv)).convert("RGB")).astype(int)
    assert np.abs(got.astype(int) - ref).mean() < 1


def test_no_preview_or_another_shape_leaves_the_thumbnail():
    img = make_image()
    small = np.asarray(img.resize((80, 60)).convert("RGB"))
    got, rescued, note = preview_rgb(make_jpeg(img), small, None, (img.height, img.width))
    assert got is small and rescued is None and note is None
    tall = _with_preview(make_jpeg(img), _preview_bytes(200, 400))  # portrait preview of a landscape photo
    got, rescued, note = preview_rgb(tall, small, None, (img.height, img.width))
    assert got is small and note is None


def test_rotted_preview_lays_its_real_part_over_the_thumbnail():
    img = make_image()
    pv = bytearray(make_jpeg(img, restart_rows=1, quality=85))
    start = len(pv) * 2 // 3
    pv[start : start + 600] = bytes(600)  # a rotted stretch near the end of the scan
    data = _with_preview(make_jpeg(img), bytes(pv))
    small = np.asarray(img.resize((64, 48)).convert("RGB"))
    got, rescued, note = preview_rgb(data, small, None, (img.height, img.width))
    assert note is not None and "where it survived" in note
    assert got is not None and got.shape[:2] == (img.height, img.width) and rescued is None
    got, rescued, note = preview_rgb(data, None, None, (img.height, img.width))  # no thumbnail: a rescued one
    assert got is None and rescued[1].dtype == bool and 0.1 < rescued[1].mean() < 1


def test_preview_pieces_off_the_thumbnail_are_not_trusted():
    """Pieces after a rotted stretch on a wrong level (white bands, build 163) are dropped;
    without a thumbnail only the rows before the first hole count."""
    from fotomedik.ai.enhance import _preview_trusted

    img = np.asarray(make_image().convert("RGB").resize((256, 192)))
    big = img.copy()
    big[128:] = np.clip(big[128:].astype(int) + 70, 0, 255).astype(np.uint8)  # a piece on a wrong level
    real = np.ones((192, 256), bool)
    real[96:112] = False  # the rotted stretch
    base = np.asarray(Image.fromarray(img).resize((64, 48)).resize((256, 192), Image.BICUBIC))
    kept = _preview_trusted(big, real, base, np.ones((192, 256), bool))
    assert kept[:96].mean() > 0.9 and kept[128:].mean() < 0.2
    alone = _preview_trusted(big, real, np.zeros_like(big), np.zeros((192, 256), bool))
    assert alone[:96].all() and not alone[96:].any()
