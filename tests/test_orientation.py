import io

from PIL import Image

from fotomedik.recipes.base import Context
from fotomedik.recipes.r13_orientation import RestoreOrientation, find_orientation, has_orientation


def _exif_jpeg(v: int, order: str = ">") -> bytes:
    ex = Image.Exif()
    ex[0x0112] = v
    buf = io.BytesIO()
    Image.new("RGB", (32, 16), (90, 120, 150)).save(buf, "JPEG", exif=ex.tobytes())
    return buf.getvalue()


def test_finds_a_rotted_orientation():
    d = bytearray(_exif_jpeg(6))
    i = bytes(d).find(b"\x01\x12\x00\x03") if bytes(d).find(b"\x01\x12\x00\x03") >= 0 else bytes(d).find(b"\x12\x01\x03\x00")
    d[i + 2] ^= 0x10  # a rotted bit in the entry
    v_at = i + 9 if d[i] == 0x01 else i + 8
    d[v_at] ^= 0x08  # and one in the value (6 -> 14)
    assert find_orientation(bytes(d)) == 6


def test_puts_orientation_back_on_a_rebuilt_file():
    original = _exif_jpeg(6)
    buf = io.BytesIO()
    Image.new("RGB", (32, 16), (90, 120, 150)).save(buf, "JPEG")
    ctx = Context(data=buf.getvalue(), name="x.jpg", original=original)
    r = RestoreOrientation()
    got = r.detect(ctx)
    assert got == 6
    r.fix(ctx, got)
    assert has_orientation(ctx.data)
    assert Image.open(io.BytesIO(ctx.data)).getexif()[0x0112] == 6
