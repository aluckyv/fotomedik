"""The Exif thumbnail (IFD1 of the APP1 block): where it sits in the file, whether it is
damaged, and how to put a fresh one in its place.

Cameras write a small JPEG preview into the header, and Explorer, galleries and photo
managers show that preview, not the picture. A preview rotted along with the header stays
a mess however well the picture itself is repaired, so the repaired copy gets a new one
made from the picture. The TIFF block is edited in place: only the two IFD1 fields that
point at the preview (JPEGInterchangeFormat and its length) and the APP1 length change,
every other tag keeps its bytes and offsets.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

from PIL import Image

from . import markers as M

JPEG_IF = 0x0201  # IFD1: offset of the thumbnail JPEG (from the TIFF header)
JPEG_IF_LEN = 0x0202  # ... and its length
THUMB_BOX = (160, 120)  # the Exif thumbnail size cameras write
THUMB_QUALITY = 85
APP1_MAX = 65533  # payload + the two length bytes must fit 16 bits
REAL_SHARE = 0.95  # a thumbnail whose scan decodes cleanly to this share counts as whole


@dataclass
class ThumbSlot:
    app1: int  # file offset of the FF E1 marker
    app1_end: int  # file offset right after the segment
    tiff: int  # file offset of the TIFF header ("II" / "MM")
    little: bool
    off_field: int  # file offset of the 4-byte value of JPEGInterchangeFormat
    len_field: int  # ... of JPEGInterchangeFormatLength
    start: int  # file offset of the thumbnail JPEG
    length: int

    @property
    def blob_end(self) -> int:
        return self.start + self.length


def find_slot(data: bytes) -> ThumbSlot | None:
    """The thumbnail's place in the Exif block, or None (no Exif, no IFD1, no thumbnail)."""
    try:
        res = M.parse_segments(data)
    except Exception:  # noqa: BLE001
        return None
    for seg in res.all(M.APP1):
        if not seg.payload.startswith(b"Exif\0\0"):
            continue
        tiff = seg.offset + 4 + 6
        end = seg.end
        if tiff + 8 > end:
            return None
        order = data[tiff : tiff + 2]
        if order not in (b"II", b"MM"):
            return None
        little = order == b"II"
        e = "<" if little else ">"
        try:
            ifd0 = struct.unpack_from(e + "I", data, tiff + 4)[0]
            n0 = struct.unpack_from(e + "H", data, tiff + ifd0)[0]
            ifd1 = struct.unpack_from(e + "I", data, tiff + ifd0 + 2 + 12 * n0)[0]
            if ifd1 == 0 or tiff + ifd1 + 2 > end:
                return None
            n1 = struct.unpack_from(e + "H", data, tiff + ifd1)[0]
            off_field = len_field = None
            start = length = 0
            for k in range(n1):
                at = tiff + ifd1 + 2 + 12 * k
                tag, typ, cnt = struct.unpack_from(e + "HHI", data, at)
                if typ not in (3, 4) or cnt != 1:
                    continue
                val = struct.unpack_from(e + ("H" if typ == 3 else "I"), data, at + 8)[0]
                if tag == JPEG_IF:
                    off_field, start = at + 8, tiff + val
                elif tag == JPEG_IF_LEN:
                    len_field, length = at + 8, val
        except struct.error:
            return None
        if off_field is None or len_field is None or length <= 0 or start < tiff or start + length > end:
            return None
        return ThumbSlot(seg.offset, end, tiff, little, off_field, len_field, start, length)
    return None


def thumbnail_damaged(data: bytes, slot: ThumbSlot | None = None) -> bool | None:
    """True when the embedded thumbnail does not decode cleanly (rotted with the header),
    False when it is whole, None when there is none."""
    slot = slot or find_slot(data)
    if slot is None:
        return None
    blob = data[slot.start : slot.blob_end]
    if not blob.startswith(b"\xff\xd8"):
        return True
    try:
        im = Image.open(io.BytesIO(blob))
        im.load()
    except Exception:  # noqa: BLE001
        return True
    from ..recipes.dcfield import _real_part

    real = _real_part(blob, (im.size[1], im.size[0]))
    return real is not None and float(real.mean()) < REAL_SHARE


def make_thumbnail(picture: Image.Image, box=THUMB_BOX) -> bytes:
    """A camera-like thumbnail JPEG of the picture (as stored, not rotated: the Exif
    orientation applies to both alike)."""
    t = picture.convert("RGB").copy()
    t.thumbnail(box, Image.LANCZOS)
    out = io.BytesIO()
    t.save(out, "JPEG", quality=THUMB_QUALITY)
    return out.getvalue()


def renew(data: bytes, picture: Image.Image, slot: ThumbSlot | None = None) -> bytes | None:
    """The file with a fresh thumbnail of `picture` in the Exif block, or None when there
    is no thumbnail slot or the block cannot take it."""
    slot = slot or find_slot(data)
    if slot is None:
        return None
    new = make_thumbnail(picture)
    e = "<" if slot.little else ">"
    buf = bytearray(data)
    if len(new) <= slot.length:
        # fits where the old one was; the rest of the old bytes are zeroed
        buf[slot.start : slot.start + len(new)] = new
        buf[slot.start + len(new) : slot.blob_end] = bytes(slot.blob_end - slot.start - len(new))
        struct.pack_into(e + "I", buf, slot.len_field, len(new))
        return bytes(buf)
    payload_len = slot.app1_end - slot.app1 - 2
    at_end = slot.blob_end >= slot.app1_end - 2  # the usual layout: the preview closes the block
    grow = len(new) - (slot.length if at_end else 0)
    if payload_len + grow > APP1_MAX:
        return None
    if at_end:
        buf[slot.start : slot.app1_end] = new
    else:
        buf[slot.app1_end : slot.app1_end] = new
        struct.pack_into(e + "I", buf, slot.off_field, slot.app1_end - slot.tiff)
    struct.pack_into(e + "I", buf, slot.len_field, len(new))
    struct.pack_into(">H", buf, slot.app1 + 2, payload_len + grow)
    return bytes(buf)
