"""Orientation: keep portrait photos upright after the header was rebuilt.

Phones store a portrait photo sideways and set the Exif Orientation tag; a viewer
turns it upright. When the damaged Exif block could not be kept, the repaired copy
loses that tag and shows sideways. The tag sits in the first IFD as a 12-byte entry
(tag 0112, type SHORT, count 1, value) that is easy to find even with a few rotted
bits; its value is read as the nearest valid orientation and written into a small
new Exif block.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ..jpeg import markers as M
from .base import Context, Recipe
from .registry import register

SEARCH = 1 << 16
MAX_BITS = 3  # rotted bits tolerated in the 9 bytes that identify the entry
VALID = (1, 3, 6, 8)  # upright, upside down, turned right, turned left (mirrored ones are rare)
# the entry's first bytes: tag 0112, type 3 (SHORT), count 1, then the value
ENTRIES = (
    bytes.fromhex("011200030000000100"),  # big endian (MM): value high byte 00, low byte next
    bytes.fromhex("1201030001000000"),  # little endian (II): value low byte next
)
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def has_orientation(data: bytes) -> bool:
    try:
        return Image.open(io.BytesIO(data)).getexif().get(0x0112) is not None
    except Exception:
        return False


def find_orientation(data: bytes) -> int | None:
    """The Orientation value of a (possibly rotted) Exif block, or None."""
    head = np.frombuffer(data[: min(len(data), SEARCH)], dtype=np.uint8)
    best = None
    for pat in ENTRIES:
        p = np.frombuffer(pat, dtype=np.uint8)
        L = len(p)
        if len(head) <= L + 2:
            continue
        dist = sum(_POP[head[j : len(head) - L - 2 + j] ^ p[j]] for j in range(L))
        i = int(np.argmin(dist))
        if dist[i] > MAX_BITS:
            continue
        raw = int(head[i + L])  # the value's low byte
        if best is None or dist[i] < best[0]:
            best = (int(dist[i]), raw)
    if best is None:
        return None
    raw = best[1]
    ham = [(bin(raw ^ v).count("1"), VALID.index(v), v) for v in VALID]
    d, _k, v = min(ham)
    return v if d <= 2 else None


@register
class RestoreOrientation(Recipe):
    name = "orientation"
    priority = 97  # last: on the final file
    description = "Put the Exif orientation back when the rebuilt header lost it"

    def detect(self, ctx: Context):
        if ctx.memo.get("orientation_done") or ctx.fallback is not None or ctx.data == ctx.original:
            return None
        if has_orientation(ctx.data) or not ctx.data.startswith(b"\xff\xd8"):
            return None
        try:
            if any(s.payload.startswith(b"Exif") for s in M.parse_segments(ctx.data).all(M.APP1)):
                return None  # an Exif block is there; leave it alone
        except Exception:
            return None
        v = find_orientation(ctx.original)
        return v if v not in (None, 1) else None

    def fix(self, ctx: Context, v: int) -> None:
        ctx.memo["orientation_done"] = True
        ex = Image.Exif()
        ex[0x0112] = v
        payload = ex.tobytes()
        if not payload.startswith(b"Exif"):
            payload = b"Exif\x00\x00" + payload
        seg = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
        new = ctx.data[:2] + seg + ctx.data[2:]
        names = {3: "upside down", 6: "turned right", 8: "turned left"}
        ctx.replace_data(new, self, f"Exif orientation restored ({names.get(v, v)}), so the photo shows upright",
                         orientation=v)
