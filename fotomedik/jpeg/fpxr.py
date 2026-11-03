"""The FlashPix preview ("screennail") some cameras store in APP2 FPXR segments.

Kodak cameras of the 2000s keep a 640x480 JPEG preview there besides the 160x120 Exif
thumbnail: sixteen times as many pixels to rebuild holes from. Its bytes are split over
the stream-data segments (type 2) of one FlashPix stream, each with its offset in it.
Read-only; the segments are found by their signature, so a header rotted around them
does not hide them.
"""

from __future__ import annotations

import re

SEARCH_LIMIT = 1 << 20
_FPXR = re.compile(rb"\xff\xe2(..)FPXR\x00", re.DOTALL)
HEAD = 13  # "FPXR\0", version, type, stream index (2), offset in the stream (4)


def screennail(data: bytes, limit: int = SEARCH_LIMIT) -> bytes | None:
    """The preview JPEG (SOI up to the last EOI of its stream), or None."""
    streams: dict[int, list[tuple[int, bytes]]] = {}
    for m in _FPXR.finditer(data[:limit]):
        length = int.from_bytes(m.group(1), "big")
        p = data[m.start() + 4 : m.start() + 2 + length]
        if len(p) < HEAD or p[6] != 2:
            continue
        idx = int.from_bytes(p[7:9], "big")
        off = int.from_bytes(p[9:13], "big")
        streams.setdefault(idx, []).append((off, p[HEAD:]))
    for idx in sorted(streams):
        buf = b"".join(c for _, c in sorted(streams[idx], key=lambda t: t[0]))
        i = buf.find(b"\xff\xd8\xff")
        if i >= 0:
            j = buf.rfind(b"\xff\xd9")
            return buf[i : j + 2] if j > i else buf[i:]
    return None
