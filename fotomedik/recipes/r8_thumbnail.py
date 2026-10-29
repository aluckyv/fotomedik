"""R8 — last resort: the embedded Exif thumbnail.

Cameras store a small JPEG preview inside the header. When nothing of the
main scan can be recovered, that preview is still a picture of the moment.
"""

from __future__ import annotations

import io
import os

from PIL import Image

from ..jpeg import markers as M
from .base import Context, Recipe
from .r7_reference import _recovered
from .registry import register

USELESS_FRACTION = 0.03


def extract_thumbnail(data: bytes) -> bytes | None:
    """The first embedded JPEG inside APPn segments that Pillow can open."""
    res = M.parse_segments(data)
    segs = [s for s in res.segments if 0xE0 <= s.marker <= 0xEF]
    if not segs and len(data) > 4:
        # header may be unparsable; scan the first 256 KB raw
        segs = []
        raw = data[: 256 * 1024]
        return _first_jpeg(raw, skip_leading=True)
    for s in segs:
        found = _first_jpeg(s.payload, skip_leading=False)
        if found:
            return found
    return None


def _first_jpeg(blob: bytes, skip_leading: bool) -> bytes | None:
    p = 2 if skip_leading else 0
    while True:
        i = blob.find(b"\xff\xd8\xff", p)
        if i < 0:
            return None
        j = blob.find(b"\xff\xd9", i + 3)
        if j < 0:
            return None
        cand = blob[i : j + 2]
        try:
            im = Image.open(io.BytesIO(cand))
            im.load()
            if im.size[0] >= 16 and im.size[1] >= 16:
                return cand
        except Exception:
            pass
        p = i + 3


@register
class ThumbnailFallback(Recipe):
    name = "thumbnail_fallback"
    priority = 90
    description = "Use the embedded Exif thumbnail when the main image is beyond recovery"

    def detect(self, ctx: Context):
        if ctx.fallback is not None:
            return None
        frac, sr = _recovered(ctx.data)
        if sr is not None and frac >= USELESS_FRACTION:
            return None
        return extract_thumbnail(ctx.data)

    def fix(self, ctx: Context, thumb: bytes) -> None:
        im = Image.open(io.BytesIO(thumb))
        ctx.fallback = "thumbnail"
        stem, _ = os.path.splitext(ctx.name)
        ctx.suggested_name = f"{stem}.thumbnail.jpg"
        ctx.recovered_fraction = 0.0
        ctx.replace_data(thumb, self, f"main image unrecoverable; embedded thumbnail {im.size[0]}x{im.size[1]} used instead", size=list(im.size))
