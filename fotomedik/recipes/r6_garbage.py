"""R6 — foreign data at the end of the file (wrong clusters, overwritten region).

The FF-rule detector (scan.garbage) finds where the image data stops looking
like entropy-coded JPEG. Everything after that is cut and an EOI is restored;
the missing image rows become holes for R9 to fill.
"""

from __future__ import annotations

from ..jpeg import markers as M
from ..scan.garbage import garbage_tail_start
from .base import Context, Recipe
from .registry import register


@register
class GarbageTail(Recipe):
    name = "garbage_tail"
    priority = 30
    description = "Cut non-image data from the end of the file and restore EOI"

    def detect(self, ctx: Context):
        parse = M.parse_segments(ctx.data)
        if parse.scan_start is None:
            return None
        tail = garbage_tail_start(ctx.data, parse.scan_start)
        if tail is None:
            return None
        # If a real EOI sits before the tail, cut there instead (slack after EOI).
        eoi = parse.eoi_offset
        if eoi is not None and parse.scan_start < eoi < tail:
            tail = eoi
        return tail

    def fix(self, ctx: Context, tail: int) -> None:
        d = ctx.data
        removed = len(d) - tail
        new = d[:tail]
        if not new.endswith(b"\xff\xd9"):
            new += b"\xff\xd9"
        ctx.replace_data(new, self, f"removed {removed} bytes of non-image data at the end", removed_bytes=removed, cut_at=tail)
