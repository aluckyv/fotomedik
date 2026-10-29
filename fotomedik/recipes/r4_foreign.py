"""Foreign sectors: card sectors inside the photo that are not photo data.

A dying card or a recovery program can leave sectors of zeros (unreadable
sectors), a repeated byte, text, file-system tables or the start of another file
inside a photo. Entropy-coded JPEG data looks random: it never holds a run of one
byte value, hardly any zero bytes and no long stretches of plain text. A sector of
zeros is the worst case, because it even decodes without an error, as silent
garbage that drags everything after it out of place.

Such sectors are blanked with fill bytes ending in a reserved marker (FF .. FF 01).
The length of the file stays the same, so the real data keeps its position, and
every decoder stops exactly there; the scan repairs (tails, segment placement,
hole filling) then treat the sector as the damage it is.
"""

from __future__ import annotations

import numpy as np

from ..jpeg import markers as M
from .base import Context, Recipe
from .registry import register

SECTOR = 512
SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF", b"PK\x03\x04", b"%PDF", b"Exif\x00", b"II*\x00", b"MM\x00*")
ZERO_SHARE = 0.4  # JPEG data has ~0.4 % zero bytes (after FF00 stuffing a little more), never this many
TEXT_SHARE = 0.95  # printable share of a text sector (random data: ~37 %)


def sector_kind(b: bytes) -> str | None:
    """Why a sector cannot be entropy-coded JPEG data, or None when it can."""
    a = np.frombuffer(b, dtype=np.uint8)
    if len(a) == 0:
        return None
    if (a == a[0]).all():
        return "zeros" if a[0] == 0 else f"fill {a[0]:02X}"
    if any(b.startswith(s) for s in SIGNATURES):
        return "other file"
    if (a == 0).mean() > ZERO_SHARE:
        return "zeros"
    printable = ((a >= 32) & (a < 127)) | (a == 9) | (a == 10) | (a == 13)
    gaps = (a == 32) | (a == 10) | (a == 13) | (a == 9)
    if printable.mean() > TEXT_SHARE and gaps.mean() > 0.05:  # words, not a repeated byte pattern
        return "text"
    return None


def foreign_runs(data: bytes, start: int, end: int) -> list[tuple[int, int, str]]:
    """(first byte, end byte, kind) of runs of foreign sectors in data[start:end].
    Sectors are aligned to the file start (photos start on a sector boundary)."""
    runs: list[list] = []
    first = -(-start // SECTOR) * SECTOR
    for o in range(first, end - SECTOR + 1, SECTOR):
        k = sector_kind(data[o : o + SECTOR])
        if k is None:
            continue
        if runs and runs[-1][1] == o:
            runs[-1][1] = o + SECTOR
            if k not in runs[-1][2]:
                runs[-1][2].append(k)
        else:
            runs.append([o, o + SECTOR, [k]])
    return [(a, b, "/".join(k)) for a, b, k in runs]


def blank(data: bytes, runs) -> bytes:
    buf = bytearray(data)
    for a, b, _k in runs:
        buf[a:b] = b"\xff" * (b - a - 1) + b"\x01"
    return bytes(buf)


@register
class ForeignSectors(Recipe):
    name = "foreign_sectors"
    priority = 35  # once the header is sound (scan start known), before any scan repair
    description = "Blank card sectors inside the scan that are not photo data (zeros, text, other files)"

    def detect(self, ctx: Context):
        if ctx.memo.get("foreign_done"):
            return None
        try:
            parse = M.parse_segments(ctx.data)
        except Exception:
            return None
        if parse.scan_start is None:
            return None
        end = parse.eoi_offset if parse.eoi_offset is not None else len(ctx.data)
        runs = foreign_runs(ctx.data, parse.scan_start, end)
        return runs or None

    def fix(self, ctx: Context, runs) -> None:
        ctx.memo["foreign_done"] = True
        n = sum((b - a) // SECTOR for a, b, _k in runs)
        kinds = sorted({k for _a, _b, ks in runs for k in ks.split("/")})
        where = ", ".join(f"{a}" for a, _b, _k in runs[:5]) + (" …" if len(runs) > 5 else "")
        ctx.replace_data(blank(ctx.data, runs), self,
                         f"{n} sector(s) of foreign data ({', '.join(kinds)}) blanked at byte {where}",
                         sectors=n, runs=[[a, b, k] for a, b, k in runs])
