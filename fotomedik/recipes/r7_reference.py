"""R7 — transplant the header from a healthy photo of the same camera.

When the header is too damaged for bit-level repair (tables full of flipped
values, SOS unreadable) the scan data may still be fine. A CameraProfile from a
healthy shot supplies DQT/DHT/SOF/DRI/SOS; only the dimensions and the scan
start have to be recovered from the damaged file.
"""

from __future__ import annotations

import numpy as np

from ..jpeg import decoder as D
from ..jpeg import markers as M
from ..jpeg.tables import FrameTables, TableError, parse_sof
from ..learn.profile import CameraProfile
from .base import Context, Recipe
from .r2_header import _header_healthy, ham_bytes
from .registry import register

SEARCH_LIMIT = 512 * 1024
MAX_SOS_MISMATCH = 0.25  # fraction of bits allowed to differ when locating the SOS by pattern
MIN_GAIN = 0.02
MIN_MATCH = 3  # profile.matches(): 10 = same Exif model, 3 = same dimensions

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def find_scan_start(data: bytes, sos_bytes: bytes, limit: int = SEARCH_LIMIT) -> int | None:
    """Offset right after the best approximate match of the reference SOS segment."""
    n = min(len(data), limit)
    L = len(sos_bytes)
    if n <= L:
        return None
    arr = np.frombuffer(data[:n], dtype=np.uint8)
    dist = np.zeros(n - L + 1, dtype=np.int32)
    for j in range(L):
        dist += _POPCOUNT[np.bitwise_xor(arr[j : n - L + 1 + j], sos_bytes[j])]
    i = int(np.argmin(dist))
    if dist[i] > MAX_SOS_MISMATCH * 8 * L:
        return None
    return i + L


def _recovered(data: bytes) -> tuple[float, D.ScanResult | None]:
    parse = M.parse_segments(data)
    if not parse.ok:
        return 0.0, None
    try:
        ft = FrameTables.from_parse(parse)
        sr = D.decode_scan(data, parse, ft)
    except (TableError, D.MissingTables, D.Unsupported):
        return 0.0, None
    return sr.recovered_fraction, sr


def _tables_near(data: bytes, prof: CameraProfile) -> bool:
    """True when the file's tables differ from the profile's by a few bits (silent damage)."""
    res = M.parse_segments(data)
    for kind, refs in ((M.DQT, prof.dqt), (M.DHT, prof.dht)):
        segs = res.all(kind)
        if len(segs) != len(refs):
            continue
        for s, r in zip(segs, refs):
            if len(s.payload) == len(r):
                h = ham_bytes(s.payload, r)
                if 0 < h <= max(2, len(r) // 8):
                    return True
    return False


@register
class ReferenceHeader(Recipe):
    name = "reference_header"
    priority = 25  # after bit-level header repair, before scan repairs
    description = "Replace a destroyed header with one learned from a healthy photo of the same camera"

    def _profiles(self, ctx: Context) -> list[CameraProfile]:
        out: list[CameraProfile] = []
        if ctx.ref is not None:
            out.append(ctx.ref)
        out += list(ctx.options.get("ref_candidates", []))
        return out

    def detect(self, ctx: Context):
        profiles = self._profiles(ctx)
        if not profiles or ctx.memo.get("reference_tried"):
            return None
        d = ctx.data
        needs = not _header_healthy(d)
        if not needs:
            cur, _ = _recovered(d)
            needs = cur < 0.05 or any(_tables_near(d, p) for p in profiles)
        if not needs:
            return None
        # An explicit --ref is trusted; auto-learned candidates must look like the same
        # camera (Exif model or identical dimensions), otherwise a foreign header would
        # turn the scan into coloured noise that still "decodes".
        if ctx.ref is not None:
            return ctx.ref
        scored = sorted(((p.matches(d), p) for p in profiles), key=lambda t: t[0], reverse=True)
        if not scored or scored[0][0] < MIN_MATCH:
            ctx.memo["reference_tried"] = True
            return None
        return scored[0][1]

    def fix(self, ctx: Context, prof: CameraProfile) -> None:
        ctx.memo["reference_tried"] = True
        d = ctx.data
        parse = M.parse_segments(d)
        current, _ = _recovered(d)

        # scan start: trust the parsed one if the reference header decodes from there
        starts: list[int] = []
        if parse.scan_start is not None:
            starts.append(parse.scan_start)
        found = find_scan_start(d, prof.sos_bytes)
        if found is not None and found not in starts:
            starts.append(found)
        if not starts:
            ctx.note(self, "reference available but the scan start could not be located")
            return

        # dimensions: the damaged SOF if plausible, else the reference
        w, h = prof.width, prof.height
        if parse.sof is not None:
            try:
                fi = parse_sof(parse.sof.marker, parse.sof.payload)
                if 0 < fi.width <= 20000 and 0 < fi.height <= 20000:
                    near = ham_bytes(fi.width.to_bytes(2, "big") + fi.height.to_bytes(2, "big"),
                                     prof.width.to_bytes(2, "big") + prof.height.to_bytes(2, "big")) <= 2
                    if not near:
                        w, h = fi.width, fi.height
            except TableError:
                pass

        # keep the damaged file's Exif block when it parsed, so metadata survives
        extra: list[bytes] = []
        for seg in parse.all(M.APP1):
            if seg.payload.startswith(b"Exif") and seg.offset < 65536:
                extra.append(d[seg.offset : seg.end])
                break

        best: tuple[float, bytes, int] | None = None
        for start in starts:
            cand = prof.header_bytes(w, h, extra) + d[start:]
            frac, sr = _recovered(cand)
            if sr is not None and (best is None or frac > best[0]):
                best = (frac, cand, start)
        if best is None:
            ctx.note(self, "reference header did not produce a decodable file")
            return
        frac, cand, start = best
        if frac + MIN_GAIN < current:
            ctx.note(self, f"reference header kept aside: {frac:.1%} vs current {current:.1%}")
            return
        who = prof.model or prof.source or "reference"
        ctx.replace_data(
            cand,
            self,
            f"header rebuilt from reference ({who}); scan data from offset {start}; {frac:.1%} of image decodes",
            reference=who,
            scan_start=start,
            width=w,
            height=h,
        )
