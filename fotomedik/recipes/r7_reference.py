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
CLOSE_SOS = 0.15  # fraction of differing bits for a certain SOS match
# A header that parses is replaced only by one that decodes clearly further than it: this
# many MCUs more, and this share more. Photos saved with optimised Huffman tables differ
# from their neighbours' by a few bits (they look like "silently damaged" tables), and with
# dense rot every header, right or wrong, stops at the first rotted sector; before, the
# rounded decoded share tied at 0.00 on 16 MP photos and 5 of 15 got another photo's tables.
# Wrong tables decode garbage for up to about 80 MCUs before an impossible code shows, and
# the best of fifteen such runs beats a right header that stopped at a rotted sector 20 MCUs
# in; a right header for silently damaged tables decodes hundreds further.
STRICT_MCUS = 64
STRICT_SHARE = 1.0
STRICT_MIN_OWN = 8  # an own header that decodes fewer MCUs than this is judged the old way (build 160 kept a
# header whose repaired tables decoded nothing, DSC_4403 of the dense-head bench, and the photo came out broken)

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def find_scan_start(data: bytes, sos_bytes: bytes, limit: int = SEARCH_LIMIT) -> int | None:
    """Offset right after the main image's SOS segment, found by approximate match of the
    reference SOS. The embedded thumbnail has an SOS too (often written with the very
    same bytes), earlier in the file: the last close match is the main image's."""
    n = min(len(data), limit)
    L = len(sos_bytes)
    if n <= L:
        return None
    arr = np.frombuffer(data[:n], dtype=np.uint8)
    dist = np.zeros(n - L + 1, dtype=np.int32)
    for j in range(L):
        dist += _POPCOUNT[np.bitwise_xor(arr[j : n - L + 1 + j], sos_bytes[j])]
    close = np.flatnonzero(dist <= CLOSE_SOS * 8 * L)
    if len(close):
        return int(close[-1]) + L
    i = int(np.argmin(dist))
    if dist[i] > MAX_SOS_MISMATCH * 8 * L:
        return None
    return i + L


def row_misfit(sr: D.ScanResult, ft) -> float | None:
    """Mean luma DC jump between vertically adjacent decoded blocks: small when the
    width is right, large when rows are sheared. None with too little decoded data."""
    comps, _per_row, _rows = D._scan_components(ft)
    arr = sr.dc[comps[0].cid]
    a, b = arr[1:], arr[:-1]
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 64:
        return None
    return float(np.mean(np.abs(a - b)[ok]))


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


def _tables_near(data: bytes, prof: CameraProfile, kinds=(M.DQT, M.DHT)) -> bool:
    """True when the file's tables differ from the profile's by a few bits (silent damage).
    Huffman tables optimised per photo pass this test too (neighbouring shots differ by a
    few bits), so a DHT match alone is only a reason to look, not to replace."""
    res = M.parse_segments(data)
    for kind, refs in ((M.DQT, prof.dqt), (M.DHT, prof.dht)):
        if kind not in kinds:
            continue
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
        ctx.memo["reference_strict"] = False
        if not needs:
            # a header that parses is replaced when its quantisation tables are a few flipped
            # bits away from a reference; when only its Huffman tables are near one, or no
            # reference has its tables and almost nothing decodes, a reference is tried but
            # kept only if it decodes clearly further (fix: the own tables may be the right
            # ones, optimised for this photo)
            res = M.parse_segments(d)
            own = ([sg.payload for sg in res.all(M.DQT)], [sg.payload for sg in res.all(M.DHT)])
            known = any((p.dqt, p.dht) == own for p in profiles)
            if any(_tables_near(d, p, (M.DQT,)) for p in profiles):
                needs = True
            else:
                cur, _ = _recovered(d)
                near_dht = any(_tables_near(d, p, (M.DHT,)) for p in profiles)
                needs = cur < 0.05 and (near_dht or not known)
                ctx.memo["reference_strict"] = True
        if not needs:
            return None
        # An explicit --ref is trusted; auto-learned candidates must look like the same
        # camera (Exif model or identical dimensions), otherwise a foreign header would
        # turn the scan into coloured noise that still "decodes". When the damaged header
        # says nothing any more, a set taken with a single camera is trusted as a whole.
        if ctx.ref is not None:
            return [ctx.ref]
        scored = sorted(((p.matches(d), p) for p in profiles), key=lambda t: t[0], reverse=True)
        best = scored[0][0] if scored else 0
        if best >= MIN_MATCH:
            eligible = [p for sc, p in scored if sc == best]
        elif len({(p.make, p.model) for p in profiles if p.model}) <= 1 and not _header_healthy(d):
            eligible = list(profiles)
        else:
            ctx.memo["reference_tried"] = True
            return None
        return eligible[:6]

    def fix(self, ctx: Context, profiles: list[CameraProfile]) -> None:
        ctx.memo["reference_tried"] = True
        d = ctx.data
        parse = M.parse_segments(d)
        current, own_sr = _recovered(d)
        own_mcus = own_sr.decoded_mcus if own_sr is not None else 0
        strict = bool(ctx.memo.get("reference_strict"))

        # dimensions the damaged SOF still states, if plausible
        sof_dims = None
        if parse.sof is not None:
            try:
                fi = parse_sof(parse.sof.marker, parse.sof.payload)
                if 0 < fi.width <= 20000 and 0 < fi.height <= 20000:
                    sof_dims = (fi.width, fi.height)
            except TableError:
                pass

        # keep the damaged file's Exif block when it parsed, so metadata survives
        extra: list[bytes] = []
        for seg in parse.all(M.APP1):
            if seg.payload.startswith(b"Exif") and seg.offset < 65536:
                extra.append(d[seg.offset : seg.end])
                break

        trials = []  # (frac, -row_misfit, votes, cand, start, prof, dims)
        for prof in profiles:
            starts: list[int] = []
            if parse.scan_start is not None:
                starts.append(parse.scan_start)
            found = find_scan_start(d, prof.sos_bytes)
            if found is not None and found not in starts:
                starts.append(found)
            dims = [(prof.width, prof.height), (prof.height, prof.width)]  # portrait shots too
            dims += [tuple(d) for d in prof.alt_dims if tuple(d) not in dims]
            if sof_dims is not None and sof_dims not in dims:
                near = ham_bytes(sof_dims[0].to_bytes(2, "big") + sof_dims[1].to_bytes(2, "big"),
                                 prof.width.to_bytes(2, "big") + prof.height.to_bytes(2, "big")) <= 2
                if not near:
                    dims.insert(0, sof_dims)
            for start in starts:
                for w, h in dims:
                    cand = prof.header_bytes(w, h, extra) + d[start:]
                    frac, sr = _recovered(cand)
                    if sr is None:
                        continue
                    p2 = M.parse_segments(cand)
                    misfit = row_misfit(sr, FrameTables.from_parse(p2))
                    clean_ivs = sum(1 for iv in sr.intervals if iv.ok) if len(sr.intervals) > 1 else 0
                    trials.append((frac, misfit, prof.votes, cand, start, prof, (w, h), clean_ivs, sr.decoded_mcus))
        if not trials:
            ctx.note(self, "reference header did not produce a decodable file")
            return

        def rank(t):
            frac, misfit, votes, clean_ivs, mcus = t[0], t[1], t[2], t[7], t[8]
            # restart intervals that decode cleanly from marker to marker prove the row
            # length (a wrong one fails every interval, however much of each prefix decodes);
            # then the decoded MCUs (the share rounds to nothing on a big photo whose first
            # pass stops at the first rotted sector), rows that line up, the commoner header
            return (clean_ivs, mcus, -(misfit if misfit is not None else 1e9), votes)

        frac, _misfit, _votes, cand, start, prof, (w, h), _clean, mcus = max(trials, key=rank)
        if own_sr is None or own_mcus < STRICT_MIN_OWN:
            strict = False  # the own tables decode nothing: they are not the right ones
        if strict and mcus <= own_mcus + max(STRICT_MCUS, STRICT_SHARE * own_mcus):
            ctx.note(self, f"own header kept: it decodes {own_mcus} MCU(s), the best reference {mcus}")
            return
        if frac + MIN_GAIN < current:
            ctx.note(self, f"reference header kept aside: {frac:.1%} vs current {current:.1%}")
            return
        who = prof.model or prof.source or "reference"
        ctx.replace_data(
            cand,
            self,
            f"header rebuilt from reference ({who}, {w}x{h}); scan data from offset {start}; {frac:.1%} of image decodes",
            reference=who,
            scan_start=start,
            width=w,
            height=h,
        )
