"""R2/R3 — header repair: flipped marker bytes, broken segment lengths, damaged tables.

Bit-rot in the header is the one place where damage is cheap to undo: markers,
lengths and table structure are highly redundant, so a byte that is one or two
bits away from the only value that makes the chain consistent is almost
certainly that value. Each fix is logged; nothing is touched beyond the header.
"""

from __future__ import annotations

from ..jpeg import markers as M
from ..jpeg.tables import FrameTables, TableError, parse_dht, standard_table_bytes
from .base import Context, Recipe
from .registry import register

MAX_MARKER_HAM = 2
MAX_LENGTH_HAM = 2
MAX_STD_TABLE_HAM = 4
LENGTH_SEARCH = 256

_STD_TABLES = [(cls, i, standard_table_bytes(cls, i)) for cls in (0, 1) for i in (0, 1)]
_STD_BY_LEN: dict[int, list[tuple[int, int, bytes]]] = {}
for _c, _i, _b in _STD_TABLES:
    _STD_BY_LEN.setdefault(len(_b), []).append((_c, _i, _b))


def ham(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def ham_bytes(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def _looks_like_jpeg(d: bytes) -> bool:
    if len(d) < 16:
        return False
    if ham(d[0], 0xFF) + ham(d[1], 0xD8) > 3:
        return False
    return d[2] == 0xFF or ham(d[2], 0xFF) <= 1


def _header_healthy(d: bytes) -> bool:
    parse = M.parse_segments(d)
    if not parse.ok:
        return False
    try:
        ft = FrameTables.from_parse(parse)
    except TableError:
        return False
    # silent damage inside tables that match a standard table except for a few bits
    for seg in parse.all(M.DHT):
        if _std_fixes(bytearray(seg.payload), 0, len(seg.payload), dry=True):
            return False
    if not _sof_sane(ft) or not _sos_sane(ft):
        return False
    return True


def _sof_sane(ft: FrameTables) -> bool:
    f = ft.frame
    if f.precision != 8 or f.height == 0 or f.width == 0:
        return False
    for c in f.components:
        if c.h not in (1, 2, 4) or c.v not in (1, 2, 4):
            return False
    return True


def _sos_sane(ft: FrameTables) -> bool:
    s = ft.scan
    if not ft.frame.progressive and (s.ss, s.se, s.ah, s.al) != (0, 63, 0, 0):
        return False
    ids = {c.id for c in ft.frame.components}
    return all(c.id in ids for c in s.components)


# --------------------------------------------------------------------------- table content


def _std_fixes(buf: bytearray, lo: int, hi: int, dry: bool = False) -> list[str]:
    """Replace DHT tables that are a few bits away from an Annex K standard table."""
    fixes: list[str] = []
    p = lo
    while p < hi:
        matched = False
        for ln, cands in _STD_BY_LEN.items():
            if p + ln > hi:
                continue
            chunk = bytes(buf[p : p + ln])
            scored = sorted((ham_bytes(chunk, std), cls, i, std) for cls, i, std in cands)
            h, cls, i, std = scored[0]
            if h == 0:
                matched = True
                p += ln
                break
            # must be close to exactly one standard table; long AC tables tolerate more flips
            limit = MAX_STD_TABLE_HAM if ln < 64 else max(MAX_STD_TABLE_HAM, ln // 12)
            if h <= limit and (len(scored) == 1 or scored[1][0] > h + 1):
                if not dry:
                    buf[p : p + ln] = std
                fixes.append(f"DHT table {cls}/{i} restored from standard ({h} bit(s))")
                matched = True
                p += ln
                break
        if matched:
            continue
        # not a standard table: step over it using its own counts (if sane)
        if p + 17 > hi:
            break
        total = sum(buf[p + 1 : p + 17])
        if total > 256 or p + 17 + total > hi:
            break
        p += 17 + total
    return fixes


def _fix_dqt(buf: bytearray, lo: int, hi: int) -> list[str]:
    fixes: list[str] = []
    n = hi - lo
    # all-8-bit tables: payload is k * 65 bytes
    if n % 65 == 0:
        for k in range(n // 65):
            at = lo + 65 * k
            pq, tq = buf[at] >> 4, buf[at] & 0x0F
            if pq != 0 or tq > 3:
                want = (0 << 4) | min(k, 3)
                if ham(buf[at], want) <= 2:
                    buf[at] = want
                    fixes.append(f"DQT table header byte at {at} -> 0x{want:02X}")
    return fixes


def _fix_dht(buf: bytearray, lo: int, hi: int) -> list[str]:
    fixes = _std_fixes(buf, lo, hi)
    # header bytes of non-standard tables: class/id nibbles must be 0/1 and <= 3
    p = lo
    while p + 17 <= hi:
        tc, th = buf[p] >> 4, buf[p] & 0x0F
        if tc > 1 or th > 3:
            for want in (0x00, 0x10, 0x01, 0x11):
                if ham(buf[p], want) <= 2:
                    buf[p] = want
                    fixes.append(f"DHT class/id byte at {p} -> 0x{want:02X}")
                    break
        total = sum(buf[p + 1 : p + 17])
        if total > 256 or p + 17 + total > hi:
            break
        p += 17 + total
    return fixes


def _fix_sof(buf: bytearray, lo: int, hi: int, length: int) -> list[str]:
    fixes: list[str] = []
    if hi - lo < 6:
        return fixes
    if buf[lo] != 8 and ham(buf[lo], 8) <= 2:
        buf[lo] = 8
        fixes.append("SOF precision -> 8")
    nc = buf[lo + 5]
    exp_nc = (length - 8) // 3 if (length - 8) % 3 == 0 else None
    if exp_nc in (1, 3, 4) and nc != exp_nc and ham(nc, exp_nc) <= 2:
        buf[lo + 5] = exp_nc
        fixes.append(f"SOF component count -> {exp_nc}")
        nc = exp_nc
    for i in range(min(nc, (hi - lo - 6) // 3)):
        at = lo + 6 + 3 * i
        hv = buf[at + 1]
        h, v = hv >> 4, hv & 0x0F
        if h not in (1, 2, 4) or v not in (1, 2, 4):
            best = None
            for wh in (1, 2, 4):
                for wv in (1, 2, 4):
                    cand = (wh << 4) | wv
                    d = ham(hv, cand)
                    if d <= 2 and (best is None or d < best[0]):
                        best = (d, cand)
            if best:
                buf[at + 1] = best[1]
                fixes.append(f"SOF component {i} sampling -> {best[1] >> 4}x{best[1] & 15}")
        tq = buf[at + 2]
        if tq > 3:
            want = 0 if i == 0 else 1
            if ham(tq, want) <= 2:
                buf[at + 2] = want
                fixes.append(f"SOF component {i} quant table -> {want}")
    return fixes


def _fix_sos(buf: bytearray, lo: int, hi: int, baseline: bool, sof_ids: list[int]) -> list[str]:
    fixes: list[str] = []
    ns = buf[lo]
    if hi - lo != 1 + 2 * ns + 3:
        exp = (hi - lo - 4) // 2
        if 1 <= exp <= 4 and (hi - lo) == 1 + 2 * exp + 3 and ham(ns, exp) <= 2:
            buf[lo] = exp
            fixes.append(f"SOS component count -> {exp}")
            ns = exp
        else:
            return fixes
    for i in range(ns):
        at = lo + 1 + 2 * i
        t = buf[at + 1]
        td, ta = t >> 4, t & 0x0F
        if td > 3 or ta > 3:
            want = 0x00 if i == 0 else 0x11
            if ham(t, want) <= 2:
                buf[at + 1] = want
                fixes.append(f"SOS component {i} tables -> 0x{want:02X}")
    if baseline:
        p = lo + 1 + 2 * ns
        for off, want, label in ((0, 0, "Ss"), (1, 63, "Se"), (2, 0, "Ah/Al")):
            if buf[p + off] != want and ham(buf[p + off], want) <= 2:
                buf[p + off] = want
                fixes.append(f"SOS {label} -> {want}")
    return fixes


# --------------------------------------------------------------------------- walk


def _ham16(a: int, b: int) -> int:
    return ham(a >> 8, b >> 8) + ham(a & 0xFF, b & 0xFF)


def _marker_distance(buf: bytearray, q: int) -> int:
    """Bits that must flip for buf[q:q+2] to read as FF + a segment marker (99 = hopeless)."""
    if q + 1 >= len(buf):
        return 99
    d_ff = ham(buf[q], 0xFF)
    if d_ff > MAX_MARKER_HAM:
        return 99
    d_m = min(ham(buf[q + 1], k) for k in M.SEGMENT_MARKERS)
    return d_ff + d_m


def _expected_length(buf: bytearray, p: int, m: int, L: int) -> int | None:
    """Length implied by the segment's own content, when the type pins it down."""
    n = len(buf)
    if m in M.SOF_MARKERS and p + 9 < n:
        nc = buf[p + 9]
        if 1 <= nc <= 4:
            return 8 + 3 * nc
    if m == M.DRI:
        return 4
    if m == M.SOS and p + 4 < n:
        ns = buf[p + 4]
        if 1 <= ns <= 4:
            return 6 + 2 * ns
    if m == M.APP0 and buf[p + 4 : p + 9] == b"JFIF\x00":
        return 16
    if m == M.DQT:
        # all-8-bit tables: 2 + 65k; pick k closest to L
        best = None
        for k in (1, 2, 3, 4):
            cand = 2 + 65 * k
            d = _ham16(cand, L)
            if best is None or d < best[0]:
                best = (d, cand)
        if best and best[0] <= MAX_LENGTH_HAM:
            return best[1]
    return None


def repair_header(buf: bytearray) -> list[str]:
    """Walk and repair the header in place. Returns human-readable fix descriptions."""
    fixes: list[str] = []
    n = len(buf)
    if n < 16:
        return fixes
    if buf[0] != 0xFF or buf[1] != 0xD8:
        if ham(buf[0], 0xFF) + ham(buf[1], 0xD8) <= 3:
            buf[0], buf[1] = 0xFF, 0xD8
            fixes.append("SOI marker restored")
        else:
            return fixes

    sof_marker: int | None = None
    sof_ids: list[int] = []
    sof_ids_at: list[int] = []
    p = 2
    while p + 4 <= n:
        # --- marker prefix
        if buf[p] != 0xFF:
            if ham(buf[p], 0xFF) <= MAX_MARKER_HAM and _marker_distance(buf, p) <= MAX_MARKER_HAM:
                buf[p] = 0xFF
                fixes.append(f"marker prefix at {p} -> FF")
            else:
                break
        while p + 2 < n and buf[p + 1] == 0xFF:
            p += 1  # fill bytes
        # --- marker byte
        m = buf[p + 1]
        if m not in M.SEGMENT_MARKERS:
            cands = sorted((ham(k, m), k) for k in M.SEGMENT_MARKERS if ham(k, m) <= MAX_MARKER_HAM)
            if not cands:
                break
            L = (buf[p + 2] << 8) | buf[p + 3]
            validated = []
            for d, k in cands:
                if k == M.SOS:
                    ns = buf[p + 4] if p + 4 < n else 0
                    ok = sof_marker is not None and 1 <= ns <= 4 and L == 6 + 2 * ns
                else:
                    exp = _expected_length(buf, p, k, L)
                    ok = _marker_distance(buf, p + 2 + L) <= MAX_MARKER_HAM or (exp is not None and _ham16(exp, L) <= MAX_LENGTH_HAM)
                if ok:
                    validated.append((d, k))
            if validated:
                k = validated[0][1]
            elif cands[0][0] == 1 and (len(cands) == 1 or cands[1][0] > 1):
                k = cands[0][1]  # unique closest candidate, accept without validation
            else:
                break
            buf[p + 1] = k
            fixes.append(f"marker at {p}: 0x{m:02X} -> {M.marker_name(k)}")
            m = k

        # --- length
        L = (buf[p + 2] << 8) | buf[p + 3]
        exp = _expected_length(buf, p, m, L)
        if m == M.SOS:
            if exp is not None and L != exp and _ham16(L, exp) <= MAX_LENGTH_HAM:
                buf[p + 2], buf[p + 3] = exp >> 8, exp & 0xFF
                fixes.append(f"SOS length {L} -> {exp}")
                L = exp
            if p + 2 + L > n:
                break
            fixes += _fix_sos(buf, p + 4, p + 2 + L, sof_marker in (0xC0, 0xC1, None), sof_ids)
            # reconcile component ids between SOF and SOS (typical ids are 1..n)
            ns = buf[p + 4]
            sos_ids_at = [p + 5 + 2 * i for i in range(ns)]
            if sof_ids_at and len(sof_ids_at) == ns:
                for i, (a, b) in enumerate(zip(sof_ids_at, sos_ids_at)):
                    if buf[a] != buf[b]:
                        typical = i + 1
                        if buf[b] == typical and ham(buf[a], typical) <= 2:
                            fixes.append(f"SOF component {i} id 0x{buf[a]:02X} -> {typical}")
                            buf[a] = typical
                        elif buf[a] == typical and ham(buf[b], typical) <= 2:
                            fixes.append(f"SOS component {i} id 0x{buf[b]:02X} -> {typical}")
                            buf[b] = typical
                        elif ham(buf[a], buf[b]) <= 2 and ham(buf[a], typical) < ham(buf[b], typical):
                            fixes.append(f"SOS component {i} id 0x{buf[b]:02X} -> 0x{buf[a]:02X}")
                            buf[b] = buf[a]
            break

        q = p + 2 + L
        if L < 2 or _marker_distance(buf, q) > 0:
            # Candidates: content-implied length, or any near-marker in the window.
            best: tuple[int, int] | None = None
            if exp is not None and _ham16(exp, L) <= MAX_LENGTH_HAM and p + 2 + exp + 1 < n:
                best = (_ham16(exp, L) + min(_marker_distance(buf, p + 2 + exp), 3), exp)
            for q2 in range(max(p + 4, q - LENGTH_SEARCH), min(n - 1, q + LENGTH_SEARCH)):
                dm = _marker_distance(buf, q2)
                if dm > MAX_MARKER_HAM:
                    continue
                L2 = q2 - p - 2
                dl = _ham16(L2, L)
                if dl <= MAX_LENGTH_HAM and (best is None or dl + dm < best[0]):
                    best = (dl + dm, L2)
            if best is None or best[0] > MAX_LENGTH_HAM + MAX_MARKER_HAM:
                break
            L2 = best[1]
            if L2 != L:
                buf[p + 2], buf[p + 3] = L2 >> 8, L2 & 0xFF
                fixes.append(f"{M.marker_name(m)} length {L} -> {L2}")
                L = L2
            q = p + 2 + L

        # --- content
        lo, hi = p + 4, p + 2 + L
        if m == M.DQT:
            fixes += _fix_dqt(buf, lo, hi)
        elif m == M.DHT:
            fixes += _fix_dht(buf, lo, hi)
        elif m in M.SOF_MARKERS:
            fixes += _fix_sof(buf, lo, hi, L)
            sof_marker = m
            nc = buf[lo + 5] if hi - lo >= 6 else 0
            sof_ids_at = [lo + 6 + 3 * i for i in range(min(nc, (hi - lo - 6) // 3))]
            sof_ids = [buf[a] for a in sof_ids_at]
        p = q
    return fixes


@register
class HeaderRepair(Recipe):
    name = "header_repair"
    priority = 20
    description = "Repair flipped bits in markers, segment lengths and tables"

    def detect(self, ctx: Context):
        d = ctx.data
        if ctx.cache.get("header_repair_failed"):
            return None
        if not _looks_like_jpeg(d):
            return None
        return None if _header_healthy(d) else True

    def fix(self, ctx: Context, finding) -> None:
        buf = bytearray(ctx.data)
        fixes = repair_header(buf)
        if fixes and bytes(buf) != ctx.data:
            ctx.replace_data(bytes(buf), self, "; ".join(fixes), fixes=fixes)
            if not _header_healthy(ctx.data):
                ctx.cache["header_repair_failed"] = True
                ctx.note(self, "header still inconsistent after repair")
        else:
            ctx.cache["header_repair_failed"] = True
            ctx.note(self, "header damaged, no safe fix found")
