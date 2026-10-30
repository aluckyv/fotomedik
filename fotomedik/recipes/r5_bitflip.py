"""R5 — undo flipped bits in the entropy-coded data by trial decoding.

A flipped bit desyncs the Huffman stream and the decoder notices a little
later (invalid code, coefficient overflow, implausible DC, marker in the wrong
place). We try flipping every bit in a window before the detection point and
keep the flip that lets the interval decode exactly up to its restart marker.

All candidate evaluation runs in the kernel (C when compiled): one call scores
hundreds of flips in place, no copies. Python only decides where to look and
which candidate to keep:

* a candidate is *complete* when every MCU of the interval decodes and the data
  ends exactly at the marker;
* among complete candidates the best has 1-bit padding before the marker (the
  standard requires it, a wrong flip rarely reproduces it), then the smoothest
  DC field against its neighbours, then a 0->1 repair (flash loses charge, so
  the true damage is mostly 1->0);
* when nothing completes, the most promising partial fixes are explored deeper
  (small beam search) because the interval may hold several flips.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..jpeg import decoder as D
from ..jpeg import kernel as K
from ..jpeg import markers as M
from ..jpeg.tables import FrameTables
from .base import Context, Recipe
from .r9_holes import decode_for
from .registry import register

W_BACK = 96  # bytes before the detection point searched first
W_FWD = 2  # bytes at/after the detection point
WIDEN = (1, 3, 8)  # window multipliers tried while nothing acceptable turns up
TWO_BIT_WINDOW = 24
MAX_ROUNDS = 8  # flips per interval
MIN_GAIN = 3  # MCUs of extra progress needed to accept a partial fix
BEAM = 3  # distinct partial outcomes explored per round when nothing completes
PER_OUTCOME = 3  # equivalent-looking flips explored per outcome
MAX_EXPANSIONS = 40  # search nodes per interval
ENOUGH_COMPLETE = 8  # stop widening once this many *distinct* complete outcomes exist
CHUNK = 64  # offsets per kernel call in the wider windows
PER_INTERVAL_MIN = 0.1  # seconds; each failed interval gets at least this much of the budget
PER_INTERVAL_MAX = 1.0
DEFAULT_TIME_BUDGET = 5.0  # seconds per file ("thorough" mode raises it)
SMALL_DAMAGE = 600  # bytes between the detection point and a recovered tail worth a flip search
ANOMALY_LEVELS = 64

_ONE_BIT = np.array([1 << b for b in range(8)], dtype=np.uint8)
_TWO_BIT = np.array([(1 << a) | (1 << b) for a in range(8) for b in range(a + 1, 8)], dtype=np.uint8)

REPAIRABLE = ("bad_huffman", "coef_overflow", "dc_range", "unexpected_marker", "bad_rst")
ROT_WINDOW = 2048  # bytes after the detection point inspected for dense damage
ROT_BAD_FF = 2  # broken FF-stuffing pairs in that window that mark a rotted stretch


def rotted(data: bytes, lo: int, hi: int) -> bool:
    """Dense damage (several broken FF00 stuffing pairs close together) cannot be undone by
    flipping a few bits; R10/R9 deal with it. Isolated flips leave at most one."""
    bad = 0
    p = data.find(b"\xff", lo, hi)
    while 0 <= p < hi - 1:
        n = data[p + 1]
        if not (n == 0x00 or 0xD0 <= n <= 0xD9 or n == 0xFF):
            bad += 1
            if bad >= ROT_BAD_FF:
                return True
        p = data.find(b"\xff", p + 1, hi)
    return False


# --------------------------------------------------------------------------- smoothness


class IntervalGeom:
    """Per-block neighbourhood of one restart interval, in kernel (scan) order.

    up/dn: DC of the block above/below from the full-file decode where that MCU
    decoded cleanly (NaN otherwise); left: index of the left neighbour inside the
    interval (-1 if none); w: quantiser weight (DC step in gray levels / 8).
    """

    def __init__(self, ft: FrameTables, pk: K.Packed, base: D.ScanResult, first: int, count: int):
        comps, per_row, _rows = D._scan_components(ft)
        bpm = pk.blocks_per_mcu
        n = count * bpm
        self.first, self.count, self.bpm = first, count, bpm
        self.cid = np.zeros(n, dtype=np.int32)
        self.brow = np.zeros(n, dtype=np.int64)
        self.bcol = np.zeros(n, dtype=np.int64)
        self.up = np.full(n, np.nan)
        self.dn = np.full(n, np.nan)
        self.left = np.full(n, -1, dtype=np.int32)
        self.w = np.zeros(n, dtype=np.float64)
        offs = []
        o = 0
        for c in comps:
            offs.append(o)
            o += c.h * c.v
        mcu_ok = base.mcu_ok
        k = 0
        for j in range(count):
            mi = first + j
            mrow, mcol = divmod(mi, per_row)
            for ci, c in enumerate(comps):
                arr = base.dc[c.cid]
                rows = arr.shape[0]
                q = pk.qdc[ci] / 8.0
                for by in range(c.v):
                    for bx in range(c.h):
                        brow, bcol = mrow * c.v + by, mcol * c.h + bx
                        self.cid[k], self.brow[k], self.bcol[k], self.w[k] = c.cid, brow, bcol, q
                        if bx > 0:
                            self.left[k] = k - 1
                        elif mcol > 0 and j > 0:
                            self.left[k] = (j - 1) * bpm + offs[ci] + by * c.h + (c.h - 1)
                        if by == 0 and brow > 0:
                            um = (mrow - 1) * per_row + mcol
                            if mcu_ok[um] and not math.isnan(arr[brow - 1, bcol]):
                                self.up[k] = arr[brow - 1, bcol]
                        if by == c.v - 1 and brow + 1 < rows:
                            dm = (mrow + 1) * per_row + mcol
                            if dm < len(mcu_ok) and mcu_ok[dm] and not math.isnan(arr[brow + 1, bcol]):
                                self.dn[k] = arr[brow + 1, bcol]
                        k += 1

    def from_mcu(self, m0: int):
        """Geometry for the blocks of MCUs m0.. (left links into earlier MCUs dropped)."""
        s = (m0 - self.first) * self.bpm
        left = self.left[s:] - s
        left[left < 0] = -1
        return (np.ascontiguousarray(self.up[s:]), np.ascontiguousarray(self.dn[s:]),
                np.ascontiguousarray(left.astype(np.int32)), np.ascontiguousarray(self.w[s:]))

    def gather(self, r: D.ScanResult) -> np.ndarray:
        out = np.full(len(self.cid), np.nan)
        for cid in np.unique(self.cid):
            sel = self.cid == cid
            out[sel] = r.dc[int(cid)][self.brow[sel], self.bcol[sel]]
        return out


def rough_vec(dc: np.ndarray, geom) -> float:
    """Same metric as the kernel's roughness(), NaN entries (undecoded blocks) skipped."""
    up, dn, left, w = geom
    ok = ~np.isnan(dc)
    terms = []
    lk = left >= 0
    sel = lk & ok
    sel[lk] &= ok[left[lk]]
    terms.append(2.0 * np.abs(dc[sel] - dc[left[sel]]) * w[sel])
    for nb in (up, dn):
        s2 = ok & ~np.isnan(nb)
        terms.append(np.abs(dc[s2] - nb[s2]) * w[s2])
    allt = np.concatenate(terms)
    return float(allt.mean()) if allt.size else 0.0


def image_roughness(sr: D.ScanResult, ft: FrameTables) -> float:
    """Typical DC roughness of the cleanly decoded part of the image (vectorised)."""
    comps, per_row, rows_m = D._scan_components(ft)
    total = 0.0
    n = 0
    ok_grid = sr.mcu_ok.reshape(rows_m, per_row) if sr.mcu_ok.size == rows_m * per_row else None
    for c in comps:
        arr = sr.dc[c.cid]
        fc = ft.frame.component(c.cid)
        w = ft.quant[fc.tq].dc / 8.0
        h = np.abs(arr[:, 1:] - arr[:, :-1])
        h = h[~np.isnan(h)]
        total += 2.0 * w * h.sum()
        n += h.size
        if ok_grid is not None and c.v and arr.shape[0] > c.v:
            okb = np.repeat(np.repeat(ok_grid, c.v, axis=0), c.h, axis=1)[: arr.shape[0], : arr.shape[1]]
            r = np.arange(c.v, arr.shape[0], c.v)
            d = np.abs(arr[r] - arr[r - 1])
            valid = okb[r] & okb[r - 1] & ~np.isnan(d)
            total += 2.0 * w * d[valid].sum()  # counted from above and from below
            n += 2 * int(valid.sum())
    return total / n if n else 0.0


def _first_anomaly(cur: D.ScanResult, base: D.ScanResult, ft: FrameTables, first: int, count: int) -> int | None:
    """MCU index in [first, first+count) whose luma DC jumps vs. the row above, or None."""
    frame = ft.frame
    luma = ft.scan.components[0].id
    fc = frame.component(luma)
    interleaved = len(ft.scan.components) > 1
    h, v = (fc.h, fc.v) if interleaved else (1, 1)
    per_row = cur.dc[luma].shape[1] // h
    q = ft.quant[fc.tq].dc
    for mi in range(first, first + count):
        mrow, mcol = divmod(mi, per_row)
        brow, bcol = mrow * v, mcol * h
        if brow == 0:
            continue
        a = cur.dc[luma][brow, bcol]
        b = base.dc[luma][brow - 1, bcol]
        if math.isnan(a) or math.isnan(b):
            continue
        if abs(a - b) * q / 8 > ANOMALY_LEVELS:
            return mi
    return None


# --------------------------------------------------------------------------- search


class _Interval:
    """Search bounds of one failed interval.

    marker_pos: where the interval must end (its restart marker / EOI), or None when
    the resync marker closes a *later* interval (this one's own marker is damaged
    beyond recognition). Then `expect` is the marker byte it should end with and a
    candidate must end on something within a few bits of it.
    """

    def __init__(self, iv: D.Interval, parse: M.ParseResult, data_len: int, dri: int):
        self.first = iv.mcu_first
        self.count = iv.mcu_count
        self.start = iv.byte_start
        self.expect = -1
        bound: int | None
        if iv.resync_at is not None:
            bound = iv.resync_at
            self.marker_pos: int | None = iv.resync_at if iv.resync_skipped == 0 else None
            if iv.resync_skipped and dri:
                self.expect = 0xD0 + (iv.index % 8)
        elif parse.eoi_offset is not None and parse.eoi_offset > iv.byte_start:
            bound = self.marker_pos = parse.eoi_offset
        else:
            bound = self.marker_pos = None
        self.decode_end = (bound + 2) if bound is not None else data_len


def _aligned(data, p: int) -> int:
    while p + 1 < len(data) and data[p] == 0xFF and data[p + 1] == 0x00:
        p += 2
    return p


def _complete(r: D.ScanResult, spec: "_Interval", data) -> bool:
    if r.error is not None or not r.intervals or not all(iv.ok for iv in r.intervals):
        return False
    if sum(iv.decoded for iv in r.intervals) != spec.count:
        return False
    end = r.intervals[-1].byte_end
    if spec.marker_pos is not None:
        return end == spec.marker_pos
    if spec.expect >= 0:
        if end is None:
            return False
        p = _aligned(data, end)
        return p + 1 < len(data) and K.near_marker(data[p], data[p + 1], spec.expect)
    return True


def _decoded(r: D.ScanResult) -> int:
    return sum(iv.decoded for iv in r.intervals)


GAIN_BUCKET = 64  # MCUs of progress treated as equal when ranking partial fixes of a long interval
DIVERGE_FWD = 4 * W_BACK  # bytes after the thumbnail's divergence point searched (the flip lies before it)


def err_mcu_of(cur: D.ScanResult) -> int:
    return cur.error.mcu_index if cur.error is not None else cur.total_mcus


DIVERGE_T = 8.0  # strength (t statistic) a step in the thumbnail residual needs
DIVERGE_STEP = 3.0  # gray levels it must at least shift the residual by


def thumbnail_divergence(data: bytes, ft: FrameTables, sr: D.ScanResult, thumb_refs=None):
    """A function giving, for a decode, the MCU where it starts to disagree with the
    thumbnail, or None when there is no usable thumbnail.

    A flipped bit breaks the chain of DC differences: from it on, a component is off by
    a constant (and the blocks may sit a few places off). Against the thumbnail that is a
    step in the residual (decoded minus thumbnail) of some component; the strongest step
    before the decoder's error point - by the t statistic of the two means - is where to
    look for the flip."""
    from .dcfield import thumbnail_base

    first_err = err_mcu_of(sr)
    clean = sr.mcu_ok.copy()
    clean[first_err:] = False
    try:
        thumb = thumbnail_base(data, ft, sr, clean, thumb_refs)
    except Exception:  # noqa: BLE001
        return None
    if thumb is None or thumb.twin:
        return None
    comps_d, per_row, _rows = D._scan_components(ft)
    comps = [(c, ft.quant[ft.frame.component(c.cid).tq].dc) for c in comps_d if c.cid in thumb.dc]

    def resid(cur: D.ScanResult, a: int, b: int) -> np.ndarray:
        """Mean residual (decoded minus thumbnail, gray levels) per component over MCUs [a, b)."""
        out = []
        for c, q in comps:
            arr, t = cur.dc[c.cid], thumb.dc[c.cid]
            vals = []
            for m in range(a, b):
                r0, c0 = divmod(m, per_row)
                blk = arr[r0 * c.v : (r0 + 1) * c.v, c0 * c.h : (c0 + 1) * c.h] - t[r0 * c.v : (r0 + 1) * c.v,
                                                                                    c0 * c.h : (c0 + 1) * c.h]
                fin = np.isfinite(blk)
                vals.append(float(blk[fin].mean()) if fin.any() else np.nan)
            v = np.array(vals, dtype=float) * q / 8.0
            out.append(float(np.nanmean(v)) if np.isfinite(v).any() else np.nan)
        return np.array(out)

    def diverge(cur: D.ScanResult):
        e = err_mcu_of(cur)
        best_t, best_k = 0.0, None
        for c, q in comps:
            arr, t = cur.dc[c.cid], thumb.dc[c.cid]
            rows = arr.shape[0] // c.v
            d = ((arr - t) * q / 8.0)[: rows * c.v, : per_row * c.h].reshape(rows, c.v, per_row, c.h)
            ok = np.isfinite(d)
            r = (np.where(ok, d, 0.0).sum(axis=(1, 3)) / np.maximum(ok.sum(axis=(1, 3)), 1)).ravel()[:e]
            n = len(r)
            if n < 2 * per_row:
                continue
            cs = np.cumsum(r)
            k = np.arange(per_row, n - per_row)  # a step with at least a row on each side
            m1, m2 = cs[k - 1] / k, (cs[-1] - cs[k - 1]) / (n - k)
            tstat = np.abs(m1 - m2) * np.sqrt(k * (n - k) / n) / (float(np.std(r)) + 1e-6)
            i = int(np.argmax(tstat))
            if tstat[i] > max(DIVERGE_T, best_t) and abs(m1[i] - m2[i]) > DIVERGE_STEP:
                best_t, best_k = float(tstat[i]), int(k[i])
        return best_k

    diverge.resid = resid
    diverge.per_row = per_row
    return diverge


STEP_ROWS = 3  # MCU rows decoded after a candidate flip to see whether the step is gone
STEP_BACK = 4  # MCUs before the step in which the flipped bit is looked for
STEP_FWD = 1


def flips_by_thumbnail(buf: bytearray, parse, ft: FrameTables, diverge, deadline: float,
                       rounds: int = 8) -> list[tuple[int, int]]:
    """Undo flips of a long interval one by one where the thumbnail shows them.

    For a scan without restart markers the decoder notices a flip thousands of MCUs
    later, and a wrong flip nearer the detection point often decodes just as far. The
    thumbnail's step (see thumbnail_divergence) pins the flip to a few MCUs; every bit
    there is tried, decoding a few rows on, and the one that makes the step vanish -
    the colours after it agree with the thumbnail as well as those before - is kept."""
    per_row = diverge.per_row
    found: list[tuple[int, int]] = []
    for _ in range(rounds):
        if time.monotonic() > deadline:
            break
        cur = D.decode_scan(bytes(buf), parse, ft)
        md = diverge(cur)
        e = err_mcu_of(cur)
        if md is None or md >= e or md < STEP_BACK + per_row:
            break
        m0 = md - STEP_BACK
        n = min(STEP_ROWS * per_row, e - m0 - 1)
        if n < per_row:
            break
        before = diverge.resid(cur, md - per_row, md)
        a = cur.mcus[m0].bit_start
        lo, hi = a // 8, cur.mcus[min(len(cur.mcus) - 1, md + STEP_FWD)].bit_start // 8 + 1
        preds = D.preds_before_mcu(cur, ft, m0)

        def step_of(data: bytes) -> float:
            r = D._decode_scan(data, parse, ft, start_byte=a // 8, start_bit=a % 8, first_mcu=m0, max_mcus=n,
                               init_preds=preds, resync=False)
            if r.error is not None or int(r.mcu_ok.sum()) < n:
                return math.inf
            after = diverge.resid(r, md + 1, m0 + n)
            return float(np.nansum(np.abs(after - before)))

        base = step_of(bytes(buf))
        best = (base, None)
        for off in range(lo, hi):
            for bit in range(8):
                buf[off] ^= 1 << bit
                sc = step_of(bytes(buf))
                buf[off] ^= 1 << bit
                if sc < best[0]:
                    best = (sc, (off, 1 << bit))
        if best[1] is None or not best[0] < 0.4 * base:
            break
        off, mask = best[1]
        buf[off] ^= mask
        after = D.decode_scan(bytes(buf), parse, ft)
        if err_mcu_of(after) <= e or diverge(after) == md:
            # the true flip lets the decode run on past where it stopped and takes the step
            # away for good; one that only hides it for a few rows (a lost stretch, not a flip) is not it
            buf[off] ^= mask
            break
        found.append((off, mask))
    return found


def repair_interval(
    buf: bytearray,
    parse: M.ParseResult,
    ft: FrameTables,
    base: D.ScanResult,
    iv: D.Interval,
    deadline: float,
    baseline: float = 0.0,
    stats: dict | None = None,
    diverge=None,
) -> tuple[bool, list[tuple[int, int]], str]:
    """Try to make one failed interval decode cleanly. Mutates buf. Returns (ok, flips, note).

    `baseline` is the typical DC roughness of the undamaged part of this image; a
    'complete' candidate much rougher than that is a wrong flip that happens to decode.
    `diverge(scan) -> MCU or None`: where a decode stops agreeing with the thumbnail. Without
    restart markers a flip is often noticed thousands of MCUs after it, far beyond the
    windows searched around the detection point; the thumbnail shows where it really is.
    """
    spec = _Interval(iv, parse, len(buf), ft.restart_interval)
    pk = K.packed_for(ft)
    arr = np.frombuffer(buf, dtype=np.uint8)
    geom_all = IntervalGeom(ft, pk, base, spec.first, spec.count)
    limit_rough = max(1.5 * baseline, baseline + 6.0)
    marker_pos = spec.marker_pos if spec.marker_pos is not None else -1
    expansions = [0]
    evaluated = [0]

    def decode_interval() -> D.ScanResult:
        return D.decode_scan(bytes(buf), parse, ft, start_byte=spec.start, end_byte=spec.decode_end,
                             first_mcu=spec.first, max_mcus=spec.count)

    def search(depth: int, flips: list[tuple[int, int]]):
        """Depth-limited beam search. Returns (complete, progress, score, flips)."""
        cur = decode_interval()
        if _complete(cur, spec, buf):
            return (True, spec.count, rough_vec(geom_all.gather(cur), geom_all.from_mcu(spec.first)), list(flips))
        progress = _decoded(cur)
        if depth >= MAX_ROUNDS or time.monotonic() > deadline or expansions[0] >= MAX_EXPANSIONS:
            return (False, progress, math.inf, list(flips))
        expansions[0] += 1

        # Where to look: the detection point, and the first visibly implausible MCU
        # before it (a desync is often noticed hundreds of bytes after the flip).
        if cur.error is not None and cur.error.reason in ("bad_huffman", "coef_overflow", "dc_range", "eof"):
            e_byte = cur.error.byte_offset
        elif cur.error is not None and cur.error.reason == "unexpected_marker":
            e_byte = cur.error.byte_offset + 1
        else:
            e_byte = cur.intervals[-1].byte_end if cur.intervals and cur.intervals[-1].byte_end else spec.decode_end
        mi = _first_anomaly(cur, base, ft, spec.first, spec.count)
        hint = e_byte
        if mi is not None and mi - spec.first < len(cur.mcus):
            # an image edge can look like an anomaly too: never trust it further back
            # than a few windows before the detection point
            hint = max(min(e_byte, cur.mcus[mi - spec.first].bit_start // 8 + 2), e_byte - 4 * W_BACK)
        hi = min(e_byte + W_FWD, spec.decode_end - (2 if spec.marker_pos is not None else 0))
        if diverge is not None:
            md = diverge(cur)
            if md is not None and spec.first <= md < err_mcu_of(cur) and md - spec.first < len(cur.mcus):
                # the flipped bit sits in the MCUs just before the step, not after it
                hint = cur.mcus[max(0, md - 2 - spec.first)].bit_start // 8
                hi = min(hi, hint + DIVERGE_FWD)
        if hi <= max(spec.start, hint - W_BACK):
            return (False, progress, math.inf, list(flips))
        err_mcu = cur.error.mcu_index if cur.error is not None else spec.first + spec.count
        g = geom_all.gather(cur)
        cur_dc_all = np.where(np.isnan(g), 0, g).astype(np.int32)

        complete: list[tuple[int, float, int, int, int]] = []  # (no_pad, rough, dir, off, mask)
        partials: dict[tuple[int, float], list[tuple[int, int, int]]] = {}

        def run(offs: np.ndarray, masks: np.ndarray, window_lo: int) -> None:
            # resume point: last MCU starting before the window (its predictors are known)
            m0 = spec.first
            for m in cur.mcus:
                if m.bit_start // 8 < window_lo:
                    m0 = m.index
                else:
                    break
            if m0 > spec.first:
                s_off, s_bit = divmod(cur.mcus[m0 - spec.first].bit_start, 8)
                p = D.preds_before_mcu(cur, ft, m0)
                preds0 = np.array([p[c] for c in pk.cids], dtype=np.int32)
            else:
                s_off, s_bit, preds0 = spec.start, 0, np.zeros(pk.ncomp, dtype=np.int32)
            needed = spec.count - (m0 - spec.first)
            stage1_n = max(1, err_mcu - m0 + MIN_GAIN + 1)
            known = [m for m in cur.mcus if m.index >= m0]
            mcu_bits = np.array([m.bit_start for m in known], dtype=np.int64)
            s0 = (m0 - spec.first) * pk.blocks_per_mcu
            cur_dc = cur_dc_all[s0 : s0 + len(known) * pk.blocks_per_mcu]
            dec, comp, pad, rough = K.eval_flips(arr, s_off, s_bit, spec.decode_end, pk, preds0, needed, stage1_n,
                                                 marker_pos, MIN_GAIN, offs, masks, geom_all.from_mcu(m0),
                                                 mcu_bits, cur_dc, spec.expect)
            evaluated[0] += len(offs)
            dirs = (arr[offs] & masks) != 0  # True: the repair sets a bit to 0 (less likely)
            for i in np.flatnonzero(comp):
                complete.append((0 if pad[i] else 1, float(rough[i]), int(dirs[i]), int(offs[i]), int(masks[i])))
            gain = (m0 - spec.first) + dec
            for i in np.flatnonzero(~comp & (gain >= progress + MIN_GAIN)):
                key = (int(gain[i]), round(float(rough[i]), 3))
                partials.setdefault(key, []).append((int(dirs[i]), int(offs[i]), int(masks[i])))

        def acceptable():
            return [c for c in complete if c[1] <= limit_rough]

        def enough() -> bool:
            # many flips are equivalent (same effect, different bit): count distinct outcomes
            return len({round(c[1], 2) for c in acceptable()}) >= ENOUGH_COMPLETE

        # Single-bit flips, nearest to the hint first; widen while nothing acceptable turns up.
        scanned_lo = scanned_hi = None
        for si, mult in enumerate(WIDEN):
            lo = max(spec.start, hint - mult * W_BACK)
            whi = hi if si == 0 else min(hi, hint + mult * W_BACK)
            offs_all = [o for o in range(lo, whi) if scanned_lo is None or not (scanned_lo <= o < scanned_hi)]
            offs_all.sort(key=lambda o: abs(o - hint))
            scanned_lo = lo if scanned_lo is None else min(lo, scanned_lo)
            scanned_hi = whi if scanned_hi is None else max(whi, scanned_hi)
            step = len(offs_all) if si == 0 else CHUNK  # first window always fully (quality)
            for c0 in range(0, len(offs_all), max(1, step)):
                chunk = np.array(offs_all[c0 : c0 + step], dtype=np.int64)
                if not len(chunk):
                    break
                run(np.repeat(chunk, 8), np.tile(_ONE_BIT, len(chunk)), int(chunk.min()))
                if (si > 0 and enough()) or time.monotonic() > deadline:
                    break
            if acceptable() or time.monotonic() > deadline or (lo == spec.start and whi == hi):
                break
        if not acceptable() and time.monotonic() <= deadline:
            lo2 = max(spec.start, e_byte - TWO_BIT_WINDOW)
            offs2 = np.arange(lo2, hi, dtype=np.int64)
            if len(offs2):
                run(np.repeat(offs2, len(_TWO_BIT)), np.tile(_TWO_BIT, len(offs2)), lo2)

        good = acceptable()
        if good:
            good.sort()
            _np, _r, _d, off, mask = good[0]
            buf[off] ^= mask
            res = search(depth + 1, flips + [(off, mask)])  # re-decode from the start to confirm
            buf[off] ^= mask
            if res[0]:
                return res
        # No clean completion: explore the most promising partial fixes deeper.
        best = (False, progress, math.inf, list(flips))
        if diverge is not None:
            # a long interval: a wrong flip may happen to decode a few MCUs further than the
            # true one; progress within GAIN_BUCKET counts the same, smoothness decides
            ranked = sorted(partials.items(), key=lambda kv: (-(kv[0][0] // GAIN_BUCKET), kv[0][1]))[:BEAM]
        else:
            ranked = sorted(partials.items(), key=lambda kv: (-kv[0][0], kv[0][1]))[:BEAM]
        branches: list[tuple[int, int]] = []
        for _key, lst in ranked:
            lst.sort()  # 0->1 repairs first
            branches += [(off, mask) for _d, off, mask in lst[:PER_OUTCOME]]
        for off, mask in branches:
            if time.monotonic() > deadline:
                break
            buf[off] ^= mask
            res = search(depth + 1, flips + [(off, mask)])
            buf[off] ^= mask
            if res[0] and (not best[0] or res[2] < best[2]):
                best = res
            elif not best[0] and not res[0] and res[1] > best[1]:
                best = res
        return best

    ok, _progress, _score, chosen = search(0, [])
    for off, mask in chosen:
        buf[off] ^= mask
    if ok and spec.marker_pos is None and spec.expect >= 0:
        # the interval now ends on a damaged copy of its marker: restore it too
        end = decode_interval().intervals[-1].byte_end
        p = _aligned(buf, end)
        for q, want in ((p, 0xFF), (p + 1, spec.expect)):
            if buf[q] != want:
                chosen.append((q, buf[q] ^ want))
                buf[q] = want
    if stats is not None:
        stats["candidates"] = stats.get("candidates", 0) + evaluated[0]
    if ok:
        return True, chosen, "repaired"
    if time.monotonic() > deadline:
        return False, chosen, "budget"
    return False, chosen, "unrepaired"


THUMB_CHECK_K = 4.0  # a fix whose blocks stray this many times the thumbnail's usual error ...
THUMB_CHECK_FLOOR = (12.0, 8.0)  # ... and more than this (gray levels, luma/chroma) is rejected
THUMB_CHECK_MIN = 24  # blocks after the damage needed to judge


def thumbnail_rejects(data: bytes, before: D.ScanResult, after: D.ScanResult, ft: FrameTables,
                      m_from: int, m_to: int, thumb_refs=None) -> str | None:
    """Why the decode `after` a fix cannot be right in MCUs [m_from, m_to), or None.

    A flip that lets the interval decode to its end is not always the true one: where
    another photo's sectors or a rotted stretch sit in the data, some flip can make the
    garbage decode "cleanly" - and every block after it then has the wrong colour. The
    thumbnail knows each block's colour; the fixed blocks must agree with it about as
    well as the clean blocks before the damage do."""
    from .dcfield import block_mask, thumbnail_base

    clean = before.mcu_ok.copy()
    clean[m_from:] = False
    try:
        thumb = thumbnail_base(data, ft, before, clean, thumb_refs)
    except Exception:  # noqa: BLE001 - no verdict without a usable thumbnail
        return None
    if thumb is None or thumb.twin:
        return None
    comps_d, per_row, _rows = D._scan_components(ft)
    comps = [(c.cid, c.h, c.v) for c in comps_d]
    span = np.zeros_like(after.mcu_ok)
    span[m_from:m_to] = True
    span &= after.mcu_ok
    for ci, (cid, h, v) in enumerate(comps[:3]):
        if cid not in thumb.dc:
            continue
        arr = after.dc[cid]
        m = block_mask(span, arr, h, v, per_row) & np.isfinite(thumb.dc[cid])
        if int(m.sum()) < THUMB_CHECK_MIN:
            continue
        q = ft.quant[ft.frame.component(cid).tq].dc
        err = float(np.median(np.abs(arr[m] - thumb.dc[cid][m]))) * q / 8.0
        limit = max(THUMB_CHECK_K * float(thumb.sigma[cid]), THUMB_CHECK_FLOOR[min(ci, 1)])
        if err > limit:
            what = "brightness" if ci == 0 else "colour"
            return f"{what} off by {err:.0f} gray levels from the thumbnail (usual {thumb.sigma[cid]:.0f})"
    return None


DENSE_SHARE = 0.05  # broken FF00 pairs above this: rot everywhere, single flips are not the problem


def broken_stuffing_share(data: bytes, start: int | None) -> float:
    """Share of FF bytes in the scan not followed by 00, a restart marker or more FF.
    Clean entropy-coded data has none; a card that rots everywhere has many."""
    a = np.frombuffer(data, dtype=np.uint8)[(start or 0) :]
    if len(a) < 2:
        return 0.0
    i = np.flatnonzero(a[:-1] == 0xFF)
    if len(i) < 50:
        return 0.0
    nx = a[i + 1]
    ok = (nx == 0) | ((nx >= 0xD0) & (nx <= 0xD7)) | (nx == 0xFF) | (nx == 0xD9)
    return float((~ok).mean())


@register
class BitFlipRepair(Recipe):
    name = "bitflip_repair"
    priority = 50
    description = "Undo flipped bits in scan data by trial decoding up to the next restart marker"

    def _candidates(self, sr: D.ScanResult, attempted: set[int], data: bytes | None = None,
                    rot: set[int] | None = None, tails: dict | None = None,
                    thorough: bool = False) -> list[D.Interval]:
        out = []
        for iv in sr.intervals:
            if iv.ok or iv.error is None or iv.error.reason not in REPAIRABLE:
                continue
            if iv.index in attempted:
                continue
            if iv.byte_end is not None and iv.byte_end <= iv.byte_start and iv.resync_at is None:
                continue
            if tails is not None and iv.index not in tails and not thorough:
                # no intact tail found: the damage runs (nearly) to the marker - rot, or
                # several flips; a long search rarely pays off outside thorough mode
                attempted.add(iv.index)
                if rot is not None:
                    rot.add(iv.index)
                continue
            if tails is not None and iv.index in tails:
                # R10 found where intact data resumes: a long damaged stretch before it is rot
                gap = tails[iv.index]["start_bit"] // 8 - iv.error.byte_offset
                if gap > SMALL_DAMAGE:
                    attempted.add(iv.index)
                    if rot is not None:
                        rot.add(iv.index)
                    continue
            if data is not None:
                lo = max(iv.byte_start, iv.error.byte_offset - 64)
                hi = min(len(data), iv.error.byte_offset + ROT_WINDOW)
                if iv.resync_at is not None:
                    hi = min(hi, iv.resync_at)
                if rotted(data, lo, hi):
                    attempted.add(iv.index)
                    if rot is not None:
                        rot.add(iv.index)
                    continue
            out.append(iv)
        return out

    def detect(self, ctx: Context):
        got = decode_for(ctx)
        if got is None:
            return None
        attempted = ctx.memo.setdefault("bitflip_attempted", set())
        thorough = float(ctx.options.get("time_budget", DEFAULT_TIME_BUDGET)) >= 30
        if not thorough and not ctx.memo.get("dense_checked"):
            ctx.memo["dense_checked"] = True
            share = broken_stuffing_share(ctx.data, got[0].scan_start)
            if share > DENSE_SHARE:
                ctx.memo["dense_damage"] = share
                ctx.note(self, f"dense damage ({share:.0%} of FF00 pairs broken): flipped bits are too many "
                               "to search for, bit-flip search skipped")
        if ctx.memo.get("dense_damage") and not thorough:
            return None
        return got if self._candidates(got[2], attempted, ctx.data, None, ctx.memo.get("tails"), thorough) else None

    def fix(self, ctx: Context, got) -> None:
        parse, ft, sr = got
        attempted: set[int] = ctx.memo.setdefault("bitflip_attempted", set())
        budget = float(ctx.options.get("time_budget", DEFAULT_TIME_BUDGET))
        deadline = time.monotonic() + budget
        buf = bytearray(ctx.data)
        baseline = image_roughness(sr, ft)
        diverge = thumbnail_divergence(ctx.data, ft, sr, ctx.options.get("thumb_refs")) if not ft.restart_interval else None
        pre_flips: list[tuple[int, int]] = []
        if diverge is not None:
            pre_flips = flips_by_thumbnail(buf, parse, ft, diverge, time.monotonic() + budget / 2)
            if pre_flips:
                sr = D.decode_scan(bytes(buf), parse, ft)
        all_flips: list[list[int]] = [[o, m] for o, m in pre_flips]
        repaired = unrepaired = 0
        notes: dict[str, int] = {}
        skipped_for_time = 0
        stats: dict = {}
        rotted_total = 0
        t0 = time.monotonic()
        # Repairing an interval can expose the next one (its marker was not trusted
        # before), so re-decode and continue until nothing new appears.
        while time.monotonic() <= deadline:
            rot: set[int] = set()
            cands = self._candidates(sr, attempted, bytes(buf), rot, ctx.memo.get("tails"), budget >= 30)
            rotted_total += len(rot)
            if not cands:
                break
            remaining = deadline - time.monotonic()
            per_interval = min(PER_INTERVAL_MAX, max(PER_INTERVAL_MIN, remaining / max(1, len(cands))))
            for iv in cands:
                if time.monotonic() > deadline:
                    skipped_for_time += 1
                    continue
                attempted.add(iv.index)
                iv_deadline = min(deadline, time.monotonic() + per_interval)
                ok, flips, note = repair_interval(buf, parse, ft, sr, iv, iv_deadline, baseline, stats, diverge)
                if flips:
                    after = D.decode_scan(bytes(buf), parse, ft)
                    m_from = iv.mcu_first + (iv.decoded or 0)
                    why = thumbnail_rejects(bytes(buf), sr, after, ft, m_from, iv.mcu_first + iv.mcu_count,
                                            ctx.options.get("thumb_refs"))
                    if why:  # the decode completes but is not the photo: undo, leave it to the other recipes
                        for o, m in flips:
                            buf[o] ^= m
                        ok, flips, note = False, [], "rejected by the thumbnail"
                        ctx.memo.setdefault("bitflip_rejected", []).append(why)
                all_flips += [[o, m] for o, m in flips]
                if ok:
                    repaired += 1
                    if ctx.memo.get("tails"):
                        ctx.memo["tails"].pop(iv.index, None)
                else:
                    unrepaired += 1
                    notes[note] = notes.get(note, 0) + 1
            parse = M.parse_segments(bytes(buf))
            sr = D.decode_scan(bytes(buf), parse, ft)
        msg = f"{repaired} interval(s) repaired with {len(all_flips)} bit flip(s); {unrepaired} unrepaired"
        if pre_flips:
            msg += f" ({len(pre_flips)} flip(s) found where the thumbnail shows a colour step)"
        if notes:
            msg += " (" + ", ".join(f"{k}: {v}" for k, v in notes.items()) + ")"
        if skipped_for_time:
            msg += f"; {skipped_for_time} skipped: time budget {budget:.0f}s exhausted"
        if rotted_total:
            msg += f"; {rotted_total} left to tail recovery (densely damaged)"
        extra = dict(repaired=repaired, unrepaired=unrepaired, candidates=stats.get("candidates", 0),
                     seconds=round(time.monotonic() - t0, 2))
        if all_flips:
            ctx.replace_data(bytes(buf), self, msg, flips=all_flips, **extra)
        else:
            ctx.note(self, msg, **extra)
