"""R10 — recover the tail of an interval after a damaged stretch.

Dying flash rarely flips isolated bits: it rots whole sectors (several KB with a
few % of bits wrong). Nothing inside such a stretch can be repaired, but the data
after it is intact. Its MCU boundaries are unknown, yet Huffman streams
resynchronise by themselves: decoding from almost any bit after the damage
converges on the true parse and ends exactly on the interval's restart marker
(kernel.tail_scan). That fixes how many MCUs the tail holds and so where they
belong. The DC values come out relative to an unknown predictor; one offset per
colour component is estimated from the rows above/below (median) and handed to
R9, which splices the tail behind the filled hole.

Spans: when a resync jumped over intervals (their markers rotted too), the scan
runs from the span's last marker backwards; if a whole interval comes back, the
one before it is tried from that interval's start, and so on.
"""

from __future__ import annotations

import math

import numpy as np

from ..jpeg import decoder as D
from ..jpeg import kernel as K
from ..scan.garbage import last_broken_stuffing
from .base import Context, Recipe
from .r9_holes import decode_for
from .registry import register

DROP_FIRST = 3  # first MCUs of a tail may be junk that resynchronised onto the true parse
GIVE_UP = 512  # bytes scanned past the last convergent start before stopping
MAX_HEAD = 6  # MCUs a start may take to land on the reference parse
MIN_TAIL = 2  # MCUs worth splicing
ANOMALY_LEVELS = 96  # gray levels: leading tail MCUs this far off their neighbours are dropped
MAX_TRIM = 8


def _neighbour_offsets(sr: D.ScanResult, ft, first_mcu: int, dc_rel: np.ndarray, pk: K.Packed) -> dict[int, float | None]:
    """Per component: median(neighbour DC - relative DC) over tail blocks with a clean vertical neighbour."""
    comps, per_row, _rows = D._scan_components(ft)
    n = dc_rel.shape[0]
    out: dict[int, float | None] = {}
    col = 0
    for ci, c in enumerate(comps):
        arr = sr.dc[c.cid]
        diffs: list[float] = []
        for j in range(n):
            mi = first_mcu + j
            mrow, mcol = divmod(mi, per_row)
            for by in range(c.v):
                for bx in range(c.h):
                    bi = col + by * c.h + bx
                    brow, bcol = mrow * c.v + by, mcol * c.h + bx
                    rel = float(dc_rel[j, bi])
                    if by == 0 and brow > 0:
                        um = (mrow - 1) * per_row + mcol
                        if sr.mcu_ok[um] and not math.isnan(arr[brow - 1, bcol]):
                            diffs.append(arr[brow - 1, bcol] - rel)
                    if by == c.v - 1 and brow + 1 < arr.shape[0]:
                        dm = (mrow + 1) * per_row + mcol
                        if dm < len(sr.mcu_ok) and sr.mcu_ok[dm] and not math.isnan(arr[brow + 1, bcol]):
                            diffs.append(arr[brow + 1, bcol] - rel)
        out[c.cid] = float(np.median(diffs)) if diffs else None
        col += c.h * c.v
    return out


def _left_offsets(sr: D.ScanResult, ft, first_mcu: int, dc_rel: np.ndarray) -> dict[int, float | None]:
    """Fallback: continue the DC of the nearest clean block to the left in the same row."""
    comps, per_row, _rows = D._scan_components(ft)
    out: dict[int, float | None] = {}
    col = 0
    mrow, mcol = divmod(first_mcu, per_row)
    for c in comps:
        arr = sr.dc[c.cid]
        val = None
        for mc in range(mcol - 1, -1, -1):
            mi = mrow * per_row + mc
            v = arr[mrow * c.v, mc * c.h + c.h - 1]
            if sr.mcu_ok[mi] and not math.isnan(v):
                val = float(v)
                break
        out[c.cid] = None if val is None else val - float(dc_rel[0, col])
        col += c.h * c.v
    return out


def find_interval_start(data: bytes, pk: K.Packed, lo: int, hi: int, mpos: int, count: int,
                        known: list[int], prev_marker: int | None = None) -> int | None:
    """Byte offset where a whole interval ending at `mpos` starts, or None.

    An interval starts on a byte boundary right after the previous marker, with all
    DC predictors at 0, and holds exactly `count` MCUs. Candidates are byte offsets
    in [lo, hi); each is decoded a few MCUs until it lands on the reference parse
    `known` (MCU start bits of the tail, ending at mpos), which fixes its MCU count.
    Every candidate passing a full decode with all checks on is collected. Junk that
    starts inside the first true MCU can pass too, so the winner is the one preceded
    by the most marker-like byte pair (the previous interval's damaged RST), then the
    earliest.
    """
    if not known:
        return None
    rem = {b: len(known) - i for i, b in enumerate(known)}
    found: list[tuple[int, int]] = []
    for s in range(lo, hi):
        preds = np.zeros(pk.ncomp, dtype=np.int32)
        n, err, _eb, _ebit, _mk, end_b, end_bit, bits, _dc = K.decode_mcus(data, s, 0, mpos + 2, pk, preds, 8)
        total = None
        for i in range(1, len(bits)):
            if int(bits[i]) in rem:
                total = i + rem[int(bits[i])]
                break
        if total is None and err is None:
            e = end_b * 8 + end_bit
            if e in rem:
                total = n + rem[e]
        if total != count:
            continue
        preds = np.zeros(pk.ncomp, dtype=np.int32)
        n, err, _eb, _ebit, _mk, end_b, end_bit, _bits, _dc = K.decode_mcus(data, s, 0, mpos + 2, pk, preds, count)
        if n == count and err is None:
            a, pad = K.py_aligned_end(data, len(data), end_b, end_bit)
            if a == mpos and pad:
                if s >= 2 and prev_marker is not None:
                    ham = bin(data[s - 2] ^ 0xFF).count("1") + bin(data[s - 1] ^ prev_marker).count("1")
                else:
                    ham = 16
                found.append((ham, s))
    if not found:
        return None
    found.sort()
    return found[0][1]


@register
class TailRecovery(Recipe):
    name = "tail_recovery"
    priority = 45  # before bit-flip repair: it tells R5 where damage is small enough to be worth it
    description = "Recover the intact end of an interval after a rotted stretch by resynchronising on its marker"

    def _spans(self, sr: D.ScanResult, parse) -> list[tuple[D.Interval, int, int]]:
        """(failed interval, index of the last interval it covers, marker offset closing it)."""
        out = []
        for iv in sr.intervals:
            if iv.ok or iv.error is None or iv.error.reason == "skipped":
                continue
            if iv.resync_at is not None:
                out.append((iv, iv.index + iv.resync_skipped, iv.resync_at))
            elif parse.eoi_offset is not None and parse.eoi_offset > iv.byte_start:
                out.append((iv, iv.index, parse.eoi_offset))  # last interval / no restart markers
        return out

    def detect(self, ctx: Context):
        if ctx.memo.get("tails_done"):
            return None
        got = decode_for(ctx)
        if got is None:
            return None
        parse, ft, sr = got
        return got if self._spans(sr, parse) else None

    def fix(self, ctx: Context, got) -> None:
        parse, ft, sr = got
        ctx.memo["tails_done"] = True
        pk = K.packed_for(ft)
        data = ctx.data
        dri = ft.restart_interval
        total = sr.total_mcus
        tails: dict[int, dict] = {}
        recovered = 0
        for iv, last, marker_pos in self._spans(sr, parse):
            j = last
            mpos = marker_pos
            while j >= iv.index:
                first_of_j = j * dri if dri else iv.mcu_first
                count_j = min(dri, total - first_of_j) if dri else total - iv.mcu_first
                if count_j <= 0:
                    j -= 1
                    continue
                lo = iv.byte_start
                bad = last_broken_stuffing(data, lo, mpos)
                if bad is not None:
                    lo = bad + 2  # certain damage: intact data resumes after it
                n, start_bit, bits = K.tail_scan(data, lo, mpos, pk, count_j, GIVE_UP, MAX_HEAD)
                if n < MIN_TAIL + DROP_FIRST:
                    break
                full = False
                if dri and j > iv.index:
                    # a later interval of the span may be intact from its very start
                    avg = max(64, (iv.byte_end or mpos) - iv.byte_start) / max(1, last - iv.index + 1)
                    # the true start may lie a few MCUs inside the found tail (junk head)
                    hi_s = bits[min(len(bits) - 1, DROP_FIRST + MAX_HEAD)] // 8 + 1
                    s0 = find_interval_start(data, pk, max(lo, int(mpos - 2.5 * avg)), min(mpos, hi_s),
                                             mpos, count_j, bits, 0xD0 + ((j - 1) % 8))
                    if s0 is not None:
                        full = True
                        n, start_bit = count_j, s0 * 8
                        preds = np.zeros(pk.ncomp, dtype=np.int32)
                        bits = [int(b) for b in K.decode_mcus(data, s0, 0, mpos + 2, pk, preds, count_j)[7]]
                drop = 0 if full else DROP_FIRST
                n_tail = n - drop
                s_bit = bits[drop]
                first_tail = first_of_j + count_j - n_tail
                preds = np.zeros(pk.ncomp, dtype=np.int32)
                dn, err, *_rest, end_b, end_bit, _bs, dc_rel = K.decode_mcus(data, s_bit // 8, s_bit % 8, mpos + 2, pk, preds,
                                                                             n_tail, check_dc=False)
                if dn != n_tail or err is not None:
                    break
                if full:
                    delta = {cid: 0.0 for cid in pk.cids}
                else:
                    delta = _neighbour_offsets(sr, ft, first_tail, dc_rel, pk)
                    left = _left_offsets(sr, ft, first_tail, dc_rel)
                    delta = {cid: (delta[cid] if delta[cid] is not None else (left[cid] if left[cid] is not None else 0.0))
                             for cid in pk.cids}
                # trim leading MCUs that clash with their neighbours (junk before convergence)
                trim = 0
                if not full:
                    luma = pk.cids[0]
                    q = pk.qdc[0] / 8.0
                    abs0 = dc_rel[:, 0] + delta[luma]
                    comps, per_row, _r = D._scan_components(ft)
                    c0 = comps[0]
                    arr = sr.dc[luma]
                    while trim < min(MAX_TRIM, n_tail - MIN_TAIL):
                        mi = first_tail + trim
                        mrow, mcol = divmod(mi, per_row)
                        ref = []
                        for drow in (-1, c0.v):
                            r = mrow * c0.v + drow
                            if 0 <= r < arr.shape[0] and not math.isnan(arr[r, mcol * c0.h]):
                                ref.append(arr[r, mcol * c0.h])
                        if ref and abs(abs0[trim] - float(np.mean(ref))) * q > ANOMALY_LEVELS:
                            trim += 1
                        else:
                            break
                if trim:
                    s_bit = bits[drop + trim]
                    n_tail -= trim
                    first_tail += trim
                    dc_rel_t = dc_rel[trim:]
                    # predictor shifts with the dropped MCUs: relative values restart at 0
                    base = dc_rel[trim - 1][[sum(h * v for h, v in pk.hv[:i + 1]) - 1 for i in range(pk.ncomp)]]
                    delta = {cid: delta[cid] + float(base[i]) for i, cid in enumerate(pk.cids)}
                    dc_rel = dc_rel_t - base[np.repeat(np.arange(pk.ncomp), [h * v for h, v in pk.hv])]
                # keep every spliced DC inside the legal range (|DC*q| <= 1024+q): a bad
                # neighbour estimate must not produce a file decoders reject
                ok = True
                col = 0
                for ci, cid in enumerate(pk.cids):
                    q = pk.qdc[ci]
                    lim = (1024 + q) // q
                    nb_c = pk.hv[ci][0] * pk.hv[ci][1]
                    rel = dc_rel[:, col : col + nb_c]
                    lo_d, hi_d = -lim - int(rel.min()), lim - int(rel.max())
                    if lo_d > hi_d:
                        ok = False
                        break
                    delta[cid] = float(min(max(delta[cid], lo_d), hi_d))
                    col += nb_c
                if not ok:
                    break
                tails[j] = {
                    "start_bit": int(s_bit),
                    "end_bit": int(end_b * 8 + end_bit),
                    "n": int(n_tail),
                    "first_mcu": int(first_tail),
                    "delta": {int(k): int(round(v)) for k, v in delta.items()},
                    "full": bool(full),
                    "dc_rel": dc_rel.astype(np.int32),
                }
                recovered += n_tail
                if not full:
                    break
                # the whole interval came back: its own (damaged) marker sits just before it
                mpos = start_bit // 8 - 2
                j -= 1
        if tails:
            ctx.memo["tails"] = tails
            ctx.note(self, f"recovered {recovered} MCU(s) after damaged stretches in {len(tails)} interval(s)",
                     recovered_mcus=recovered, intervals=sorted(tails))
