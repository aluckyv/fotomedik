"""R10 — recover the tail of an interval after a damaged stretch.

Dying flash rarely flips isolated bits: it rots whole sectors (several KB with a
few % of bits wrong). Nothing inside such a stretch can be repaired, but the data
after it is intact. Its MCU boundaries are unknown, yet Huffman streams
resynchronise by themselves: decoding from almost any bit after the damage
converges on the true parse and ends exactly on the interval's restart marker.
kernel.tail_flow follows the parse most start positions flow into back from the
marker. That fixes how many MCUs the tail holds and so where they belong. Leading
MCUs whose DC clashes with the rows above/below (junk decoded inside the damage)
are trimmed. The DC values come out relative to an unknown predictor; one offset per
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
from .. import parallel
from ..jpeg import kernel as K
from ..scan.garbage import last_dense_damage
from .base import Context, Recipe
from .dcfield import block_mask, build_field, thumbnail_base
from .r9_holes import decode_for, junk_onset, tail_blocks
from .registry import register

DROP_FIRST = 2  # the first MCUs of a tail start inside the damage
MIN_TAIL = 2  # MCUs worth splicing
ANOMALY_MIN = 20  # gray levels: a leading MCU at least this far off its neighbours is junk
TRIM_SCAN = 48  # leading MCUs examined for junk
MAX_REF_DIST = 8  # block rows: farther clean rows say little about a tail's DC


class References:
    """Per component: the DC the surroundings suggest for every block (nearest clean rows
    above/below, guided by the Exif thumbnail when it has one - see dcfield), from the
    decode before any repair."""

    def __init__(self, sr: D.ScanResult, ft, thumb=None):
        self.comps, self.per_row, _rows = D._scan_components(ft)
        self.ref = {}
        for c in self.comps:
            arr = sr.dc[c.cid]
            q = ft.quant[ft.frame.component(c.cid).tq].dc
            f = build_field(arr, block_mask(sr.mcu_ok, arr, c.h, c.v, self.per_row), q,
                            thumb.dc[c.cid] if thumb else None, thumb.sigma[c.cid] if thumb else None)
            self.ref[c.cid] = (f.val, f.dist)

    def blocks(self, first_mcu: int, n: int):
        """Per component (cid, column slice in dc_rel, reference values, distances), shape (n, h*v)."""
        col = 0
        for c in self.comps:
            br, bc = tail_blocks(first_mcu, n, c.h, c.v, self.per_row)
            val, dist = self.ref[c.cid]
            yield c.cid, slice(col, col + c.h * c.v), val[br, bc], dist[br, bc]
            col += c.h * c.v

    def offsets(self, first_mcu: int, dc_rel: np.ndarray, max_dist: float = MAX_REF_DIST) -> dict[int, float | None]:
        """Per component: median(reference - relative DC) over blocks with a reference
        within max_dist block rows."""
        out: dict[int, float | None] = {}
        for cid, sl, rv, dv in self.blocks(first_mcu, dc_rel.shape[0]):
            good = (dv <= max_dist) & ~np.isnan(rv)
            out[cid] = float(np.median(rv[good] - dc_rel[:, sl][good])) if good.any() else None
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


def head_anomaly(refs: References, first_mcu: int, dc_rel: np.ndarray, delta: dict, pk: K.Packed,
                 max_dist: float = MAX_REF_DIST) -> np.ndarray:
    """Gray levels by which each tail MCU misses the DC its surroundings suggest (first
    block of every component, worst component); NaN where no reference is near."""
    n = dc_rel.shape[0]
    out = np.full(n, np.nan)
    for ci, (cid, sl, rv, dv) in enumerate(refs.blocks(first_mcu, n)):
        a = np.abs(dc_rel[:, sl][:, 0] + delta[cid] - rv[:, 0]) * (pk.qdc[ci] / 8.0)
        a = np.where((dv[:, 0] <= max_dist) & ~np.isnan(rv[:, 0]), a, np.nan)
        out = np.where(np.isnan(out), a, np.where(np.isnan(a), out, np.maximum(out, a)))
    return out


def leading_junk(anom: np.ndarray, floor: float = ANOMALY_MIN, gap: int = 4) -> int:
    """How many leading MCUs to drop: through the last anomalous one that is followed by
    fewer than `gap` normal ones before the next anomaly. The threshold adapts to the image:
    max(floor, 4 x the median anomaly of the whole tail)."""
    vals = anom[~np.isnan(anom)]
    if not len(vals):
        return 0
    thr = max(floor, 4.0 * float(np.median(vals)))
    trim = 0
    normal = 0
    for i, a in enumerate(anom):
        if not math.isnan(a) and a > thr:
            trim = i + 1
            normal = 0
        else:
            normal += 1
            if normal >= gap:
                break
    return trim


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

    def _span(self, data, sr, ft, pk, refs, span) -> dict[int, dict]:
        """Tails recovered for one failed span (interval index -> tail)."""
        iv, last, marker_pos = span
        dri = ft.restart_interval
        total = sr.total_mcus
        tails: dict[int, dict] = {}
        j = last
        mpos = marker_pos
        while j >= iv.index:
            first_of_j = j * dri if dri else iv.mcu_first
            count_j = min(dri, total - first_of_j) if dri else total - iv.mcu_first
            if count_j <= 0:
                j -= 1
                continue
            lo = iv.byte_start
            bad = last_dense_damage(data, lo, mpos)
            if bad is not None:
                lo = bad + 2  # end of a rotted stretch: intact data resumes after it
            bits, _flows = K.tail_flow(data, lo, mpos, pk, count_j)
            n = len(bits)
            if n < MIN_TAIL + DROP_FIRST:
                break
            start_bit = bits[0]
            full = False
            if dri and j > iv.index:
                # a later interval of the span may be intact from its very start
                avg = max(64, (iv.byte_end or mpos) - iv.byte_start) / max(1, last - iv.index + 1)
                # the true start may lie a few MCUs inside the found tail (junk head)
                hi_s = bits[min(len(bits) - 1, DROP_FIRST + 12)] // 8 + 1
                s0 = K.find_interval_start(data, max(lo, int(mpos - 2.5 * avg)), min(mpos, hi_s),
                                           mpos, pk, count_j, bits, 0xD0 + ((j - 1) % 8))
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
                delta = refs.offsets(first_tail, dc_rel)
                left = _left_offsets(sr, ft, first_tail, dc_rel)
                delta = {cid: (delta[cid] if delta[cid] is not None else (left[cid] if left[cid] is not None else 0.0))
                         for cid in pk.cids}
            # trim leading MCUs that clash with their neighbours (junk before convergence)
            trim = 0
            if not full:
                k = min(TRIM_SCAN, n_tail - MIN_TAIL)
                anom = head_anomaly(refs, first_tail, dc_rel, delta, pk)
                trim = leading_junk(anom)
                # junk decoded inside the damage drifts: a step down in the anomaly
                clean_part = anom[n_tail // 2 :]
                sig = float(np.nanmedian(clean_part)) if not np.all(np.isnan(clean_part)) else 1.0
                onset = junk_onset((anom / max(1.0, sig))[::-1])
                if onset is not None:
                    trim = max(trim, n_tail - onset)
                trim = min(k, trim)
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
            if not full:
                break
            # the whole interval came back: its own (damaged) marker sits just before it
            mpos = start_bit // 8 - 2
            j -= 1
        return tails

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
        clean = sr.mcu_ok.copy()
        for iv in sr.intervals:
            if not iv.ok:
                clean[iv.mcu_first : iv.mcu_first + iv.mcu_count] = False
        thumb = thumbnail_base(data, ft, sr, clean, ctx.options.get("thumb_refs"))
        ctx.memo["thumb_base"] = thumb
        refs = References(sr, ft, thumb)
        tails: dict[int, dict] = {}
        for part in parallel.pmap(lambda sp: self._span(data, sr, ft, pk, refs, sp), self._spans(sr, parse)):
            tails.update(part)
        recovered = sum(t["n"] for t in tails.values())
        if tails:
            ctx.memo["tails"] = tails
            ctx.note(self, f"recovered {recovered} MCU(s) after damaged stretches in {len(tails)} interval(s)",
                     recovered_mcus=recovered, intervals=sorted(tails))
