"""R9 — rebuild every interval that still fails, so the whole image renders.

For each failed restart interval:
* the MCUs decoded before the error are kept, minus a safety margin (a desync is
  usually noticed a few MCUs after it happened) and minus MCUs whose DC clashes
  with their neighbours;
* a tail recovered by R10 (the intact end after a rotted stretch) is spliced in
  bit-exactly, with the DC predictor set so its colours come out right;
* what is left in between is synthesised as DC-only blocks whose value is
  interpolated between the nearest real rows above and below (or a marker colour
  with `mark_holes`).

The result is a standard JPEG that any viewer renders to full height; the holes
are listed in the report and the share of real (not synthesised) MCUs is the
recovered fraction.
"""

from __future__ import annotations

import math

import numpy as np

from ..jpeg import decoder as D
from ..jpeg import kernel as K
from ..jpeg import markers as M
from ..jpeg.encoder import BitWriter, encode_flat_mcus, gray_dc
from ..jpeg.tables import FrameTables, TableError
from ..jpeg.writer import assemble, rst_marker
from .base import Context, Hole, Recipe
from .registry import register

BACKOFF_MCUS = 3  # MCUs dropped before a detected error (silent desync margin)
ANOMALY_LEVELS = 96  # luma jump vs. the block above (gray levels) that marks an MCU as untrusted
ANOMALY_MAX_EXTRA = 16
MARK_YCC = (105, 212, 235)  # magenta in YCbCr


def decode_for(ctx: Context) -> tuple[M.ParseResult, FrameTables, D.ScanResult] | None:
    """Parse + decode the current data, cached on the context until the data changes."""
    hit = ctx.cache.get("decode")
    if hit is not None:
        return hit
    parse = M.parse_segments(ctx.data)
    if not parse.ok:
        return None
    try:
        ft = FrameTables.from_parse(parse)
        sr = D.decode_scan(ctx.data, parse, ft)
    except (TableError, D.MissingTables, D.Unsupported):
        return None
    ctx.cache["decode"] = (parse, ft, sr)
    return parse, ft, sr


def _stream_bits(w: BitWriter, data: bytes, start_bit: int, end_bit: int) -> None:
    """Append entropy-coded bits [start_bit, end_bit) of `data` (positions as the
    decoder reports them: byte offsets in the stuffed stream) to w, unstuffed."""
    p = start_bit // 8
    first = True
    while p * 8 < end_bit:
        b = data[p]
        lo_bit = start_bit % 8 if first else 0
        hi_bit = 8 if (p + 1) * 8 <= end_bit else end_bit - p * 8
        n = hi_bit - lo_bit
        if n > 0:
            w.put_bits((b >> (8 - hi_bit)) & ((1 << n) - 1), n)
        first = False
        if b == 0xFF and p + 1 < len(data) and data[p + 1] == 0x00:
            p += 2  # skip stuffing byte
        else:
            p += 1


@register
class FillHoles(Recipe):
    name = "fill_holes"
    priority = 60
    description = "Rebuild failed intervals: keep what decoded, splice recovered tails, interpolate the rest"

    def detect(self, ctx: Context):
        got = decode_for(ctx)
        if got is None:
            return None
        parse, ft, sr = got
        covered = sum(iv.mcu_count for iv in sr.intervals)
        if sr.error is None and all(iv.ok for iv in sr.intervals) and covered >= sr.total_mcus:
            return None
        return got

    def fix(self, ctx: Context, got) -> None:
        parse, ft, sr = got
        data = ctx.data
        frame = ft.frame
        dri = ft.restart_interval
        total = sr.total_mcus
        comps_d, mcus_per_row, _rows = D._scan_components(ft)
        comps = [(c.cid, c.h, c.v) for c in comps_d]
        mcu_info = {m.index: m for m in sr.mcus}
        mark = bool(ctx.options.get("mark_holes"))
        qdc = {sc.id: ft.quant[frame.component(sc.id).tq].dc for sc in ft.scan.components}
        luma, lh, lv = comps[0]
        tails: dict[int, dict] = ctx.memo.pop("tails", {}) if ctx.memo.get("tails") else {}

        # ---------------------------------------------------------------- pass 1: what is real
        work_dc = {cid: sr.dc[cid].copy() for cid, _h, _v in comps}
        trusted = sr.mcu_ok.copy()

        def untrust_tail(iv, keep):
            """Shrink `keep` by the safety margin and by DC anomalies near the end."""
            keep = max(0, keep - BACKOFF_MCUS)
            extra = 0
            arr = sr.dc[luma]
            while keep > 0 and extra < ANOMALY_MAX_EXTRA:
                mi = iv.mcu_first + keep - 1
                mrow, mcol = divmod(mi, mcus_per_row)
                brow, bcol = mrow * lv, mcol * lh
                cur = arr[brow, bcol]
                above = None
                r = brow - 1
                while r >= 0:
                    um = (r // lv) * mcus_per_row + bcol // lh
                    if trusted[um] and not math.isnan(arr[r, bcol]):
                        above = arr[r, bcol]
                        break
                    r -= 1
                if above is None or math.isnan(cur):
                    break
                if abs(cur - above) * qdc[luma] / 8 > ANOMALY_LEVELS:
                    keep -= 1
                    extra += 1
                else:
                    break
            return keep

        plan: dict[int, dict] = {}  # interval index -> {keep, tail}
        for iv in sr.intervals:
            if iv.ok:
                continue
            reason = iv.error.reason if iv.error else "damaged"
            if reason == "bad_rst" and not ctx.options.get("keep_damaged"):
                # decoded to the end but did not land on its marker: the desync happened
                # somewhere unknown, so nothing in it can be trusted
                keep = 0
            elif reason == "skipped":
                keep = 0
            else:
                keep = untrust_tail(iv, iv.decoded) if iv.decoded else 0
            tail = tails.get(iv.index)
            if tail is not None:
                n_tail = tail["n"]
                full = tail["full"] and n_tail == iv.mcu_count
                if not full and keep + n_tail > iv.mcu_count - 1:
                    keep = max(0, iv.mcu_count - 1 - n_tail)  # leave one MCU to carry the predictor
                if full:
                    keep = 0
                if not full and keep + n_tail > iv.mcu_count - 1:
                    tail = None  # cannot fit
            trusted[iv.mcu_first + keep : iv.mcu_first + iv.mcu_count] = False
            if tail is not None:
                f = tail["first_mcu"]
                n = tail["n"]
                trusted[f : f + n] = True
                rel = tail["dc_rel"]
                col = 0
                for cid, h, v in comps:
                    d = tail["delta"][cid]
                    idx = np.arange(f, f + n)
                    mrow, mcol = np.divmod(idx, mcus_per_row)
                    for by in range(v):
                        for bx in range(h):
                            work_dc[cid][mrow * v + by, mcol * h + bx] = rel[:, col] + d
                            col += 1
            plan[iv.index] = {"keep": keep, "tail": tail, "reason": reason}
        covered = max((iv.mcu_first + iv.mcu_count for iv in sr.intervals), default=0)
        if covered < total:
            trusted[covered:] = False

        # ---------------------------------------------------------------- interpolation sources
        interp: dict[int, np.ndarray] = {}
        for cid, h, v in comps:
            arr = work_dc[cid]
            rows, cols = arr.shape
            okm = np.repeat(np.repeat(trusted[: (rows // v) * mcus_per_row].reshape(rows // v, mcus_per_row), v, 0), h, 1)
            okm = okm[:rows, :cols] & ~np.isnan(arr)
            ridx = np.arange(rows)[:, None].repeat(cols, 1)
            above = np.where(okm, ridx, -1)
            above = np.maximum.accumulate(above, axis=0)
            below = np.where(okm, ridx, rows)
            below = np.minimum.accumulate(below[::-1], axis=0)[::-1]
            colidx = np.arange(cols)[None, :].repeat(rows, 0)
            va = np.where(above >= 0, arr[np.clip(above, 0, rows - 1), colidx], np.nan)
            vb = np.where(below < rows, arr[np.clip(below, 0, rows - 1), colidx], np.nan)
            da = (ridx - above).astype(float)
            db = (below - ridx).astype(float)
            lin = (va * db + vb * da) / np.where(da + db > 0, da + db, 1)
            val = np.where(np.isnan(va), vb, np.where(np.isnan(vb), va, lin))
            val = np.where(okm, arr, val)
            interp[cid] = np.nan_to_num(val, nan=0.0)

        def fill_targets(first, count, last_override=None):
            per_block: dict[int, list[int]] = {}
            idx = np.arange(first, first + count)
            mrow, mcol = np.divmod(idx, mcus_per_row)
            for ci, (cid, h, v) in enumerate(comps):
                if mark:
                    val = gray_dc(ft, cid, MARK_YCC[min(ci, 2)] if len(comps) > 1 else 200)
                    seq = [val] * (count * h * v)
                else:
                    cols = []
                    for by in range(v):
                        for bx in range(h):
                            cols.append(np.rint(interp[cid][mrow * v + by, mcol * h + bx]).astype(int))
                    seq = list(np.stack(cols, axis=1).reshape(-1)) if cols else []
                if last_override is not None and seq:
                    seq[-1] = int(last_override[cid])
                per_block[cid] = [int(x) for x in seq]
            return per_block

        def last_block_preds(mi):
            out = {}
            for cid, h, v in comps:
                mrow, mcol = divmod(mi, mcus_per_row)
                val = sr.dc[cid][mrow * v + v - 1, mcol * h + h - 1]
                out[cid] = 0 if math.isnan(val) else int(val)
            return out

        pk = K.packed_for(ft)

        def build_interval(iv, keep, tail, reason, preds_in):
            """Entropy bytes of one rebuilt interval: kept prefix + fill + spliced tail."""
            w = BitWriter()
            prefix = b""
            p_ = preds_in
            if keep > 0:
                cut = mcu_info[iv.mcu_first + keep]
                off, bit = divmod(cut.bit_start, 8)
                prefix = data[iv.byte_start : off]
                if prefix.endswith(b"\xff"):
                    prefix += b"\x00"
                if bit:
                    w.put_bits(data[off] >> (8 - bit), bit)
                p_ = last_block_preds(iv.mcu_first + keep - 1)
            n_tail = tail["n"] if tail is not None else 0
            gap = iv.mcu_count - keep - n_tail
            hole = None
            if gap > 0:
                override = tail["delta"] if tail is not None else None
                encode_flat_mcus(ft, gap, {}, p_, fill_targets(iv.mcu_first + keep, gap, override), writer=w)
                hole = Hole(iv.mcu_first + keep, gap, reason)
            if tail is not None:
                _stream_bits(w, data, tail["start_bit"], tail["end_bit"])
            return prefix + w.finish(), n_tail, hole

        def interval_ok(piece, iv) -> bool:
            if not dri:
                return True  # checked with the whole file below
            buf = piece + b"\xff\xd9"
            pr = np.zeros(pk.ncomp, dtype=np.int32)
            n, err, *_ = K.decode_mcus(buf, 0, 0, len(buf), pk, pr, iv.mcu_count)
            return n == iv.mcu_count and err is None

        # ---------------------------------------------------------------- pass 2: build
        pieces: list[bytes] = []
        holes: list[Hole] = []
        preds = {cid: 0 for cid, _h, _v in comps}
        spliced = 0
        last_index = -1
        for iv in sr.intervals:
            if dri:
                preds = {cid: 0 for cid, _h, _v in comps}
            if iv.ok:
                pieces.append(data[iv.byte_start : iv.byte_end])
                if iv.mcu_count:
                    preds = last_block_preds(iv.mcu_first + iv.mcu_count - 1)
            else:
                pl = plan[iv.index]
                piece, n_spliced, hole = build_interval(iv, pl["keep"], pl["tail"], pl["reason"], dict(preds))
                if pl["tail"] is not None and not interval_ok(piece, iv):
                    # the spliced tail does not decode in place: rebuild without it
                    f = pl["tail"]["first_mcu"]
                    trusted[f : f + pl["tail"]["n"]] = False
                    piece, n_spliced, hole = build_interval(iv, pl["keep"], None, pl["reason"], dict(preds))
                pieces.append(piece)
                spliced += n_spliced
                if hole is not None:
                    holes.append(hole)
            last_index = iv.index
            if dri and iv.mcu_first + iv.mcu_count < total:
                pieces.append(rst_marker(iv.index))

        # MCUs after the last interval the decoder reached (no RST to resync on, or EOF)
        if covered < total:
            first = covered
            idx = last_index + 1
            while first < total:
                count = min(dri, total - first) if dri else total - first
                if dri:
                    preds = {cid: 0 for cid, _h, _v in comps}
                blob = encode_flat_mcus(ft, count, {}, preds, fill_targets(first, count))
                pieces.append(blob)
                holes.append(Hole(first, count, "missing"))
                first += count
                if dri and first < total:
                    pieces.append(rst_marker(idx))
                idx += 1

        header = data[: parse.scan_start]
        new = assemble(header, b"".join(pieces))

        # Self-check: the rebuilt file must decode cleanly with our own decoder.
        parse2 = M.parse_segments(new)
        ft2 = FrameTables.from_parse(parse2)
        sr2 = D.decode_scan(new, parse2, ft2)
        if sr2.error is not None or not all(iv.ok for iv in sr2.intervals):
            ctx.note(self, f"rebuild self-check failed: {sr2.error}")
            return

        kept = int(trusted.sum())
        ctx.holes.extend(holes)
        ctx.recovered_fraction = kept / total if total else 0.0
        filled = sum(h.mcu_count for h in holes)
        msg = f"filled {len(holes)} hole(s) covering {filled} of {total} MCUs; {kept} MCUs kept"
        if spliced:
            msg += f" (incl. {spliced} recovered after damaged stretches)"
        ctx.replace_data(new, self, msg, holes=[h.to_dict() for h in holes], kept_mcus=kept,
                         total_mcus=total, spliced_mcus=spliced)


def _unstuff(blob: bytes) -> bytes:
    """Remove FF00 stuffing so the bytes can be re-streamed through a BitWriter."""
    return blob.replace(b"\xff\x00", b"\xff")
