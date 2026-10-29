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
JUNK_LEVEL = 6.0  # median anomaly (in units of the vertical prediction error) of a junk prefix
MARK_YCC = (105, 212, 235)  # magenta in YCbCr
TAIL_REF_ROUNDS = (2.0, 6.0, math.inf)  # mean reference distance (block rows) accepted per round


def block_mask(mcu_ok: np.ndarray, arr: np.ndarray, h: int, v: int, per_row: int) -> np.ndarray:
    """Per-block mask of one component: the block's MCU is trusted and its DC known."""
    rows, cols = arr.shape
    okm = np.repeat(np.repeat(mcu_ok[: (rows // v) * per_row].reshape(rows // v, per_row), v, 0), h, 1)
    return okm[:rows, :cols] & ~np.isnan(arr)


def vertical_reference(arr: np.ndarray, okm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per block: DC interpolated linearly between the nearest trusted blocks above and
    below in the same column (the block itself where trusted), and the distance in
    block rows to the nearer of them (0 where trusted, inf where the column has none)."""
    rows, cols = arr.shape
    ridx = np.arange(rows)[:, None].repeat(cols, 1)
    above = np.maximum.accumulate(np.where(okm, ridx, -1), axis=0)
    below = np.minimum.accumulate(np.where(okm, ridx, rows)[::-1], axis=0)[::-1]
    colidx = np.arange(cols)[None, :].repeat(rows, 0)
    va = np.where(above >= 0, arr[np.clip(above, 0, rows - 1), colidx], np.nan)
    vb = np.where(below < rows, arr[np.clip(below, 0, rows - 1), colidx], np.nan)
    da = (ridx - above).astype(float)
    db = (below - ridx).astype(float)
    lin = (va * db + vb * da) / np.where(da + db > 0, da + db, 1)
    val = np.where(np.isnan(va), vb, np.where(np.isnan(vb), va, lin))
    dist = np.minimum(np.where(above >= 0, da, np.inf), np.where(below < rows, db, np.inf))
    val = np.where(okm, arr, val)
    dist = np.where(okm, 0.0, dist)
    return val, dist


def tail_blocks(first: int, n: int, h: int, v: int, per_row: int) -> tuple[np.ndarray, np.ndarray]:
    """(block rows, block cols) of a component for MCUs first..first+n, shape (n, h*v),
    in the order the blocks appear in the MCU."""
    mrow, mcol = np.divmod(np.arange(first, first + n), per_row)
    br = np.stack([mrow * v + by for by in range(v) for _bx in range(h)], axis=1)
    bc = np.stack([mcol * h + bx for _by in range(v) for bx in range(h)], axis=1)
    return br, bc


def junk_onset(s: np.ndarray, min_t: float = 6.0, ratio: float = 2.5, floor: float = 4.0) -> int | None:
    """Where junk begins at the end of a per-MCU anomaly series (in noise units), or None.

    After a silent desync the DC predictor is corrupted and stays so: the anomaly steps
    up and does not come back. The split maximising the step statistic
    (mean after - mean before) * sqrt(k (n-k) / n) is accepted when it is significant
    and the level after it is clearly junk; the part before is searched again, since
    junk often comes in two stages (a small drift, then a big jump).
    """
    s = np.nan_to_num(np.asarray(s, dtype=float), nan=0.0)
    onset = None
    n = len(s)
    while n >= 4:
        cs = np.cumsum(s[:n])
        k = np.arange(1, n)
        pre = cs[:-1] / k
        post = (cs[-1] - cs[:-1]) / (n - k)
        t = (post - pre) * np.sqrt(k * (n - k) / n)
        i = int(np.argmax(t))
        if t[i] < min_t or post[i] < max(floor, ratio * pre[i]):
            break
        onset = int(k[i])
        n = onset
    return onset


def vertical_noise(arr: np.ndarray, okm: np.ndarray, q: int) -> float:
    """Typical gray-level error of predicting a clean block from the blocks above and below."""
    m = okm[1:-1] & okm[:-2] & okm[2:]
    if not m.any():
        return 1.0
    err = np.abs(arr[1:-1] - (arr[:-2] + arr[2:]) / 2.0)[m]
    return max(1.0, float(np.median(err)) * q / 8.0)


def anomaly_series(refs: dict, comps, per_row: int, qdc: dict, noise: dict, dc: dict, first: int, n: int,
                   max_dist: float = 8) -> np.ndarray:
    """Per MCU: worst component of |DC - reference| in units of that component's typical
    vertical prediction error (`noise`). `refs`: cid -> (values, distances) from
    vertical_reference; dc: cid -> the MCUs' first-block DC values. NaN without reference."""
    out = np.full(n, np.nan)
    for cid, h, v in comps:
        val, dist = refs[cid]
        br, bc = tail_blocks(first, n, h, v, per_row)
        rv, dv = val[br[:, 0], bc[:, 0]], dist[br[:, 0], bc[:, 0]]
        a = np.abs(dc[cid] - rv) * qdc[cid] / 8.0 / noise[cid]
        a = np.where((dv <= max_dist) & ~np.isnan(rv), a, np.nan)
        out = np.where(np.isnan(out), a, np.where(np.isnan(a), out, np.maximum(out, a)))
    return out


def legal_offset(rel: np.ndarray, q: int, d: float) -> float | None:
    """Clamp offset d so every rel + d stays a legal DC value; None when impossible."""
    lim = (1024 + q) // q
    lo_d, hi_d = -lim - int(rel.min()), lim - int(rel.max())
    if lo_d > hi_d:
        return None
    return float(min(max(d, lo_d), hi_d))


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

        # references for judging kept prefixes: clean intervals only (a prefix may be junk)
        clean = sr.mcu_ok.copy()
        for iv in sr.intervals:
            if not iv.ok:
                clean[iv.mcu_first : iv.mcu_first + iv.mcu_count] = False
        clean_masks = {cid: block_mask(clean, sr.dc[cid], h, v, mcus_per_row) for cid, h, v in comps}
        prefix_refs = {cid: vertical_reference(sr.dc[cid], clean_masks[cid]) for cid, h, v in comps}
        noise = {cid: vertical_noise(sr.dc[cid], clean_masks[cid], qdc[cid]) for cid, h, v in comps}

        def untrust_tail(iv, keep):
            """Shrink `keep` by the safety margin, and to just before a silent desync:
            from there on the DC drifts away from what the rows around suggest."""
            n = keep
            keep = max(0, keep - BACKOFF_MCUS)
            if n >= 4:
                idx = np.arange(iv.mcu_first, iv.mcu_first + n)
                mrow, mcol = np.divmod(idx, mcus_per_row)
                first_dc = {cid: sr.dc[cid][mrow * v, mcol * h] for cid, h, v in comps}
                s_ = anomaly_series(prefix_refs, comps, mcus_per_row, qdc, noise, first_dc, iv.mcu_first, n)
                onset = junk_onset(s_)
                if onset is not None:
                    keep = min(keep, max(0, onset - 1))  # the MCU before the jump is already suspect
                # junk from the very start (the interval's beginning rotted): no step to find,
                # but the MCUs sit far from the rows right next to them all along
                near = anomaly_series(prefix_refs, comps, mcus_per_row, qdc, noise, first_dc, iv.mcu_first, n,
                                      max_dist=1)[:keep]
                near = near[~np.isnan(near)]
                if len(near) >= 4 and float(np.median(near)) >= JUNK_LEVEL:
                    keep = 0
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
            plan[iv.index] = {"keep": keep, "tail": tail, "reason": reason}

        # Tails: the DC offset of each is re-estimated against the rows around it, as they
        # will look in the output - tails close to real rows first, the others then also
        # lean on those (dense damage often leaves no clean row next to a tail).
        pending = [pl for pl in plan.values() if pl["tail"] is not None]
        for max_dist in TAIL_REF_ROUNDS:
            if not pending:
                break
            refs = {cid: vertical_reference(work_dc[cid], block_mask(trusted, work_dc[cid], h, v, mcus_per_row))
                    for cid, h, v in comps}
            later = []
            for pl in pending:
                tail = pl["tail"]
                f, n, rel = tail["first_mcu"], tail["n"], tail["dc_rel"]
                deltas = {}
                col = 0
                dists = []
                for cid, h, v in comps:
                    if tail["full"]:
                        deltas[cid], col = 0.0, col + h * v  # a whole interval: DC values are absolute
                        dists.append(0.0)
                        continue
                    br, bc = tail_blocks(f, n, h, v, mcus_per_row)
                    val, dist = refs[cid]
                    rv, dv = val[br, bc], dist[br, bc]
                    r = rel[:, col : col + h * v]
                    good = np.isfinite(dv) & ~np.isnan(rv)
                    dists.append(float(np.mean(dv[good])) if good.any() else math.inf)
                    deltas[cid] = float(np.median(rv[good] - r[good])) if good.any() else None
                    col += h * v
                if max(dists) > max_dist:
                    later.append(pl)
                    continue
                col = 0
                ok = True
                for cid, h, v in comps:
                    d = deltas[cid] if deltas[cid] is not None else float(tail["delta"][cid])
                    d = legal_offset(rel[:, col : col + h * v], qdc[cid], d)
                    if d is None:
                        ok = False
                        break
                    deltas[cid] = d
                    col += h * v
                if not ok:
                    pl["tail"] = None
                    continue
                tail["delta"] = {cid: int(round(d)) for cid, d in deltas.items()}
                trusted[f : f + n] = True
                col = 0
                for cid, h, v in comps:
                    br, bc = tail_blocks(f, n, h, v, mcus_per_row)
                    work_dc[cid][br, bc] = rel[:, col : col + h * v] + tail["delta"][cid]
                    col += h * v
            pending = later
        covered = max((iv.mcu_first + iv.mcu_count for iv in sr.intervals), default=0)
        if covered < total:
            trusted[covered:] = False

        # ---------------------------------------------------------------- interpolation sources
        interp: dict[int, np.ndarray] = {}
        for cid, h, v in comps:
            val, _dist = vertical_reference(work_dc[cid], block_mask(trusted, work_dc[cid], h, v, mcus_per_row))
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
