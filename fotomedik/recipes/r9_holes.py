"""R9 — fill what could not be recovered so the file opens completely.

Every restart interval that still fails after the other recipes is rebuilt:
the MCUs decoded before the error are kept (minus a safety margin, because a
flipped bit is usually noticed a few MCUs after it happened), the rest of the
interval is replaced by DC-only blocks continuing the colour of the row above
(or a marker colour when `mark_holes` is set). The result is a standard JPEG
that any viewer renders to full height; the holes are listed in the report.
"""

from __future__ import annotations

import math

import numpy as np

from ..jpeg import decoder as D
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


@register
class FillHoles(Recipe):
    name = "fill_holes"
    priority = 60
    description = "Replace unrecoverable intervals with flat blocks so the whole image renders"

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
        comps = [(sc.id, frame.component(sc.id)) for sc in ft.scan.components]
        interleaved = len(comps) > 1
        mcus_per_row = frame.mcus_per_row if interleaved else sr.dc[comps[0][0]].shape[1]
        mcu_info = {m.index: m for m in sr.mcus}
        trusted = sr.mcu_ok.copy()
        mark = bool(ctx.options.get("mark_holes"))
        qdc = {cid: ft.quant[fc.tq].dc for cid, fc in comps}
        luma = comps[0][0]

        def block_geom(cid, fc):
            return (fc.h, fc.v) if interleaved else (1, 1)

        def dc_above(cid, fc, brow, bcol):
            """DC of the nearest trusted block above (same column), or None."""
            h, v = block_geom(cid, fc)
            arr = sr.dc[cid]
            r = brow - 1
            while r >= 0:
                mi = (r // v) * mcus_per_row + (bcol // h)
                if trusted[mi] and not math.isnan(arr[r, bcol]):
                    return int(arr[r, bcol])
                r -= 1
            return None

        def untrust_tail(iv, keep):
            """Shrink `keep` by the safety margin and by DC anomalies near the end."""
            keep = max(0, keep - BACKOFF_MCUS)
            extra = 0
            h, v = block_geom(luma, comps[0][1])
            while keep > 0 and extra < ANOMALY_MAX_EXTRA:
                mi = iv.mcu_first + keep - 1
                mrow, mcol = divmod(mi, mcus_per_row)
                brow, bcol = mrow * v, mcol * h
                cur = sr.dc[luma][brow, bcol]
                above = dc_above(luma, comps[0][1], brow, bcol) if brow > 0 else None
                if above is None or math.isnan(cur):
                    break
                if abs(cur - above) * qdc[luma] / 8 > ANOMALY_LEVELS:
                    keep -= 1
                    extra += 1
                else:
                    break
            return keep

        def fill_targets(first, count):
            per_block: dict[int, list[int]] = {}
            for cid, fc in comps:
                h, v = block_geom(cid, fc)
                seq: list[int] = []
                for mi in range(first, first + count):
                    mrow, mcol = divmod(mi, mcus_per_row)
                    for by in range(v):
                        for bx in range(h):
                            brow, bcol = mrow * v + by, mcol * h + bx
                            if mark:
                                idx = [c for c, _ in comps].index(cid)
                                seq.append(gray_dc(ft, cid, MARK_YCC[min(idx, 2)] if interleaved else 200))
                            else:
                                val = dc_above(cid, fc, brow, bcol)
                                seq.append(0 if val is None else val)
                per_block[cid] = seq
            return per_block

        pieces: list[bytes] = []
        holes: list[Hole] = []
        preds = {cid: 0 for cid, _ in comps}

        def last_block_preds(mi):
            out = {}
            for cid, fc in comps:
                h, v = block_geom(cid, fc)
                mrow, mcol = divmod(mi, mcus_per_row)
                val = sr.dc[cid][mrow * v + v - 1, mcol * h + h - 1]
                out[cid] = 0 if math.isnan(val) else int(val)
            return out

        def emit_fill(first, count, preds, reason):
            if count <= 0:
                return
            blob = encode_flat_mcus(ft, count, {}, preds, fill_targets(first, count))
            pieces.append(blob)
            holes.append(Hole(first, count, reason))
            trusted[first : first + count] = False

        covered = 0
        last_index = -1
        for iv in sr.intervals:
            covered = max(covered, iv.mcu_first + iv.mcu_count)
            if dri:
                preds = {cid: 0 for cid, _ in comps}
            if iv.ok:
                chunk = data[iv.byte_start : iv.byte_end]
                pieces.append(chunk)
                if iv.mcu_count:
                    preds = last_block_preds(iv.mcu_first + iv.mcu_count - 1)
            else:
                reason = iv.error.reason if iv.error else "damaged"
                if reason in ("bad_rst", "unexpected_marker") and not ctx.options.get("keep_damaged"):
                    # The interval decoded to the end but did not land on its marker: the
                    # desync happened somewhere unknown, so nothing in it can be trusted.
                    keep = 0
                else:
                    keep = untrust_tail(iv, iv.decoded) if iv.decoded else 0
                if keep > 0:
                    cut = mcu_info[iv.mcu_first + keep]
                    off, bit = divmod(cut.bit_start, 8)
                    prefix = data[iv.byte_start : off]
                    if prefix.endswith(b"\xff"):
                        prefix += b"\x00"
                    preds = last_block_preds(iv.mcu_first + keep - 1)
                    rest = iv.mcu_count - keep
                    # partial last byte of the kept part + synthesised fill, re-streamed bit-exact
                    w = BitWriter()
                    if bit:
                        w.put_bits(data[off] >> (8 - bit), bit)
                    if rest > 0:
                        blob = encode_flat_mcus(ft, rest, {}, preds, fill_targets(iv.mcu_first + keep, rest))
                        for b in _unstuff(blob):
                            w.put_bits(b, 8)
                        holes.append(Hole(iv.mcu_first + keep, rest, reason))
                        trusted[iv.mcu_first + keep : iv.mcu_first + iv.mcu_count] = False
                    pieces.append(prefix + w.finish())
                else:
                    trusted[iv.mcu_first : iv.mcu_first + iv.mcu_count] = False
                    emit_fill(iv.mcu_first, iv.mcu_count, preds, reason)
            last_index = iv.index
            if dri and iv.mcu_first + iv.mcu_count < total:
                pieces.append(rst_marker(iv.index))

        # MCUs after the last interval the decoder reached (no RST to resync on, or EOF)
        if covered < total:
            first = covered
            if dri:
                idx = last_index + 1
                while first < total:
                    count = min(dri, total - first)
                    preds = {cid: 0 for cid, _ in comps}
                    trusted[first : first + count] = False
                    emit_fill(first, count, preds, "missing")
                    first += count
                    if first < total:
                        pieces.append(rst_marker(idx))
                    idx += 1
            else:
                trusted[first:total] = False
                emit_fill(first, total - first, preds, "missing")

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
        ctx.replace_data(
            new,
            self,
            f"filled {len(holes)} hole(s) covering {filled} of {total} MCUs; {kept} MCUs kept",
            holes=[h.to_dict() for h in holes],
            kept_mcus=kept,
            total_mcus=total,
        )


def _unstuff(blob: bytes) -> bytes:
    """Remove FF00 stuffing so the bytes can be re-streamed through a BitWriter."""
    return blob.replace(b"\xff\x00", b"\xff")
