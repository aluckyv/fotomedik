"""Smooth holes: a soft surface instead of flat 8x8 squares.

Holes are filled with DC-only blocks: one flat colour per 8x8 block (per 16x16 for
subsampled chroma), which reads as a mosaic of squares. Their DC values already
follow the embedded thumbnail (or the rows around), so the coarse picture is right;
only the steps between blocks are wrong.

Here every component's block averages are treated as samples at the block centres
and interpolated with a cubic (Catmull-Rom) surface across the whole image, real
blocks included, so holes blend into the real picture around them. The surface over
each hole block is transformed (DCT), quantised with the file's own table and
written back as that block's AC coefficients. The DC stays exactly as it was.

Real image data is never re-encoded: the bits of every real MCU are copied as they
are, and only the hole MCUs are written anew. The file keeps its header, tables and
restart markers; the DC chain is unchanged, so every real block decodes as before.
"""

from __future__ import annotations

import numpy as np

from ..jpeg import decoder as D
from ..jpeg import kernel as K
from ..jpeg import markers as M
from ..jpeg.encoder import BitWriter, encode_blocks
from ..jpeg.tables import FrameTables, TableError
from ..jpeg.writer import assemble, rst_marker
from .base import Context, Recipe
from .dcfield import block_mask
from .registry import register

KEEP = 21  # zigzag coefficients kept (the 6 lowest diagonals): the surface is smooth, the rest is noise

ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
])  # zigzag index -> natural (row * 8 + col) index


def _dct_matrix() -> np.ndarray:
    d = np.zeros((8, 8))
    for u in range(8):
        cu = np.sqrt(0.5) if u == 0 else 1.0
        for x in range(8):
            d[u, x] = 0.5 * cu * np.cos((2 * x + 1) * u * np.pi / 16)
    return d


DCT = _dct_matrix()


def upsample_weights(n: int) -> np.ndarray:
    """(8n, n) Catmull-Rom weights: pixel i sits at (i + 0.5) / 8 - 0.5 in block-centre
    coordinates; samples beyond the edges repeat the edge block."""
    out = np.zeros((8 * n, n))
    t = (np.arange(8 * n) + 0.5) / 8 - 0.5
    base = np.floor(t).astype(int)
    f = t - base
    w = (
        ((-f + 2) * f - 1) * f / 2,
        ((3 * f - 5) * f * f + 2) / 2,
        ((-3 * f + 4) * f + 1) * f / 2,
        (f - 1) * f * f / 2,
    )
    rows = np.arange(8 * n)
    for j, wj in enumerate(w):
        idx = np.clip(base - 1 + j, 0, n - 1)
        np.add.at(out, (rows, idx), wj)
    return out


def surface(mean: np.ndarray, rounds: int = 6) -> np.ndarray:
    """Smooth pixel plane (8x the block grid) whose 8x8 averages are the block averages.

    A cubic through the block centres averages a little lower than a peak block
    (and higher than a dip); the difference is fed back a few times, so the surface
    over each block averages to that block's own DC and blocks meet without steps."""
    rows, cols = mean.shape
    wr, wc = upsample_weights(rows), upsample_weights(cols)
    target = mean.astype(float)
    plane = np.zeros((8 * rows, 8 * cols))
    for _ in range(rounds):
        got = plane.reshape(rows, 8, cols, 8).mean(axis=(1, 3))
        plane += wr @ (target - got) @ wc.T
    return plane


def hole_coefficients(dc: np.ndarray, holes: np.ndarray, quant: np.ndarray) -> np.ndarray:
    """AC coefficients (zigzag, quantised; DC slot 0) for the hole blocks of one component.

    dc: (rows, cols) quantised DC per block; holes: (rows, cols) bool;
    quant: 64 quantisation values in zigzag order. Returns int32 (rows, cols, 64)."""
    rows, cols = dc.shape
    out = np.zeros((rows, cols, 64), dtype=np.int32)
    if not holes.any():
        return out
    plane = surface(dc * (quant[0] / 8.0))
    q_nat = np.empty(64)
    q_nat[ZIGZAG] = quant
    by, bx = np.nonzero(holes)
    blocks = plane.reshape(rows, 8, cols, 8).transpose(0, 2, 1, 3)[by, bx]  # (n, 8, 8)
    coef = DCT @ blocks @ DCT.T  # (n, 8, 8), [v, u]
    coef = coef.reshape(-1, 64) / q_nat
    zz = np.rint(coef[:, ZIGZAG]).astype(np.int32)
    zz[:, 0] = 0
    zz[:, KEEP:] = 0
    out[by, bx] = zz
    return out


def smooth_holes(data: bytes, hole_mcus: np.ndarray) -> bytes | None:
    """`data` with the given MCUs rewritten as smooth blocks, or None when the scan
    does not decode cleanly or nothing would change."""
    parse = M.parse_segments(data)
    try:
        ft = FrameTables.from_parse(parse)
    except TableError:
        return None
    if ft.frame.progressive or parse.scan_start is None:
        return None
    sr = D.decode_scan(data, parse, ft)
    total = sr.total_mcus
    if sr.error is not None or not all(iv.ok for iv in sr.intervals) or len(hole_mcus) != total:
        return None
    comps, per_row, _rows = D._scan_components(ft)
    pk = K.packed_for(ft)
    # the AC coefficients of every hole block, per component
    grids = {}
    for c in comps:
        arr = sr.dc[c.cid]
        if np.isnan(arr).any():
            return None
        quant = np.asarray(ft.quant[ft.frame.component(c.cid).tq].values, dtype=float)
        holes = block_mask(hole_mcus, arr, c.h, c.v, per_row)
        grids[c.cid] = (np.rint(arr).astype(np.int32), hole_coefficients(arr, holes, quant))
    if not any(g[1].any() for g in grids.values()):
        return None

    def mcu_coefs(first: int, count: int) -> np.ndarray:
        """int32 [count, blocks_per_mcu, 64] for MCUs first..first+count in scan order."""
        m = np.arange(first, first + count)
        r, col = m // per_row, m % per_row
        parts = []
        for c in comps:
            dcg, acg = grids[c.cid]
            for yy in range(c.v):
                for xx in range(c.h):
                    blk = acg[r * c.v + yy, col * c.h + xx].copy()
                    blk[:, 0] = dcg[r * c.v + yy, col * c.h + xx]
                    parts.append(blk)
        return np.stack(parts, axis=1)

    def preds_before(m: int, first: int) -> dict[int, int]:
        if m <= first:
            return {c.cid: 0 for c in comps}
        r, col = (m - 1) // per_row, (m - 1) % per_row
        return {c.cid: int(grids[c.cid][0][r * c.v + c.v - 1, col * c.h + c.h - 1]) for c in comps}

    bit_start = {mi.index: mi.bit_start for mi in sr.mcus}
    pieces: list[bytes] = []
    n_iv = len(sr.intervals)
    for k, iv in enumerate(sr.intervals):
        a, b = iv.mcu_first, iv.mcu_first + iv.mcu_count
        # where the interval's last MCU ends (its padding is not data)
        last = bit_start[b - 1]
        got = K.decode_mcus(data, last // 8, last % 8, iv.byte_end if iv.byte_end is not None else len(data), pk,
                            np.zeros(pk.ncomp, dtype=np.int32), 1, check_dc=False)
        if got[0] != 1:
            return None
        end_bit = got[5] * 8 + got[6]
        w = BitWriter()
        m = a
        while m < b:
            run_end = m
            hole = bool(hole_mcus[m])
            while run_end < b and bool(hole_mcus[run_end]) == hole:
                run_end += 1
            if hole:
                encode_blocks(ft, mcu_coefs(m, run_end - m), preds_before(m, a), w)
            else:
                w.copy_bits(data, bit_start[m], bit_start[run_end] if run_end < b else end_bit)
            m = run_end
        pieces.append(w.finish())
        if ft.restart_interval and k < n_iv - 1:
            pieces.append(rst_marker(iv.index))
    new = assemble(data[: parse.scan_start], b"".join(pieces))

    # Self-check: decodes cleanly and every block keeps its DC.
    parse2 = M.parse_segments(new)
    ft2 = FrameTables.from_parse(parse2)
    sr2 = D.decode_scan(new, parse2, ft2)
    if sr2.error is not None or not all(iv.ok for iv in sr2.intervals):
        return None
    for c in comps:
        if not np.array_equal(sr2.dc[c.cid], sr.dc[c.cid]):
            return None
    return new


@register
class SmoothHoles(Recipe):
    name = "smooth_holes"
    priority = 95  # last: after holes are filled and the thumbnail fallback has decided
    description = "Give filled holes a smooth surface instead of flat 8x8 squares"

    def detect(self, ctx: Context):
        if (not ctx.holes or ctx.fallback is not None or ctx.options.get("mark_holes")
                or not ctx.options.get("smooth_holes", True) or ctx.memo.get("smoothed")):
            return None
        return True

    def fix(self, ctx: Context, _got) -> None:
        ctx.memo["smoothed"] = True
        try:
            parse = M.parse_segments(ctx.data)
            ft = FrameTables.from_parse(parse)
            _comps, _per_row, _rows = D._scan_components(ft)
            total = D.decode_scan(ctx.data, parse, ft).total_mcus
        except Exception:
            return
        mask = np.zeros(total, dtype=bool)
        for h in ctx.holes:
            mask[h.mcu_first : h.mcu_first + h.mcu_count] = True
        new = smooth_holes(ctx.data, mask)
        if new is None:
            ctx.note(self, "holes left flat (scan did not re-encode cleanly)")
            return
        n = int(mask.sum())
        ctx.replace_data(new, self, f"smoothed {n} filled MCU(s) (soft surface instead of flat squares)", mcus=n)
