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


def _block_coefficients(mean: np.ndarray, rounds: int = 6) -> np.ndarray:
    """Coefficients x (block grid) of the surface W_r x W_c^T whose 8x8 averages are
    `mean`: a cubic through the block centres averages a little lower than a peak block
    (and higher than a dip), so the difference is fed back a few times. Worked out on
    the block grid (cheap); pixels are made only where needed."""
    rows, cols = mean.shape
    ar = upsample_weights(rows).reshape(rows, 8, rows).mean(axis=1)
    ac = upsample_weights(cols).reshape(cols, 8, cols).mean(axis=1)
    target = mean.astype(float)
    x = np.zeros_like(target)
    for _ in range(rounds):
        x += target - ar @ x @ ac.T
    return x


def surface(mean: np.ndarray, rounds: int = 6) -> np.ndarray:
    """Smooth pixel plane (8x the block grid) whose 8x8 averages are the block averages,
    so blocks meet without steps."""
    rows, cols = mean.shape
    x = _block_coefficients(mean, rounds)
    return upsample_weights(rows) @ x @ upsample_weights(cols).T


def _local(n_side: int = 5) -> np.ndarray:
    """(8, 5) weights of the pixel rows of one block on the coefficients of blocks
    -2..+2 around it (interior blocks; edges are handled by clamped indices)."""
    w = upsample_weights(9)  # block 4 of 9 is far from both edges
    return w[32:40, 2:7]


def hole_coefficients(dc: np.ndarray, holes: np.ndarray, quant: np.ndarray) -> np.ndarray:
    """AC coefficients (zigzag, quantised; DC slot 0) for the hole blocks of one component.

    dc: (rows, cols) quantised DC per block; holes: (rows, cols) bool;
    quant: 64 quantisation values in zigzag order. Returns int16 (rows, cols, 64)."""
    rows, cols = dc.shape
    out = np.zeros((rows, cols, 64), dtype=np.int16)
    if not holes.any():
        return out
    x = _block_coefficients(dc * (quant[0] / 8.0))
    k = DCT @ _local()  # (8, 5): DCT of a block's pixels from its 5x5 neighbourhood
    q_nat = np.empty(64)
    q_nat[ZIGZAG] = quant
    by, bx = np.nonzero(holes)
    off = np.arange(-2, 3)
    ry = np.clip(by[:, None] + off, 0, rows - 1)  # (n, 5)
    rx = np.clip(bx[:, None] + off, 0, cols - 1)
    nb = x[ry[:, :, None], rx[:, None, :]]  # (n, 5, 5)
    coef = k @ nb @ k.T  # (n, 8, 8), [v, u]
    coef = coef.reshape(-1, 64) / q_nat
    zz = np.rint(coef[:, ZIGZAG]).astype(np.int16)
    zz[:, 0] = 0
    zz[:, KEEP:] = 0
    out[by, bx] = zz
    return out


BRIDGE_MAX = 40  # pixels: gaps up to this tall are bridged from the real pixels above and below


def component_planes(data: bytes, ft, comps) -> dict | None:
    """Decoded pixels per component at its own resolution (block grid size), level-shifted."""
    import io

    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        im = Image.open(io.BytesIO(data))
        im.draft("YCbCr", im.size)
        im = im.convert("YCbCr") if im.mode not in ("YCbCr", "L") else im
        arr = np.asarray(im, dtype=np.float32)
    except Exception:
        return None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
    if arr.ndim == 2:
        arr = arr[:, :, None]
    fr = ft.frame
    out = {}
    for i, c in enumerate(comps):
        ch = arr[:, :, min(i, arr.shape[2] - 1)] - 128.0
        fc = fr.component(c.cid)
        sy, sx = fr.vmax // fc.v, fr.hmax // fc.h
        if sy > 1 or sx > 1:
            hh, ww = ch.shape[0] // sy * sy, ch.shape[1] // sx * sx
            ch = ch[:hh, :ww].reshape(hh // sy, sy, ww // sx, sx).mean(axis=(1, 3))
        H, W = c.blocks_h * 8, c.blocks_w * 8
        ch = np.pad(ch, ((0, max(0, H - ch.shape[0])), (0, max(0, W - ch.shape[1]))), mode="edge")[:H, :W]
        out[c.cid] = ch
    return out


def bridge_coefficients(plane: np.ndarray, holes: np.ndarray, quant: np.ndarray, max_gap: int = BRIDGE_MAX):
    """For hole blocks inside a short vertical gap: AC coefficients (zigzag, quantised,
    DC slot 0) of a linear blend, pixel column by pixel column, between the last real
    pixel above and the first real pixel below - edges and texture run through the gap
    instead of stopping at it. Returns (coefficients (rows, cols, 64), bridged mask)."""
    rows, cols = holes.shape
    hp = np.repeat(np.repeat(holes, 8, 0), 8, 1)
    known = ~hp
    H = rows * 8
    ridx = np.arange(H)[:, None].repeat(hp.shape[1], 1)
    above = np.maximum.accumulate(np.where(known, ridx, -1), axis=0)
    below = np.minimum.accumulate(np.where(known, ridx, H)[::-1], axis=0)[::-1]
    ok = (above >= 0) & (below < H) & (below - above <= max_gap + 1)
    colidx = np.arange(hp.shape[1])[None, :].repeat(H, 0)
    va = plane[np.clip(above, 0, H - 1), colidx]
    vb = plane[np.clip(below, 0, H - 1), colidx]
    t = (ridx - above) / np.maximum(below - above, 1)
    target = va + (vb - va) * t
    good = ok.reshape(rows, 8, cols, 8).all(axis=(1, 3)) & holes
    out = np.zeros((rows, cols, 64), dtype=np.int16)
    if not good.any():
        return out, good
    q_nat = np.empty(64)
    q_nat[ZIGZAG] = quant
    by, bx = np.nonzero(good)
    blocks = target.reshape(rows, 8, cols, 8).transpose(0, 2, 1, 3)[by, bx]
    coef = (DCT @ blocks @ DCT.T).reshape(-1, 64) / q_nat
    zz = np.clip(np.rint(coef[:, ZIGZAG]), -1023, 1023).astype(np.int16)
    zz[:, 0] = 0
    out[by, bx] = zz
    return out, good


def reference_coefficients(plane: np.ndarray, holes: np.ndarray, quant: np.ndarray):
    """Quantised coefficients (zigzag, DC in slot 0) of the reference picture's own
    pixels for every hole block: (coefficients (rows, cols, 64) int16, DC (rows, cols) int32)."""
    rows, cols = holes.shape
    q_nat = np.empty(64)
    q_nat[ZIGZAG] = quant
    out = np.zeros((rows, cols, 64), dtype=np.int16)
    dc = np.zeros((rows, cols), dtype=np.int32)
    by, bx = np.nonzero(holes)
    if len(by) == 0:
        return out, dc
    blocks = plane.reshape(rows, 8, cols, 8).transpose(0, 2, 1, 3)[by, bx]
    coef = (DCT @ blocks @ DCT.T).reshape(-1, 64) / q_nat
    zz = np.clip(np.rint(coef[:, ZIGZAG]), -1023, 1023).astype(np.int16)
    lim = 1024 // max(1, int(quant[0]))
    dc[by, bx] = np.clip(zz[:, 0], -lim, lim)
    zz[:, 0] = 0
    out[by, bx] = zz
    return out, dc


def smooth_holes(data: bytes, hole_mcus: np.ndarray, reference=None) -> bytes | None:
    """`data` with the given MCUs rewritten as smooth blocks, or None when the scan
    does not decode cleanly or nothing would change.

    reference: YCbCr picture(s) of the same scene (fotomedik.reference): the hole blocks
    then carry its pixels (aligned and calibrated against the decoded ones) instead of
    the smooth surface; decoded MCUs are copied bit for bit either way."""
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
    planes = component_planes(data, ft, comps)
    hole_blocks = {}
    for c in comps:
        arr = sr.dc[c.cid]
        if np.isnan(arr).any():
            return None
        hole_blocks[c.cid] = block_mask(hole_mcus, arr, c.h, c.v, per_row)
    planes_ok = planes is not None and all(
        planes[c.cid].shape == (sr.dc[c.cid].shape[0] * 8, sr.dc[c.cid].shape[1] * 8) for c in comps)
    ref_planes = None
    if reference is not None and planes_ok:
        from ..reference import aligned_planes, best_candidate

        cands = reference if isinstance(reference, list) else [reference]
        y0 = planes[comps[0].cid]
        trusted = ~np.repeat(np.repeat(hole_blocks[comps[0].cid], 8, 0), 8, 1)[: y0.shape[0], : y0.shape[1]]
        try:
            ref_planes = aligned_planes(best_candidate(cands, y0, trusted), planes, comps, ft.frame, hole_blocks)
        except Exception:
            ref_planes = None
    for c in comps:
        arr = sr.dc[c.cid]
        quant = np.asarray(ft.quant[ft.frame.component(c.cid).tq].values, dtype=float)
        holes = hole_blocks[c.cid]
        dcg = np.rint(arr).astype(np.int32)
        if ref_planes is not None:
            ac, dc_ref = reference_coefficients(ref_planes[c.cid], holes, quant)
            dcg = np.where(holes, dc_ref, dcg)
        else:
            ac = hole_coefficients(arr, holes, quant)
            if planes_ok:
                bridged, good = bridge_coefficients(planes[c.cid], holes, quant)
                ac = np.where(good[..., None], bridged, ac)
        grids[c.cid] = (dcg, ac, np.rint(arr).astype(np.int32))
    if not any(g[1].any() for g in grids.values()):
        return None

    # every hole MCU's blocks in scan order, built once
    hole_idx = np.flatnonzero(hole_mcus)
    slot = np.full(total, -1, dtype=np.int64)
    slot[hole_idx] = np.arange(len(hole_idx))
    r_all, c_all = hole_idx // per_row, hole_idx % per_row
    # a hole MCU followed by a real one in the same interval must end on the DC the real
    # MCU's bits were coded against (its decoded DC); elsewhere a hole's DC is free
    iv_end = np.zeros(total + 1, dtype=bool)
    for iv in sr.intervals:
        iv_end[iv.mcu_first + iv.mcu_count] = True
    nxt = hole_idx + 1
    keep_dc = (nxt < total) & ~iv_end[nxt] & ~hole_mcus[np.minimum(nxt, total - 1)]
    parts = []
    for c in comps:
        dcg, acg, decoded = grids[c.cid]
        for yy in range(c.v):
            for xx in range(c.h):
                ry, rx = r_all * c.v + yy, c_all * c.h + xx
                blk = acg[ry, rx].astype(np.int32)
                dcs = dcg[ry, rx]
                if yy == c.v - 1 and xx == c.h - 1:  # the component's last block of the MCU
                    dcs = np.where(keep_dc, decoded[ry, rx], dcs)
                blk[:, 0] = dcs
                parts.append(blk)
    table = np.stack(parts, axis=1)  # (n_holes, bpm, 64)

    def mcu_coefs(first: int, count: int) -> np.ndarray:
        """int32 [count, blocks_per_mcu, 64] for hole MCUs first..first+count in scan order."""
        i = slot[first]
        return table[i : i + count]

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
        real = ~hole_blocks[c.cid]
        if not np.array_equal(sr2.dc[c.cid][real], sr.dc[c.cid][real]):
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
        from ..reference import for_context

        reference = for_context(ctx, ft)
        new = smooth_holes(ctx.data, mask, reference)
        if new is None:
            ctx.note(self, "holes left flat (scan did not re-encode cleanly)")
            return
        n = int(mask.sum())
        if reference:
            ctx.memo["reference_used"] = True
            ctx.replace_data(new, self, f"{n} filled MCU(s) carry the reference picture's pixels", mcus=n,
                             reference=str(ctx.options.get("reference")))
        else:
            ctx.replace_data(new, self, f"smoothed {n} filled MCU(s) (soft surface instead of flat squares)", mcus=n)
