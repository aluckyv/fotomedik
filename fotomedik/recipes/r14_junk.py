"""Junk blocks: decoded data that is not picture.

Foreign bytes in the stream (a rotted or misplaced card sector, text, another file) decode
into valid-looking blocks: a Huffman stream accepts almost anything, and the decoder
notices nothing until it hits an impossible code - a few MCUs later on average, often
never before the next restart marker. The blocks pass the thumbnail test when their
colour happens to fit, and show up as stretches of coloured stripes, checkerboards and
black-and-white patches inside otherwise good rows.

Three things give them away, each measured against the picture's own statistics (the
95th percentile over the trusted MCUs, so a camera with fine texture is judged by its
own texture):
* luma spectrum: the mean |coefficient| over the high zigzag positions (28..63) or the
  middle ones (6..27) relative to the low ones (1..5). In a real block the energy falls
  off steeply with frequency; in garbage it does not (Abdat et al. 1996). A sharp edge
  has high-frequency energy too, but with far more in the low band, so the high band
  must also reach a share of the low one (RATIO);
* chroma spectrum: real chroma blocks are smooth, garbage chroma is not. The energy beyond
  the chroma low band counts, relative to the luma low band (coloured texture brings luma
  detail with it) and to the chroma low band (the edge of a yellow sign is a step, with a
  spectrum that falls off as steeply as any edge's);
* saturation: garbage whose DC chain ran away decodes as plain black or white blocks,
  which jump away from the trusted, unsaturated rows above and below (a black doorway
  under a black lintel does not).

A cue three times the picture's 95th percentile is "strong", twice is "weak". Junk comes
in runs - a spliced piece or a kept prefix that is garbage from some point on - so the
decision is made per run of trusted MCUs between holes: a stretch in which strong MCUs
are dense is junk (widened over its weak neighbours); a short head or tail of weak MCUs
next to a hole with a strong one among them is junk; an isolated MCU only when it is very
strong; short gaps between junk MCUs are junk too. Junk MCUs become holes (reason "junk"), the colours of all holes are worked out
again without them (fotomedik.recipes.dcfield), and smooth_holes writes them.

Measured on healthy photos with stretches of random bytes or text written into their
streams: about 40 percent of the junk MCUs are caught (most of the rest are saturated
patches inside junk regions, with no real neighbour to jump from), and 1 in 2000 real
MCUs is lost.
"""

from __future__ import annotations

import numpy as np

from ..jpeg import decoder as D
from ..jpeg import markers as M
from ..jpeg.tables import FrameTables, TableError
from ..reference import for_context as reference_for
from ..reference import twin_loader
from .base import Context, Hole, Recipe
from .dcfield import interpolation_sources, thumbnail_base
from .r12_smooth import DCT, ZIGZAG, component_planes
from .registry import register

FLOORS = {"flat": 0.2, "high": 1.0, "chroma": 0.5}  # the 95th percentile counts at least this much
MID_MIN = 2.0  # mean |coefficient| in the middle band a block needs for the flatness to mean anything
RATIO = 0.25  # high-band (or chroma) energy relative to the luma low band: edges have far less
CHROMA_RATIO = 0.5  # chroma energy beyond the low band relative to within it: a colour edge stays below
K_STRONG, K_WEAK, K_ISOLATED = 3.0, 2.0, 5.0  # multiples of the 95th percentile
SATURATED = 0.5  # share of luma pixels at 0 or 255 for a saturated MCU
SAT_JUMP = 50.0  # gray levels a saturated MCU must differ from the rows above and below
WINDOW, WINDOW_SHARE = 9, 0.45  # MCUs; a window with this share of strong MCUs is junk
PEEL_MAX, PEEL_GAP = 16, 1  # a head or tail next to a hole: weak MCUs, gaps of this many allowed
GAP = 3  # MCUs: a gap this short between junk MCUs is junk too
GROW_ROUNDS = 3  # how far junk spreads through saturated patches beside and below it

_POS = np.empty(64, dtype=int)
_POS[ZIGZAG] = np.arange(64)  # natural index -> zigzag position
LOW = (_POS >= 1) & (_POS < 6)
MID = (_POS >= 6) & (_POS < 28)
HIGH = _POS >= 28
CHROMA_HIGH = _POS >= 6
_DCT32 = DCT.astype(np.float32)


def comp_blocks(plane: np.ndarray, h: int, v: int, rows: int, per_row: int) -> np.ndarray:
    """(rows, per_row, h*v, 8, 8): one component's decoded blocks, MCU by MCU."""
    H, W = rows * v * 8, per_row * h * 8
    p = np.pad(plane, ((0, max(0, H - plane.shape[0])), (0, max(0, W - plane.shape[1]))), mode="edge")[:H, :W]
    return p.reshape(rows, v, 8, per_row, h, 8).transpose(0, 3, 1, 4, 2, 5).reshape(rows, per_row, v * h, 8, 8)


def features(planes: dict, comps, rows: int, per_row: int) -> dict:
    """Per MCU (rows, per_row), from the decoded pixels: luma band means low/mid/high,
    the chroma band means beyond and within the low band (of the chroma component with
    more energy beyond it; 0 without chroma), the share of saturated luma pixels and the
    mean luma level."""
    out: dict = {}
    for ci, c in enumerate(comps):
        blocks = comp_blocks(planes[c.cid], c.h, c.v, rows, per_row).astype(np.float32)
        if ci == 0:
            out["sat"] = ((blocks <= -126.5) | (blocks >= 125.5)).mean(axis=(2, 3, 4))
            out["level"] = blocks.mean(axis=(2, 3, 4))
        coef = np.abs(np.einsum("ab,rcnbd,ed->rcnae", _DCT32, blocks, _DCT32))
        coef = coef.reshape(rows, per_row, blocks.shape[2], 64)
        if ci == 0:
            out["low"] = coef[..., LOW].mean(axis=(2, 3))
            out["mid"] = coef[..., MID].mean(axis=(2, 3))
            out["high"] = coef[..., HIGH].mean(axis=(2, 3))
        else:
            lo, hi = coef[..., LOW].mean(axis=(2, 3)), coef[..., CHROMA_HIGH].mean(axis=(2, 3))
            if "chroma" in out:  # the chroma component with more energy beyond the low band
                pick = hi > out["chroma"]
                out["chroma"], out["chroma_low"] = np.where(pick, hi, out["chroma"]), np.where(pick, lo, out["chroma_low"])
            else:
                out["chroma"], out["chroma_low"] = hi, lo
    if "chroma" not in out:
        out["chroma"] = out["chroma_low"] = np.zeros_like(out["low"])
    return out


def saturated_jumps(f: dict, trusted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Saturated MCUs (black or white) that jump away from the trusted, unsaturated rows
    above and below: (both sides jump - one suffices at the picture's edge, either side)."""
    sat, level = f["sat"], f["level"]
    ok = trusted & (sat < 0.1)
    up = np.zeros_like(ok)
    dn = np.zeros_like(ok)
    up[1:] = ok[:-1] & (np.abs(level[1:] - level[:-1]) > SAT_JUMP)
    dn[:-1] = ok[1:] & (np.abs(level[:-1] - level[1:]) > SAT_JUMP)
    top = np.zeros_like(ok)
    top[0] = True
    bottom = np.zeros_like(ok)
    bottom[-1] = True
    is_sat = sat >= SATURATED
    return is_sat & ((up & dn) | (up & bottom) | (dn & top)), is_sat & (up | dn)


def cues(f: dict, trusted: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(strong, weak, isolated) per MCU, judged against the trusted MCUs' statistics, and
    the thresholds used (cue -> the value counting 1x)."""
    low = f["low"]
    raw = {"flat": f["mid"] / (low + 1.0), "high": f["high"], "chroma": f["chroma"]}
    applies = {"flat": f["mid"] > MID_MIN, "high": f["high"] / (low + 1.0) >= RATIO,
               "chroma": (f["chroma"] / (low + 1.0) >= RATIO) & (f["chroma"] / (f["chroma_low"] + 1.0) >= CHROMA_RATIO)}
    thr = {}
    score = np.zeros_like(low)
    for k, r in raw.items():
        thr[k] = max(FLOORS[k], float(np.percentile(r[trusted], 95))) if trusted.any() else FLOORS[k]
        score = np.maximum(score, np.where(applies[k], r / thr[k], 0.0))
    both, either = saturated_jumps(f, trusted)
    strong = (score >= K_STRONG) | both
    weak = (score >= K_WEAK) | either
    isolated = (score >= K_ISOLATED) | both
    return strong, weak, isolated, thr


def junk_runs(strong: np.ndarray, weak: np.ndarray, isolated: np.ndarray, trusted: np.ndarray) -> np.ndarray:
    """Junk MCUs decided per run of trusted MCUs within a row (see the module doc)."""
    out = np.zeros_like(trusted)
    rows, cols = trusted.shape
    half = WINDOW // 2
    for r in range(rows):
        c = 0
        while c < cols:
            if not trusted[r, c]:
                c += 1
                continue
            c0 = c
            while c < cols and trusted[r, c]:
                c += 1
            n = c - c0
            st, wk = strong[r, c0:c], weak[r, c0:c]
            if not wk.any():
                continue
            seg = np.zeros(n, dtype=bool)
            if n >= 3 and st.any():
                # dense stretches: the strong share over a sliding window
                cs = np.concatenate([[0], np.cumsum(st)])
                lo = np.clip(np.arange(n) - half, 0, n)
                hi = np.clip(np.arange(n) + half + 1, 0, n)
                dense = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1) >= WINDOW_SHARE
                for i in np.flatnonzero(dense):
                    a, b = max(0, i - half), min(n, i + half + 1)
                    seg[a:b] |= wk[a:b]
            # a head or tail next to a hole (not the picture's edge)
            for side in (0, 1):
                if not ((c0 > 0) if side == 0 else (c < cols)):
                    continue
                order = range(n) if side == 0 else range(n - 1, -1, -1)
                gap, last = 0, -1
                for k, i in enumerate(order):
                    if k >= PEEL_MAX:
                        break
                    if wk[i]:
                        last, gap = i, 0
                    else:
                        gap += 1
                        if gap > PEEL_GAP:
                            break
                if last >= 0:
                    sl = slice(0, last + 1) if side == 0 else slice(last, n)
                    if st[sl].any():
                        seg[sl] = True
            # isolated very strong MCUs, grown along weak neighbours
            grown = isolated[r, c0:c].copy()
            for _ in range(3):
                grown[1:] |= grown[:-1] & wk[1:]
                grown[:-1] |= grown[1:] & wk[:-1]
            seg |= grown
            if seg.any():
                # junk is one piece: short gaps between junk MCUs are junk too (not the rest of
                # a mostly-junk run, though: a kept prefix may end in junk and still be real before it)
                idx = np.flatnonzero(seg)
                for a, b in zip(idx[:-1], idx[1:]):
                    if 1 < b - a <= GAP + 1:
                        seg[a + 1 : b] = True
            out[r, c0:c] = seg
    return out


def grow_saturated(junk: np.ndarray, f: dict, trusted: np.ndarray, rounds: int = GROW_ROUNDS) -> np.ndarray:
    """Saturated MCUs (black or white patches) inside a junk region: junk above or below
    them and junk beside them. Garbage whose DC chain ran away leaves such patches with
    no real neighbour to jump from; a real dark region touching junk on one side only
    is not taken (it would need junk both above/below and beside it)."""
    out = junk.copy()
    cand = trusted & (f["sat"] >= SATURATED) & ~junk
    for _ in range(rounds):
        up = np.zeros_like(out)
        dn = np.zeros_like(out)
        lf = np.zeros_like(out)
        rt = np.zeros_like(out)
        up[1:] = out[:-1]
        dn[:-1] = out[1:]
        lf[:, 1:] = out[:, :-1]
        rt[:, :-1] = out[:, 1:]
        new = cand & ~out & (up | dn) & (lf | rt)
        if not new.any():
            break
        out |= new
    return out


def junk_mask(f: dict, trusted: np.ndarray) -> tuple[np.ndarray, dict]:
    """MCUs (rows, per_row) judged junk among the trusted ones, and the thresholds used.
    The thresholds come from the trusted MCUs themselves, so a second pass leaves out
    what the first one found (a lot of junk would have raised them)."""
    if not trusted.any():
        return np.zeros_like(trusted), dict(FLOORS)
    reference = trusted
    junk = np.zeros_like(trusted)
    for _ in range(2):
        strong, weak, isolated, thr = cues(f, reference)
        strong, weak, isolated = strong & trusted, weak & trusted, isolated & trusted
        junk = junk_runs(strong, weak, isolated, trusted) if strong.any() else strong
        if junk.any():
            junk = grow_saturated(junk, f, trusted)
        if not junk.any() or not (trusted & ~junk).any():
            break
        reference = trusted & ~junk
    return junk, thr


THUMB_FAR_K = 4.0  # a block this many times the thumbnail's usual error away from it is far off ...
THUMB_FAR_FLOOR = (24.0, 16.0)  # ... and at least this far (gray levels, luma/chroma)
THUMB_FAR_DENSE = 0.6  # share of far-off MCUs over one MCU row that makes a stretch junk


def thumbnail_far(sr, thumb, ft: FrameTables, comps, per_row: int, rows: int, trusted: np.ndarray) -> np.ndarray:
    """MCUs (rows, per_row) in long stretches that do not look like the photo at all.

    Another photo's sectors in the stream - same camera, same tables - decode into
    perfectly normal blocks: right spectrum, plausible colours, only of another scene.
    The thumbnail is the one witness: a whole stretch of blocks far from it, row after
    row, is not this photo. Single blocks at edges stray too (the thumbnail is small),
    so only dense stretches count."""
    far = np.zeros(rows * per_row, dtype=bool)
    for ci, c in enumerate(comps[:3]):
        if c.cid not in thumb.dc:
            continue
        arr, tdc = sr.dc[c.cid], thumb.dc[c.cid]
        q = ft.quant[ft.frame.component(c.cid).tq].dc
        lim = max(THUMB_FAR_K * float(thumb.sigma[c.cid]), THUMB_FAR_FLOOR[min(ci, 1)]) * 8.0 / q
        br, bc = (arr.shape[0] // c.v) * c.v, per_row * c.h
        d = np.abs(arr[:br, :bc] - tdc[:br, :bc])
        d = np.where(np.isfinite(d), d, 0.0).reshape(br // c.v, c.v, per_row, c.h)
        m = (np.median(d.transpose(0, 2, 1, 3).reshape(br // c.v, per_row, -1), axis=-1) > lim).ravel()
        far[: len(m)] |= m
    far = far.reshape(rows, per_row) & trusted
    flat = far.ravel().astype(float)
    dense = np.convolve(flat, np.ones(per_row) / per_row, mode="same") > THUMB_FAR_DENSE
    # the whole stretch: dense windows and the far MCUs touching them
    grown = np.convolve(dense.astype(float), np.ones(per_row), mode="same") > 0
    return (grown & far.ravel()).reshape(rows, per_row) | dense.reshape(rows, per_row) & trusted


@register
class JunkBlocks(Recipe):
    name = "junk_blocks"
    priority = 92  # after the holes are filled and the thumbnail fallback has decided, before smoothing
    description = "Decoded blocks whose spectrum or colour says garbage become holes"

    def detect(self, ctx: Context):
        if not ctx.holes or ctx.fallback is not None or ctx.options.get("mark_holes") or ctx.memo.get("junk_checked"):
            return None
        return True

    def fix(self, ctx: Context, _got) -> None:
        ctx.memo["junk_checked"] = True
        try:
            parse = M.parse_segments(ctx.data)
            ft = FrameTables.from_parse(parse)
            comps, per_row, _rows = D._scan_components(ft)
            sr = D.decode_scan(ctx.data, parse, ft)
        except (TableError, Exception):
            return
        if sr.error is not None or not all(iv.ok for iv in sr.intervals):
            return
        total = sr.total_mcus
        rows = -(-total // per_row)
        planes = component_planes(ctx.data, ft, comps)
        if planes is None:
            return
        hole = np.zeros(rows * per_row, dtype=bool)
        for h in ctx.holes:
            hole[h.mcu_first : h.mcu_first + h.mcu_count] = True
        hole[total:] = True  # padding beyond the last MCU (none for a whole number of rows)
        trusted = ~hole.reshape(rows, per_row)
        try:
            junk, thr = junk_mask(features(planes, comps, rows, per_row), trusted)
        except (ValueError, MemoryError):
            return
        thumb = ctx.memo.get("thumb_base")
        far = 0
        if thumb is not None and not getattr(thumb, "twin", False):
            alien = thumbnail_far(sr, thumb, ft, comps, per_row, rows, trusted) & ~junk
            far = int(alien.sum())
            junk = junk | alien
        n = int(junk.sum())
        thr_text = ", ".join(f"{k} {v:.2f}" for k, v in thr.items())
        if n == 0:
            ctx.note(self, f"no junk blocks (thresholds: {thr_text})")
            return
        junk_flat = junk.reshape(-1)[:total]
        flat_idx = np.flatnonzero(junk_flat)
        ranges: list[Hole] = []
        start = prev = int(flat_idx[0])
        for i in flat_idx[1:]:
            i = int(i)
            if i == prev + 1:
                prev = i
                continue
            ranges.append(Hole(start, prev - start + 1, "junk"))
            start = prev = i
        ranges.append(Hole(start, prev - start + 1, "junk"))
        ctx.holes.extend(ranges)
        ctx.holes.sort(key=lambda h: h.mcu_first)
        ctx.memo["junk_mcus"] = junk_flat.copy()
        ctx.memo["hole_dc"] = self.hole_colours(ctx, ft, sr, comps, per_row, trusted.reshape(-1)[:total] & ~junk_flat)
        if ctx.recovered_fraction is not None:
            ctx.recovered_fraction = max(0.0, ctx.recovered_fraction - n / total)
        ctx.note(self, f"{n} decoded MCU(s) in {len(ranges)} run(s) are junk (thresholds: {thr_text})"
                       + (f", {far} of them far from the thumbnail (another photo's data)" if far else "")
                       + ": they become holes", mcus=n, runs=len(ranges), thresholds={k: round(v, 2) for k, v in thr.items()})

    @staticmethod
    def hole_colours(ctx: Context, ft: FrameTables, sr, comps, per_row: int, trusted: np.ndarray) -> dict | None:
        """The DC of every block as the hole filling (R9) would have it with the junk
        among the holes: it is no source any more, neither for the thumbnail's
        calibration nor for the detail the real rows carry into the holes."""
        clean = ctx.memo.get("clean_mcus")
        clean = (clean if clean is not None and len(clean) == len(trusted) else trusted) & trusted
        qdc = {c.cid: ft.quant[ft.frame.component(c.cid).tq].dc for c in comps}
        thumb = ctx.memo.get("thumb_base")
        if thumb is not None:  # R9 found the thumbnail (or the reference picture) usable: calibrate it anew
            thumb = thumbnail_base(ctx.data, ft, sr, clean, ctx.options.get("thumb_refs"), reference_for(ctx, ft),
                                   twin_loader(ctx, ft)) or thumb
        try:
            return interpolation_sources(sr.dc, trusted, [(c.cid, c.h, c.v) for c in comps], qdc, thumb, per_row)
        except Exception:
            return None
