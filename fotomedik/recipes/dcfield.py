"""DC references for damaged regions: what the 8x8 block averages there probably are.

Two sources:
* the nearest clean rows above and below (linear interpolation between them);
* the embedded Exif thumbnail - a small picture of the whole frame, at 160x120
  about one pixel per one or two MCUs. It knows the coarse picture (colour,
  brightness, large shapes) even where nothing nearby survived.

With a usable thumbnail the reference is the thumbnail's DC plus its error at the
nearest clean rows, interpolated and fading with distance: clean rows add local
accuracy, the thumbnail keeps far-away holes and tails right. Every reference
carries its expected error (gray levels), so anomalies can be judged fairly
whether they were measured next to a clean row or only against the thumbnail.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..jpeg import decoder as D

THUMB_MIN_R = 0.8  # luma correlation with the clean blocks needed to trust the thumbnail
THUMB_MIN_BLOCKS = 200
UNCAL_MIN_R = 0.5  # agreement needed with whatever decoded when there is too little clean data
UNCAL_SIGMA = (6.0, 3.0)  # expected error (gray levels, luma/chroma) of an uncalibrated thumbnail
RESID_DECAY = 4.0  # block rows over which a clean row's correction to the thumbnail fades
FADE_ROWS = 6.0  # block rows over which a one-sided fill fades to the average colour
RESCUED_MIN_R = 0.6  # a repaired thumbnail must agree this well with what decoded
THUMB_DIST = 3.0  # block rows a thumbnail-only reference counts as (for ordering by reliability)


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


def pull_push(arr: np.ndarray, known: np.ndarray) -> np.ndarray:
    """`arr` with the unknown blocks filled smoothly from all known ones around them,
    in every direction (a pyramid: averages are pulled down to coarse levels, then
    pushed back up with bilinear interpolation). Unlike interpolating each column
    between the rows above and below, it leaves no vertical streaks."""
    v = np.where(known, arr, 0.0).astype(float)
    w = known.astype(float)
    levels = [(v, w)]
    while min(v.shape) > 2:
        h, wd = v.shape
        ph, pw = h % 2, wd % 2
        v2 = np.pad(v, ((0, ph), (0, pw)), mode="constant")
        w2 = np.pad(w, ((0, ph), (0, pw)), mode="constant")
        v = v2.reshape(v2.shape[0] // 2, 2, v2.shape[1] // 2, 2).sum(axis=(1, 3))
        w = w2.reshape(w2.shape[0] // 2, 2, w2.shape[1] // 2, 2).sum(axis=(1, 3))
        levels.append((v, w))
    v, w = levels[-1]
    filled = np.where(w > 0, v / np.maximum(w, 1e-9), float(np.mean(arr[known])) if known.any() else 0.0)
    for v, w in reversed(levels[:-1]):
        h, wd = v.shape
        up = np.asarray(Image.fromarray(filled.astype(np.float32), mode="F").resize((wd, h), Image.BILINEAR))
        a = np.clip(w, 0.0, 1.0)  # fully trusted where this level has data
        filled = np.where(w > 0, (v / np.maximum(w, 1e-9)) * a + up * (1 - a), up)
    return np.where(known, arr, filled)


def vertical_noise(arr: np.ndarray, okm: np.ndarray, q: int) -> float:
    """Typical gray-level error of predicting a clean block from the blocks above and below."""
    m = okm[1:-1] & okm[:-2] & okm[2:]
    if not m.any():
        return 2.0
    err = np.abs(arr[1:-1] - (arr[:-2] + arr[2:]) / 2.0)[m]
    return max(1.0, float(np.median(err)) * q / 8.0)


@dataclass
class Field:
    val: np.ndarray  # reference DC (quantised units) per block, NaN where unknown
    dist: np.ndarray  # block rows to the nearest clean row (capped at THUMB_DIST with a thumbnail)
    sigma: np.ndarray  # expected |error| of val, gray levels


@dataclass
class ThumbBase:
    dc: dict  # cid -> thumbnail DC on the component's block grid (calibrated)
    sigma: dict  # cid -> typical |error| of it vs clean blocks, gray levels
    r: float  # luma correlation with the clean blocks
    fit: dict  # cid -> (gain, offset in gray levels) of the calibration
    rescued: bool = False  # the thumbnail was damaged too and has been repaired first


def build_field(arr: np.ndarray, okm: np.ndarray, q: int, base: np.ndarray | None = None,
                base_sigma: float | None = None) -> Field:
    sv = vertical_noise(arr, okm, q)
    if base is None:
        val, dist = vertical_reference(arr, okm)
        # Known on one side only (e.g. everything below the last real row is lost): a
        # copied row turns into vertical streaks, so fade towards the picture's average
        # colour with the distance instead.
        rows = arr.shape[0]
        ridx = np.arange(rows)[:, None]
        has_above = np.maximum.accumulate(np.where(okm, ridx, -1), axis=0) >= 0
        has_below = (np.minimum.accumulate(np.where(okm, ridx, rows)[::-1], axis=0)[::-1]) < rows
        one_side = ~okm & (has_above ^ has_below)
        if one_side.any() and okm.any():
            mean = float(np.mean(arr[okm]))
            w = np.exp(-np.maximum(0.0, dist - 1.0) / FADE_ROWS)
            val = np.where(one_side, mean + (val - mean) * w, val)
        sigma = sv * (1.0 + 0.5 * np.maximum(0.0, dist - 1.0))
        return Field(val, dist, np.where(np.isfinite(dist), sigma, np.inf))
    known = np.isfinite(base)
    resid = np.where(okm & known, arr - base, np.nan)
    rv, dist = vertical_reference(np.nan_to_num(resid), okm & known)
    w = np.where(np.isfinite(dist), np.exp(-np.maximum(0.0, dist - 1.0) / RESID_DECAY), 0.0)
    val = base + np.where(np.isfinite(dist), rv, 0.0) * w
    val = np.where(okm, arr, val)
    bs = max(base_sigma or sv, sv)
    sigma = bs + (sv - bs) * w
    dist = np.minimum(dist, THUMB_DIST)
    if not known.all():
        # where the (rescued) thumbnail has no data, only the clean rows can tell
        plain = build_field(arr, okm, q)
        val = np.where(known, val, plain.val)
        dist = np.where(known, dist, plain.dist)
        sigma = np.where(known, sigma, plain.sigma)
    return Field(val, dist, sigma)


def _crop_black_bars(img: np.ndarray, mask: np.ndarray | None = None):
    """Letterboxed thumbnails (16:9 photo in a 4:3 preview) carry black bars."""
    y = img[:, :, 0]
    rows = np.where(y.max(axis=1) > 24)[0]
    cols = np.where(y.max(axis=0) > 24)[0]
    if len(rows) < 8 or len(cols) < 8 or mask is not None:
        return img, mask
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1], None


RESCUE_MIN = 0.25  # share of a damaged thumbnail that must come back to be of use


def thumbnail_image(data: bytes, thumb_refs: list | None = None, landscape: bool | None = None):
    """(YCbCr pixels, mask of real pixels or None when all are) of the embedded
    thumbnail, or None. A thumbnail that is damaged itself - it sits in the same rotted
    sectors as the header - is repaired first: its header comes from a healthy
    thumbnail of the set (thumb_refs), its scan goes through tail recovery and hole
    filling, and only its real parts count."""
    from .r8_thumbnail import extract_thumbnail

    blob = extract_thumbnail(data)
    if blob is not None:
        try:
            im = Image.open(io.BytesIO(blob))
            im.load()
            return np.asarray(im.convert("YCbCr")), None
        except Exception:
            pass
    if not thumb_refs:
        return None
    if landscape is not None:
        # a thumbnail header of the other orientation would decode the scan sheared
        thumb_refs = [r for r in thumb_refs if (r.width >= r.height) == landscape] or thumb_refs
    return _rescue_thumbnail(data, thumb_refs)


def _rescue_thumbnail(data: bytes, thumb_refs: list):
    from .r7_reference import CLOSE_SOS, _POPCOUNT

    head = data[: min(len(data), 1 << 16)]
    arr = np.frombuffer(head, dtype=np.uint8)
    best = None
    for ref in thumb_refs[:4]:
        sos = ref.sos_bytes
        L = len(sos)
        if len(arr) <= L:
            continue
        dist = np.zeros(len(arr) - L + 1, dtype=np.int32)
        for j in range(L):
            dist += _POPCOUNT[np.bitwise_xor(arr[j : len(arr) - L + 1 + j], sos[j])]
        close = np.flatnonzero(dist <= CLOSE_SOS * 8 * L)
        starts = [int(close[0]) + L] if len(close) else []  # the thumbnail's SOS comes first
        # its SOS may be rotted too: thumbnails of one camera share their layout, so the
        # scan starts a fixed distance after the thumbnail's SOI (FF D8 FF, a bit may be off)
        if ref.scan_offset:
            sig = np.frombuffer(b"\xff\xd8\xff", dtype=np.uint8)
            dsig = sum(_POPCOUNT[np.bitwise_xor(arr[j : len(arr) - 2 + j], sig[j])] for j in range(3))
            for p0 in np.flatnonzero(dsig <= 2)[:8]:
                if 64 <= p0 and p0 + ref.scan_offset < len(head) and int(p0) + ref.scan_offset not in starts:
                    starts.append(int(p0) + ref.scan_offset)
        for start in starts:
            cand = _rescue_from(head, start, ref)
            if cand is not None and (best is None or cand[0] > best[0]):
                best = cand
    if best is None:
        return None
    return best[1], best[2]


def _rescue_from(head: bytes, start: int, ref):
    """(real share, YCbCr pixels, mask of real pixels) of the thumbnail whose scan starts
    at `start`, decoded with the reference thumbnail header; None when too little is real."""
    from ..jpeg import markers as M
    from .base import Context
    from .r9_holes import FillHoles
    from .r10_tails import TailRecovery
    from .registry import run_pipeline

    if True:
        end = head.find(b"\xff\xd9", start)
        end = end + 2 if end > 0 else len(head)
        blob = ref.header_bytes(ref.width, ref.height, []) + head[start:end]
        if not blob.endswith(b"\xff\xd9"):
            blob += b"\xff\xd9"
        ctx = Context(data=blob, name="thumbnail", options={"time_budget": 0.5})
        try:
            ctx = run_pipeline(ctx, recipes=[TailRecovery(), FillHoles()])
            im = Image.open(io.BytesIO(ctx.data))
            im.load()
        except Exception:
            return None
        if ctx.recovered_fraction is not None:
            frac = ctx.recovered_fraction
        else:  # nothing needed repair: the scan decodes as it is
            from ..scan.scanner import scan_bytes

            frac = scan_bytes(ctx.data).recovered_fraction
        if frac < RESCUE_MIN:
            return None
        res = M.parse_segments(ctx.data)
        from ..jpeg.tables import FrameTables

        frame = FrameTables.from_parse(res).frame
        mw, mh = frame.mcu_width, frame.mcu_height
        per_row = -(-frame.width // mw)
        mask = np.ones((im.size[1], im.size[0]), dtype=bool)
        for hole in ctx.holes:
            for mi in range(hole.mcu_first, hole.mcu_first + hole.mcu_count):
                r, c = divmod(mi, per_row)
                mask[r * mh : (r + 1) * mh, c * mw : (c + 1) * mw] = False
        return frac, np.asarray(im.convert("YCbCr")), mask


def thumbnail_base(data: bytes, ft, sr: D.ScanResult, clean_mcus: np.ndarray,
                   thumb_refs: list | None = None) -> ThumbBase | None:
    """The embedded thumbnail mapped onto each component's block grid as DC values,
    calibrated against the clean blocks (dc = a * thumb + b). None when there is no
    thumbnail, its shape does not match the frame, or it does not agree with the clean
    part of the image."""
    got = thumbnail_image(data, thumb_refs, landscape=ft.frame.width >= ft.frame.height)
    if got is None:
        return None
    img, mask = got
    img, mask = _crop_black_bars(img, mask)
    frame = ft.frame
    th, tw = img.shape[:2]
    if abs((tw / th) / (frame.width / frame.height) - 1.0) > 0.03:
        return None
    comps, per_row, _rows = D._scan_components(ft)
    hmax, vmax = frame.hmax, frame.vmax
    out: dict = {}
    sig: dict = {}
    fit: dict = {}
    r_luma = 0.0
    for ci, c in enumerate(comps):
        arr = sr.dc[c.cid]
        rows, cols = arr.shape
        # blocks covering the real image area of this component
        comp_w = -(-frame.width * c.h // hmax)  # ceil: component size in samples
        comp_h = -(-frame.height * c.v // vmax)
        bw, bh = -(-comp_w // 8), -(-comp_h // 8)
        ch = min(ci, 2) if len(comps) == 3 else 0
        plane = Image.fromarray(np.ascontiguousarray(img[:, :, ch]))
        up = np.asarray(plane.resize((max(1, bw), max(1, bh)), Image.BILINEAR), dtype=float)
        grid = np.empty((rows, cols))
        grid[: min(rows, bh), : min(cols, bw)] = up[: min(rows, bh), : min(cols, bw)]
        if bh < rows:
            grid[bh:, :] = grid[bh - 1 : bh, :]
        if bw < cols:
            grid[:, bw:] = grid[:, bw - 1 : bw]
        q = ft.quant[frame.component(c.cid).tq].dc
        tdc = (grid - 128.0) * 8.0 / q
        if mask is not None:
            mplane = Image.fromarray((mask * 255).astype(np.uint8))
            mup = np.asarray(mplane.resize((max(1, bw), max(1, bh)), Image.BILINEAR), dtype=float) / 255.0
            mgrid = np.zeros((rows, cols))
            mgrid[: min(rows, bh), : min(cols, bw)] = mup[: min(rows, bh), : min(cols, bw)]
            if bh < rows:
                mgrid[bh:, :] = mgrid[bh - 1 : bh, :]
            if bw < cols:
                mgrid[:, bw:] = mgrid[:, bw - 1 : bw]
            tdc = np.where(mgrid > 0.99, tdc, np.nan)  # only where the thumbnail itself is real
        okm = block_mask(clean_mcus, arr, c.h, c.v, per_row) & np.isfinite(tdc)
        x, y = tdc[okm], arr[okm]
        if len(x) < THUMB_MIN_BLOCKS:
            # too little clean data to calibrate: cameras render the thumbnail from the
            # same picture (measured gain 0.98-1.1, offset < 4 gray levels), so use it as
            # is - unless whatever did decode contradicts it
            anyok = block_mask(sr.mcu_ok, arr, c.h, c.v, per_row) & np.isfinite(tdc)
            need_r = RESCUED_MIN_R if mask is not None else UNCAL_MIN_R
            if ci == 0 and int(anyok.sum()) >= 50:
                xa, ya = tdc[anyok], arr[anyok]
                if np.std(xa) < 1e-6 or np.std(ya) < 1e-6 or np.corrcoef(xa, ya)[0, 1] < need_r:
                    return None
            out[c.cid] = tdc
            sig[c.cid] = UNCAL_SIGMA[min(ci, 1)]
            fit[c.cid] = (1.0, 0.0)
            continue
        sx = float(np.std(x))
        if sx > 1e-6 and float(np.std(y)) > 1e-6:
            r = float(np.corrcoef(x, y)[0, 1])
            a = float(np.clip(np.cov(x, y)[0, 1] / (sx * sx), 0.7, 1.4))
        else:
            r, a = 0.0, 1.0
        if ci == 0:
            if r < THUMB_MIN_R:
                return None
            r_luma = r
        b = float(np.median(y - a * x))
        out[c.cid] = a * tdc + b
        fit[c.cid] = (round(a, 3), round(b * q / 8.0, 1))
        sig[c.cid] = max(1.0, float(np.median(np.abs(y - out[c.cid][okm]))) * q / 8.0)
    return ThumbBase(out, sig, r_luma, fit, mask is not None)
