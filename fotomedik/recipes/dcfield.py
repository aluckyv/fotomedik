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


def build_field(arr: np.ndarray, okm: np.ndarray, q: int, base: np.ndarray | None = None,
                base_sigma: float | None = None) -> Field:
    sv = vertical_noise(arr, okm, q)
    if base is None:
        val, dist = vertical_reference(arr, okm)
        sigma = sv * (1.0 + 0.5 * np.maximum(0.0, dist - 1.0))
        return Field(val, dist, np.where(np.isfinite(dist), sigma, np.inf))
    resid = np.where(okm, arr - base, np.nan)
    rv, dist = vertical_reference(np.nan_to_num(resid), okm)
    w = np.where(np.isfinite(dist), np.exp(-np.maximum(0.0, dist - 1.0) / RESID_DECAY), 0.0)
    val = base + np.where(np.isfinite(dist), rv, 0.0) * w
    val = np.where(okm, arr, val)
    bs = max(base_sigma or sv, sv)
    sigma = bs + (sv - bs) * w
    return Field(val, np.minimum(dist, THUMB_DIST), sigma)


def _crop_black_bars(img: np.ndarray) -> np.ndarray:
    """Letterboxed thumbnails (16:9 photo in a 4:3 preview) carry black bars."""
    y = img[:, :, 0]
    rows = np.where(y.max(axis=1) > 24)[0]
    cols = np.where(y.max(axis=0) > 24)[0]
    if len(rows) < 8 or len(cols) < 8:
        return img
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def thumbnail_base(data: bytes, ft, sr: D.ScanResult, clean_mcus: np.ndarray) -> ThumbBase | None:
    """The embedded thumbnail mapped onto each component's block grid as DC values,
    calibrated against the clean blocks (dc = a * thumb + b). None when there is no
    thumbnail, its shape does not match the frame, or it does not agree with the clean
    part of the image."""
    from .r8_thumbnail import extract_thumbnail

    blob = extract_thumbnail(data)
    if blob is None:
        return None
    try:
        im = Image.open(io.BytesIO(blob))
        im.load()
        img = np.asarray(im.convert("YCbCr"))
    except Exception:
        return None
    img = _crop_black_bars(img)
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
        okm = block_mask(clean_mcus, arr, c.h, c.v, per_row)
        x, y = tdc[okm], arr[okm]
        if len(x) < THUMB_MIN_BLOCKS:
            # too little clean data to calibrate: cameras render the thumbnail from the
            # same picture (measured gain 0.98-1.1, offset < 4 gray levels), so use it as
            # is - unless whatever did decode contradicts it
            anyok = block_mask(sr.mcu_ok, arr, c.h, c.v, per_row)
            if ci == 0 and int(anyok.sum()) >= 50:
                xa, ya = tdc[anyok], arr[anyok]
                if np.std(xa) > 1e-6 and np.std(ya) > 1e-6 and np.corrcoef(xa, ya)[0, 1] < UNCAL_MIN_R:
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
    return ThumbBase(out, sig, r_luma, fit)
