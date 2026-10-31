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
TWIN_MIN_R = 0.5  # luma agreement a neighbouring shot needs (people move between shots)
TWIN_TILE_PX = 64  # the twin is checked against the decoded blocks per tile of this size
TWIN_TILE_K = 2.0  # a tile whose error exceeds this many times the typical one is dropped
TWIN_TILE_FLOOR = (6.0, 4.0)  # gray levels (luma, chroma) below which a tile always counts as agreeing
TWIN_TILE_MIN = 12  # decoded blocks a tile needs to be judged
TWIN_MAX_BAD = 0.5  # share of judged tiles that may disagree before the twin is not used at all
TWIN_NEAR = 4.0  # block rows: holes this close to real rows follow those rows, not the twin
TWIN_ECC = 0.6  # correlation the aligned block pictures must reach for the alignment to be used


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


MELT = 0.75  # blur (block columns, as sigma) per block row of distance from the real row


def _blur_rows(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of every row of `a` (edges repeat)."""
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, ((0, 0), (r, r)), mode="edge")
    out = np.zeros_like(a, dtype=float)
    for j, kj in enumerate(k):
        out += kj * pad[:, j : j + a.shape[1]]
    return out


def melt(resid: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """The detail a real row carries into a hole (`resid`, per column) blurred along the
    row the deeper it goes: sigma MELT * (distance - 1) block columns. Carried in as
    it is, a row of heads turns into vertical streaks that reach down the whole hole;
    melted, it fades into soft shapes like an out-of-focus picture would."""
    sigma = np.clip(MELT * (np.nan_to_num(dist, nan=0.0, posinf=0.0) - 1.0), 0.0, None)
    if not (sigma > 0.25).any():
        return resid
    levels = [1.0, 2.0, 4.0, 8.0, 16.0]
    blurred = [resid] + [_blur_rows(resid, s) for s in levels]
    bounds = [0.0] + levels
    out = resid.copy()
    for k in range(len(levels)):
        lo, hi = bounds[k], bounds[k + 1]
        sel = (sigma > lo) & (sigma <= hi) if k < len(levels) - 1 else sigma > lo
        if sel.any():  # between two levels: the mix that gives the wanted blur
            t = np.clip((sigma - lo) / (hi - lo), 0.0, 1.0)
            out = np.where(sel, blurred[k] * (1 - t) + blurred[k + 1] * t, out)
    return out


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
    twin: bool = False  # taken from a neighbouring shot of the same scene, not the thumbnail


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
    val = base + melt(np.where(np.isfinite(dist), rv, 0.0), dist) * w
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


def interpolation_sources(work_dc: dict, trusted: np.ndarray, comps, qdc: dict, thumb: ThumbBase | None,
                          per_row: int) -> dict:
    """Per component (cid -> block grid): the DC every block probably has, in quantised
    units within the legal range. Trusted blocks keep theirs; the others follow the
    thumbnail (or the reference picture) corrected by the real rows around them, and
    where that says nothing, the rows above and below - bridged with an all-sides fill
    the taller the gap, so thin gaps stay sharp and tall ones leave no vertical streaks.
    comps: (cid, h, v) per component; trusted: per MCU."""
    out = {}
    for cid, h, v in comps:
        arr = work_dc[cid]
        known = block_mask(trusted, arr, h, v, per_row)
        f = build_field(arr, known, qdc[cid], thumb.dc[cid] if thumb else None, thumb.sigma[cid] if thumb else None)
        if thumb is not None and thumb.twin:
            # thin holes between real rows are told best by those rows; a neighbouring shot
            # (another moment, another framing) only for what lies far from any
            plain = build_field(arr, known, qdc[cid])
            near = plain.dist <= TWIN_NEAR
            f = Field(np.where(near, plain.val, f.val), np.where(near, plain.dist, f.dist),
                      np.where(near, plain.sigma, f.sigma))
        lim = (1024 + qdc[cid]) // qdc[cid] - 1
        val = f.val
        guide = thumb.dc[cid] if thumb is not None else None
        unguided = ~np.isfinite(guide) if guide is not None else np.ones_like(val, dtype=bool)
        if unguided.any() and known.any():
            pp = pull_push(np.nan_to_num(arr), known)
            wv = np.where(np.isfinite(f.dist), np.exp(-np.maximum(0.0, f.dist - 2.0) / 2.0), 0.0)
            mixed = np.where(np.isfinite(val), wv * val + (1 - wv) * pp, pp)
            val = np.where(unguided & ~known, mixed, val)
        out[cid] = np.clip(np.nan_to_num(val, nan=0.0), -lim, lim)
    return out


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
            img = np.asarray(im.convert("YCbCr"))
        except Exception:
            img = None
        if img is not None:
            # a thumbnail in rotted sectors still opens - from the damage on in wrong colours
            real = _real_part(blob, img.shape[:2])
            if real is None:
                return img, None
            if float(real.mean()) >= RESCUE_MIN:
                return img, real
    if not thumb_refs:
        return None
    if landscape is not None:
        # a thumbnail header of the other orientation would decode the scan sheared
        thumb_refs = [r for r in thumb_refs if (r.width >= r.height) == landscape] or thumb_refs
    return _rescue_thumbnail(data, thumb_refs)


def _real_part(blob: bytes, shape) -> np.ndarray | None:
    """Mask (float, 1 = real) of the thumbnail's pixels from data that decodes cleanly, or
    None when all of it does. The rows just before a decoding error are suspect too."""
    from ..jpeg import markers as M
    from ..jpeg.tables import FrameTables

    try:
        res = M.parse_segments(blob)
        ft = FrameTables.from_parse(res)
        sr = D.decode_scan(blob, res, ft)
    except Exception:  # noqa: BLE001 - PIL opened it; nothing better to say
        return None
    if sr.error is None and all(iv.ok for iv in sr.intervals):
        return None
    _comps, per_row, _rows = D._scan_components(ft)
    ok = sr.mcu_ok.copy()
    for iv in sr.intervals:
        if not iv.ok:
            end = iv.mcu_first + iv.mcu_count
            ok[max(iv.mcu_first, end - 2 * per_row) : end] = False
            if iv.error is not None:
                e = iv.error.mcu_index
                ok[max(iv.mcu_first, e - 2 * per_row) : end] = False
    mw, mh = 8 * ft.frame.hmax, 8 * ft.frame.vmax
    real = np.zeros(shape, dtype=np.float32)
    for m in np.flatnonzero(ok):
        y, x = divmod(int(m), per_row)
        real[y * mh : (y + 1) * mh, x * mw : (x + 1) * mw] = 1.0
    return real


class RescuedMask(np.ndarray):
    """The mask of a repaired thumbnail whose colours cannot be trusted: nothing of its
    scan decoded before the damage, so the colour level of every repaired part is a
    guess (a whole picture can come out green). Its brightness still counts."""

    def __new__(cls, mask):
        return np.asarray(mask).view(cls)


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
    img, mask, colour_ok = best[1], best[2], best[3]
    if not colour_ok:
        mask = RescuedMask(mask)
    return img, mask


def _rescue_from(head: bytes, start: int, ref):
    """(real share, YCbCr pixels, mask of real pixels, colours known) of the thumbnail whose
    scan starts at `start`, decoded with the reference thumbnail header; None when too
    little is real."""
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
        # the repaired parts take their colour level from the rows decoded before the damage;
        # with none of those (the scan rotted from its first row) every level is a guess
        first_hole = min((h.mcu_first for h in ctx.holes), default=per_row)
        return frac, np.asarray(im.convert("YCbCr")), mask, first_hole >= per_row


def thumbnail_base(data: bytes, ft, sr: D.ScanResult, clean_mcus: np.ndarray,
                   thumb_refs: list | None = None, reference: list | None = None,
                   twin=None) -> ThumbBase | None:
    """The embedded thumbnail mapped onto each component's block grid as DC values,
    calibrated against the clean blocks (dc = a * thumb + b). None when there is no
    thumbnail, its shape does not match the frame, or it does not agree with the clean
    part of the image.

    reference: full-size pictures of the same scene (YCbCr arrays, see fotomedik.reference)
    that stand in for the thumbnail; of several (the two turns of a copy lying the other
    way round) the one agreeing best with the clean blocks is used.

    twin: a callable giving (YCbCr pixels, mask of real pixels) of a neighbouring shot of
    the same scene, asked for only when the photo's own thumbnail is of no use. It says
    what colour and brightness the holes have, nothing finer: it is aligned to the photo,
    and every part of it that disagrees with what decoded is left out (people move)."""
    if reference:
        bases = [_base_from(img, None, ft, sr, clean_mcus, GAIN_WIDE) for img in reference]
        bases = [b for b in bases if b is not None]
        return max(bases, key=lambda b: b.r) if bases else None
    got = thumbnail_image(data, thumb_refs, landscape=ft.frame.width >= ft.frame.height)
    if got is not None:
        img, mask = got
        img, mask = _crop_black_bars(img, mask)
        base = _base_from(img, mask, ft, sr, clean_mcus, GAIN_THUMB)
        if base is not None and not base.rescued:
            return base
    else:
        base = None
    if twin is not None:
        t = twin()
        if t is not None:
            tb = twin_base(t[0], t[1], ft, sr, clean_mcus)
            if tb is not None and (base is None or tb.r > base.r):
                return tb
    return base


def twin_base(img: np.ndarray, real: np.ndarray, ft, sr: D.ScanResult, clean_mcus: np.ndarray) -> ThumbBase | None:
    """A neighbouring shot (YCbCr pixels of the frame's aspect, mask of its real pixels)
    as the DC reference of this photo: aligned to the decoded blocks, calibrated like a
    thumbnail, and cut back to the tiles where it agrees with them. Without enough clean
    blocks (no restart markers: nothing decodes cleanly end to end) everything that
    decoded is what it is checked against."""
    if int(clean_mcus.sum()) < THUMB_MIN_BLOCKS:
        clean_mcus = sr.mcu_ok
    img, real = _align_twin(img, real, ft, sr, clean_mcus)
    base = _base_from(img, real, ft, sr, clean_mcus, GAIN_WIDE, min_r=TWIN_MIN_R)
    if base is None:
        return None
    base = _guard_twin(base, ft, sr, clean_mcus)
    if base is not None:
        base.twin = True
        base.rescued = False
    return base


def _luma_blocks(ft, sr: D.ScanResult, clean_mcus: np.ndarray):
    """(luma DC in gray levels, mask of clean blocks) of the frame's real area."""
    comps, per_row, _rows = D._scan_components(ft)
    c = comps[0]
    arr = sr.dc[c.cid]
    q = ft.quant[ft.frame.component(c.cid).tq].dc
    okm = block_mask(clean_mcus, arr, c.h, c.v, per_row) & np.isfinite(arr)
    frame = ft.frame
    comp_w = -(-frame.width * c.h // frame.hmax)
    comp_h = -(-frame.height * c.v // frame.vmax)
    bw, bh = -(-comp_w // 8), -(-comp_h // 8)
    return (np.nan_to_num(arr) * q / 8.0 + 128.0)[:bh, :bw], okm[:bh, :bw]


def _align_twin(img: np.ndarray, real: np.ndarray, ft, sr: D.ScanResult, clean_mcus: np.ndarray):
    """The twin warped onto this photo's frame (a homography found on the luma block
    pictures, only where this photo decoded), or as it is when no alignment holds."""
    try:
        import cv2
    except ImportError:
        return img, real
    luma, okm = _luma_blocks(ft, sr, clean_mcus)
    if okm.sum() < THUMB_MIN_BLOCKS:
        return img, real
    bh, bw = luma.shape
    th, tw = img.shape[:2]
    small = np.asarray(Image.fromarray(np.ascontiguousarray(img[:, :, 0])).resize((bw, bh), Image.BILINEAR), np.float32)
    gt = cv2.GaussianBlur(luma.astype(np.float32), (0, 0), 1.5)
    gd = cv2.GaussianBlur(small, (0, 0), 1.5)
    warp = np.eye(3, dtype=np.float32)
    try:
        rho, warp = cv2.findTransformECC(gd, gt, warp, cv2.MOTION_HOMOGRAPHY,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
                                         okm.astype(np.uint8), 5)
    except cv2.error:
        return img, real
    if rho < TWIN_ECC:
        return img, real
    # the warp maps block coordinates of the twin's small copy to those of the photo (the
    # masked input): carry it to the twin's own pixels
    s_b = np.diag([bw / tw, bh / th, 1.0])
    h = np.linalg.inv(s_b) @ warp.astype(np.float64) @ s_b  # twin pixels -> photo (in twin pixel units)
    flags = cv2.INTER_LINEAR
    out = cv2.warpPerspective(np.ascontiguousarray(img), h, (tw, th), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    m = real.astype(np.float32) if real is not None else np.ones((th, tw), np.float32)
    mw = cv2.warpPerspective(m, h, (tw, th), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, mw


def _guard_twin(base: ThumbBase, ft, sr: D.ScanResult, clean_mcus: np.ndarray) -> ThumbBase | None:
    """Leave out every tile where the twin disagrees with the decoded blocks (in any
    component): there the scene changed between the shots. None when most of it does."""
    frame = ft.frame
    comps, per_row, _rows = D._scan_components(ft)
    ty, tx = -(-frame.height // TWIN_TILE_PX), -(-frame.width // TWIN_TILE_PX)
    bad = np.zeros(ty * tx, dtype=bool)
    judged = np.zeros(ty * tx, dtype=bool)
    ids = {}
    for ci, c in enumerate(comps):
        arr, dc = sr.dc[c.cid], base.dc[c.cid]
        rows, cols = arr.shape
        py = 8.0 * frame.vmax / c.v  # pixels per block
        px = 8.0 * frame.hmax / c.h
        ri = np.clip((np.arange(rows) * py // TWIN_TILE_PX).astype(int), 0, ty - 1)
        cj = np.clip((np.arange(cols) * px // TWIN_TILE_PX).astype(int), 0, tx - 1)
        tid = ri[:, None] * tx + cj[None, :]
        ids[c.cid] = tid
        okm = block_mask(clean_mcus, arr, c.h, c.v, per_row) & np.isfinite(dc) & np.isfinite(arr)
        if not okm.any():
            continue
        q = ft.quant[frame.component(c.cid).tq].dc
        res = np.abs(arr - dc)[okm] * q / 8.0
        t = tid[okm]
        lim = max(TWIN_TILE_K * float(np.median(res)), TWIN_TILE_FLOOR[min(ci, 1)])
        n = np.bincount(t, minlength=ty * tx)
        order = np.argsort(t, kind="stable")
        starts = np.concatenate(([0], np.cumsum(n)[:-1]))
        rs = res[order]
        for k in np.flatnonzero(n >= TWIN_TILE_MIN):
            judged[k] = True
            if np.median(rs[starts[k] : starts[k] + n[k]]) > lim:
                bad[k] = True
    if judged.any() and bad[judged].mean() > TWIN_MAX_BAD:
        return None
    if not judged.any():
        return None
    for cid, tid in ids.items():
        base.dc[cid] = np.where(bad[tid], np.nan, base.dc[cid])
    return base


GAIN_THUMB = (0.7, 1.4)  # calibration gain allowed for a camera thumbnail
GAIN_WIDE = (0.5, 2.0)  # for a reference picture (edited copies are brightened or darkened)


FLAT_STD = 0.5  # DC units: below this a stretch is flat and its pattern cannot be compared
FLAT_LEVEL = 24.0
CLOSE_LEVEL = 12.0  # gray levels: blocks this close to the thumbnail (median) agree with it  # gray levels a flat stretch may differ from the thumbnail's level
AGREE_FLOOR = 10.0  # gray levels of luma a decoded MCU may stray from the thumbnail before it counts as garbage
AGREE_K = 4.0  # ... or this many times the typical stray of the run's first part


STEP_T = 8.0  # t statistic a step in the residual needs
STEP_MIN = 3.0  # gray levels it must shift the residual by


def _step(r: np.ndarray, per_row: int, with_t: bool = False, min_t: float = STEP_T, min_step: float = STEP_MIN):
    """Index of the strongest step in the mean of `r` (at least a row on each side), when
    it is clear (t statistic of the two means) and large enough; None otherwise.
    with_t: (index, t) instead."""
    n = len(r)
    if n < 3 * per_row:
        return None
    cs = np.cumsum(r)
    k = np.arange(per_row, n - per_row)
    m1, m2 = cs[k - 1] / k, (cs[-1] - cs[k - 1]) / (n - k)
    t = np.abs(m1 - m2) * np.sqrt(k * (n - k) / n) / (float(np.std(r)) + 1e-6)
    i = int(np.argmax(t))
    if not (t[i] > min_t and abs(m1[i] - m2[i]) > min_step):
        return None
    return (int(k[i]), float(t[i])) if with_t else int(k[i])


def agreeing_mcus(tdc: np.ndarray, arr: np.ndarray, mcu_ok: np.ndarray, h: int, v: int, per_row: int,
                  q: float) -> np.ndarray:
    """`mcu_ok` without the decoded garbage at the end of each run of decoded MCUs.

    A damaged stretch is often noticed only hundreds of MCUs after it begins: until
    then the decoder happily turns the wrong bits into blocks of wrong brightness. The
    (uncalibrated) thumbnail luma `tdc` shows where a run stops agreeing with the photo
    for good; from there to the end of the run nothing is trusted."""
    rows, _cols = arr.shape
    n = min(len(mcu_ok), (rows // v) * per_row)
    out = mcu_ok.copy()
    diff = (arr - tdc)[: (rows // v) * v, : per_row * h] * q / 8.0
    known = block_mask(mcu_ok, arr, h, v, per_row)[: (rows // v) * v, : per_row * h] & np.isfinite(diff)
    if not known.any():
        return out
    off = float(np.median(diff[known]))  # the thumbnail's overall brightness offset
    dev = np.where(known, np.abs(diff - off), 0.0).reshape(rows // v, v, per_row, h).sum(axis=(1, 3))
    sgn = np.where(known, diff - off, 0.0).reshape(rows // v, v, per_row, h).sum(axis=(1, 3))
    cnt = known.reshape(rows // v, v, per_row, h).sum(axis=(1, 3))
    per_mcu = (dev / np.maximum(cnt, 1)).ravel()[:n]
    signed = (sgn / np.maximum(cnt, 1)).ravel()[:n]
    ok = mcu_ok[:n]
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= 2 * per_row:
            sm = np.convolve(per_mcu[i:j], np.ones(per_row) / per_row, mode="same")  # over one MCU row
            limit = max(AGREE_FLOOR, AGREE_K * float(np.median(sm[: max(per_row, (j - i) // 4)])))
            bad = sm > limit
            # the first point from which the rest of the run is mostly off
            rest_bad = np.cumsum(bad[::-1])[::-1] / np.arange(j - i, 0, -1)
            cut = np.flatnonzero(bad & (rest_bad > 0.8))
            at = int(cut[0]) if len(cut) else None
            # a step in the signed residual: from a lost or wrong bit on, the brightness is off
            # by a constant and the blocks sit a few places off - smaller than the floor above
            st = _step(signed[i:j], per_row)
            if st is not None and (at is None or st < at):
                at = st
            if at is not None:
                out[i + at : j] = False
        i = j
    return out


def _base_from(img: np.ndarray, mask, ft, sr: D.ScanResult, clean_mcus: np.ndarray, gain_range,
               min_r: float = THUMB_MIN_R, _trim: bool = True) -> ThumbBase | None:
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
            # judged by what decoded before the first error: everything after it may be
            # garbage that decoded (dense rot leaves little else), and garbage contradicts
            # any thumbnail
            first_bad = np.flatnonzero(~sr.mcu_ok)
            prefix = sr.mcu_ok.copy()
            if len(first_bad):
                prefix[int(first_bad[0]) :] = False
            anyok = block_mask(prefix, arr, c.h, c.v, per_row) & np.isfinite(tdc)
            if int(anyok.sum()) < 50 and mask is not None:
                anyok = block_mask(sr.mcu_ok, arr, c.h, c.v, per_row) & np.isfinite(tdc)
            need_r = RESCUED_MIN_R if mask is not None else UNCAL_MIN_R

            def contradicts(xa, ya) -> bool:
                if float(np.median(np.abs(xa - ya))) * q / 8.0 <= CLOSE_LEVEL:
                    return False  # close block by block: a low-contrast stretch whose pattern is noise
                if np.std(xa) < FLAT_STD or np.std(ya) < FLAT_STD:
                    # a flat stretch (sky, a wall) says nothing about the pattern, only the level
                    return abs(float(np.mean(xa) - np.mean(ya))) * q / 8.0 > FLAT_LEVEL
                return float(np.corrcoef(xa, ya)[0, 1]) < need_r

            if ci == 0 and int(anyok.sum()) >= 50 and contradicts(tdc[anyok], arr[anyok]):
                # perhaps only the garbage decoded before an error disagrees
                agree = agreeing_mcus(tdc, arr, prefix, c.h, c.v, per_row, q)
                anyok = block_mask(agree, arr, c.h, c.v, per_row) & np.isfinite(tdc)
                # what remains must not contradict it; when nothing remains, rot began at the
                # top and nothing decoded can speak against the camera's own thumbnail
                if int(anyok.sum()) >= 50 and contradicts(tdc[anyok], arr[anyok]):
                    return None
            if ci > 0 and isinstance(mask, RescuedMask):
                # a repaired thumbnail with no clean rows: its colours are guesses and nothing here
                # can check them, so they say nothing; the picture's own rows give the colour
                tdc = np.full_like(tdc, np.nan)
            out[c.cid] = tdc
            sig[c.cid] = UNCAL_SIGMA[min(ci, 1)]
            fit[c.cid] = (1.0, 0.0)
            continue
        sx = float(np.std(x))
        if sx > 1e-6 and float(np.std(y)) > 1e-6:
            r = float(np.corrcoef(x, y)[0, 1])
            a = float(np.clip(np.cov(x, y)[0, 1] / (sx * sx), *gain_range))
        else:
            r, a = 0.0, 1.0
        if ci == 0:
            if r < min_r:
                if _trim:  # perhaps the "clean" part holds garbage decoded before an error
                    agree = agreeing_mcus(tdc, arr, clean_mcus, c.h, c.v, per_row, q)
                    if int(agree.sum()) < int(clean_mcus.sum()):
                        return _base_from(img, mask, ft, sr, agree, gain_range, min_r, _trim=False)
                return None
            r_luma = r
        b = float(np.median(y - a * x))
        out[c.cid] = a * tdc + b
        fit[c.cid] = (round(a, 3), round(b * q / 8.0, 1))
        sig[c.cid] = max(1.0, float(np.median(np.abs(y - out[c.cid][okm]))) * q / 8.0)
    return ThumbBase(out, sig, r_luma, fit, mask is not None)
