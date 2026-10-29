"""Fill holes with real pixels from a neighbouring shot of the same scene.

People often take the same picture two or three times within a minute. Where
one copy has a hole, another copy may have the real thing. The donor is aligned
to the photo with a homography (ORB features + RANSAC on the parts both copies
really have), its colours are matched, and a hole block takes the donor's pixels
only when the donor agrees with what is known about that block (its average from
the thumbnail) - a person who moved between the shots is not pasted in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

MAX_GAP_S = 180  # shots further apart in time are not considered the same scene
WORK = 1024  # longest side of the images features are matched on
MIN_INLIERS = 40
MIN_INLIER_SHARE = 0.3
LUMA_TOL = 14.0  # gray levels a donor block may differ from the block's known average
CHROMA_TOL = 10.0
MAX_RESIDUAL = 14.0  # median |difference| on shared real pixels for an alignment to count
MIN_ISLAND = 3  # accepted blocks in a group smaller than this are dropped


@dataclass
class Shot:
    """One repaired photo of the set: pixels, what is real, when it was taken."""

    rgb: np.ndarray  # uint8 (h, w, 3)
    real: np.ndarray  # bool (h, w): decoded from real data (not a hole)
    known: np.ndarray  # bool (h, w): real, or a hole whose average is known (thumbnail)
    when: datetime | None
    name: str = ""


_STAMP = re.compile(r"(\d{4})[-_.:]?(\d{2})[-_.:]?(\d{2})[ _T-]?(\d{2})[._:\-]?(\d{2})[._:\-]?(\d{2})")


def taken_at(path: Path, data: bytes | None = None) -> datetime | None:
    """When the photo was taken: EXIF DateTimeOriginal, else a time stamp in the name."""
    try:
        im = Image.open(path) if data is None else Image.open(__import__("io").BytesIO(data))
        exif = im.getexif()
        sub = exif.get_ifd(0x8769)
        s = sub.get(0x9003) or exif.get(0x0132)
        if s:
            return datetime.strptime(str(s).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    m = _STAMP.search(path.name)
    if m:
        try:
            return datetime(*map(int, m.groups()))
        except ValueError:
            return None
    return None


def candidates(target: Shot, shots: list[Shot]) -> list[Shot]:
    """Shots taken close to the target, nearest first."""
    if target.when is None:
        return []
    out = []
    for s in shots:
        if s is target or s.when is None:
            continue
        gap = abs((s.when - target.when).total_seconds())
        if gap <= MAX_GAP_S and s.real.mean() > 0.05:
            out.append((gap, s))
    return [s for _g, s in sorted(out, key=lambda t: t[0])]


def _small(a: np.ndarray, scale: float, nearest: bool = False) -> np.ndarray:
    import cv2

    h, w = a.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(a.astype(np.uint8), size, interpolation=interp)


def align(target: Shot, donor: Shot) -> np.ndarray | None:
    """3x3 homography mapping donor pixel coordinates to target ones, or None.
    Features on the parts both copies really have; when there are too few of them
    (most of the target is a hole), the whole frames are matched at low resolution,
    the target's holes standing in with their thumbnail-based content."""
    h = align_features(target, donor)
    return h if h is not None else align_frames(target, donor)


ECC_SIZES = (192, 384, 768)
MIN_ECC = 0.75  # correlation the aligned low-resolution frames must reach


def align_frames(target: Shot, donor: Shot) -> np.ndarray | None:
    import cv2

    ht, wt = target.rgb.shape[:2]
    hd, wd = donor.rgb.shape[:2]
    if abs(ht / wt - hd / wd) > 0.05:  # portrait vs landscape: not the same framing
        return None
    warp = np.eye(3, dtype=np.float32)
    prev = None
    rho = 0.0
    for size in ECC_SIZES:
        st, sd = size / max(ht, wt), size / max(hd, wd)
        gt = cv2.GaussianBlur(cv2.cvtColor(_small(target.rgb, st), cv2.COLOR_RGB2GRAY), (0, 0), 1.5).astype(np.float32)
        gd = cv2.GaussianBlur(cv2.cvtColor(_small(donor.rgb, sd), cv2.COLOR_RGB2GRAY), (0, 0), 1.5).astype(np.float32)
        if prev is not None:  # carry the warp to this scale
            k = size / prev
            warp = warp * np.array([[1, 1, k], [1, 1, k], [1 / k, 1 / k, 1]], dtype=np.float32)
        mask = (_small(target.known, st, True) > 0).astype(np.uint8)
        try:
            rho, warp = cv2.findTransformECC(gd, gt, warp, cv2.MOTION_HOMOGRAPHY,
                                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), mask, 5)
        except cv2.error:
            return None
        prev = size
    if rho < MIN_ECC:
        return None
    # warp maps target coords to donor coords at the last scale: invert and rescale
    st, sd = prev / max(ht, wt), prev / max(hd, wd)
    s_t = np.diag([st, st, 1.0])
    s_d = np.diag([sd, sd, 1.0])
    h_td = np.linalg.inv(s_d) @ warp.astype(np.float64) @ s_t  # target -> donor, full res
    h_full = np.linalg.inv(h_td)
    return h_full / h_full[2, 2]


def align_features(target: Shot, donor: Shot) -> np.ndarray | None:
    import cv2

    st = WORK / max(target.rgb.shape[:2])
    sd = WORK / max(donor.rgb.shape[:2])
    gt = cv2.cvtColor(_small(target.rgb, st), cv2.COLOR_RGB2GRAY)
    gd = cv2.cvtColor(_small(donor.rgb, sd), cv2.COLOR_RGB2GRAY)
    mt = (_small(target.real, st, True) > 0).astype(np.uint8) * 255
    md = (_small(donor.real, sd, True) > 0).astype(np.uint8) * 255
    k = np.ones((9, 9), np.uint8)
    mt, md = cv2.erode(mt, k), cv2.erode(md, k)  # keep features away from hole edges
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=10)
    kt, dt = orb.detectAndCompute(gt, mt)
    kd, dd = orb.detectAndCompute(gd, md)
    if dt is None or dd is None or len(kt) < MIN_INLIERS or len(kd) < MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dd, dt, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([kd[m.queryIdx].pt for m in good]) / sd
    dst = np.float32([kt[m.trainIdx].pt for m in good]) / st
    h, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0 / st, maxIters=4000, confidence=0.999)
    if h is None or inl is None:
        return None
    n = int(inl.sum())
    if n < MIN_INLIERS or n < MIN_INLIER_SHARE * len(good):
        return None
    return h


def _block_means(a: np.ndarray, b: int) -> np.ndarray:
    h, w = a.shape[:2]
    hh, ww = h // b * b, w // b * b
    a = a[:hh, :ww].astype(np.float32)
    shape = (hh // b, b, ww // b, b) + a.shape[2:]
    return a.reshape(shape).mean(axis=(1, 3))


def fill_from(target: Shot, donor: Shot, holes: np.ndarray, block: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """(new rgb, filled mask): hole blocks of `target` that the aligned donor really has
    and that agree with the block's known average."""
    import cv2

    hgr = align(target, donor)
    if hgr is None:
        return target.rgb, np.zeros_like(holes)
    h, w = holes.shape
    warped = cv2.warpPerspective(donor.rgb, hgr, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpPerspective(donor.real.astype(np.uint8), hgr, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    # colours: a straight line per channel fitted on what both copies really have
    both = target.real & valid
    if both.sum() < 5000:
        return target.rgb, np.zeros_like(holes)
    idx = np.flatnonzero(both)
    if len(idx) > 200_000:
        idx = idx[:: len(idx) // 200_000 + 1]
    src = warped.reshape(-1, 3)[idx].astype(np.float32)
    ref = target.rgb.reshape(-1, 3)[idx].astype(np.float32)
    fitted = np.empty_like(warped, dtype=np.float32)
    for c in range(3):
        a, b = np.polyfit(src[:, c], ref[:, c], 1)
        if not 0.6 < a < 1.6:
            return target.rgb, np.zeros_like(holes)
        fitted[..., c] = warped[..., c] * a + b
    resid = np.median(np.abs(fitted.reshape(-1, 3)[idx] - ref))
    if resid > MAX_RESIDUAL:
        return target.rgb, np.zeros_like(holes)
    fitted = np.clip(fitted, 0, 255)

    # per block: in a hole, fully covered by real donor pixels, and consistent
    bh, bw = h // block, w // block
    hole_b = _block_means(holes, block) > 0.5
    valid_b = _block_means(valid, block) > 0.98
    cand = hole_b & valid_b
    known_b = _block_means(target.known & holes, block) > 0.5
    ycc_t = _block_means(cv2.cvtColor(target.rgb, cv2.COLOR_RGB2YCrCb), block)
    ycc_d = _block_means(cv2.cvtColor(fitted.astype(np.uint8), cv2.COLOR_RGB2YCrCb), block)
    dy = np.abs(ycc_t[..., 0] - ycc_d[..., 0])
    dc = np.abs(ycc_t[..., 1:] - ycc_d[..., 1:]).max(axis=-1)
    agree = (dy < LUMA_TOL) & (dc < CHROMA_TOL)
    accept = cand & np.where(known_b, agree, True)
    # unknown blocks (nothing to check against) only next to accepted known ones
    if (~known_b & cand).any():
        grown = cv2.dilate(accept.astype(np.uint8) & known_b.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        accept &= known_b | grown
    n, lab, stats, _ = cv2.connectedComponentsWithStats(accept.astype(np.uint8), connectivity=4)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_ISLAND:
            accept[lab == i] = False
    if not accept.any():
        return target.rgb, np.zeros_like(holes)
    mask = np.zeros((h, w), dtype=bool)
    mask[: bh * block, : bw * block] = np.repeat(np.repeat(accept, block, 0), block, 1)
    mask &= holes
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0) * mask
    soft = np.maximum(soft, cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32))
    out = target.rgb.astype(np.float32) * (1 - soft[..., None]) + fitted * soft[..., None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), mask


def burst_fill(target: Shot, shots: list[Shot], holes: np.ndarray, say=None) -> tuple[np.ndarray, np.ndarray]:
    """Fill as much of `holes` as the neighbouring shots can. Returns (rgb, filled mask)."""
    rgb = target.rgb
    filled = np.zeros_like(holes)
    for donor in candidates(target, shots):
        left = holes & ~filled
        if not left.any():
            break
        cur = Shot(rgb, target.real | filled, target.known | filled, target.when, target.name)
        rgb, got = fill_from(cur, donor, left)
        if got.any() and say:
            say(f"{target.name}: {100 * got.sum() / max(1, holes.sum()):.0f} % of the holes filled from {donor.name}")
        filled |= got
    return rgb, filled
