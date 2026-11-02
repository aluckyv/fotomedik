"""Holes textured with the photo's own pixels, the way Photoshop's Content-Aware Fill does it.

The steps before this one fill a hole with the right colours (the thumbnail, the guided
network, LaMa), but smooth: a sky without its grain, a field without its grass. Here
every hole pixel is matched with a place of the real picture that looks the same at a
coarse scale (PatchMatch, Barnes et al. 2009: random offsets that spread to the
neighbours and are searched around, on a working copy of about 1000 pixels), and the
matches are made coherent by a few rounds of voting (Wexler, Shechtman and Irani 2007).
Only the fine detail of the matched places is carried over, on top of the fill's own
coarse picture: the colours and shapes stay those of the thumbnail, the grain and the
texture come from the photo itself. Nothing is invented, and faces and the date stamp
are left as they are (no patch of someone else's skin on a face).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

WORK = 1024  # long side of the working copy the matches are found on
RADIUS = 2  # 5x5 patches on the working copy
ITERS = 5  # PatchMatch rounds per voting round
VOTES = 3  # voting rounds (coarse match first, then coherent texture)
GUIDE_W = 1.0  # weight of the coarse look (colour and its slope around) in a match
TEXTURE_W = 0.5  # weight of the texture already voted in (from the second round on)
MIN_REAL = 0.05  # share of real pixels below which there is too little to take texture from
DETAIL_MAX = 20.0  # gray levels: the carried detail is grain and texture, never a new edge
EDGE_LO, EDGE_HI = 6.0, 18.0  # slope of the coarse picture (levels per working pixel): flat gets all the detail, an edge none
MATCH_MAX = 14.0  # gray levels (rms of the coarse look): a worse match carries nothing
SEED = 1234


def box_blur(img: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, edges repeated; float32, any number of channels."""
    if r <= 0:
        return img.astype(np.float32)
    a = img.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
        n = a.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        a = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return a


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        m = out.copy()
        m[1:] &= out[:-1]
        m[:-1] &= out[1:]
        m[:, 1:] &= out[:, :-1]
        m[:, :-1] &= out[:, 1:]
        out = m
    return out


def _features(img: np.ndarray) -> np.ndarray:
    """Coarse look of every pixel: the blurred colour there and 4 pixels away on each side."""
    b = box_blur(img, 2)
    h, w = b.shape[:2]
    pad = np.pad(b, ((4, 4), (4, 4), (0, 0)), mode="edge")
    parts = [pad[4 + dy : 4 + dy + h, 4 + dx : 4 + dx + w] for dy, dx in ((0, 0), (-4, 0), (4, 0), (0, -4), (0, 4))]
    return np.concatenate(parts, axis=-1)


class _Matcher:
    """PatchMatch of target pixels (ty, tx) against valid source pixels, vectorised."""

    def __init__(self, feat: np.ndarray, src: np.ndarray, valid: np.ndarray, targets: np.ndarray, rng):
        self.feat, self.src, self.valid = feat, src, valid
        self.h, self.w = valid.shape
        self.ty, self.tx = np.nonzero(targets)
        vy, vx = np.nonzero(valid)
        pick = rng.integers(0, len(vy), len(self.ty))
        self.sy, self.sx = vy[pick].astype(np.int64), vx[pick].astype(np.int64)
        self.tex = None  # the voted picture (texture term), None in the first round
        self.rng = rng
        self.cost = self.costs(self.sy, self.sx)
        self.index = np.full((self.h, self.w), -1, dtype=np.int64)
        self.index[self.ty, self.tx] = np.arange(len(self.ty))

    def costs(self, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
        d = self.feat[self.ty, self.tx] - self.feat[sy, sx]
        c = GUIDE_W * (d * d).sum(-1)
        if self.tex is not None:
            t = np.zeros_like(c)
            for dy in (-RADIUS, 0, RADIUS):
                for dx in (-RADIUS, 0, RADIUS):
                    py = np.clip(self.ty + dy, 0, self.h - 1)
                    px = np.clip(self.tx + dx, 0, self.w - 1)
                    qy = np.clip(sy + dy, 0, self.h - 1)
                    qx = np.clip(sx + dx, 0, self.w - 1)
                    e = self.tex[py, px] - self.src[qy, qx]
                    t += (e * e).sum(-1)
            c = c + TEXTURE_W * t / 9.0
        return c

    def _try(self, sy: np.ndarray, sx: np.ndarray, ok: np.ndarray) -> None:
        sy = np.clip(sy, 0, self.h - 1)
        sx = np.clip(sx, 0, self.w - 1)
        ok = ok & self.valid[sy, sx]
        if not ok.any():
            return
        idx = np.flatnonzero(ok)
        c = self.costs(sy[idx], sx[idx]) if len(idx) == len(sy) else self._costs_at(idx, sy[idx], sx[idx])
        better = c < self.cost[idx]
        j = idx[better]
        self.sy[j], self.sx[j], self.cost[j] = sy[j], sx[j], c[better]

    def _costs_at(self, idx, sy, sx):
        ty, tx = self.ty, self.tx
        self.ty, self.tx = ty[idx], tx[idx]
        try:
            return self.costs(sy, sx)
        finally:
            self.ty, self.tx = ty, tx

    def run(self, iters: int) -> None:
        for it in range(iters):
            for step in (8, 4, 2, 1) if it == 0 else (1,):
                for dy, dx in ((0, step), (step, 0), (0, -step), (-step, 0)):
                    ny, nx = self.ty - dy, self.tx - dx
                    inside = (ny >= 0) & (ny < self.h) & (nx >= 0) & (nx < self.w)
                    nb = np.where(inside, self.index[np.clip(ny, 0, self.h - 1), np.clip(nx, 0, self.w - 1)], -1)
                    ok = nb >= 0
                    nbc = np.maximum(nb, 0)
                    self._try(self.sy[nbc] + dy, self.sx[nbc] + dx, ok)
            r = max(self.h, self.w) // 2
            while r >= 1:
                oy = self.rng.integers(-r, r + 1, len(self.ty))
                ox = self.rng.integers(-r, r + 1, len(self.ty))
                self._try(self.sy + oy, self.sx + ox, np.ones(len(self.ty), bool))
                r //= 4

    def vote(self) -> np.ndarray:
        """Every target pixel: the mean of what the patches over it bring (their sources)."""
        acc = np.zeros((len(self.ty), self.src.shape[2]), np.float32)
        n = np.zeros(len(self.ty), np.float32)
        for dy in range(-RADIUS, RADIUS + 1):
            for dx in range(-RADIUS, RADIUS + 1):
                py, px = self.ty - dy, self.tx - dx  # the patch centred there covers this pixel
                inside = (py >= 0) & (py < self.h) & (px >= 0) & (px < self.w)
                nb = np.where(inside, self.index[np.clip(py, 0, self.h - 1), np.clip(px, 0, self.w - 1)], -1)
                ok = nb >= 0
                nbc = np.maximum(nb, 0)
                qy = np.clip(self.sy[nbc] + dy, 0, self.h - 1)
                qx = np.clip(self.sx[nbc] + dx, 0, self.w - 1)
                acc[ok] += self.src[qy[ok], qx[ok]]
                n[ok] += 1
        return acc / np.maximum(n, 1)[:, None]


def skin(arr: np.ndarray) -> np.ndarray:
    """Pixels of skin colour (arms, backs, hands): no texture of other places goes there."""
    y, cb, cr = np.moveaxis(np.asarray(Image.fromarray(arr).convert("YCbCr"), dtype=np.int16), -1, 0)
    m = (cr >= 135) & (cr <= 178) & (cb >= 78) & (cb <= 130) & (y >= 50)
    return box_blur(m.astype(np.float32)[..., None], 6)[..., 0] > 0.15  # a margin around it


def texture_holes(arr: np.ndarray, holes: np.ndarray, avoid: np.ndarray | None = None,
                  stats: dict | None = None) -> np.ndarray:
    """`arr` (rgb uint8, holes already filled with the right colours) with the holes given
    the fine detail of matching places of the real picture. `avoid`: pixels that are
    neither textured nor taken from (faces, the date stamp)."""
    h, w = holes.shape
    avoid = (np.zeros_like(holes) if avoid is None else avoid) | skin(arr)
    targets_full = holes & ~avoid
    source_full = ~holes & ~avoid
    if not targets_full.any() or source_full.mean() < MIN_REAL:
        return arr
    s = min(1.0, WORK / max(h, w))
    sw, sh = max(1, round(w * s)), max(1, round(h * s))
    small = np.asarray(Image.fromarray(arr).resize((sw, sh), Image.BILINEAR), dtype=np.float32)
    hole_s = np.asarray(Image.fromarray(holes.astype(np.uint8) * 255).resize((sw, sh), Image.BILINEAR)) > 0
    avoid_s = np.asarray(Image.fromarray(avoid.astype(np.uint8) * 255).resize((sw, sh), Image.BILINEAR)) > 0
    targets = hole_s & ~avoid_s
    valid = _erode(~hole_s & ~avoid_s, RADIUS + 1)
    if not targets.any() or valid.sum() < 64:
        return arr
    rng = np.random.default_rng(SEED)
    feat = _features(small)
    m = _Matcher(feat, small, valid, targets, rng)
    for v in range(VOTES):
        m.run(ITERS)
        tex = small.copy()
        tex[m.ty, m.tx] = m.vote()
        m.tex = tex
        m.cost = m.costs(m.sy, m.sx)
    # offsets of the working copy, to the full picture (nearest cell); a poor match carries nothing
    oy = np.zeros((sh, sw), np.float32)
    ox = np.zeros((sh, sw), np.float32)
    gd = feat[m.ty, m.tx] - feat[m.sy, m.sx]
    good = np.sqrt((gd * gd).mean(-1)) <= MATCH_MAX
    oy[m.ty, m.tx] = (m.sy - m.ty) / s
    ox[m.ty, m.tx] = (m.sx - m.tx) / s
    use = np.zeros((sh, sw), bool)
    use[m.ty[good], m.tx[good]] = True
    # how much detail: all on flat and textured surfaces, none on the coarse picture's edges
    coarse = box_blur(small, 2).mean(-1)
    gy, gx = np.gradient(coarse)
    amount = np.clip((EDGE_HI - np.hypot(gy, gx)) / (EDGE_HI - EDGE_LO), 0.0, 1.0)
    yy = np.minimum((np.arange(h) * s).astype(np.int64), sh - 1)
    xx = np.minimum((np.arange(w) * s).astype(np.int64), sw - 1)
    r = max(1, int(round(1.0 / s)) * 2)  # finer than the working copy: detail, coarser: the fill's own
    real = arr.astype(np.float32)
    fine = real - box_blur(real, r)  # the real picture's detail
    base = box_blur(real, r)  # the fill's coarse picture (and the real one's, outside the holes)
    out = real.copy()
    ys, xs = np.nonzero(targets_full)
    cy, cx = yy[ys], xx[xs]
    qy = np.rint(ys + oy[cy, cx]).astype(np.int64)
    qx = np.rint(xs + ox[cy, cx]).astype(np.int64)
    ok = use[cy, cx] & (qy >= 0) & (qy < h) & (qx >= 0) & (qx < w)
    ok[ok] &= source_full[qy[ok], qx[ok]]
    ys, xs, qy, qx, cy, cx = ys[ok], xs[ok], qy[ok], qx[ok], cy[ok], cx[ok]
    a = amount[cy, cx][:, None]
    textured = base[ys, xs] + np.clip(fine[qy, qx], -DETAIL_MAX, DETAIL_MAX)
    out[ys, xs] = (1.0 - a) * real[ys, xs] + a * textured  # at an edge the fill stays as it was
    if stats is not None:
        stats["textured"] = int(ok.sum())
        stats["work"] = (sw, sh)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
