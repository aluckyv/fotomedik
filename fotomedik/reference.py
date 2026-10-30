"""A reference photo: an approximate copy of the same picture that stands in for the
160x120 thumbnail - an AI-restored version, a scan of a print, a copy from a web album.

Two things use it. The DC field (recipes.dcfield.thumbnail_base) takes it instead of
the thumbnail: pieces are placed and holes coloured against a full-size guide instead
of a 25-pixel blur. And smooth_holes (recipes.r12_smooth) writes the reference's own
pixels into the hole blocks, so the holes carry its detail while every decoded MCU
stays as it came from the card.

Such copies are rarely aligned with the file: editors and image models resample,
shift by pixels that vary across the frame, brighten and sharpen. So the reference is
aligned tile by tile against the decoded (trusted) pixels, with the shift field
interpolated across the holes, and its colours are calibrated per component against
the same pixels before anything is taken from it.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SEARCH = 16  # pixels: largest local shift looked for
TILE = 128  # pixels: the shift is estimated per window of this size (luma)
MAX_WINDOWS = 250  # about this many windows whatever the photo's size (the field is smooth anyway)
MIN_TRUSTED = 0.15  # share of a tile's pixels that must be trusted to estimate its shift
MIN_TEXTURE = 1.5  # standard deviation the trusted pixels of a window need (flat sky says nothing)
MIN_MATCH = 0.5  # correlation the best shift must reach
MIN_CONFIDENCE = 0.1  # ... and by how much it must beat the typical shift's correlation
TRIM = 2.0  # pixels: a window this far from the shift surface is a wrong match and is dropped
GAIN_RANGE = (0.5, 2.0)  # calibration gain allowed (edited copies are brightened or darkened)
TONE_TILE = 256  # pixels: tones are matched per window of this size (luma)
PAIR_MIN_R = 0.75  # correlation at 32x24 that pairs a reference with a photo by content
REFERENCE_SUFFIX = "_ref"  # output of a repair guided by a reference: <stem>_ref<suffix>

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ----------------------------------------------------------------------------- loading


def load_candidates(path: Path, width: int, height: int) -> list[np.ndarray]:
    """The reference as YCbCr uint8 arrays of the frame's size. A copy that lies the
    other way round comes in both turns; the caller keeps the one that fits."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    cands = [im]
    if (im.size[0] < im.size[1]) != (width < height):
        cands = [im.transpose(Image.ROTATE_90), im.transpose(Image.ROTATE_270)]
    out = []
    for c in cands:
        if c.size != (width, height):
            c = c.resize((width, height), Image.BICUBIC)
        out.append(np.asarray(c.convert("YCbCr")))
    return out


def signature(im: Image.Image) -> np.ndarray:
    """A 32x24 luma print of a picture, zero-mean and unit-norm (landscape)."""
    if im.size[0] < im.size[1]:
        im = im.transpose(Image.ROTATE_90)
    g = np.asarray(im.convert("L").resize((32, 24), Image.BOX), np.float32)
    g -= g.mean()
    n = float(np.linalg.norm(g))
    return g / n if n else g


def _thumbnail(data: bytes) -> Image.Image | None:
    """The embedded thumbnail of a (damaged) JPEG, else whatever of the picture itself
    decodes (a viewer's view of it), else None."""
    i = data.find(b"\xff\xd8", 2)
    j = data.find(b"\xff\xd9", i) if i > 0 else -1
    if i > 0 and j > 0:
        try:
            im = Image.open(io.BytesIO(data[i : j + 2]))
            im.load()
            return im
        except Exception:
            pass
    from PIL import ImageFile

    old = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    except Exception:
        return None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = old


def pair(photos: list[Path], references: Path | list[Path]) -> dict[Path, Path]:
    """Which reference belongs to which photo: by name first (the same stem, or the
    photo's stem followed by a separator), else by content (the photo's thumbnail
    against the reference, correlation at least PAIR_MIN_R). Photos without a match
    are left out."""
    refs = sorted(p for p in Path(references).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) \
        if isinstance(references, (str, Path)) else list(references)
    out: dict[Path, Path] = {}
    taken: set[Path] = set()
    by_stem = {r.stem.lower(): r for r in refs}
    for p in photos:
        stem = p.stem.lower()
        hit = by_stem.get(stem)
        if hit is None:
            hits = [r for r in refs if r.stem.lower().startswith(stem) and len(r.stem) > len(stem)
                    and not r.stem[len(stem)].isalnum()]
            hit = hits[0] if len(hits) == 1 else (hits[0] if hits else None)
        if hit is not None and hit not in taken and hit.resolve() != p.resolve():
            out[p] = hit
            taken.add(hit)
    rest = [p for p in photos if p not in out]
    free = [r for r in refs if r not in taken]
    if rest and free:
        sigs = []
        for r in free:
            try:
                sigs.append(signature(Image.open(r)))
            except Exception:
                sigs.append(None)
        for p in rest:
            try:
                th = _thumbnail(p.read_bytes())
            except OSError:
                th = None
            if th is None:
                continue
            s = signature(th)
            best = None
            for r, rs in zip(free, sigs):
                if rs is None or r in taken:
                    continue
                c = float((s * rs).sum())
                if best is None or c > best[0]:
                    best = (c, r)
            if best is not None and best[0] >= PAIR_MIN_R:
                out[p] = best[1]
                taken.add(best[1])
    return out


# ----------------------------------------------------------------------------- alignment


def _half(a: np.ndarray) -> np.ndarray:
    """2x2 box reduction (odd edges dropped)."""
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def _ncc_all(a: np.ndarray, m: np.ndarray, pad: np.ndarray, radius: int) -> np.ndarray:
    """Normalised correlation of the window `a` (masked by m) with the padded reference at
    every shift within `radius`: array (2r+1, 2r+1) indexed [sy + r, sx + r]."""
    from numpy.lib.stride_tricks import sliding_window_view

    th, tw = a.shape
    am = a[m]
    an = (am - am.mean()) / (am.std() + 1e-6)
    wins = sliding_window_view(pad[: th + 2 * radius, : tw + 2 * radius], (th, tw))  # (2r+1, 2r+1, th, tw)
    b = wins[:, :, m]  # (2r+1, 2r+1, n)
    b = b - b.mean(axis=2, keepdims=True)
    sd = b.std(axis=2) + 1e-6
    return (b @ an) / (an.size * sd)


def _tile_shifts(ref: np.ndarray, real: np.ndarray, trusted: np.ndarray, tile: int, search: int):
    """Per window (size `tile`, stride tile/2): (dy, dx) that moves the reference onto the
    real pixels, NaN where the window cannot tell (too few trusted pixels, or the match
    is no sharper at its best than elsewhere, as in flat sky). The search runs at half
    resolution, the best shift is then refined at full resolution. Correlation is used,
    so the copy's brightness and contrast do not matter."""
    H, W = real.shape
    stride = max(8, tile // 2, int(np.sqrt(H * W / MAX_WINDOWS)))  # big photos: windows further apart
    ys = list(range(0, max(1, H - tile) + 1, stride))
    xs = list(range(0, max(1, W - tile) + 1, stride))
    if ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    if xs[-1] + tile < W:
        xs.append(max(0, W - tile))
    dy = np.full((len(ys), len(xs)), np.nan)
    dx = np.full((len(ys), len(xs)), np.nan)
    r2 = -(-search // 2)
    ref2, real2 = _half(ref), _half(real)
    trusted2 = _half(trusted.astype(np.float32)) > 0.75
    pad2 = np.pad(ref2, ((r2, r2), (r2, r2)), mode="edge")
    pad = np.pad(ref, ((search + 1, search + 1), (search + 1, search + 1)), mode="edge")
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
            m = trusted[y0:y1, x0:x1]
            if m.mean() < MIN_TRUSTED:
                continue
            a = real[y0:y1, x0:x1]
            if a[m].std() < MIN_TEXTURE:
                continue
            # coarse: half resolution, every shift
            h0, h1, w0, w1 = y0 // 2, y1 // 2, x0 // 2, x1 // 2
            m2 = trusted2[h0:h1, w0:w1]
            if m2.sum() < 16:
                continue
            c = _ncc_all(real2[h0:h1, w0:w1], m2, pad2[h0 : h1 + 2 * r2, w0 : w1 + 2 * r2], r2)
            k = int(np.argmax(c))
            by, bx = (k // c.shape[1] - r2) * 2, (k % c.shape[1] - r2) * 2
            typical = float(np.median(c))
            # fine: full resolution around it
            f = _ncc_all(a, m, pad[y0 + search + 1 + by - 1 : y1 + search + 1 + by + 1,
                                   x0 + search + 1 + bx - 1 : x1 + search + 1 + bx + 1], 1)
            kf = int(np.argmax(f))
            best = float(f.flat[kf])
            if best < MIN_MATCH or best - typical < MIN_CONFIDENCE:
                continue  # the match is no better here than anywhere: nothing to anchor on
            dy[i, j] = by + kf // 3 - 1
            dx[i, j] = bx + kf % 3 - 1
    centres = (np.array(ys) + np.minimum(tile, H) / 2.0, np.array(xs) + np.minimum(tile, W) / 2.0)
    return dy, dx, centres


def _fit_field(vals: np.ndarray, cy: np.ndarray, cx: np.ndarray, H: int, W: int) -> np.ndarray:
    """A smooth shift field through the window estimates: a quadratic in (y, x) fitted
    with outliers trimmed (a window that disagrees with the surface by over 3 px is
    dropped, three rounds); with few windows a plane, with fewer a constant; zeros
    when nothing was estimated. Image models and editors drift smoothly, so a low-order
    surface is the right model and single wrong windows do not bend it."""
    known = np.isfinite(vals)
    if not known.any():
        return np.zeros((H, W), np.float32)
    iy, ix = np.nonzero(known)
    y = cy[iy] / max(1.0, H)
    x = cx[ix] / max(1.0, W)
    v = vals[known]

    def design(y, x, order):
        cols = [np.ones_like(y)]
        if order >= 1:
            cols += [y, x]
        if order >= 2:
            cols += [y * y, x * x, y * x]
        return np.stack(cols, axis=1)

    # start from the median shift: windows far from it are wrong matches, not drift
    keep = np.abs(v - np.median(v)) <= TRIM
    if keep.sum() < 3:
        keep = np.ones(len(v), bool)
    order = 2 if keep.sum() >= 12 else 1 if keep.sum() >= 5 else 0
    coef = None
    for _ in range(4):
        A = design(y[keep], x[keep], order)
        coef, *_rest = np.linalg.lstsq(A, v[keep], rcond=None)
        resid = np.abs(design(y, x, order) @ coef - v)
        new_keep = resid <= TRIM
        if new_keep.sum() < max(3, order * 3) or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    yy, xx = np.mgrid[0:H, 0:W]
    return (design(yy.ravel() / max(1.0, H), xx.ravel() / max(1.0, W), order) @ coef).reshape(H, W).astype(np.float32)


def shift_field(ref_y: np.ndarray, real_y: np.ndarray, trusted: np.ndarray, tile: int = TILE,
                search: int = SEARCH) -> tuple[np.ndarray, np.ndarray]:
    """Dense (dy, dx) per pixel of the luma plane that moves the reference onto the real
    pixels: estimated per window where trusted texture allows, a smooth surface through
    those estimates everywhere."""
    dy, dx, (cy, cx) = _tile_shifts(ref_y.astype(np.float32), real_y.astype(np.float32), trusted, tile, search)
    H, W = real_y.shape
    return _fit_field(dy, cy, cx, H, W), _fit_field(dx, cy, cx, H, W)


def warp(plane: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """plane sampled at (y + dy, x + dx) (nearest pixel, edges clamped)."""
    H, W = plane.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    sy = np.clip(np.rint(yy + dy).astype(int), 0, H - 1)
    sx = np.clip(np.rint(xx + dx).astype(int), 0, W - 1)
    return plane[sy, sx]


def _window_grid(H: int, W: int, tile: int) -> tuple[list[int], list[int]]:
    stride = max(8, tile // 2)
    ys = list(range(0, max(1, H - tile) + 1, stride))
    xs = list(range(0, max(1, W - tile) + 1, stride))
    if ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    if xs[-1] + tile < W:
        xs.append(max(0, W - tile))
    return ys, xs


def _dense(vals: np.ndarray, cy: np.ndarray, cx: np.ndarray, H: int, W: int) -> np.ndarray:
    """Window values spread over the picture: NaN windows take a distance-weighted mean of
    the estimated ones, then linear interpolation between window centres."""
    known = np.isfinite(vals)
    if not known.any():
        return np.zeros((H, W), np.float32)
    f = vals.copy()
    if not known.all():
        iy, ix = np.nonzero(known)
        v = vals[known]
        for i, j in zip(*np.nonzero(~known)):
            w = np.exp(-((iy - i) ** 2 + (ix - j) ** 2) / 8.0)
            f[i, j] = float((w * v).sum() / w.sum()) if w.sum() > 1e-9 else v[int(np.argmin((iy - i) ** 2 + (ix - j) ** 2))]
    yy = np.interp(np.arange(H), cy, np.arange(len(cy))) if len(cy) > 1 else np.zeros(H)
    xx = np.interp(np.arange(W), cx, np.arange(len(cx))) if len(cx) > 1 else np.zeros(W)
    iy, ix = np.floor(yy).astype(int), np.floor(xx).astype(int)
    iy1, ix1 = np.minimum(iy + 1, len(cy) - 1), np.minimum(ix + 1, len(cx) - 1)
    fy, fx = (yy - iy)[:, None], (xx - ix)[None, :]
    return (f[iy][:, ix] * (1 - fy) * (1 - fx) + f[iy1][:, ix] * fy * (1 - fx)
            + f[iy][:, ix1] * (1 - fy) * fx + f[iy1][:, ix1] * fy * fx).astype(np.float32)


def calibrate_local(ref: np.ndarray, real: np.ndarray, trusted: np.ndarray, tile: int = TONE_TILE) -> np.ndarray:
    """`ref` brought to `real`'s tones region by region: gain and offset are fitted per
    window on the trusted pixels and interpolated between windows (an image model
    re-renders tones differently in the sky and on a facade; one global curve leaves
    bands where real and reference rows alternate)."""
    H, W = real.shape
    tile = min(tile, H, W)
    ys, xs = _window_grid(H, W, tile)
    gain = np.full((len(ys), len(xs)), np.nan)
    off = np.full((len(ys), len(xs)), np.nan)
    a0, b0 = calibrate(ref, real, trusted)
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
            m = trusted[y0:y1, x0:x1]
            if m.mean() < MIN_TRUSTED or m.sum() < 256:
                continue
            x, y = ref[y0:y1, x0:x1][m].astype(np.float64), real[y0:y1, x0:x1][m].astype(np.float64)
            if np.std(x) < MIN_TEXTURE:  # flat: only the offset is known here
                gain[i, j], off[i, j] = a0, float(np.median(y - a0 * x))
                continue
            a = float(np.clip(np.cov(x, y)[0, 1] / np.var(x), *GAIN_RANGE))
            d = y - a * x
            lo, hi = np.percentile(d, [10, 90])
            core = d[(d >= lo) & (d <= hi)]
            gain[i, j], off[i, j] = a, float(core.mean()) if core.size else float(d.mean())
    if not np.isfinite(gain).any():
        return np.clip(a0 * ref + b0, -128.0, 127.0)
    cy = np.array(ys) + min(tile, H) / 2.0
    cx = np.array(xs) + min(tile, W) / 2.0
    g, o = _dense(gain, cy, cx, H, W), _dense(off, cy, cx, H, W)
    return np.clip(g * ref + o, -128.0, 127.0)


def calibrate(ref: np.ndarray, real: np.ndarray, trusted: np.ndarray) -> tuple[float, float]:
    """(gain, offset) with real ~ gain * ref + offset over the trusted pixels (the 5 %
    tails of the difference left out), gain within GAIN_RANGE."""
    x, y = ref[trusted].astype(np.float64), real[trusted].astype(np.float64)
    if x.size < 64 or np.std(x) < 1e-6:
        return 1.0, float(np.median(y - x)) if x.size else 0.0
    a = float(np.clip(np.cov(x, y)[0, 1] / np.var(x), *GAIN_RANGE))
    d = y - a * x
    lo, hi = np.percentile(d, [5, 95])
    core = d[(d >= lo) & (d <= hi)]
    return a, float(core.mean()) if core.size else float(d.mean())


def aligned_planes(ref_ycc: np.ndarray, planes: dict, comps, frame, hole_blocks: dict) -> dict:
    """The reference's planes per component (level-shifted, at the component's own
    resolution and block-grid size), aligned to and calibrated against the decoded
    planes outside the holes. planes/hole_blocks: as recipes.r12_smooth has them."""
    first = comps[0]
    real_y = planes[first.cid]
    H, W = real_y.shape
    ref_y = ref_ycc[:, :, 0].astype(np.float32) - 128.0
    ref_y = np.pad(ref_y, ((0, max(0, H - ref_y.shape[0])), (0, max(0, W - ref_y.shape[1]))), mode="edge")[:H, :W]
    hb = hole_blocks[first.cid]
    trusted = ~np.repeat(np.repeat(hb, 8, 0), 8, 1)[:H, :W]
    dy, dx = shift_field(ref_y, real_y, trusted)
    out = {}
    for i, c in enumerate(comps):
        plane = planes[c.cid]
        h, w = plane.shape
        ch = min(i, 2) if ref_ycc.shape[2] >= 3 else 0
        r = ref_ycc[:, :, ch].astype(np.float32) - 128.0
        fc = frame.component(c.cid)
        sy, sx = frame.vmax // fc.v, frame.hmax // fc.h
        if sy > 1 or sx > 1:
            hh, ww = r.shape[0] // sy * sy, r.shape[1] // sx * sx
            r = r[:hh, :ww].reshape(hh // sy, sy, ww // sx, sx).mean(axis=(1, 3))
        r = np.pad(r, ((0, max(0, h - r.shape[0])), (0, max(0, w - r.shape[1]))), mode="edge")[:h, :w]
        fy = np.asarray(Image.fromarray(dy, mode="F").resize((w, h), Image.BILINEAR), np.float32) / sy
        fx = np.asarray(Image.fromarray(dx, mode="F").resize((w, h), Image.BILINEAR), np.float32) / sx
        r = warp(r, fy, fx)
        t = ~np.repeat(np.repeat(hole_blocks[c.cid], 8, 0), 8, 1)[:h, :w]
        out[c.cid] = calibrate_local(r, plane, t, max(32, TONE_TILE // max(sy, sx)))
    return out


# ----------------------------------------------------------------------------- in a repair


def for_context(ctx, ft) -> list[np.ndarray] | None:
    """The reference pictures of this repair (options["reference"]: a path), in the
    frame's size, cached on the context; None without one or when it cannot be read."""
    path = ctx.options.get("reference")
    if not path:
        return None
    key = f"reference_img_{ft.frame.width}x{ft.frame.height}"
    if key not in ctx.memo:
        try:
            ctx.memo[key] = load_candidates(Path(path), ft.frame.width, ft.frame.height)
            ctx.memo["reference_used"] = True
        except Exception as e:
            ctx.memo[key] = None
            ctx.note("reference", f"{Path(path).name} could not be read: {e}")
    return ctx.memo[key]


def best_candidate(cands: list[np.ndarray], real_y: np.ndarray, trusted: np.ndarray) -> np.ndarray:
    """Of the turns of a reference, the one whose luma agrees best with the trusted pixels."""
    if len(cands) == 1:
        return cands[0]
    H, W = real_y.shape

    def score(c):
        r = c[:, :, 0].astype(np.float32) - 128.0
        r = np.pad(r, ((0, max(0, H - r.shape[0])), (0, max(0, W - r.shape[1]))), mode="edge")[:H, :W]
        x, y = r[trusted], real_y[trusted]
        if x.size < 64 or np.std(x) < 1e-6 or np.std(y) < 1e-6:
            return -1.0
        return float(np.corrcoef(x, y)[0, 1])

    return max(cands, key=score)


# ----------------------------------------------------------------------------- a twin shot


TWIN_WIDTH = 160  # pixels: a twin is used at about a thumbnail's size (colour, not detail)
TWIN_REAL = 0.2  # share of a twin pixel's area that must be real data for it to count


def twin_loader(ctx, ft):
    """A callable giving (YCbCr pixels, real mask) of the neighbouring shot named by
    options["twin"], at about a thumbnail's size - or None. That is the twin's own
    thumbnail when it has a usable one; otherwise the twin is repaired first (in memory,
    without a twin of its own) and only its real parts count: its holes are guesses.
    Computed on the first call only, then kept on the context."""
    path = ctx.options.get("twin")
    if not path:
        return None

    def load():
        if "twin_img" not in ctx.memo:
            ctx.memo["twin_img"] = _twin_picture(ctx, Path(path), ft.frame.width, ft.frame.height)
        return ctx.memo["twin_img"]

    return load


def _twin_picture(ctx, path: Path, width: int, height: int):
    from .jpeg import decoder as D
    from .jpeg import markers as M
    from .jpeg.tables import FrameTables
    from .recipes.base import Context
    from .recipes.registry import run_pipeline

    from .recipes.dcfield import thumbnail_image

    try:
        data = path.read_bytes()
    except OSError as e:
        ctx.note("twin", f"{path.name} could not be read: {e}")
        return None
    got = thumbnail_image(data)
    if got is not None and got[1] is None and (got[0].shape[1] >= got[0].shape[0]) == (width >= height):
        ctx.memo["twin_name"] = path.name
        return got[0], np.ones(got[0].shape[:2], dtype=np.float32)
    opts = {k: v for k, v in ctx.options.items() if k not in ("twin", "reference", "reference_dir", "mark_holes")}
    try:
        tctx = Context(data=data, name=path.name, ref=ctx.ref, options=opts)
        run_pipeline(tctx)
        im = Image.open(io.BytesIO(tctx.data))
        im.load()
        res = M.parse_segments(tctx.data)
        tft = FrameTables.from_parse(res)
    except Exception as e:
        ctx.note("twin", f"{path.name} could not be repaired: {e}")
        return None
    if (im.size[0] >= im.size[1]) != (width >= height):
        return None
    real = np.ones(im.size[::-1], dtype=np.float32)
    if tctx.holes:
        _comps, per_row, _rows = D._scan_components(tft)
        mw, mh = 8 * tft.frame.hmax, 8 * tft.frame.vmax
        for h in tctx.holes:
            for m in range(h.mcu_first, h.mcu_first + h.mcu_count):
                y, x = divmod(m, per_row)
                real[y * mh : (y + 1) * mh, x * mw : (x + 1) * mw] = 0.0
    w = TWIN_WIDTH
    hgt = max(1, round(w * im.size[1] / im.size[0]))
    ycc = np.asarray(im.convert("YCbCr"), dtype=np.float32)
    share = np.asarray(Image.fromarray(real, mode="F").resize((w, hgt), Image.BOX), np.float32)
    planes = []
    for c in range(3):
        s = np.asarray(Image.fromarray(ycc[:, :, c] * real, mode="F").resize((w, hgt), Image.BOX), np.float32)
        planes.append(s / np.maximum(share, 1e-6))
    small = np.clip(np.stack(planes, axis=2), 0, 255).astype(np.uint8)
    ctx.memo["twin_name"] = path.name
    return small, (share >= TWIN_REAL).astype(np.float32)
