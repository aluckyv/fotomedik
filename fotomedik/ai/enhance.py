"""AI copy of a repaired photo: sharper rebuilt areas, filled unknown holes.

Only pixels of holes change; pixels decoded from real data are carried over (the
copy is saved as a JPEG at quality 95 without chroma subsampling, so they differ
from the repaired file only by that re-compression).

1. Holes nothing is known about (no thumbnail there) are filled by LaMa from the
   picture around them. LaMa works on a 512x512 view of the whole frame; its fill
   is scaled back up.
2. Holes are sharpened by Real-ESRGAN: the photo is reduced to a quarter (the
   holes carry no more detail than that) and enlarged 4x again by the network,
   tile by tile, only where holes are.
3. Data consistency (the range/null-space idea of DDNM, Wang et al. 2022): what
   is known about a hole is its 8x8 block averages, so after the networks every
   block gets its known average back (as a smooth correction surface); the
   networks contribute only the detail on top of it.
4. The result is blended into the holes with a soft edge.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

TILE = 128  # Real-ESRGAN input tile (low resolution side)
PAD = 8  # context around every tile, cut away after enlarging
LAMA = 512  # LaMa input size
GROW_UNKNOWN = 8  # pixels the unknown mask is grown by (LaMa needs a clean border)
FEATHER = 3.0  # blur radius of the blend edge inside holes


# ----------------------------------------------------------------------------- masks


def mcu_geometry(data: bytes) -> tuple[int, int, int, int, int]:
    """(mcu_w, mcu_h, mcus_per_row, width, height) of a JPEG."""
    from ..jpeg import decoder as D
    from ..jpeg import markers as M
    from ..jpeg.tables import FrameTables

    parse = M.parse_segments(data)
    ft = FrameTables.from_parse(parse)
    comps, per_row, _rows = D._scan_components(ft)
    if len(comps) > 1:
        mw = 8 * max(c.h for c in comps)
        mh = 8 * max(c.v for c in comps)
    else:
        mw = mh = 8
    return mw, mh, per_row, ft.frame.width, ft.frame.height


def mcu_mask(ranges, geometry) -> np.ndarray:
    """Pixel mask (height, width) of MCU ranges [(first, count) or {"mcu_first", "mcu_count"}]."""
    mw, mh, per_row, w, h = geometry
    rows = -(-h // mh)
    grid = np.zeros(rows * per_row, dtype=bool)
    for r in ranges:
        first, count = (r["mcu_first"], r["mcu_count"]) if isinstance(r, dict) else r
        grid[first : first + count] = True
    grid = grid.reshape(rows, per_row)
    return np.repeat(np.repeat(grid, mh, 0), mw, 1)[:h, :w]


def _soft(mask: np.ndarray, radius: float) -> np.ndarray:
    """Blend weight inside `mask` that rises from its edge over `radius` pixels."""
    if radius <= 0:
        return mask.astype(np.float32)
    im = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))
    soft = np.asarray(im, dtype=np.float32) / 255.0
    return np.where(mask, np.clip(2 * soft - 1, 0, 1), 0).astype(np.float32)


def _grow(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    im = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * px + 1))
    return np.asarray(im) > 127


# ----------------------------------------------------------------------------- engine


class Engine:
    """The two networks. Sessions are ONNX Runtime sessions or anything with the same
    `run(None, {name: array})` / `get_inputs()` interface (the tests use stand-ins)."""

    def __init__(self, sr=None, inpaint=None, device: str = "CPU"):
        self.sr = sr
        self.inpaint_s = inpaint
        self.device = device

    @classmethod
    def load(cls, progress=None, stop=None) -> "Engine":
        import onnxruntime as ort

        from . import models

        avail = ort.get_available_providers()
        providers = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider") if p in avail]
        providers.append("CPUExecutionProvider")

        def session(model):
            path = models.ensure(model, progress, stop)
            so = ort.SessionOptions()
            if "DmlExecutionProvider" in providers:  # DirectML wants these off
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            try:
                return ort.InferenceSession(str(path), sess_options=so, providers=providers)
            except Exception:  # the graphics card refused the model: use the processor
                return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        sr = session(models.SR)
        inp = session(models.INPAINT)
        used = sr.get_providers()[0] if hasattr(sr, "get_providers") else "CPUExecutionProvider"
        device = {"DmlExecutionProvider": "GPU (DirectML)", "CUDAExecutionProvider": "GPU (CUDA)"}.get(used, "CPU")
        return cls(sr, inp, device)

    def enlarge4(self, tile: np.ndarray) -> np.ndarray:
        """float32 (h, w, 3) in 0..1 -> (4h, 4w, 3) in 0..1."""
        name = self.sr.get_inputs()[0].name
        x = np.ascontiguousarray(tile.transpose(2, 0, 1)[None], dtype=np.float32)
        y = self.sr.run(None, {name: x})[0][0]
        return np.clip(y.transpose(1, 2, 0), 0, 1)

    def fill(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """uint8 (512, 512, 3) and bool (512, 512) -> uint8 (512, 512, 3), masked part filled."""
        ins = self.inpaint_s.get_inputs()
        img_name, mask_name = ins[0].name, ins[1].name
        if len(ins[0].shape) == 4 and ins[0].shape[1] == 1:  # mask first in some exports
            img_name, mask_name = mask_name, img_name
        img = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        m = mask.astype(np.float32)[None, None]
        out = self.inpaint_s.run(None, {img_name: np.ascontiguousarray(img), mask_name: np.ascontiguousarray(m)})[0][0]
        out = out.transpose(1, 2, 0)
        if out.max() <= 1.5:  # some exports give 0..1
            out = out * 255.0
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- steps


def inpaint_unknown(arr: np.ndarray, unknown: np.ndarray, engine: Engine) -> np.ndarray:
    h, w = unknown.shape
    grown = _grow(unknown, GROW_UNKNOWN)
    small = np.asarray(Image.fromarray(arr).resize((LAMA, LAMA), Image.BICUBIC))
    m = np.asarray(Image.fromarray((grown * 255).astype(np.uint8)).resize((LAMA, LAMA), Image.BILINEAR)) > 20
    filled = engine.fill(small, m)
    up = np.asarray(Image.fromarray(filled).resize((w, h), Image.BICUBIC), dtype=np.float32)
    a = _soft(unknown, FEATHER)[..., None]
    return np.clip(arr * (1 - a) + up * a, 0, 255).astype(np.uint8)


def sharpen_holes(arr: np.ndarray, holes: np.ndarray, engine: Engine, progress=None) -> np.ndarray:
    h, w = holes.shape
    lw, lh = -(-w // 4), -(-h // 4)
    lo = np.asarray(Image.fromarray(arr).resize((lw, lh), Image.LANCZOS), dtype=np.float32) / 255.0
    need = np.asarray(Image.fromarray((holes * 255).astype(np.uint8)).resize((lw, lh), Image.BOX)) > 0
    hi = np.asarray(Image.fromarray(arr).resize((4 * lw, 4 * lh), Image.BICUBIC), dtype=np.float32) / 255.0
    hi = hi.copy()
    tiles = [(y, x) for y in range(0, lh, TILE) for x in range(0, lw, TILE)
             if need[y : y + TILE, x : x + TILE].any()]
    padded = np.pad(lo, ((PAD, PAD + TILE), (PAD, PAD + TILE), (0, 0)), mode="reflect")
    for i, (y, x) in enumerate(tiles):
        th, tw = min(TILE, lh - y), min(TILE, lw - x)
        src = padded[y : y + th + 2 * PAD, x : x + tw + 2 * PAD]
        big = engine.enlarge4(src)
        hi[4 * y : 4 * (y + th), 4 * x : 4 * (x + tw)] = big[4 * PAD : 4 * (PAD + th), 4 * PAD : 4 * (PAD + tw)]
        if progress:
            progress(i + 1, len(tiles))
    hi = hi[:h, :w] * 255.0
    a = _soft(holes, FEATHER)[..., None]
    return np.clip(np.rint(arr * (1 - a) + hi * a), 0, 255).astype(np.uint8)


def keep_block_means(out: np.ndarray, ref: np.ndarray, holes: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Give every 8x8 block of `out` the average it has in `ref` where `known`, through
    a smooth correction (no steps between blocks). Only pixels inside holes move."""
    from ..recipes.r12_smooth import surface

    h, w = holes.shape
    bh, bw = -(-h // 8), -(-w // 8)
    pad = ((0, bh * 8 - h), (0, bw * 8 - w), (0, 0))
    o = np.pad(out.astype(np.float32), pad, mode="edge")
    r = np.pad(ref.astype(np.float32), pad, mode="edge")
    k = np.pad(known, pad[:2], mode="edge").reshape(bh, 8, bw, 8).any(axis=(1, 3))
    diff = (r - o).reshape(bh, 8, bw, 8, 3).mean(axis=(1, 3))
    diff[~k] = 0.0
    corr = np.stack([surface(diff[..., c]) for c in range(3)], axis=-1)[:h, :w]
    res = out.astype(np.float32) + corr * holes[..., None]
    return np.clip(np.rint(res), 0, 255).astype(np.uint8)


def enhance(img: Image.Image, holes: np.ndarray, unknown: np.ndarray, engine: Engine, progress=None) -> Image.Image:
    """AI copy: unknown holes filled (LaMa), all holes sharpened (Real-ESRGAN).
    Pixels outside `holes` are returned unchanged."""
    arr = np.asarray(img.convert("RGB"))
    real = arr.copy()
    unknown = unknown & holes
    if unknown.any() and engine.inpaint_s is not None:
        arr = inpaint_unknown(arr, unknown, engine)
    if holes.any() and engine.sr is not None:
        before = arr
        arr = sharpen_holes(arr, holes, engine, progress)
        arr = keep_block_means(arr, before, holes, holes & ~unknown)
    arr = np.where(holes[..., None], arr, real)
    return Image.fromarray(arr)


def enhance_file(data: bytes, holes, unknown, engine: Engine, progress=None) -> bytes | None:
    """JPEG bytes of the AI copy of a repaired file, or None when it has no holes.
    holes/unknown: MCU ranges. EXIF is kept."""
    if not holes:
        return None
    geo = mcu_geometry(data)
    hm = mcu_mask(holes, geo)
    um = mcu_mask(unknown, geo) if unknown else np.zeros_like(hm)
    img = Image.open(io.BytesIO(data))
    exif = img.info.get("exif")
    out = enhance(img, hm, um, engine, progress)
    buf = io.BytesIO()
    kw = {"quality": 95, "subsampling": 0}
    if exif:
        kw["exif"] = exif
    out.save(buf, "JPEG", **kw)
    return buf.getvalue()
