"""AI copy of a repaired photo: sharper rebuilt areas, filled unknown holes.

Only pixels of holes change; pixels decoded from real data are carried over (the
copy is saved as a JPEG at quality 95 without chroma subsampling, so they differ
from the repaired file only by that re-compression).

1. Holes are healed by LaMa from the real pixels around them, at full resolution in
   overlapping 512x512 windows (a window that is mostly hole sees a twice larger view):
   thin holes between real rows get the structure running through them.
2. Without LaMa, holes are sharpened by Real-ESRGAN: the photo is reduced to a quarter (the
   holes carry no more detail than that) and enlarged 4x again by the network,
   tile by tile, only where holes are; on a graphics card several tiles go in one
   call. The same enlarging brings the holes Stable Diffusion repainted at a
   reduced size back to full size (diffusion.repaint_holes).
3. Data consistency (the range/null-space idea of DDNM, Wang et al. 2022): what
   is known about a hole is its 8x8 block averages, so after the networks every
   block gets its known average back (as a smooth correction surface); the
   networks contribute only the detail on top of it.
4. The result is blended into the holes with a soft edge.
"""

from __future__ import annotations

import io
import time

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


ORT_DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int64)": np.int64,
              "tensor(int32)": np.int32, "tensor(double)": np.float64}
GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")
DEVICE_NAMES = {"DmlExecutionProvider": "GPU (DirectML)", "CUDAExecutionProvider": "GPU (CUDA)",
                "CPUExecutionProvider": "CPU"}
SR_BATCH = 4  # tiles enlarged in one go on a graphics card (when the model allows a batch)


def _batch_dynamic(session) -> bool:
    """Whether the network takes more than one picture at a time (a symbolic batch size)."""
    try:
        shape = session.get_inputs()[0].shape
    except Exception:
        return False
    return bool(shape) and not isinstance(shape[0], int)


class Stopped(InterruptedError):
    """The user stopped the run in the middle of a photo."""


def _check(stop) -> None:
    if stop is not None and stop.is_set():
        raise Stopped("stopped")


class Engine:
    """The networks the chosen steps need. Sessions are ONNX Runtime sessions or anything
    with the same `run(None, {name: array})` / `get_inputs()` interface (the tests use
    stand-ins). face_det is the path of the YuNet model (OpenCV runs it).

    An engine is kept between runs (see run.engine_for) and `ensure` loads the networks
    a later run needs on top. `devices` says where every network runs and `notes`
    collects what went differently than hoped (a network the graphics card refused)."""

    def __init__(self, sr=None, inpaint=None, device: str = "CPU", deblock=None, face_det=None, face=None, sd=None):
        self.sr = sr
        self.inpaint_s = inpaint
        self.device = device
        self.deblock_s = deblock
        self.face_det = face_det
        self.face_s = face
        self.sd = sd
        self.devices: dict[str, str] = {}
        self.notes: list[str] = []
        self.loaded: set[str] = set()
        self.sr_batch = 1

    @classmethod
    def load(cls, progress=None, stop=None, steps=("ai",), half: bool = True, say=None) -> "Engine":
        return cls().ensure(steps, progress, stop, half, say)

    def has(self, step: str) -> bool:
        return step in self.loaded

    def ensure(self, steps, progress=None, stop=None, half: bool = True, say=None) -> "Engine":
        """Load the networks of `steps` that are not loaded yet (downloading them first).
        progress(done, total, model) follows the downloads, say(text) tells about slow
        one-off work. half: fp16 copies of the fp32 networks on a graphics card (see .half)."""
        import onnxruntime as ort

        from . import models

        if hasattr(ort, "preload_dlls"):  # a CUDA build finds its CUDA/cuDNN libraries (pip packages)
            try:
                ort.preload_dlls()
            except Exception:
                pass
        avail = ort.get_available_providers()
        gpu = [p for p in GPU_PROVIDERS if p in avail]
        providers = gpu + ["CPUExecutionProvider"]

        def make(path):
            so = ort.SessionOptions()
            if "DmlExecutionProvider" in providers:  # DirectML wants these off
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            try:
                return ort.InferenceSession(str(path), sess_options=so, providers=providers)
            except Exception:  # the graphics card refused the model: use the processor
                if not gpu:
                    raise
                return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        def session(model):
            path = models.ensure(model, progress, stop)
            if half and gpu:
                from . import half as H

                s = H.session(model, path, make, say, self.notes)
            else:
                s = make(path)
            prov = s.get_providers()[0] if hasattr(s, "get_providers") else "CPUExecutionProvider"
            self.devices[model.name] = DEVICE_NAMES.get(prov, prov)
            if gpu and prov not in GPU_PROVIDERS:
                self.notes.append(f"{model.name} runs on the processor: the graphics card refused it")
            return s

        steps = set(steps) - self.loaded
        if "ai" in steps:
            if self.sr is None:
                self.sr = session(models.SR)
            if self.inpaint_s is None:
                self.inpaint_s = session(models.INPAINT)
        if "deblock" in steps:
            self.deblock_s = session(models.DEBLOCK)
        if "faces" in steps:
            self.face_det = str(models.ensure(models.FACE_DET, progress, stop))
            self.face_s = session(models.FACE)
        if "diffusion" in steps:
            from .diffusion import Diffusion

            if self.inpaint_s is None:
                self.inpaint_s = session(models.INPAINT)
            if self.sr is None:  # brings the repainted holes back to full size
                self.sr = session(models.SR)
            tok = None
            try:  # the prompt's words -> token ids; without it the painting is unconditioned
                from .tokenizer import ClipTokenizer

                tok = ClipTokenizer.load(models.ensure(models.CLIP_VOCAB, progress, stop),
                                         models.ensure(models.CLIP_MERGES, progress, stop))
            except InterruptedError:
                raise
            except Exception as e:
                self.notes.append(f"Stable Diffusion paints without a prompt (tokenizer files: {e})")
            self.sd = Diffusion(session(models.SD_UNET), session(models.SD_VAE_ENC), session(models.SD_VAE_DEC),
                                session(models.SD_TEXT), tok)
        self.loaded |= steps
        on_gpu = [d for d in self.devices.values() if d != "CPU"]
        self.device = on_gpu[0] if on_gpu else "CPU"
        self.sr_batch = SR_BATCH if (self.sr is not None and on_gpu and _batch_dynamic(self.sr)) else 1
        return self

    def enlarge4(self, tile: np.ndarray) -> np.ndarray:
        """float32 (h, w, 3) in 0..1 -> (4h, 4w, 3) in 0..1."""
        return self.enlarge4_many([tile])[0]

    def enlarge4_many(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        """enlarge4 of several same-sized tiles, `sr_batch` at a time in one network call."""
        name = self.sr.get_inputs()[0].name
        out: list[np.ndarray] = []
        step = max(1, self.sr_batch)
        for k in range(0, len(tiles), step):
            group = tiles[k : k + step]
            x = np.ascontiguousarray(np.stack([t.transpose(2, 0, 1) for t in group]), dtype=np.float32)
            y = self.sr.run(None, {name: x})[0]
            out.extend(np.clip(yi.transpose(1, 2, 0), 0, 1) for yi in y)
        return out

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
    # nothing inside an unknown hole is worth keeping: the fill replaces it entirely
    # (blending with the placeholder would leave a frame around it)
    return np.where(unknown[..., None], np.clip(np.rint(up), 0, 255), arr).astype(np.uint8)


HEAL_TILE = 512  # LaMa window (pixels of the picture at full size)
HEAL_STEP = 384  # windows overlap by a quarter; their fills are blended with a smooth weight
HEAL_WIDE = 0.6  # share of a window that is hole above which a twice larger view is filled instead


def heal_holes(arr: np.ndarray, holes: np.ndarray, engine: Engine, progress=None, stop=None,
               stats: dict | None = None) -> np.ndarray:
    """Holes filled by LaMa from the real pixels around them, at full resolution, window
    by window (like a healing brush): thin holes between real rows get the structure and
    texture that runs through them. Where a window is mostly hole, a twice larger view
    (reduced to the window's size) is filled instead, so wide holes still see context."""
    h, w = holes.shape
    T = HEAL_TILE
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w), np.float32)

    def starts(n, size):
        if n <= size:
            return [0]
        out = list(range(0, n - size + 1, HEAL_STEP))
        if out[-1] != n - size:
            out.append(n - size)
        return out

    jobs = []
    for y in starts(h, T):
        for x in starts(w, T):
            m = holes[y : y + T, x : x + T]
            if m.any():
                jobs.append((y, x, float(m.mean())))
    if stats is not None:
        stats["tiles"] = len(jobs)
    for i, (y, x, share) in enumerate(jobs):
        _check(stop)
        if share > HEAL_WIDE and h >= 2 * T and w >= 2 * T:
            # a view twice the size, centred on the window
            y2 = min(max(0, y - T // 2), h - 2 * T)
            x2 = min(max(0, x - T // 2), w - 2 * T)
            view = arr[y2 : y2 + 2 * T, x2 : x2 + 2 * T]
            vm = holes[y2 : y2 + 2 * T, x2 : x2 + 2 * T]
            small = np.asarray(Image.fromarray(view).resize((T, T), Image.LANCZOS))
            sm = np.asarray(Image.fromarray((vm * 255).astype(np.uint8)).resize((T, T), Image.BOX)) > 0
            big = np.asarray(Image.fromarray(engine.fill(small, sm)).resize((2 * T, 2 * T), Image.BICUBIC), np.float32)
            out = big[y - y2 : y - y2 + T, x - x2 : x - x2 + T]
        else:
            tile = arr[y : y + T, x : x + T]
            m = holes[y : y + T, x : x + T]
            th, tw = m.shape
            if (th, tw) != (T, T):  # a small picture: pad to the window
                tile = np.pad(tile, ((0, T - th), (0, T - tw), (0, 0)), mode="reflect")
                m = np.pad(m, ((0, T - th), (0, T - tw)))
            out = engine.fill(tile, m).astype(np.float32)[:th, :tw]
        th, tw = out.shape[:2]
        win = np.outer(np.hanning(th + 2)[1:-1], np.hanning(tw + 2)[1:-1]).astype(np.float32) + 1e-3
        acc[y : y + th, x : x + tw] += out * win[..., None]
        wsum[y : y + th, x : x + tw] += win
        if progress:
            progress(i + 1, len(jobs))
    filled = np.where(wsum[..., None] > 0, acc / np.maximum(wsum, 1e-6)[..., None], arr)
    return np.where(holes[..., None], np.clip(np.rint(filled), 0, 255), arr).astype(np.uint8)


def reduce_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """`mask` reduced `factor` times: a reduced pixel is set when any of its pixels is."""
    h, w = mask.shape
    lw, lh = -(-w // factor), -(-h // factor)
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((lw, lh), Image.BOX)) > 0


def reduce(arr: np.ndarray, holes: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """`arr` (uint8) and `holes` reduced `factor` times: (uint8 picture, mask of the
    reduced pixels that touch a hole)."""
    h, w = holes.shape
    lw, lh = -(-w // factor), -(-h // factor)
    lo = np.asarray(Image.fromarray(arr).resize((lw, lh), Image.LANCZOS))
    return lo, reduce_mask(holes, factor)


def enlarge_holes(lo: np.ndarray, need: np.ndarray, engine: Engine, factor: int = 4, progress=None,
                  stop=None, stats: dict | None = None) -> np.ndarray:
    """`lo` (uint8, reduced `factor` times) enlarged `factor` times: by Real-ESRGAN in the
    tiles that touch `need`, bicubic elsewhere. factor 2 or 4 (the network enlarges 4x;
    for 2 its output is halved). Returns float32 0..1, (factor*lh, factor*lw, 3)."""
    lh, lw = need.shape
    lof = lo.astype(np.float32) / 255.0
    hi = np.asarray(Image.fromarray(lo).resize((factor * lw, factor * lh), Image.BICUBIC), dtype=np.float32) / 255.0
    hi = hi.copy()
    tiles = [(y, x) for y in range(0, lh, TILE) for x in range(0, lw, TILE)
             if need[y : y + TILE, x : x + TILE].any()]
    if stats is not None:
        stats["tiles"] = len(tiles)
    padded = np.pad(lof, ((PAD, PAD + TILE), (PAD, PAD + TILE), (0, 0)), mode="reflect")
    side = TILE + 2 * PAD
    step = max(1, engine.sr_batch)
    for k in range(0, len(tiles), step):
        _check(stop)
        group = tiles[k : k + step]
        bigs = engine.enlarge4_many([padded[y : y + side, x : x + side] for y, x in group])
        for (y, x), big in zip(group, bigs):
            th, tw = min(TILE, lh - y), min(TILE, lw - x)
            big = big[4 * PAD : 4 * (PAD + th), 4 * PAD : 4 * (PAD + tw)]
            if factor != 4:
                q = 4 // factor  # 4x output -> factor x: average q x q pixels
                big = big.reshape(th * factor, q, tw * factor, q, 3).mean(axis=(1, 3))
            hi[factor * y : factor * (y + th), factor * x : factor * (x + tw)] = big
        if progress:
            progress(min(k + step, len(tiles)), len(tiles))
    return hi


def sharpen_holes(arr: np.ndarray, holes: np.ndarray, engine: Engine, progress=None, factor: int = 4,
                  stop=None, stats: dict | None = None) -> np.ndarray:
    """Holes of `arr` sharpened: the photo is reduced `factor` times (the holes carry no
    more detail than that) and enlarged again by Real-ESRGAN where holes are."""
    h, w = holes.shape
    lo, need = reduce(arr, holes, factor)
    hi = enlarge_holes(lo, need, engine, factor, progress, stop, stats)[:h, :w] * 255.0
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


DEBLOCK_TILE = 512
DEBLOCK_PAD = 16


def deblock(arr: np.ndarray, engine: Engine, progress=None, stop=None, stats: dict | None = None) -> np.ndarray:
    """Remove JPEG blocking and ringing from the whole picture (FBCNN), tile by tile."""
    s = engine.deblock_s
    name = s.get_inputs()[0].name
    h, w = arr.shape[:2]
    x = arr.astype(np.float32) / 255.0
    out = np.empty_like(x)
    p = DEBLOCK_PAD
    padded = np.pad(x, ((p, p + DEBLOCK_TILE), (p, p + DEBLOCK_TILE), (0, 0)), mode="reflect")
    tiles = [(y, x0) for y in range(0, h, DEBLOCK_TILE) for x0 in range(0, w, DEBLOCK_TILE)]
    if stats is not None:
        stats["tiles"] = len(tiles)
    for i, (y, x0) in enumerate(tiles):
        _check(stop)
        th, tw = min(DEBLOCK_TILE, h - y), min(DEBLOCK_TILE, w - x0)
        ph, pw = -(-(th + 2 * p) // 8) * 8, -(-(tw + 2 * p) // 8) * 8  # the network halves 3 times
        tile = padded[y : y + ph, x0 : x0 + pw]
        inp = np.ascontiguousarray(tile.transpose(2, 0, 1)[None], dtype=np.float32)
        res = [r for r in s.run(None, {name: inp}) if getattr(r, "ndim", 0) == 4][0][0].transpose(1, 2, 0)
        out[y : y + th, x0 : x0 + tw] = res[p : p + th, p : p + tw]
        if progress:
            progress(i + 1, len(tiles))
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def _took(t0: float) -> str:
    s = time.perf_counter() - t0
    return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"


def enhance_steps(arr: np.ndarray, holes: np.ndarray, unknown: np.ndarray, engine: Engine, steps,
                  burst=None, progress=None, quality: str | None = None, stop=None,
                  prompt: str | None = None) -> tuple[np.ndarray, list[str]]:
    """The enhanced copy, step by step. Returns (rgb, what was done, with the time it took).

    burst: callable(rgb, holes) -> (rgb, filled mask) giving real pixels from other shots.
    progress(done, total, what) reports the tiles or windows of the slow steps.
    quality: diffusion preset (see diffusion.PRESETS); prompt: what the scene shows, for
    the diffusion (None: a generic prompt). Raises Stopped when `stop` is set."""
    steps = set(steps)
    done: list[str] = []
    real = arr.copy()
    unknown = unknown & holes

    def report(what):
        return (lambda i, n: progress(i, n, what)) if progress else None

    if "burst" in steps and burst is not None and holes.any():
        t0 = time.perf_counter()
        arr, filled = burst(arr, holes)
        if filled.any():
            done.append(f"similar shots: {100 * filled.sum() / max(1, holes.sum()):.0f} % of the holes ({_took(t0)})")
            real = np.where(filled[..., None], arr, real)
            holes = holes & ~filled
            unknown = unknown & ~filled
    _check(stop)
    if "deblock" in steps and engine.deblock_s is not None:
        t0, st = time.perf_counter(), {}
        arr = deblock(arr, engine, report("JPEG clean-up tile"), stop, st)
        real = arr.copy()  # the cleaned picture is the base from here on
        done.append(f"JPEG blocks and noise removed ({st.get('tiles', 0)} tiles, {_took(t0)})")
    if holes.any():
        before = arr
        if "diffusion" in steps and engine.sd is not None:
            from .diffusion import repaint_holes

            t0 = time.perf_counter()
            if unknown.any() and engine.inpaint_s is not None:  # a start for holes the thumbnail left blank
                arr = inpaint_unknown(arr, unknown, engine)
            arr, what = repaint_holes(arr, holes, unknown, engine, quality, report("Stable Diffusion window"), stop,
                                      prompt)
            arr = keep_block_means(arr, before, holes, holes & ~unknown)
            done.append(f"holes repainted (Stable Diffusion, follows the thumbnail; {what}, {_took(t0)})")
        elif "ai" in steps and engine.inpaint_s is not None:
            # healing: every hole filled from the real pixels around it at full size; where
            # the thumbnail knows a hole's block averages, they are given back afterwards
            t0, st = time.perf_counter(), {}
            arr = heal_holes(arr, holes, engine, report("LaMa window"), stop, st)
            arr = keep_block_means(arr, before, holes, holes & ~unknown)
            done.append(f"holes healed from the picture around them (LaMa, {st.get('tiles', 0)} windows, {_took(t0)})")
        elif "ai" in steps and engine.sr is not None:
            if engine.sr is not None:
                t0, st = time.perf_counter(), {}
                filled_arr = arr
                arr = sharpen_holes(arr, holes, engine, report("Real-ESRGAN tile"), stop=stop, stats=st)
                arr = keep_block_means(arr, filled_arr, holes, holes & ~unknown)
                done.append(f"holes sharpened (Real-ESRGAN, {st.get('tiles', 0)} tiles, {_took(t0)})")
        arr = np.where(holes[..., None], arr, real)
    _check(stop)
    if "faces" in steps and engine.face_s is not None:
        from .faces import restore_faces

        t0 = time.perf_counter()
        arr, n = restore_faces(arr, engine)
        if n:
            done.append(f"{n} face(s) restored (GFPGAN, {_took(t0)})")
    return arr, done


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


def photo_masks(data: bytes, holes, unknown) -> tuple[np.ndarray, np.ndarray]:
    """Pixel masks (holes, unknown holes) of a repaired JPEG from its MCU ranges."""
    geo = mcu_geometry(data)
    hm = mcu_mask(holes, geo) if holes else np.zeros((geo[4], geo[3]), dtype=bool)
    um = mcu_mask(unknown, geo) if unknown else np.zeros_like(hm)
    return hm, um


def save_jpeg(arr: np.ndarray, exif: bytes | None) -> bytes:
    buf = io.BytesIO()
    kw = {"quality": 95, "subsampling": 0}
    if exif:
        kw["exif"] = exif
    Image.fromarray(arr).save(buf, "JPEG", **kw)
    return buf.getvalue()


def enhance_file(data: bytes, holes, unknown, engine: Engine, progress=None) -> bytes | None:
    """JPEG bytes of the AI copy of a repaired file, or None when it has no holes.
    holes/unknown: MCU ranges. EXIF is kept."""
    if not holes:
        return None
    hm, um = photo_masks(data, holes, unknown)
    img = Image.open(io.BytesIO(data))
    exif = img.info.get("exif")
    out = enhance(img, hm, um, engine, progress)
    buf = io.BytesIO()
    kw = {"quality": 95, "subsampling": 0}
    if exif:
        kw["exif"] = exif
    out.save(buf, "JPEG", **kw)
    return buf.getvalue()
