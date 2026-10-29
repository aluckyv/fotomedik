"""Half-precision (fp16) copies of the fp32 networks for the graphics card.

A graphics card runs fp16 networks about twice as fast as fp32 ones. Real-ESRGAN,
FBCNN and GFPGAN come as fp32 files. The first time one of them is loaded on a
graphics card, a copy with fp16 weights is written next to it (ONNX Runtime's own
converter; inputs and outputs stay fp32, so the calling code does not change) and
tried against the original on a test picture. The copy is used only when it is both
accurate and clearly faster on this machine; otherwise it is dropped and a marker file
says not to try again. LaMa (Fourier transforms inside) and Stable Diffusion (fp16
already) are left alone.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import models

HALF = (models.SR, models.DEBLOCK, models.FACE)
TOL_MEAN = 0.01  # mean |fp16 - fp32| as a share of the fp32 output's range
TOL_MAX = 0.10  # largest |fp16 - fp32|, same share
SPEEDUP = 1.1  # the fp16 copy must be at least this much faster to be used
TEST_SIDE = 128  # test picture side where the network takes any size


def half_path(path: Path) -> Path:
    return path.with_name(path.stem + ".fp16.onnx")


def marker(path: Path, kind: str) -> Path:
    """kind "ok": the fp16 copy is in use; "bad": it was tried and dropped."""
    return path.with_name(path.stem + f".fp16.{kind}")


def convert(path: Path, out: Path) -> None:
    """Write the fp16 copy of the fp32 network at `path` to `out`."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(path))
    half = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(half, str(out))


def test_picture(model: models.Model, session) -> np.ndarray:
    """A smooth test picture (gradients and waves) in the network's input layout."""
    inp = session.get_inputs()[0]
    shape = []
    for i, d in enumerate(inp.shape):
        if isinstance(d, int) and d > 0:
            shape.append(d)
        else:
            shape.append(1 if i == 0 else 3 if i == 1 else TEST_SIDE)
    b, c, h, w = shape[:4]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / max(h, w)
    planes = [0.5 + 0.4 * np.sin(6 * xx + 2 * yy), 0.5 + 0.3 * np.cos(5 * yy + xx), 0.1 + 0.8 * xx]
    pic = np.stack([planes[k % 3] for k in range(c)])[None].repeat(b, axis=0)
    pic = np.clip(pic + np.random.default_rng(0).normal(0, 0.02, pic.shape), 0, 1).astype(np.float32)
    if model is models.FACE:  # GFPGAN takes -1..1
        pic = (pic - 0.5) / 0.5
    return np.ascontiguousarray(pic)


def _picture_output(session, x: np.ndarray) -> np.ndarray:
    name = session.get_inputs()[0].name
    outs = session.run(None, {name: x})
    return next(o for o in outs if getattr(o, "ndim", 0) == 4).astype(np.float32)


def _seconds(session, x: np.ndarray) -> float:
    """Best of two timed runs after a warm-up (the first run on a graphics card compiles)."""
    name = session.get_inputs()[0].name
    session.run(None, {name: x})
    best = float("inf")
    for _ in range(2):
        t = time.perf_counter()
        session.run(None, {name: x})
        best = min(best, time.perf_counter() - t)
    return best


def compare(ref: np.ndarray, got: np.ndarray) -> tuple[float, float]:
    """(mean, largest) |got - ref| as shares of ref's range; inf when got is not finite."""
    if not np.isfinite(got).all():
        return float("inf"), float("inf")
    rng = float(ref.max() - ref.min()) or 1.0
    d = np.abs(got - ref)
    return float(d.mean()) / rng, float(d.max()) / rng


def session(model: models.Model, path: Path, make, say=None, notes: list[str] | None = None):
    """A session for `model`: its fp16 copy when that is accurate and faster here, else
    the fp32 original. make(path) creates a session on the preferred device."""
    from .enhance import GPU_PROVIDERS

    if model not in HALF:
        return make(path)
    hp, ok, bad = half_path(path), marker(path, "ok"), marker(path, "bad")
    if ok.exists() and hp.exists():
        try:
            s = make(hp)
            if s.get_providers()[0] in GPU_PROVIDERS:
                return s
        except Exception:
            pass
        ok.unlink(missing_ok=True)
    full = make(path)
    if bad.exists() or full.get_providers()[0] not in GPU_PROVIDERS:
        return full  # tried before, or on the processor (where fp16 is no faster)
    if say:
        say(f"Preparing the fast (fp16) copy of {model.name} — once…")
    half_s = None
    try:
        convert(path, hp)
        half_s = make(hp)
        x = test_picture(model, full)
        ref, got = _picture_output(full, x), _picture_output(half_s, x)
        mean, worst = compare(ref, got)
        t32, t16 = _seconds(full, x), _seconds(half_s, x)
        accurate = mean <= TOL_MEAN and worst <= TOL_MAX and half_s.get_providers()[0] in GPU_PROVIDERS
        if accurate and t32 >= SPEEDUP * t16:
            ok.write_text(f"fp32 {t32:.3f} s, fp16 {t16:.3f} s, mean difference {mean:.4f}, largest {worst:.3f}\n",
                          encoding="utf-8")
            return half_s
        reason = (f"fp32 {t32:.3f} s, fp16 {t16:.3f} s, mean difference {mean:.4f}, largest {worst:.3f}"
                  + ("" if accurate else " (not accurate enough)"))
    except Exception as e:  # conversion or the fp16 session failed
        reason = f"{type(e).__name__}: {e}"
    del half_s
    bad.write_text(reason + "\n", encoding="utf-8")
    try:
        hp.unlink(missing_ok=True)
    except OSError:
        pass
    if notes is not None:
        notes.append(f"{model.name}: fp32 kept ({reason})")
    return full
