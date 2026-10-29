"""The enhancement phase of a run: after all photos are repaired, an enhanced copy
("<photo>_enhanced.jpg") of every repaired photo is made, one photo at a time, with
the steps chosen:

  burst      holes filled with real pixels from a neighbouring shot of the scene
  deblock    JPEG blocks and noise removed from the whole picture (FBCNN)
  ai         unknown holes filled (LaMa), holes sharpened (Real-ESRGAN)
  diffusion  holes repainted by Stable Diffusion (instead of "ai"; graphics card)
  faces      faces restored (GFPGAN)

The faithful repair is never touched; everything here goes into the extra copy.
"""

from __future__ import annotations

import io
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from ..report.report import FileReport

STEPS = ("burst", "deblock", "ai", "diffusion", "faces")
STEP_NAMES = {
    "burst": "similar shots",
    "deblock": "JPEG clean-up",
    "ai": "AI sharpen and fill",
    "diffusion": "maximum quality (diffusion)",
    "faces": "faces",
}
NETWORK_STEPS = {"deblock", "ai", "diffusion", "faces"}


def enhanced_name(output_name: str) -> str:
    """'sub/x.jpg' -> 'sub/x_enhanced.jpg' (separators kept as given)."""
    return os.path.splitext(output_name)[0] + "_enhanced.jpg"


ai_name = enhanced_name  # older name


class _Photos:
    """Repaired photos of the set as burst donors, loaded on demand (a few kept)."""

    def __init__(self, items: list[tuple[FileReport, Path | None]], out: Path, keep: int = 6):
        from .burst import taken_at

        self.out = out
        self.items = []
        for rep, src in items:
            if not rep.output_name or rep.fallback is not None:
                continue
            path = out / rep.output_name
            when = taken_at(src if src is not None else path) or taken_at(path)
            self.items.append((rep, path, when))
        self.cache: OrderedDict = OrderedDict()
        self.keep = keep

    def shot(self, i: int):
        from .burst import Shot
        from .enhance import photo_masks

        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        rep, path, when = self.items[i]
        data = path.read_bytes()
        rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        holes, unknown = photo_masks(data, rep.holes, rep.unknown)
        s = Shot(rgb, ~holes, ~unknown, when, Path(rep.output_name).name)
        self.cache[i] = s
        if len(self.cache) > self.keep:
            self.cache.popitem(last=False)
        return s

    def burst_for(self, i: int, say):
        """callable(rgb, holes) -> (rgb, filled) for photo i, or None without neighbours."""
        from .burst import MAX_GAP_S

        _rep, _path, when = self.items[i]
        if when is None:
            return None
        near = [j for j, (_r, _p, w) in enumerate(self.items)
                if j != i and w is not None and abs((w - when).total_seconds()) <= MAX_GAP_S]
        if not near:
            return None

        def run(rgb, holes):
            from .burst import Shot, burst_fill

            me = self.shot(i)
            target = Shot(rgb, me.real, me.known, me.when, me.name)
            return burst_fill(target, [self.shot(j) for j in near], holes, say)

        return run


def run_extras(items: list[tuple[FileReport, Path | None]], out: Path, steps, say=None, stop=None,
               on_done=None, engine=None) -> str | None:
    """Make the enhanced copy of every repaired photo that needs one.
    items: (report, original path). Returns the device the networks ran on, "" when
    no network was needed, or None when nothing could be done (reasons go to say)."""
    say = say or (lambda s: None)
    steps = [s for s in STEPS if s in set(steps)]
    if "diffusion" in steps and "ai" in steps:
        steps.remove("ai")  # diffusion does the holes
    if not steps:
        return None
    whole_picture = bool({"deblock", "faces"} & set(steps))
    todo = [i for i, (rep, _src) in enumerate(items)
            if rep.output_name and rep.fallback is None and (rep.holes or (whole_picture and rep.findings))]
    if not todo:
        return None
    if "burst" in steps:
        try:
            import cv2  # noqa: F401
        except Exception:
            say("Similar shots: OpenCV is not available here")
            steps.remove("burst")
    nets = [s for s in steps if s in NETWORK_STEPS]
    if nets and engine is None:
        from . import models
        from .enhance import Engine

        try:
            import onnxruntime  # noqa: F401
        except Exception as e:  # not bundled / not installed
            say(f"AI steps are not available here ({type(e).__name__}: {e})")
            nets = []
        if nets:
            miss = models.missing(nets)
            if miss:
                mb = sum(m.size_mb for m in miss)
                say(f"Downloading the AI models (about {mb} MB, only the first time)…")

            def progress(done, total, model):
                say(f"Downloading {model.name}: {100 * done // max(1, total)} %")

            try:
                engine = Engine.load(progress, stop, nets)
            except InterruptedError:
                return None
            except Exception as e:
                say(f"AI models could not be loaded: {e}")
                nets = []
        steps = [s for s in steps if s not in NETWORK_STEPS or s in nets]
    if not steps:
        return None
    if engine is None:
        from .enhance import Engine

        engine = Engine()
    from .enhance import enhance_steps, photo_masks, save_jpeg

    photos = _Photos(items, out)
    index = {id(rep): k for k, (rep, _p, _w) in enumerate(photos.items)}
    device = engine.device if nets else ""
    for n, i in enumerate(todo, 1):
        if stop is not None and stop.is_set():
            break
        rep = items[i][0]
        k = index.get(id(rep))
        if k is None:
            continue
        say(f"Enhanced copy {n}/{len(todo)}" + (f" on {device}" if device else "") + f": {Path(rep.output_name).name}")
        src = out / rep.output_name
        try:
            data = src.read_bytes()
            img = Image.open(io.BytesIO(data))
            exif = img.info.get("exif")
            arr = np.asarray(img.convert("RGB"))
            holes, unknown = photo_masks(data, rep.holes, rep.unknown)
            burst = photos.burst_for(k, say) if "burst" in steps else None
            new, done = enhance_steps(arr, holes, unknown, engine, steps, burst)
        except Exception as e:  # one photo must not stop the rest
            say(f"Enhanced copy of {rep.name} failed: {type(e).__name__}: {e}")
            continue
        if not done:
            continue
        name = enhanced_name(rep.output_name)
        (out / name).write_bytes(save_jpeg(new, exif))
        rep.ai_output = name
        rep.enhanced = done
        if on_done:
            on_done(rep)
    return device


def run_ai(reports: list[FileReport], out: Path, say=None, stop=None, on_done=None, engine=None) -> str | None:
    """AI sharpen and fill only (the first version of the enhanced copy)."""
    return run_extras([(r, None) for r in reports], out, ("ai",), say, stop, on_done, engine)
