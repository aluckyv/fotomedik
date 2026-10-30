"""The enhancement phase of a run: an enhanced copy ("<photo>_enhanced.jpg") of every
repaired photo is made, one photo at a time, with the steps chosen:

  burst      holes filled with real pixels from a neighbouring shot of the scene
  deblock    JPEG blocks and noise removed from the whole picture (FBCNN)
  ai         holes healed from the picture around them (LaMa, full size)
  diffusion  holes repainted by Stable Diffusion (instead of "ai"; graphics card)
  faces      faces restored (GFPGAN)

The copies are made on a thread of their own while the repairs are still running
(`Enhancer`): the networks load while the first photos are repaired and, on a
graphics card, the copies follow the repairs closely. A photo whose holes "burst"
could fill waits until the other shots of its scene are repaired. On the processor
alone the copies wait for the repairs to finish, so the two do not fight for the cores.

The networks stay loaded between runs (`engine_for`): loading them takes long.
The faithful repair is never touched; everything here goes into the extra copy.
"""

from __future__ import annotations

import io
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from .. import errlog
from ..report.report import FileReport

STEPS = ("burst", "deblock", "ai", "diffusion", "faces")
STEP_NAMES = {
    "burst": "similar shots",
    "deblock": "JPEG clean-up",
    "ai": "AI healing",
    "diffusion": "maximum quality (diffusion)",
    "faces": "faces",
}
NETWORK_STEPS = {"deblock", "ai", "diffusion", "faces"}
NEAR_FILES = 10  # photos this close in the list may be the same scene when their time is unknown
PROGRESS_EVERY = 1.0  # seconds between progress lines of a slow step
RESTARTS = 3  # times the enhanced copies start again after an unexpected error


def enhanced_name(output_name: str) -> str:
    """'sub/x.jpg' -> 'sub/x_enhanced.jpg' (separators kept as given)."""
    return os.path.splitext(output_name)[0] + "_enhanced.jpg"


ai_name = enhanced_name  # older name


# ----------------------------------------------------------------------------- engine


_ENGINE = None  # the Engine of this process, kept between runs
_ENGINE_LOCK = threading.Lock()


def engine_for(nets, say=None, stop=None, half: bool = True, out: Path | None = None):
    """The shared engine with the networks of `nets` loaded (downloaded first when
    needed), or None when they cannot be used here (the reason goes to say, unless
    the run was stopped)."""
    global _ENGINE
    say = say or (lambda s: None)
    from . import runtime

    def rt_progress(done, total, what):
        say(f"Downloading {what}: {100 * done // max(1, total)} %")

    try:
        if runtime.load(say, rt_progress, stop) is None:
            return None
    except Exception as e:  # not bundled / not reachable
        errlog.log_error(out, "starting the AI (onnxruntime)", e)
        say(f"AI steps are not available here ({type(e).__name__}: {e})")
        return None
    from . import models
    from .enhance import Engine

    with _ENGINE_LOCK:
        eng = _ENGINE if _ENGINE is not None else Engine()
        todo = [s for s in nets if not eng.has(s)]
        if todo:
            miss = models.missing(todo)
            if miss:
                mb = sum(m.size_mb for m in miss)
                say(f"Downloading the AI models (about {mb} MB, only the first time)…")

            def progress(done, total, model):
                say(f"Downloading {model.name}: {100 * done // max(1, total)} %")

            try:
                eng.ensure(todo, progress, stop, half, say)
            except InterruptedError:
                return None
            except Exception as e:
                errlog.log_error(out, f"loading the AI models ({', '.join(todo)})", e)
                say(f"AI models could not be loaded ({type(e).__name__}); details in {errlog.LOG_NAME}")
                return None
        _ENGINE = eng
    return eng


def release_engine() -> None:
    """Forget the loaded networks (frees their memory)."""
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = None


# ----------------------------------------------------------------------------- donors


class _Photos:
    """Repaired photos of the set as burst donors, loaded on demand (a few kept)."""

    def __init__(self, out: Path, keep: int = 6):
        self.out = out
        self.items: dict[int, tuple] = {}  # index -> (report, path, taken_at)
        self.cache: OrderedDict = OrderedDict()
        self.keep = keep

    def add(self, i: int, rep: FileReport, src: Path | None) -> None:
        from .burst import taken_at

        if not rep.output_name or rep.fallback is not None:
            return
        path = self.out / rep.output_name
        when = (taken_at(src) if src is not None else None) or taken_at(path)
        self.items[i] = (rep, path, when, src)

    def shot(self, i: int):
        from .burst import Shot
        from .enhance import photo_masks

        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        rep, path, when, _src = self.items[i]
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

        if i not in self.items:
            return None
        _rep, _path, when, src = self.items[i]
        if when is None:
            return None
        near = [j for j, (_r, _p, w, s) in self.items.items()
                if j != i and w is not None and abs((w - when).total_seconds()) <= MAX_GAP_S
                and not (src is not None and s == src)]  # not the same photo repaired another way
        if not near:
            return None

        def run(rgb, holes):
            from .burst import Shot, burst_fill

            me = self.shot(i)
            target = Shot(rgb, me.real, me.known, me.when, me.name)
            return burst_fill(target, [self.shot(j) for j in near], holes, say)

        return run


# ----------------------------------------------------------------------------- the phase


class Enhancer:
    """Makes the enhanced copies on a thread of its own while the photos are repaired.

    sources: the damaged files in list order (None where unknown). add(i, report) as
    photo i is repaired, close() when all are, finish() to wait for the last copy.
    on_done(report) is called (from the thread) as each copy is written."""

    def __init__(self, sources, out: Path, steps, say=None, stop=None, on_done=None, engine=None,
                 quality: str | None = None, half: bool = True, prompt: str | None = None):
        self.sources = list(sources)
        self.out = out
        user_say = say or (lambda s: None)

        def say_and_log(text: str) -> None:
            errlog.log_line(out, text)
            user_say(text)

        self.say = say_and_log
        self.stop = stop
        self.on_done = on_done
        self.engine = engine
        self.quality = quality
        self.half = half
        self.prompt = prompt
        steps = [s for s in STEPS if s in set(steps)]
        if "diffusion" in steps and "ai" in steps:
            steps.remove("ai")  # diffusion does the holes
        self.steps = steps
        self.device: str | None = None
        self.made = 0
        self.failed = 0
        self.cond = threading.Condition()
        self.repaired: dict[int, FileReport] = {}
        self.closed = False
        self.whens = None
        self.thread = threading.Thread(target=self._run, name="fotomedik-enhance", daemon=True)

    # -- the repairing side
    def start(self) -> None:
        if self.steps:
            self.thread.start()

    def add(self, i: int, rep: FileReport) -> None:
        with self.cond:
            self.repaired[i] = rep
            self.cond.notify_all()

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify_all()

    def finish(self) -> str | None:
        """Wait for the copies. Returns the device the networks ran on, "" when no
        network was needed, None when nothing could be done."""
        self.close()
        if self.thread.ident is not None:  # started
            self.thread.join()
        return self.device

    # -- the enhancing side
    def _stopped(self) -> bool:
        return self.stop is not None and self.stop.is_set()

    def _run(self) -> None:
        for _attempt in range(RESTARTS + 1):
            try:
                self._work()
                return
            except Exception as e:  # the repairs must not suffer from this; the copies go on
                errlog.log_error(self.out, "making the enhanced copies", e)
                self.say(f"Enhanced copies: an error, written to {errlog.LOG_NAME}; going on with the next photo")
        self.say(f"Enhanced copies stopped after repeated errors (see {errlog.LOG_NAME})")

    def _work(self) -> None:
        from ..runner import is_lost
        from .enhance import Engine

        errlog.log_line(self.out, "--- " + errlog.system_info() + f"; steps: {', '.join(self.steps)}")

        steps = list(self.steps)
        if "burst" in steps:
            try:
                import cv2  # noqa: F401

                from .burst import taken_at

                self.whens = [taken_at(s) if s is not None else None for s in self.sources]
            except Exception:
                self.say("Similar shots: OpenCV is not available here")
                steps.remove("burst")
        nets = [s for s in steps if s in NETWORK_STEPS]
        engine = self.engine
        in_hand = engine is not None or not nets  # the networks are loaded (or none is needed)
        wait_all = False
        if in_hand:
            engine = engine or Engine()
        whole_picture = bool({"deblock", "faces"} & set(steps))
        if not hasattr(self, "_state"):  # kept when the work starts again after an error
            self._state = (_Photos(self.out), set(), [])
        photos, seen, pending = self._state
        while True:
            with self.cond:
                while True:
                    if self._stopped():
                        return
                    for i in sorted(self.repaired):
                        if i in seen:
                            continue
                        seen.add(i)
                        rep = self.repaired[i]
                        photos.add(i, rep, self.sources[i])
                        if (rep.output_name and rep.fallback is None and not is_lost(rep)
                                and (rep.holes or (whole_picture and rep.findings))):
                            pending.append(i)
                    ready = None
                    if pending and not in_hand:
                        break  # the first photo that needs a copy: load the networks now
                    if self.closed or not wait_all:
                        ready = next((i for i in pending if self.closed or self._eligible(i)), None)
                    if ready is not None:
                        pending.remove(ready)
                        break
                    if self.closed:
                        return
                    self.cond.wait(0.5)
            if not in_hand:  # loading overlaps with the repairs still running
                engine = engine_for(nets, self.say, self.stop, self.half, self.out)
                if self._stopped():
                    return
                if engine is None:
                    nets = []
                    steps = [s for s in steps if s not in NETWORK_STEPS]
                    if not steps:
                        return
                    engine = Engine()
                while engine.notes:  # networks the graphics card refused, fp16 copies dropped
                    self.say(engine.notes.pop(0))
                try:
                    import onnxruntime as _ort

                    from . import runtime as _rt

                    errlog.log_line(self.out, f"AI runtime {_ort.__version__} ({_rt.kind or 'installed'}), providers "
                                    f"{_ort.get_available_providers()}; networks on: {engine.devices}")
                except Exception:
                    pass
                in_hand = True
                wait_all = bool(nets) and engine.device == "CPU"  # on the processor: after the repairs
                continue
            self.device = engine.device if nets else ""
            self._enhance(ready, engine, steps, photos)

    def _eligible(self, i: int) -> bool:
        """Whether every other shot of photo i's scene (a possible burst donor) is repaired."""
        if self.whens is None:
            return True
        from .burst import MAX_GAP_S

        wi = self.whens[i]
        for j in range(len(self.sources)):
            if j == i or j in self.repaired:
                continue
            wj = self.whens[j]
            if wi is None or wj is None:
                if abs(j - i) <= NEAR_FILES:
                    return False
            elif abs((wj - wi).total_seconds()) <= MAX_GAP_S:
                return False
        return True

    def _enhance(self, i: int, engine, steps, photos: _Photos) -> None:
        from .enhance import Stopped, enhance_steps, photo_masks, save_jpeg, thumbnail_picture

        rep = self.repaired[i]
        name = Path(rep.output_name).name
        where = f" on {self.device}" if self.device else ""
        self.say(f"Enhanced copy of {name}{where}…")
        last = [time.monotonic()]

        def progress(k, n, what):
            now = time.monotonic()
            if k >= n or now - last[0] >= PROGRESS_EVERY:
                last[0] = now
                self.say(f"Enhanced copy of {name}: {what} {k}/{n}")

        src = self.out / rep.output_name
        try:
            data = src.read_bytes()
            img = Image.open(io.BytesIO(data))
            exif = img.info.get("exif")
            arr = np.asarray(img.convert("RGB"))
            holes, unknown = photo_masks(data, rep.holes, rep.unknown)
            burst = photos.burst_for(i, self.say) if "burst" in steps else None
            new, done = enhance_steps(arr, holes, unknown, engine, steps, burst, progress, self.quality, self.stop,
                                      self.prompt, thumbnail_picture(data, arr.shape))
        except Stopped:
            return
        except Exception as e:  # one photo must not stop the rest
            errlog.log_error(self.out, f"enhanced copy of {rep.name} ({', '.join(steps)})", e)
            self.failed += 1
            self.say(f"Enhanced copy of {rep.name} failed ({type(e).__name__}); details in {errlog.LOG_NAME}")
            return
        finally:
            while engine.notes:  # e.g. a network the graphics card failed on, moved to the processor
                self.say(engine.notes.pop(0))
        if not done:
            return
        out_name = enhanced_name(rep.output_name)
        (self.out / out_name).write_bytes(save_jpeg(new, exif))
        rep.ai_output = out_name
        rep.enhanced = done
        self.made += 1
        self.say(f"Enhanced copy of {name} done ({self.made} so far): " + "; ".join(done))
        if self.on_done:
            self.on_done(rep)


def run_extras(items: list[tuple[FileReport, Path | None]], out: Path, steps, say=None, stop=None,
               on_done=None, engine=None, quality: str | None = None, prompt: str | None = None) -> str | None:
    """Make the enhanced copy of every repaired photo that needs one, after the fact.
    items: (report, original path). Returns the device the networks ran on, "" when
    no network was needed, or None when nothing could be done (reasons go to say)."""
    enh = Enhancer([src for _rep, src in items], out, steps, say, stop, on_done, engine, quality, prompt=prompt)
    if not enh.steps:
        return None
    enh.start()
    for i, (rep, _src) in enumerate(items):
        enh.add(i, rep)
    return enh.finish()


def run_ai(reports: list[FileReport], out: Path, say=None, stop=None, on_done=None, engine=None) -> str | None:
    """AI sharpen and fill only (the first version of the enhanced copy)."""
    return run_extras([(r, None) for r in reports], out, ("ai",), say, stop, on_done, engine)
