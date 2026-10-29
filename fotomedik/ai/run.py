"""The AI phase of a run: after all photos are repaired, one model session (on the
graphics card when there is one) makes the "_ai" copies one photo at a time."""

from __future__ import annotations

from pathlib import Path

from ..report.report import FileReport


def ai_name(output_name: str) -> str:
    p = Path(output_name)
    return str(p.with_name(p.stem + "_ai.jpg"))


def run_ai(reports: list[FileReport], out: Path, say=None, stop=None, on_done=None, engine=None) -> str | None:
    """Make the AI copy of every repaired photo with holes. Returns the device used
    ("GPU (DirectML)", "CPU"), or None when AI could not run (the reason goes to say)."""
    say = say or (lambda s: None)
    todo = [r for r in reports if r.output_name and r.holes and r.fallback is None]
    if not todo:
        return None
    if engine is None:
        from . import models
        from .enhance import Engine

        try:
            import onnxruntime  # noqa: F401
        except Exception as e:  # not bundled / not installed
            say(f"AI enhancement is not available here ({type(e).__name__}: {e})")
            return None
        if models.missing():
            mb = sum(m.size_mb for m in models.missing())
            say(f"Downloading the AI models (about {mb} MB, only the first time)…")

        def progress(done, total, model):
            say(f"Downloading {model.name}: {100 * done // max(1, total)} %")

        try:
            engine = Engine.load(progress, stop)
        except InterruptedError:
            return None
        except Exception as e:
            say(f"AI models could not be loaded: {e}")
            return None
    from .enhance import enhance_file

    for i, rep in enumerate(todo, 1):
        if stop is not None and stop.is_set():
            break
        say(f"AI enhancing {i}/{len(todo)} on {engine.device}: {Path(rep.output_name).name}")
        src = out / rep.output_name
        try:
            new = enhance_file(src.read_bytes(), rep.holes, rep.unknown, engine)
        except Exception as e:  # one photo must not stop the rest
            say(f"AI copy of {rep.name} failed: {type(e).__name__}: {e}")
            continue
        if new is None:
            continue
        name = ai_name(rep.output_name)
        (out / name).write_bytes(new)
        rep.ai_output = name
        if on_done:
            on_done(rep)
    return engine.device
