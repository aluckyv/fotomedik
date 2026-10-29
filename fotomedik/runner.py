"""Repair a set of photos and report each one as soon as it is done (used by the app).

Same work as `fotomedik batch`: healthy photos in the set become reference headers,
files run in parallel processes (biggest first) and use all cores inside a file.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import FIRST_COMPLETED, CancelledError, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

from .cli import IMAGE_SUFFIXES, Workers, _counted, _learn_candidates, _repair_file
from .report.report import FileReport


@dataclass
class Result:
    source: Path  # the original photo
    output: Path | None  # repaired file (None: nothing could be saved)
    report: FileReport
    seconds: float
    options: dict | None = None  # what the run used (learned references included), for replays

    @property
    def ai_output(self) -> Path | None:
        if not self.report.ai_output or self.output is None:
            return None
        return self.output.with_name(Path(self.report.ai_output).name)


LOST = ("No photo in the file", "Not repairable", "Only the thumbnail", "Mostly lost")
MOSTLY_LOST_PCT = 30.0


def verdict(rep: FileReport, has_output: bool = True) -> str:
    """The one-line result shown in the app."""
    kinds = {f["kind"] for f in rep.findings}
    pct = rep.recovered_pct
    if not has_output or not rep.output_name:
        return "No photo in the file" if "overwritten" in kinds else "Not repairable"
    if rep.fallback == "thumbnail":
        return "Only the thumbnail"
    if not rep.findings:
        return "Healthy"
    if pct is not None and pct >= 90:
        return "Repaired"
    if pct is not None and pct >= MOSTLY_LOST_PCT:
        return "Partly recovered"
    return "Mostly lost"


def is_lost(rep: FileReport) -> bool:
    """Too little of the photo survived for enhancement to be worth it."""
    return verdict(rep) in LOST


@dataclass
class Step:
    title: str
    explanation: str
    detail: str
    data: bytes | None  # the photo after this step (None: nothing to show)


def replay(r: Result) -> list[Step]:
    """The repair of one photo again, step by step: the photo after every change."""
    from .cli import repair_bytes
    from .report.report import STRINGS

    names = STRINGS["en"]["recipe"]
    original = r.source.read_bytes()
    opts = dict(r.options or {})
    opts["snapshots"] = True
    opts.pop("write_reports", None)
    ctx, _rep = repair_bytes(original, r.source.name, opts)
    steps = [Step("As it came from the card", STEP_HELP["original"], _findings_text(r.report), original)]
    for name, message, data in ctx.memo.get("snapshots", []):
        steps.append(Step(names.get(name, name).capitalize(), STEP_HELP.get(name, ""), message, data))
    notes = [e for e in ctx.log if e.extra.get("note")]
    if notes:
        steps.append(Step("Also tried", STEP_HELP["notes"], "\n".join(f"{e.recipe}: {e.message}" for e in notes), None))
    if r.ai_output is not None and r.ai_output.exists():
        steps.append(Step("Enhanced copy", STEP_HELP["enhanced"], "\n".join(r.report.enhanced),
                          r.ai_output.read_bytes()))
    return steps


def _findings_text(rep: FileReport) -> str:
    from .report.report import STRINGS

    kinds = STRINGS["en"]["kind"]
    if not rep.findings:
        return "No damage found."
    return "Damage found: " + "; ".join(kinds.get(f["kind"], f["kind"]) for f in rep.findings)


STEP_HELP = {
    "original": "The file as a normal viewer shows it. Viewers stop at the first error and paint the rest grey, "
                "or show garbage after a silent error.",
    "cardrecovery_xor": "A recovery program (CardRecovery) had scrambled the start of the file with a fixed pattern; "
                        "it is undone.",
    "header_repair": "Single flipped bits in the header (markers, lengths, tables) are put right.",
    "reference_header": "The header was beyond repair. The header of the same camera, learned from the other photos "
                        "(voted bit by bit), is put in, with this photo's size and orientation.",
    "garbage_tail": "Data after the end of the picture that belongs to another file is cut off.",
    "foreign_sectors": "Card sectors inside the picture that cannot be photo data (zeros, text, another file) are "
                       "blanked, so they count as damage instead of decoding as garbage.",
    "bitflip_repair": "Where only a few bits are wrong, every candidate flip is tried until the stretch decodes "
                      "cleanly up to its restart marker.",
    "fill_holes": "Everything that decodes is kept. After each damaged stretch the data is searched for the point "
                  "where decoding catches on again; those pieces are put back in place (following the thumbnail) "
                  "and their colour is matched to the rows around. What is still missing is filled with colours "
                  "from the embedded thumbnail.",
    "smooth_holes": "The filled holes get a soft surface instead of flat 8x8 squares. Real data is not touched.",
    "thumbnail_fallback": "Almost nothing of the picture survived, so the small embedded preview is saved instead.",
    "notes": "Repairs that were tried but did not apply.",
    "enhanced": "The separate enhanced copy (similar shots, AI steps). Detail added by AI is invented.",
}


def collect(paths: list[Path], out: Path) -> list[tuple[Path, Path]]:
    """(file, relative output path) for every photo in the given files/folders."""
    files: list[tuple[Path, Path]] = []
    out_r = out.resolve()
    for p in paths:
        if p.is_dir():
            for root, _dirs, names in os.walk(p):
                for n in sorted(names):
                    f = Path(root) / n
                    if f.suffix.lower() in IMAGE_SUFFIXES and not f.resolve().is_relative_to(out_r):
                        files.append((f, f.relative_to(p)))
        elif p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            files.append((p, Path(p.name)))
    return files


def run(paths: list[Path], out: Path, options: dict | None = None, jobs: int = 0,
        on_result=None, on_status=None, stop: threading.Event | None = None, on_ai=None) -> list[Result]:
    """Repair everything under `paths` into `out`. Originals are never written.
    on_result(Result) is called as each photo finishes; on_status(str) for phases.
    options["enhance"] (steps, see ai.run) adds an enhanced copy of every repaired photo
    after the repairs (on_ai(Result) as each is made); options["ai"] means ["ai"]."""
    out.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if p.is_dir() and p.resolve() == out.resolve():
            raise ValueError("the output folder must differ from the input folder")
    files = collect(paths, out)
    jobs = jobs or (os.cpu_count() or 1)
    options = dict(options or {})
    options.setdefault("time_budget", 5.0)
    results: list[Result] = []
    if not files:
        return results
    say = on_status or (lambda s: None)
    with Workers(jobs, len(files)) as workers:
        if len(files) > 1:
            say("Learning from the healthy photos in the set…")
            options["ref_candidates"], options["thumb_refs"] = _learn_candidates(files, say, jobs, workers)
        work = [(str(p), str(rel), str(out), options, None) for p, rel in files]
        order = sorted(range(len(work)), key=lambda i: -os.path.getsize(work[i][0]))
        say(f"Repairing {len(work)} photo(s)…")

        def failed(i, err: Exception) -> tuple[FileReport, float]:
            src = files[i][0]
            rep = FileReport(src.name, None, src.stat().st_size, 0, None,
                             findings=[{"kind": "internal_error", "severity": "high", "offset": None,
                                        "detail": f"{type(err).__name__}: {err}"}])
            return rep, 0.0

        def done(i, rep, secs):
            output = (out / rep.output_name) if rep.output_name else None
            r = Result(files[i][0], output, rep, secs, options)
            results.append(r)
            if on_result:
                on_result(r)

        if workers.ex is None:
            for i in order:
                if stop is not None and stop.is_set():
                    break
                try:
                    rep, secs = _repair_file(work[i])
                except Exception as e:  # one bad file must not stop the batch
                    rep, secs = failed(i, e)
                done(i, rep, secs)
        else:
            with workers.counter.get_lock():
                workers.counter.value = len(work)
            pending = {workers.ex.submit(_counted, _repair_file, work[i]): i for i in order}
            while pending:
                if stop is not None and stop.is_set():
                    for f in pending:
                        f.cancel()
                    break
                finished, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for f in finished:
                    i = pending.pop(f)
                    try:
                        rep, secs = f.result()
                    except (BrokenProcessPool, CancelledError):
                        # worker processes could not start: repair here, one by one
                        workers.broken()
                        try:
                            rep, secs = _repair_file(work[i])
                        except Exception as e:
                            rep, secs = failed(i, e)
                    except Exception as e:  # one bad file must not stop the batch
                        rep, secs = failed(i, e)
                    done(i, rep, secs)
    steps = list(options.get("enhance") or (["ai"] if options.get("ai") else []))
    if steps and results and not (stop is not None and stop.is_set()):
        lost = [r for r in results if is_lost(r.report)]
        if lost:
            say(f"{len(lost)} lost or mostly lost photo(s) get no enhanced copy")
        by_rep = {id(r.report): r for r in results}
        from .ai.run import run_extras

        run_extras([(r.report, r.source) for r in results], out, steps, say, stop,
                   on_done=(lambda rep: on_ai(by_rep[id(rep)])) if on_ai else None)
    return results
