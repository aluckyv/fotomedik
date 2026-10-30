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


def plan(files: list[tuple[Path, Path]], options: dict) -> list[tuple[Path, Path, dict]]:
    """The repairs to run: (file, relative output path, options) per output.

    options["reference"] (a picture of the same scene, see fotomedik.reference): every
    file is repaired guided by it, into <stem>_ref<suffix> - meant for one photo.
    options["reference_dir"] (a folder of such pictures): each photo it holds a picture
    for gets a guided repair into <stem>_ref<suffix> besides the plain one."""
    from .reference import REFERENCE_SUFFIX, pair

    def guided(rel: Path, ref: Path) -> tuple[Path, dict]:
        opts = dict(options)
        opts.pop("reference_dir", None)
        opts["reference"] = str(ref)
        return rel.with_name(rel.stem + REFERENCE_SUFFIX + rel.suffix), opts

    if options.get("reference"):
        return [(p, *guided(rel, Path(options["reference"]))) for p, rel in files]
    plain = dict(options)
    plain.pop("reference_dir", None)
    items = [(p, rel, plain) for p, rel in files]
    if options.get("reference_dir"):
        refs = pair([p for p, _rel in files], Path(options["reference_dir"]))
        items += [(p, *guided(rel, refs[p])) for p, rel in files if p in refs]
    return items


def run(paths: list[Path], out: Path, options: dict | None = None, jobs: int = 0,
        on_result=None, on_status=None, stop: threading.Event | None = None, on_ai=None,
        on_ai_status=None) -> list[Result]:
    """Repair everything under `paths` into `out`. Originals are never written.
    on_result(Result) is called as each photo finishes; on_status(str) for phases.
    options["enhance"] (steps, see ai.run) adds an enhanced copy of every repaired photo,
    made while the repairs go on (on_ai(Result) as each is made, on_ai_status(str) for
    their progress; options["quality"] is the diffusion preset, options["prompt"] what the
    scene shows); options["ai"] means ["ai"]."""
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
    items = plan(files, options)
    steps = list(options.get("enhance") or (["ai"] if options.get("ai") else []))
    enhancer = None
    by_rep: dict[int, Result] = {}
    if steps:
        from .ai.run import Enhancer

        def copy_made(rep: FileReport) -> None:
            if options.get("write_reports"):  # the report was written before the copy existed
                name = Path(rep.output_name)
                rp = out / name.parent / (name.stem + ".report.json")
                if rp.exists():
                    rp.write_text(rep.to_json(), encoding="utf-8")
            if on_ai:
                on_ai(by_rep[id(rep)])

        enhancer = Enhancer([p for p, _rel, _o in items], out, steps, on_ai_status or say, stop,
                            on_done=copy_made, quality=options.get("quality"), prompt=options.get("prompt"))
        enhancer.start()
    with Workers(jobs, len(items)) as workers:
        if len(files) > 1:
            say("Learning from the healthy photos in the set…")
            learned = _learn_candidates(files, say, jobs, workers)
            for _p, _rel, opts in items:
                opts["ref_candidates"], opts["thumb_refs"] = learned
        work = [(str(p), str(rel), str(out), opts, None) for p, rel, opts in items]
        order = list(range(len(work)))  # in list order: shots of one scene finish close together
        guided = sum(1 for _p, _rel, o in items if o.get("reference"))
        say(f"Repairing {len(files)} photo(s)…" + (f" ({guided} guided by a reference picture)" if guided else ""))

        def failed(i, err: Exception) -> tuple[FileReport, float]:
            src = items[i][0]
            rep = FileReport(src.name, None, src.stat().st_size, 0, None,
                             findings=[{"kind": "internal_error", "severity": "high", "offset": None,
                                        "detail": f"{type(err).__name__}: {err}"}])
            return rep, 0.0

        def done(i, rep, secs):
            output = (out / rep.output_name) if rep.output_name else None
            r = Result(items[i][0], output, rep, secs, items[i][2])
            results.append(r)
            by_rep[id(rep)] = r
            if on_result:
                on_result(r)
            if enhancer is not None:
                enhancer.add(i, rep)

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
    if enhancer is not None:
        if results and not (stop is not None and stop.is_set()):
            lost = [r for r in results if is_lost(r.report)]
            if lost:
                say(f"{len(lost)} lost or mostly lost photo(s) get no enhanced copy")
            say("Repairs done; finishing the enhanced copies…")
        enhancer.finish()
    return results
