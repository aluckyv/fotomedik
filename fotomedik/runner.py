"""Repair a set of photos and report each one as soon as it is done (used by the app).

Same work as `fotomedik batch`: healthy photos in the set become reference headers,
files run in parallel processes (biggest first) and use all cores inside a file.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import FIRST_COMPLETED, wait
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
        on_result=None, on_status=None, stop: threading.Event | None = None) -> list[Result]:
    """Repair everything under `paths` into `out`. Originals are never written.
    on_result(Result) is called as each photo finishes; on_status(str) for phases."""
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
            options["ref_candidates"] = _learn_candidates(files, say, jobs, workers)
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
            r = Result(files[i][0], output, rep, secs)
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
                    except Exception as e:  # one bad file must not stop the batch
                        rep, secs = failed(i, e)
                    done(i, rep, secs)
    return results
