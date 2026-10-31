"""FotoMedik command line: scan | fix | learn | batch."""

from __future__ import annotations

import csv
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path

import click

from . import parallel
from . import recipes  # noqa: F401  (registers built-in recipes)
from .jpeg import markers as M
from .jpeg.tables import TableError, parse_sof
from .learn.profile import CameraProfile
from .recipes.base import Context
from .recipes.registry import run_pipeline
from .report.report import FileReport
from .scan.scanner import scan_bytes

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".~cr"}


def _iter_files(path: Path):
    if path.is_file():
        yield path, Path(path.name)
        return
    for root, _dirs, files in os.walk(path):
        for f in sorted(files):
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_SUFFIXES:
                yield p, p.relative_to(path)


def _dims(data: bytes) -> tuple[int | None, int | None]:
    res = M.parse_segments(data)
    if res.sof is None:
        return None, None
    try:
        fi = parse_sof(res.sof.marker, res.sof.payload)
        return fi.width, fi.height
    except TableError:
        return None, None


def repair_bytes(data: bytes, name: str, options: dict, ref: CameraProfile | None = None) -> tuple[Context, FileReport]:
    diag = scan_bytes(data, name)
    ctx = Context(data=data, name=name, ref=ref, options=dict(options))
    run_pipeline(ctx)
    w, h = _dims(ctx.data)
    rep = FileReport.from_context(ctx, findings_before=[f.to_dict() for f in diag.findings], width=w, height=h)
    rep.unknown = _unknown_holes(ctx)
    if ctx.memo.get("reference_used") and options.get("reference"):
        rep.reference = Path(str(options["reference"])).name
    if ctx.memo.get("twin_used"):
        rep.twin = ctx.memo.get("twin_name")
    if rep.recovered_pct is None:
        if ctx.data != data:  # repaired (e.g. a new header) and nothing left to fill: measure the result
            after = scan_bytes(ctx.data, name)
            rep.recovered_pct = round(100 * after.recovered_fraction, 1) if after.openable else None
        else:
            rep.recovered_pct = round(100 * diag.recovered_fraction, 1) if not diag.findings or diag.openable else None
    return ctx, rep


def _unknown_holes(ctx: Context) -> list[tuple[int, int]]:
    """MCU ranges of holes the thumbnail says nothing about (all holes without one)."""
    if not ctx.holes or ctx.fallback is not None:
        return []
    ranges = [(h.mcu_first, h.mcu_count) for h in ctx.holes]
    thumb = ctx.memo.get("thumb_base")
    if thumb is None:
        return ranges
    try:
        import numpy as np

        from .jpeg import decoder as D
        from .jpeg import markers as M
        from .jpeg.tables import FrameTables

        ft = FrameTables.from_parse(M.parse_segments(ctx.data))
        comps, per_row, _rows = D._scan_components(ft)
        c = comps[0]
        dc = thumb.dc[c.cid]
        rows_m = dc.shape[0] // c.v
        nan = np.isnan(dc[: rows_m * c.v, : per_row * c.h]).reshape(rows_m, c.v, per_row, c.h).any(axis=(1, 3))
        unknown = nan.reshape(-1)
    except Exception:
        return []
    out: list[tuple[int, int]] = []
    for first, count in ranges:
        seg = unknown[first : first + count]
        i = 0
        while i < len(seg):
            if seg[i]:
                j = i
                while j < len(seg) and seg[j]:
                    j += 1
                out.append((first + i, j - i))
                i = j
            else:
                i += 1
    return out


def _nothing_to_save(ctx: Context, rep: FileReport) -> bool:
    """No photo data was found and nothing was repaired: a copy would not open either."""
    fatal = {f["kind"] for f in rep.findings} & {"not_jpeg", "overwritten"}
    return bool(fatal) and not ctx.applied and ctx.data == ctx.original


def _write_outputs(ctx: Context, rep: FileReport, out_dir: Path, rel: Path, reports: bool = False) -> Path:
    """The repaired file (unless there is nothing to save) and, on request, its report."""
    out_name = ctx.suggested_name or rel.name
    target = out_dir / rel.parent / out_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if _nothing_to_save(ctx, rep):
        rep.output_name = None
    else:
        target.write_bytes(ctx.data)
    if reports:
        (target.parent / (Path(out_name).stem + ".report.json")).write_text(rep.to_json(), encoding="utf-8")
    return target


def _profile_of(args: tuple[str, str]) -> dict | None:
    """Worker: what a file can teach about its camera - its header as a reference profile
    (with the share of its scan that decodes; a damaged scan does not make the header less
    useful), the header of its embedded thumbnail, its table block as the file has it,
    and its header region (damaged headers still vote, see learn.consensus)."""
    path, rel = args
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    from .learn.consensus import HEAD, table_block

    out: dict = {"head": data[:HEAD], "profile": None, "health": 0.0, "thumb": None, "block": None}
    try:
        d = scan_bytes(data, Path(rel).name)
        if d.openable:
            out["profile"] = CameraProfile.from_jpeg(data, source=rel)
            out["health"] = d.recovered_fraction
            out["block"] = table_block(data)
            from .recipes.r8_thumbnail import extract_thumbnail

            blob = extract_thumbnail(data)
            if blob is not None:
                out["thumb"] = CameraProfile.from_jpeg(blob, source=f"{rel} (thumbnail)")
    except Exception:
        pass
    return out


def _repair_file(args: tuple[str, str, str, dict, CameraProfile | None]) -> tuple[FileReport, float]:
    """Worker: repair one file and write its outputs. Runs in a separate process."""
    path, rel, out, options, profile = args
    t = time.perf_counter()
    data = Path(path).read_bytes()
    relp = Path(rel)
    ctx, rep = repair_bytes(data, relp.name, options, profile)
    target = _write_outputs(ctx, rep, Path(out), relp, bool(options.get("write_reports")))
    if rep.output_name is not None:
        rep.output_name = str(target.relative_to(Path(out)))
    return rep, time.perf_counter() - t


def _worker_init(counter) -> None:
    parallel.attach_batch_counter(counter)


def _counted(fn, item):
    """Run fn in a batch worker; the shared counter of unfinished files tells every
    worker how many cores it may use inside its file (see parallel.threads)."""
    try:
        return fn(item)
    finally:
        counter = parallel._remaining
        if counter is not None:
            with counter.get_lock():
                counter.value -= 1


class Workers:
    """Worker processes shared by all phases of one run (starting them costs time).
    Every core is used: files run in parallel processes, and inside a file the work
    spreads over threads as cores free up (see parallel.threads)."""

    def __init__(self, jobs: int, n_items: int):
        self.ex = None
        self.counter = None
        if jobs > 1 and n_items > 1:
            # workers start from a clean process (forking a process that runs threads can deadlock)
            mp = multiprocessing.get_context("spawn" if sys.platform == "win32" else "forkserver")
            self.counter = mp.Value("i", n_items)
            self.ex = ProcessPoolExecutor(max_workers=min(jobs, n_items), mp_context=mp,
                                          initializer=_worker_init, initargs=(self.counter,))

    def map(self, fn, items: list, cost=None, counted: bool = False) -> list:
        """Results in input order. `cost(item)` orders the submission (biggest first) so
        long files do not end up last on an otherwise idle pool. `counted`: each finished
        item lowers the count of unfinished files that sets the threads per file."""
        if self.ex is None or len(items) <= 1:
            return [fn(it) for it in items]
        if counted:
            with self.counter.get_lock():
                self.counter.value = len(items)
        order = sorted(range(len(items)), key=(lambda i: -cost(items[i])) if cost else (lambda i: i))
        out: list = [None] * len(items)
        task = partial(_counted, fn) if counted else fn
        try:
            for i, res in zip(order, self.ex.map(task, [items[i] for i in order])):
                out[i] = res
        except BrokenProcessPool:
            # worker processes could not start (or died): finish the work in this process
            self.broken()
            return [fn(it) for it in items]
        return out

    def broken(self) -> None:
        """Stop using worker processes for the rest of the run."""
        if self.ex is not None:
            self.ex.shutdown(wait=False, cancel_futures=True)
            self.ex = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.ex is not None:
            self.ex.shutdown()


def _pool_map(fn, items: list, jobs: int, cost=None):
    """One-off parallel map (see Workers)."""
    with Workers(jobs, len(items)) as w:
        return w.map(fn, items, cost, counted=True)


def _learn_candidates(files: list[tuple[Path, Path]], echo, jobs: int = 1, workers: "Workers | None" = None
                      ) -> tuple[list[CameraProfile], list[CameraProfile]]:
    """Healthy headers in the set become reference profiles for the damaged ones."""
    items = [(str(p), str(rel)) for p, rel in files]
    found = workers.map(_profile_of, items) if workers is not None else _pool_map(_profile_of, items, jobs)
    # Identical headers are counted: a header shared by many photos of the set is certainly
    # the camera's, while a lone variant may carry a flipped bit in its tables.
    def ranked(pairs) -> list[CameraProfile]:
        groups: dict[tuple, list] = {}
        for prof, health in pairs:
            key = (prof.make, prof.model, prof.width, prof.height, tuple(prof.dqt), tuple(prof.dht),
                   prof.sos, prof.restart_interval)
            g = groups.setdefault(key, [prof, 0, -1.0])
            g[1] += 1
            if health > g[2]:
                g[0], g[2] = prof, health
        out = []
        for prof, votes, _h in sorted(groups.values(), key=lambda g: (g[1], g[2]), reverse=True):
            prof.votes = votes
            out.append(prof)
        return out

    good = [g for g in found if g is not None]
    profiles = ranked([(g["profile"], g["health"]) for g in good if g["profile"] is not None])
    thumbs = ranked([(g["thumb"], 1.0) for g in good if g["thumb"] is not None])
    # every header votes, damaged ones included: the tables come out clean even when no
    # photo of the set has an intact header (bit-by-bit majority, see learn.consensus)
    from .learn.consensus import consensus

    blocks = {}
    for g in good:
        if g["profile"] is not None and g["block"]:
            key = (g["profile"].width, g["profile"].height, tuple(g["profile"].dqt), tuple(g["profile"].dht))
            blocks.setdefault(key, g["block"])
    seeds = []
    for p in profiles:
        b = blocks.get((p.width, p.height, tuple(p.dqt), tuple(p.dht)))
        if b is not None:
            seeds.append((p, b))
    if len(good) >= 3:
        voted = consensus(seeds, [g["head"] for g in good])
        if seeds:
            by_key = {id(p): v for (p, _b), v in zip(seeds, voted)}
            profiles = [by_key.get(id(p), p) for p in profiles]
        else:
            profiles = voted  # no header of the set parses: the vote is all there is
        profiles.sort(key=lambda p: p.votes, reverse=True)
    if profiles:
        echo(f"[ref] {len(profiles)} reference header(s) learned from the set")
    return profiles, thumbs


@click.group()
def main():
    """FotoMedik — diagnose and repair damaged photos (JPEG)."""


@main.command()
@click.argument("repaired", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("truth", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--picture", type=click.Path(path_type=Path), help="save truth | repaired | error map here (PNG/JPEG)")
@click.option("--runs", is_flag=True, help="list the runs of decoded MCUs that are wrong")
def compare(repaired: Path, truth: Path, picture: Path | None, runs: bool):
    """Measure a repair against an intact copy of the same picture (see fotomedik.compare).
    Uses the .report.json next to REPAIRED (written with --reports) to tell holes from decoded parts."""
    from . import compare as C

    rep = C.report_for(repaired)
    cmp = C.compare(repaired, truth, rep)
    click.echo(cmp.format_text())
    if rep is None:
        click.echo("(no .report.json next to the repaired photo: holes and decoded parts not told apart)")
    if runs:
        for r, c0, c1, e in cmp.wrong_runs():
            click.echo(f"  wrong: row {r}, MCUs {c0}-{c1 - 1}, error {e:.1f}")
    if picture:
        C.side_by_side(repaired, truth, cmp).save(picture)
        click.echo(f"picture: {picture}")


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--lang", default="sr", type=click.Choice(["sr", "en"]))
def scan(path: Path, lang: str):
    """Diagnose files without writing anything."""
    for p, rel in _iter_files(path):
        d = scan_bytes(p.read_bytes(), str(rel))
        rep = FileReport(str(rel), str(rel), d.size, d.size, round(100 * d.recovered_fraction, 1) if d.openable else None,
                         findings=[f.to_dict() for f in d.findings], width=d.width, height=d.height)
        click.echo(rep.format_text(lang))
        click.echo("")


def _fix_common(path: Path, out: Path, ref, mark_holes, keep_damaged, time_budget, lang, auto_ref: bool,
                jobs: int = 0, reports: bool = False, enhance: str = "", quality: str | None = None,
                prompt: str | None = None, reference: Path | None = None, twins: bool = False):
    if path.is_dir() and out.resolve() == path.resolve():
        raise click.ClickException("output folder must differ from the input folder (originals are never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs or (os.cpu_count() or 1)
    t0 = time.perf_counter()
    options = {"mark_holes": mark_holes, "keep_damaged": keep_damaged, "time_budget": time_budget,
               "write_reports": reports, "twins": twins}
    if reference is not None:
        options["reference_dir" if reference.is_dir() else "reference"] = str(reference)
    profile = CameraProfile.from_json(Path(ref).read_text(encoding="utf-8")) if ref else None
    files = [(p, rel) for p, rel in _iter_files(path) if not p.resolve().is_relative_to(out.resolve())]
    from .runner import plan

    items = plan(files, options)
    with Workers(jobs, len(items)) as workers:
        if auto_ref and len(files) > 1:
            learned = _learn_candidates(files, click.echo, jobs, workers)
            for _p, _rel, opts in items:
                opts["ref_candidates"], opts["thumb_refs"] = learned
        work = [(str(p), str(rel), str(out), opts, profile) for p, rel, opts in items]
        results = workers.map(_repair_file, work, cost=lambda it: os.path.getsize(it[0]), counted=True)
    steps = [x.strip() for x in enhance.split(",") if x.strip()]
    if steps:
        from .ai.run import run_extras

        def ai_done(rep: FileReport) -> None:
            if reports_wanted:  # the report was written before the AI copy existed
                rp = out / Path(rep.output_name).with_suffix(".report.json")
                if rp.exists():
                    rp.write_text(rep.to_json(), encoding="utf-8")

        reports_wanted = reports
        device = run_extras([(rep, items[i][0]) for i, (rep, _s) in enumerate(results)], out, steps, click.echo,
                            on_done=ai_done, quality=quality, prompt=prompt)
        if device:
            click.echo(f"[enhance] networks ran on {device}")
    reports: list[FileReport] = []
    for rep, secs in results:
        reports.append(rep)
        click.echo(rep.format_text(lang))
        click.echo(f"({secs:.1f} s)")
        click.echo("")
    click.echo(f"{len(reports)} file(s) in {time.perf_counter() - t0:.1f} s ({jobs} parallel job(s))")
    return reports


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--out", required=True, type=click.Path(path_type=Path), help="output folder (never the input)")
@click.option("--ref", type=click.Path(exists=True), help="camera profile JSON from `learn`")
@click.option("--mark-holes", is_flag=True, help="paint unrecovered areas magenta instead of blending")
@click.option("--keep-damaged", is_flag=True, help="keep decoded blocks of intervals that did not resync (noisier)")
@click.option("--time-budget", default=5.0, show_default=True, help="seconds per file for bit-flip search")
@click.option("--thorough", is_flag=True, help="search much longer for flipped bits (60 s per file)")
@click.option("--jobs", default=0, show_default="all CPU cores", help="files repaired in parallel")
@click.option("--reports", is_flag=True, help="also write a .report.json next to every repaired photo")
@click.option("--ai", is_flag=True, help="same as --enhance burst,ai")
@click.option("--enhance", default="", help="enhanced copy (_enhanced.jpg) with these steps, comma separated: "
              "burst (similar shots), deblock (JPEG clean-up), ai (sharpen and fill), thumbsr (holes from the "
              "thumbnail enlarged by AI), diffusion (max quality), faces. Network steps invent detail.")
@click.option("--quality", default="balanced", show_default=True, type=click.Choice(["fast", "balanced", "full"]),
              help="diffusion preset: the photo is repainted at quarter, half or full size")
@click.option("--scene", default="", help="what the photos show, for the diffusion (e.g. 'people on a square')")
@click.option("--reference", type=click.Path(exists=True, path_type=Path),
              help="a picture of the same scene (an AI-restored copy, a scan) that guides the repair into "
                   "<name>_ref; a folder of such pictures is paired with the photos by name or content")
@click.option("--twins", is_flag=True,
              help="a photo without a usable thumbnail takes the colours of its holes from a shot of the same "
                   "scene taken within 30 seconds")
@click.option("--lang", default="sr", type=click.Choice(["sr", "en"]))
def fix(path, out, ref, mark_holes, keep_damaged, time_budget, thorough, jobs, reports, ai, enhance, quality, scene,
        reference, twins, lang):
    """Repair a file or folder into OUT (--reports: a .report.json per file too)."""
    _fix_common(path, out, ref, mark_holes, keep_damaged, 60.0 if thorough else time_budget, lang,
                auto_ref=False, jobs=jobs, reports=reports, enhance=enhance or ("burst,ai" if ai else ""),
                quality=quality, prompt=scene or None, reference=reference, twins=twins)


@main.command()
@click.argument("good", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--out", required=True, type=click.Path(path_type=Path))
def learn(good: Path, out: Path):
    """Extract a camera profile (header tables) from a healthy photo."""
    try:
        prof = CameraProfile.from_jpeg(good.read_bytes(), source=good.name)
    except ValueError as e:
        raise click.ClickException(str(e))
    out.write_text(prof.to_json(), encoding="utf-8")
    click.echo(f"profile: {prof.make} {prof.model} {prof.width}x{prof.height}, DRI {prof.restart_interval} -> {out}")


@main.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--out", required=True, type=click.Path(path_type=Path))
@click.option("--ref", type=click.Path(exists=True))
@click.option("--mark-holes", is_flag=True)
@click.option("--keep-damaged", is_flag=True)
@click.option("--time-budget", default=5.0, show_default=True)
@click.option("--thorough", is_flag=True, help="search much longer for flipped bits (60 s per file)")
@click.option("--jobs", default=0, show_default="all CPU cores", help="files repaired in parallel")
@click.option("--reports", is_flag=True, help="also write a .report.json next to every repaired photo")
@click.option("--ai", is_flag=True, help="same as --enhance burst,ai")
@click.option("--enhance", default="", help="enhanced copy (_enhanced.jpg) with these steps, comma separated: "
              "burst (similar shots), deblock (JPEG clean-up), ai (sharpen and fill), thumbsr (holes from the "
              "thumbnail enlarged by AI), diffusion (max quality), faces. Network steps invent detail.")
@click.option("--quality", default="balanced", show_default=True, type=click.Choice(["fast", "balanced", "full"]),
              help="diffusion preset: the photo is repainted at quarter, half or full size")
@click.option("--scene", default="", help="what the photos show, for the diffusion (e.g. 'people on a square')")
@click.option("--reference", type=click.Path(exists=True, path_type=Path),
              help="a picture of the same scene (an AI-restored copy, a scan) that guides the repair into "
                   "<name>_ref; a folder of such pictures is paired with the photos by name or content")
@click.option("--twins", is_flag=True,
              help="a photo without a usable thumbnail takes the colours of its holes from a shot of the same "
                   "scene taken within 30 seconds")
@click.option("--lang", default="sr", type=click.Choice(["sr", "en"]))
def batch(folder, out, ref, mark_holes, keep_damaged, time_budget, thorough, jobs, reports, ai, enhance, quality,
          scene, reference, twins, lang):
    """Repair a whole folder; healthy photos in it serve as references; writes summary.csv."""
    reps = _fix_common(folder, out, ref, mark_holes, keep_damaged, 60.0 if thorough else time_budget, lang,
                       auto_ref=True, jobs=jobs, reports=reports, enhance=enhance or ("burst,ai" if ai else ""),
                       quality=quality, prompt=scene or None, reference=reference, twins=twins)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "output", "recovered_pct", "holes", "findings", "applied", "fallback"])
        for r in reps:
            w.writerow([r.name, r.output_name, r.recovered_pct, len(r.holes), ";".join(f["kind"] for f in r.findings), ";".join(r.applied), r.fallback or ""])
    click.echo(f"{len(reps)} file(s) -> {out} (summary.csv)")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
