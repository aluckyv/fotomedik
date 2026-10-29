"""FotoMedik command line: scan | fix | learn | batch."""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click

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
    if rep.recovered_pct is None:
        rep.recovered_pct = round(100 * diag.recovered_fraction, 1) if not diag.findings or diag.openable else None
    return ctx, rep


def _write_outputs(ctx: Context, rep: FileReport, out_dir: Path, rel: Path) -> Path:
    out_name = ctx.suggested_name or rel.name
    target = out_dir / rel.parent / out_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(ctx.data)
    (target.parent / (Path(out_name).stem + ".report.json")).write_text(rep.to_json(), encoding="utf-8")
    return target


def _profile_of(args: tuple[str, str]) -> CameraProfile | None:
    """Worker: a reference profile from a file whose header and most of the scan are healthy."""
    path, rel = args
    try:
        data = Path(path).read_bytes()
        d = scan_bytes(data, Path(rel).name)
        if not d.openable or d.recovered_fraction < 0.5:
            return None
        return CameraProfile.from_jpeg(data, source=rel)
    except Exception:
        return None


def _repair_file(args: tuple[str, str, str, dict, CameraProfile | None]) -> tuple[FileReport, float]:
    """Worker: repair one file and write its outputs. Runs in a separate process."""
    path, rel, out, options, profile = args
    t = time.perf_counter()
    data = Path(path).read_bytes()
    relp = Path(rel)
    ctx, rep = repair_bytes(data, relp.name, options, profile)
    target = _write_outputs(ctx, rep, Path(out), relp)
    rep.output_name = str(target.relative_to(Path(out)))
    return rep, time.perf_counter() - t


def _pool_map(fn, items: list, jobs: int, cost=None):
    """Map in parallel processes, results in input order; in-process for jobs <= 1.

    `cost(item)` orders the submission (biggest first) so long files do not end up
    last on an otherwise idle pool.
    """
    if jobs <= 1 or len(items) <= 1:
        return [fn(it) for it in items]
    order = sorted(range(len(items)), key=(lambda i: -cost(items[i])) if cost else (lambda i: i))
    out: list = [None] * len(items)
    with ProcessPoolExecutor(max_workers=min(jobs, len(items))) as ex:
        for i, res in zip(order, ex.map(fn, [items[i] for i in order])):
            out[i] = res
    return out


def _learn_candidates(files: list[tuple[Path, Path]], echo, jobs: int = 1) -> list[CameraProfile]:
    """Healthy headers in the set become reference profiles for the damaged ones."""
    profiles: list[CameraProfile] = []
    seen: set[tuple] = set()
    for prof in _pool_map(_profile_of, [(str(p), str(rel)) for p, rel in files], jobs):
        if prof is None:
            continue
        key = (prof.model, prof.width, prof.height, tuple(prof.dqt), tuple(prof.dht), prof.restart_interval)
        if key in seen:
            continue
        seen.add(key)
        profiles.append(prof)
    if profiles:
        echo(f"[ref] {len(profiles)} reference profile(s) learned from healthy files in the set")
    return profiles


@click.group()
def main():
    """FotoMedik — diagnose and repair damaged photos (JPEG)."""


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
                jobs: int = 0):
    if path.is_dir() and out.resolve() == path.resolve():
        raise click.ClickException("output folder must differ from the input folder (originals are never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs or (os.cpu_count() or 1)
    t0 = time.perf_counter()
    options = {"mark_holes": mark_holes, "keep_damaged": keep_damaged, "time_budget": time_budget}
    profile = CameraProfile.from_json(Path(ref).read_text(encoding="utf-8")) if ref else None
    files = [(p, rel) for p, rel in _iter_files(path) if not p.resolve().is_relative_to(out.resolve())]
    if auto_ref and len(files) > 1:
        options["ref_candidates"] = _learn_candidates(files, click.echo, jobs)
    work = [(str(p), str(rel), str(out), options, profile) for p, rel in files]
    results = _pool_map(_repair_file, work, jobs, cost=lambda it: os.path.getsize(it[0]))
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
@click.option("--lang", default="sr", type=click.Choice(["sr", "en"]))
def fix(path, out, ref, mark_holes, keep_damaged, time_budget, thorough, jobs, lang):
    """Repair a file or folder into OUT (+ a .report.json per file)."""
    _fix_common(path, out, ref, mark_holes, keep_damaged, 60.0 if thorough else time_budget, lang,
                auto_ref=False, jobs=jobs)


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
@click.option("--lang", default="sr", type=click.Choice(["sr", "en"]))
def batch(folder, out, ref, mark_holes, keep_damaged, time_budget, thorough, jobs, lang):
    """Repair a whole folder; healthy photos in it serve as references; writes summary.csv."""
    reports = _fix_common(folder, out, ref, mark_holes, keep_damaged, 60.0 if thorough else time_budget, lang,
                          auto_ref=True, jobs=jobs)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "output", "recovered_pct", "holes", "findings", "applied", "fallback"])
        for r in reports:
            w.writerow([r.name, r.output_name, r.recovered_pct, len(r.holes), ";".join(f["kind"] for f in r.findings), ";".join(r.applied), r.fallback or ""])
    click.echo(f"{len(reports)} file(s) -> {out} (summary.csv)")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
