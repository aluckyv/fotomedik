"""Repair jobs run in the background, one after another (for Claude, see fotomedik.mcp_server).

A job is one folder repaired into an output folder by the command line repair, in a
process of its own on this computer. Jobs are JSON files in <FotoMedik home>/jobs, so
they outlive whatever queued them: the runner (``FotoMedik.exe --jobs``) takes the
oldest queued job, runs ``FotoMedik.exe --cli batch ...`` for it with the output going
to <id>.out, and moves on to the next one; only one runner works at a time.

Safety, whoever asks: the output folder is never the source folder, inside it or
around it, never inside a protected folder (mcp.json, set by the person in the app
with "Connect to Claude"), and a job never deletes anything.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from . import install

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".~cr"}  # the same as cli.IMAGE_SUFFIXES
STEPS = ("puzzle", "burst", "deblock", "ai", "thumbsr", "diffusion", "patch", "faces", "rows",
         "holemap", "sdxl", "flank")  # ai.run.STEPS
DEFAULT_STEPS = "puzzle,burst,deblock,thumbsr"
FINISHED = ("done", "failed", "stopped", "cancelled", "interrupted")
SETTINGS = "mcp.json"


class JobError(Exception):
    """A job that cannot be queued (the message says why, for the one who asked)."""


# ----------------------------------------------------------------------------- places


def jobs_dir() -> Path:
    d = install.home() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def settings() -> dict:
    """{"protected": [folders and drives nothing may be written into]} (set in the app)."""
    try:
        data = json.loads((install.home() / SETTINGS).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        data = {}
    prot = data.get("protected") or []
    return {"protected": [str(p) for p in prot if str(p).strip()]}


def save_settings(protected: list[str]) -> Path:
    path = install.home() / SETTINGS
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"protected": [p.strip() for p in protected if p.strip()]}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _full(p: str | Path) -> Path:
    """An absolute path; a bare drive ("F:") means its root."""
    s = str(p).strip().strip('"')
    if re.fullmatch(r"[A-Za-z]:", s):
        s += "\\"
    try:
        return Path(s).expanduser().resolve()
    except OSError:
        return Path(os.path.abspath(os.path.expanduser(s)))


def _inside(p: Path, folder: Path) -> bool:
    return p == folder or p.is_relative_to(folder)


def look_at_folder(source: str, photos: bool = False) -> dict:
    """The overview of a folder (see overview.py) as plain data for Claude."""
    from .ai.cuda import nvidia_present
    from .cli import _iter_files
    from .overview import overview

    src = _full(source)
    if not src.is_dir():
        raise JobError(f"the source folder does not exist: {src}")
    files = list(_iter_files(src))
    ov = overview(files)
    gpu = nvidia_present()
    preset, why = ov.suggest(gpu)
    d = ov.damaged
    out = {"photos": len(ov.photos), "damaged": len(d),
           "damaged_header": sum(1 for p in d if p.state == "header"),
           "thumbnail_whole": sum(1 for p in d if p.thumb == "whole"),
           "thumbnail_part": sum(1 for p in d if p.thumb == "part"),
           "thumbnail_none": sum(1 for p in d if p.thumb == "none"),
           "with_faces": sum(1 for p in d if p.faces), "burst_groups": ov.bursts(),
           "unreadable": sum(1 for p in ov.photos if p.state == "unreadable"),
           "suggested_preset": preset, "suggested_steps": {"faithful": "", "standard": DEFAULT_STEPS,
                                                            "maximum": "puzzle,burst,deblock,diffusion,faces"}[preset],
           "why": why, "nvidia_driver": gpu}
    if photos:
        out["list"] = [{"photo": p.name, "state": p.state, "decodes": None if p.recovered is None else round(p.recovered, 3),
                        "thumbnail": p.thumb, "thumbnail_real": round(p.thumb_real, 2), "faces": p.faces,
                        "size": f"{p.width}x{p.height}" if p.width else None} for p in ov.photos]
    return out


def check_paths(source: str, output: str, protected: list[str] | None = None) -> tuple[Path, Path]:
    """(source, output) as absolute paths, or JobError when the job could harm a file."""
    if not str(source).strip() or not str(output).strip():
        raise JobError("both the source and the output folder are needed")
    src, out = _full(source), _full(output)
    if not src.is_dir():
        raise JobError(f"the source folder does not exist: {src}")
    if out.exists() and not out.is_dir():
        raise JobError(f"the output is a file, not a folder: {out}")
    if _inside(out, src):
        raise JobError("the output folder must not be the source folder or inside it "
                       "(originals are never overwritten)")
    if _inside(src, out):
        raise JobError("the output folder must not contain the source folder")
    for p in settings()["protected"] if protected is None else protected:
        if _inside(out, _full(p)):
            raise JobError(f"the output folder is in a protected place ({p}); nothing is written there")
    return src, out


def photos(src: Path) -> list[str]:
    """The photos of a folder, as paths relative to it (subfolders too, sorted)."""
    found = []
    for root, _dirs, files in os.walk(src):
        for f in sorted(files):
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_SUFFIXES:
                found.append(p.relative_to(src).as_posix())
    return found


def enhanced_rel(rel: str) -> str:
    return os.path.splitext(rel)[0] + "_enhanced.jpg"


def done_already(rel: str, out: Path, steps: list[str]) -> bool:
    """Whether a photo has its result in the output folder (the enhanced copy when steps
    are on, else the repaired copy) - what a run with --skip-done leaves out."""
    return (out / (enhanced_rel(rel) if steps else rel)).exists()


def parse_steps(steps: str | None) -> list[str]:
    raw = DEFAULT_STEPS if steps is None else steps
    out = [s.strip().lower() for s in raw.split(",") if s.strip()]
    bad = [s for s in out if s not in STEPS]
    if bad:
        raise JobError(f"unknown step(s): {', '.join(bad)} (known: {', '.join(STEPS)})")
    return out


# ----------------------------------------------------------------------------- jobs


def _path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9A-Za-z_-]+", job_id or ""):
        raise JobError(f"no such job: {job_id}")
    return jobs_dir() / f"{job_id}.json"


def read(job_id: str) -> dict:
    try:
        return json.loads(_path(job_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise JobError(f"no such job: {job_id}") from None


def write(job: dict) -> None:
    path = _path(job["id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    for _ in range(20):  # the other side may be reading it this very moment (Windows)
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, path)


def all_jobs() -> list[dict]:
    jobs = []
    for p in jobs_dir().glob("*.json"):
        try:
            jobs.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(jobs, key=lambda j: (j.get("created", ""), j.get("id", "")))


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_job(source: str, output: str, steps: str | None = None, only: list[str] | None = None,
            skip_done: bool = False, learn_from: list[str] | None = None) -> dict:
    """Queue a job (JobError when it is not allowed) and make sure the runner works.
    learn_from: folders of healthy photos of the same cameras the repair learns from too
    (only read)."""
    src, out = check_paths(source, output)
    learn = []
    for f in learn_from or []:
        if not str(f).strip():
            continue
        p = _full(f)
        if not p.is_dir():
            raise JobError(f"the folder to learn from does not exist: {p}")
        if _inside(p, out) or _inside(out, p):
            raise JobError(f"the folder to learn from must not be the output folder, inside it or around it: {p}")
        learn.append(str(p))
    st = parse_steps(steps)
    have = photos(src)
    if only:
        wanted = [str(n).replace("\\", "/").strip("/") for n in only if str(n).strip()]
        index = {h.lower(): h for h in have}
        missing = [n for n in wanted if n.lower() not in index]
        if missing:
            raise JobError(f"not in the source folder: {', '.join(missing[:10])}"
                           + (f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""))
        chosen = [index[n.lower()] for n in wanted]
    else:
        chosen = have
    if not chosen:
        raise JobError(f"no photos (JPEG) in {src}")
    todo = [r for r in chosen if not (skip_done and done_already(r, out, st))]
    if not todo:
        raise JobError(f"all {len(chosen)} photo(s) have their result in {out} already")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    n = 1
    while (jobs_dir() / f"{stamp}-{n}.json").exists():
        n += 1
    job = {"id": f"{stamp}-{n}", "state": "queued", "source": str(src), "output": str(out),
           "steps": ",".join(st), "only": [r for r in chosen] if only else None, "skip_done": bool(skip_done),
           "learn_from": learn or None, "photos": len(chosen), "to_do": len(todo), "created": _now(), "started": None,
           "finished": None, "pid": None, "returncode": None}
    write(job)
    ensure_runner()
    return job


def command(job: dict) -> list[str]:
    """The command line repair for a job."""
    args = ["batch", job["source"], "-o", job["output"]]
    if job.get("steps"):
        args += ["--enhance", job["steps"]]
    if job.get("only"):
        lst = jobs_dir() / f"{job['id']}.only.txt"
        lst.write_text("\n".join(job["only"]) + "\n", encoding="utf-8")
        args += ["--only-file", str(lst)]
    if job.get("skip_done"):
        args.append("--skip-done")
    for f in job.get("learn_from") or []:
        args += ["--learn-from", f]
    test = os.environ.get("FOTOMEDIK_JOB_CMD")  # the tests run something else instead
    if test:
        return json.loads(test)
    if getattr(sys, "frozen", False):
        return [sys.executable, "--cli", *args]
    return [sys.executable, "-c", "from fotomedik.cli import main; main()", *args]


def runner_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--jobs"]
    return [sys.executable, "-c", "from fotomedik.jobs import run_queue; run_queue()"]


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["FOTOMEDIK_HOME"] = str(install.home())
    here = str(Path(__file__).resolve().parent.parent)
    if not getattr(sys, "frozen", False):
        env["PYTHONPATH"] = here + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def spawn(cmd: list[str], out=None, detached: bool = True) -> subprocess.Popen:
    """Start a process that lives on by itself (it keeps working when the one that
    started it - Claude's connection - goes away)."""
    kw: dict = {"stdin": subprocess.DEVNULL, "stdout": out or subprocess.DEVNULL,
                "stderr": subprocess.STDOUT if out else subprocess.DEVNULL, "env": _env(),
                "cwd": str(install.home()), "close_fds": True}
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        if detached:
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            try:  # out of a job object that would end it together with its parent
                return subprocess.Popen(cmd, creationflags=flags | 0x01000000, **kw)  # CREATE_BREAKAWAY_FROM_JOB
            except OSError:
                pass
        return subprocess.Popen(cmd, creationflags=flags, **kw)
    return subprocess.Popen(cmd, start_new_session=True, **kw)


def ensure_runner() -> None:
    """Start the runner unless one is working (a second one simply ends)."""
    if not runner_alive():
        install.home().mkdir(parents=True, exist_ok=True)
        spawn(runner_command())


# ----------------------------------------------------------------------------- the runner


class _Lock:
    """One runner at a time: an exclusive lock on jobs/runner.lock."""

    def __init__(self):
        self.fh = None

    def acquire(self) -> bool:
        path = jobs_dir() / "runner.lock"
        try:
            self.fh = open(path, "a+b")
        except OSError:
            return False
        try:
            if sys.platform == "win32":
                import msvcrt

                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            self.fh = None
            return False
        return True

    def release(self) -> None:
        if self.fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self.fh.close()
        self.fh = None


def runner_alive() -> bool:
    lock = _Lock()
    if lock.acquire():
        lock.release()
        return False
    return True


def run_queue() -> int:
    """Work through the queued jobs, oldest first; returns at once when another runner works."""
    lock = _Lock()
    if not lock.acquire():
        return 0
    try:
        for job in all_jobs():  # a runner that was working died with the computer or was killed
            if job.get("state") == "running":
                job["state"], job["finished"] = "interrupted", _now()
                write(job)
        while True:
            queued = [j for j in all_jobs() if j.get("state") == "queued"]
            if not queued:
                return 0
            _run_one(queued[0])
    finally:
        lock.release()


def _run_one(job: dict) -> None:
    job["state"], job["started"] = "running", _now()
    write(job)
    log = jobs_dir() / f"{job['id']}.out"
    try:
        Path(job["output"]).mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as fh:
            fh.write(f"=== {_now()}  {' '.join(command(job))}\n".encode("utf-8"))
            fh.flush()
            proc = spawn(command(job), out=fh, detached=False)
            job["pid"] = proc.pid
            write(job)
            rc = proc.wait()
    except Exception as e:  # the job could not even start
        with open(log, "ab") as fh:
            fh.write(f"could not start the repair: {type(e).__name__}: {e}\n".encode("utf-8"))
        rc = -1
    now = read(job["id"])
    if now.get("state") == "running":
        now["state"] = "done" if rc == 0 else "failed"
    now["returncode"], now["finished"] = rc, _now()
    write(now)


# ----------------------------------------------------------------------------- stop / status


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    try:  # a zombie child of this process counts as gone
        done, _status = os.waitpid(int(pid), os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)  # type: ignore[attr-defined]
        return
    import signal

    try:
        os.killpg(int(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            pass


def stop(job_id: str) -> str:
    job = read(job_id)
    state = job.get("state")
    if state == "queued":
        job["state"], job["finished"] = "cancelled", _now()
        write(job)
        return f"job {job_id} taken off the queue"
    if state != "running":
        return f"job {job_id} is not running ({state})"
    job["state"], job["finished"] = "stopped", _now()
    write(job)
    if job.get("pid"):
        kill_tree(job["pid"])
    return f"job {job_id} stopped; the photos finished so far stay in {job['output']}"


def _tail(path: Path, lines: int) -> list[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 64 * 1024))
            text = fh.read().decode("utf-8", "replace").replace("\x00", "")  # UTF-16 bits of onnxruntime
    except OSError:
        return []
    keep = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return [ln for ln in keep if "onnxruntime" not in ln and "[W:" not in ln][-lines:]


def _since(ts: str | None) -> float:
    if not ts:
        return 0.0
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()


def status(job: dict, lines: int = 12) -> dict:
    """What a job has done so far (outputs written since it started, its last messages)."""
    st = dict(job)
    if st.get("state") == "running" and not pid_alive(st.get("pid")) and not runner_alive():
        st["state"] = "interrupted"
    out = Path(job["output"])
    t0 = _since(job.get("started"))
    repaired = enhanced = 0
    enh_times = []
    if t0 and out.is_dir():
        for root, _dirs, files in os.walk(out):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                try:
                    m = p.stat().st_mtime
                except OSError:
                    continue
                if m + 1 < t0:
                    continue
                if f.lower().endswith("_enhanced.jpg"):
                    enhanced += 1
                    enh_times.append(m)
                else:
                    repaired += 1
    st["repaired_since_start"] = repaired
    st["enhanced_since_start"] = enhanced
    if len(enh_times) >= 2:
        enh_times.sort()
        st["seconds_per_enhanced"] = round((enh_times[-1] - enh_times[0]) / (len(enh_times) - 1), 1)
    end = _since(job.get("finished")) or time.time()
    if t0:
        st["elapsed_min"] = round((end - t0) / 60, 1)
    errors = 0
    try:
        with open(out / "FotoMedik-greske.txt", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if ln.startswith("=== ") and job.get("started") and ln[4:23] >= job["started"]:
                    errors += 1
    except OSError:
        pass
    st["errors_since_start"] = errors
    st["last_lines"] = _tail(jobs_dir() / f"{job['id']}.out", lines)
    return st


def latest() -> dict | None:
    jobs = all_jobs()
    running = [j for j in jobs if j.get("state") == "running"]
    return (running or jobs or [None])[-1]


# ----------------------------------------------------------------------------- updating the program


def update_program(install_it: bool = True) -> dict:
    """Bring the installed FotoMedik up to the newest build on GitHub (for Claude, through
    the MCP tool update_app). Only between jobs. Most builds change only the code part:
    it is swapped in at once and every new job uses it (this server keeps the code it
    started with until Claude restarts it). A build with a new base (Python, Qt...) is
    downloaded and put in by the launcher the next time FotoMedik starts."""
    from . import install

    have = install.installed()
    if have is None:
        raise JobError("this is not the installed FotoMedik (run from source): nothing to update")
    rel = install.Release()
    out = {"installed_build": int(have.get("build", 0)), "newest_build": rel.build}
    if not install.needs_update(rel, have):
        return {**out, "updated": False, "note": "already the newest build"}
    if not install_it:
        return {**out, "updated": False, "note": "a newer build is available"}
    if runner_alive():
        raise JobError("a job is running: update after it (or stop it first)")
    parts = install.stage(rel)
    if "base" in parts:
        return {**out, "updated": False, "staged": parts,
                "note": "the new build changes the program's base; it is put in when FotoMedik is opened "
                        "from its own shortcut while Claude is closed (Claude starts the program "
                        "itself, not through the launcher): close Claude fully, open FotoMedik "
                        "once, then open Claude again"}
    install.apply_pending()
    return {**out, "updated": True, "installed_build": rel.build,
            "note": "new jobs use the new build; this connection keeps the old code until Claude restarts it"}
