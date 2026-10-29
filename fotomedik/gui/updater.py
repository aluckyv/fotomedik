"""Self-update of the packaged app from the GitHub release "app".

CI publishes FotoMedik.exe and version.json ({"build": N, "commit": ...}) on every
push to main. The app compares N with its own build stamp; updating downloads the
new exe next to the running one, swaps the files (Windows allows renaming a running
exe, not overwriting it) and starts the new version.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .._build import BUILD, COMMIT

REPO = "aluckyv/fotomedik"
TAG = "app"
API = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
TIMEOUT = 10


@dataclass
class Update:
    build: int
    commit: str
    exe_url: str
    size: int


def current_build() -> int:
    return BUILD


def current_commit() -> str:
    return COMMIT


def is_packaged() -> bool:
    return bool(getattr(sys, "frozen", False))


def _get(url: str, accept: str = "application/vnd.github+json") -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "FotoMedik-updater"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


def check() -> Update | None:
    """The newer build on GitHub, or None (up to date, offline, or not reachable)."""
    rel = json.loads(_get(API))
    assets = {a["name"]: a for a in rel.get("assets", [])}
    if "version.json" not in assets or "FotoMedik.exe" not in assets:
        return None
    info = json.loads(_get(assets["version.json"]["url"], "application/octet-stream"))
    build = int(info.get("build", 0))
    if build <= BUILD:
        return None
    exe = assets["FotoMedik.exe"]
    return Update(build, str(info.get("commit", "")), exe["url"], int(exe.get("size", 0)))


def download(upd: Update, dest: Path, progress=None, cancelled=None) -> None:
    """Download the new exe to dest; progress(done, total) is called as it arrives."""
    req = urllib.request.Request(upd.exe_url, headers={"Accept": "application/octet-stream",
                                                       "User-Agent": "FotoMedik-updater"})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length") or upd.size or 0)
        done = 0
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("download cancelled")
            chunk = r.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    if upd.size and tmp.stat().st_size != upd.size:
        tmp.unlink(missing_ok=True)
        raise OSError("the download is incomplete")
    os.replace(tmp, dest)


def install_and_restart(new_exe: Path) -> None:
    """Swap the running exe for new_exe and start it (the caller then quits)."""
    cur = Path(sys.executable)
    old = cur.with_name(cur.stem + ".old" + cur.suffix)
    if old.exists():
        old.unlink()
    os.replace(cur, old)  # a running exe can be renamed, not overwritten
    os.replace(new_exe, cur)
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    subprocess.Popen([str(cur)], creationflags=flags, close_fds=True, env=fresh_env())


def fresh_env() -> dict:
    """Environment for starting the new exe as an independent program.

    The packaged app sets private variables that tie its own sub-processes to the
    folder it unpacked itself into. Inherited by the new exe, they make it reuse
    the old app's folder, which is deleted when the old app quits, so its worker
    processes fail with "Failed to start embedded python interpreter"."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("_PYI_") and k not in ("_MEIPASS2", "_PYI_SPLASH_IPC")}
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return env


def cleanup_old() -> None:
    """Remove the previous version left behind by an update."""
    if not is_packaged():
        return
    cur = Path(sys.executable)
    old = cur.with_name(cur.stem + ".old" + cur.suffix)
    try:
        old.unlink(missing_ok=True)
    except OSError:
        pass
