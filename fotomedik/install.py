"""The installed app: a folder that updates itself part by part.

    %LOCALAPPDATA%\\FotoMedik\\
        FotoMedik.exe        the small launcher (installs, applies updates, starts the app)
        app\\base\\            Python, Qt, numpy... (PyInstaller folder build) with FotoMedik.exe
        app\\code\\            the fotomedik package - most updates change only this
        app\\version.json     what is installed: build, base_id
        runtime\\, models\\     AI runtime and networks, downloaded when first needed

The release "app" on GitHub carries version.json ({build, commit, base_id, base_sha256,
code_sha256, ...}), fotomedik-base.zip, fotomedik-code.zip and the launcher. An update
downloads the code, and the base only when its id changed (new Python, Qt, numpy...),
unpacks them next to the running ones (*.new) and lets the launcher swap them in while
the app is closed - a half-finished download never touches the working app.

Only the standard library here: the launcher runs it without the rest of the app.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "aluckyv/fotomedik"
TAG = "app"
API = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
BASE_ZIP = "fotomedik-base.zip"
CODE_ZIP = "fotomedik-code.zip"
LAUNCHER = "FotoMedik.exe"
VERSION = "version.json"
PARTS = ("base", "code")
TIMEOUT = 15


def home() -> Path:
    env = os.environ.get("FOTOMEDIK_HOME")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "FotoMedik"


def app_dir() -> Path:
    return home() / "app"


def app_exe() -> Path:
    return app_dir() / "base" / "FotoMedik.exe"


def launcher_path() -> Path:
    return home() / LAUNCHER


def installed() -> dict | None:
    """What is installed ({build, base_id, ...}), or None."""
    try:
        info = json.loads((app_dir() / VERSION).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not app_exe().exists() or not (app_dir() / "code" / "fotomedik").is_dir():
        return None
    return info


# ----------------------------------------------------------------------------- the release


def _get(url: str, accept: str = "application/vnd.github+json", timeout: float = TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "FotoMedik"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class Release:
    """The newest release: its version info and where its files are. FOTOMEDIK_ASSETS
    (a folder with the same files) stands in for GitHub - the CI tests use it."""

    def __init__(self):
        local = os.environ.get("FOTOMEDIK_ASSETS")
        if local:
            self.local = Path(local)
            self.assets = {p.name: {"url": str(p), "size": p.stat().st_size} for p in self.local.iterdir()}
            self.info = json.loads((self.local / VERSION).read_text(encoding="utf-8-sig"))
            return
        self.local = None
        rel = json.loads(_get(API))
        self.assets = {a["name"]: {"url": a["url"], "size": int(a.get("size", 0))} for a in rel.get("assets", [])}
        if VERSION not in self.assets:
            raise RuntimeError("the release has no version.json")
        self.info = json.loads(_get(self.assets[VERSION]["url"], "application/octet-stream"))

    @property
    def build(self) -> int:
        return int(self.info.get("build", 0))

    def has_folder_app(self) -> bool:
        return BASE_ZIP in self.assets and CODE_ZIP in self.assets and "base_id" in self.info

    def download(self, name: str, dest: Path, progress=None, cancelled=None, sha256: str | None = None) -> None:
        """Fetch asset `name` to dest (through dest.part), checking its size and hash."""
        a = self.assets[name]
        tmp = dest.with_name(dest.name + ".part")
        h = hashlib.sha256()
        if self.local is not None:
            src = open(a["url"], "rb")
            total = a["size"]
        else:
            req = urllib.request.Request(a["url"], headers={"Accept": "application/octet-stream", "User-Agent": "FotoMedik"})
            src = urllib.request.urlopen(req, timeout=60)
            total = int(src.headers.get("Content-Length") or a["size"] or 0)
        done = 0
        with src, open(tmp, "wb") as out:
            while True:
                if cancelled is not None and cancelled():
                    raise InterruptedError("download cancelled")
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total, name)
        if a["size"] and done != a["size"]:
            tmp.unlink(missing_ok=True)
            raise OSError(f"{name}: the download is incomplete")
        if sha256 and h.hexdigest() != sha256:
            tmp.unlink(missing_ok=True)
            raise OSError(f"{name}: the download is damaged (checksum)")
        os.replace(tmp, dest)


def needs_update(rel: Release, have: dict | None) -> bool:
    if have is None:
        return True
    return rel.build > int(have.get("build", 0))


# ----------------------------------------------------------------------------- installing


def _unzip(zip_path: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    # a zip of a folder may hold that folder at its top: unwrap it
    inner = list(tmp.iterdir())
    if len(inner) == 1 and inner[0].is_dir() and inner[0].name.lower() in ("fotomedik", "base", "code") \
            and not (tmp / "fotomedik").is_dir() and not (tmp / "FotoMedik.exe").exists():
        inner[0].rename(dest)
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        tmp.rename(dest)


def stage(rel: Release, progress=None, cancelled=None) -> list[str]:
    """Download and unpack the parts of the release that differ from what is installed,
    as app/<part>.new plus app/version.new.json. Returns the parts staged."""
    root = app_dir()
    root.mkdir(parents=True, exist_ok=True)
    have = installed() or {}
    parts = ["code"]
    if have.get("base_id") != rel.info.get("base_id") or not (root / "base").is_dir():
        parts.insert(0, "base")
    for part in parts:
        name = BASE_ZIP if part == "base" else CODE_ZIP
        z = root / name
        rel.download(name, z, progress, cancelled, rel.info.get(f"{part}_sha256"))
        try:
            _unzip(z, root / f"{part}.new")
        finally:
            z.unlink(missing_ok=True)
    (root / "version.new.json").write_text(json.dumps(rel.info), encoding="utf-8")
    return parts


def pending() -> bool:
    return (app_dir() / "version.new.json").exists()


def apply_pending() -> bool:
    """Swap staged parts in (the app must not be running). True when something changed."""
    root = app_dir()
    vnew = root / "version.new.json"
    if not vnew.exists():
        return False
    for part in PARTS:
        new = root / f"{part}.new"
        if not new.is_dir():
            continue
        cur, old = root / part, root / f"{part}.old"
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        if cur.exists():
            os.replace(cur, old)
        os.replace(new, cur)
    os.replace(vnew, root / VERSION)
    for part in PARTS:
        shutil.rmtree(root / f"{part}.old", ignore_errors=True)
    return True


def keep_launcher(current: Path) -> None:
    """Put a copy of the running launcher where restarts look for it."""
    dest = launcher_path()
    try:
        if current.resolve() == dest.resolve():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current, dest)
    except OSError:
        pass


def start_app(args: list[str], wait: bool = False) -> int:
    env = dict(os.environ)
    env["FOTOMEDIK_LAUNCHER"] = str(launcher_path())
    env["FOTOMEDIK_HOME"] = str(home())
    flags = 0
    if sys.platform == "win32" and not wait:
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    p = subprocess.Popen([str(app_exe()), *args], env=env, creationflags=flags, close_fds=True)
    return p.wait() if wait else 0


def is_folder_app() -> bool:
    """Whether this process is the installed folder app (base + code)."""
    exe = Path(sys.executable)
    return bool(getattr(sys, "frozen", False)) and exe.parent.name == "base" \
        and (exe.parent.parent / "code" / "fotomedik").is_dir()


def make_shortcuts() -> None:
    """A FotoMedik shortcut on the desktop and in the Start menu, pointing at the launcher
    (made once; a missing one is made again, an existing one is left alone)."""
    if sys.platform != "win32":
        return
    target = launcher_path()
    places = [Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop",
              Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"]
    for d in places:
        lnk = d / "FotoMedik.lnk"
        if lnk.exists() or not d.is_dir():
            continue
        ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
              f"$s.TargetPath='{target}';$s.WorkingDirectory='{target.parent}';$s.Save()")
        try:
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           creationflags=0x08000000, timeout=30, check=False)  # no console window
        except (OSError, subprocess.SubprocessError):
            pass
