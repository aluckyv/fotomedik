"""The AI runtime (ONNX Runtime) of the installed app, picked for this computer's
graphics card and downloaded when the AI is first used.

ONNX Runtime comes in builds that cannot live side by side in one program: for NVIDIA
cards (CUDA), for any DirectX 12 card (DirectML) and for the processor alone. The
program carries none of them; the first AI step fetches the right one - the official
package from PyPI, about 25-250 MB - into %LOCALAPPDATA%\\FotoMedik\\runtime and loads
it from there. With an NVIDIA card the CUDA build is tried first (its NVIDIA libraries
found on the computer or downloaded, see .cuda); when that cannot work, DirectML.

Run from source (a Python with onnxruntime installed), the installed one is used.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

VERSION = "1.22.0"  # one ONNX Runtime version for every kind (same Python API)
PACKAGE = {"cuda": "onnxruntime-gpu", "directml": "onnxruntime-directml", "cpu": "onnxruntime"}
NAMES = {"cuda": "NVIDIA (CUDA)", "directml": "DirectML", "cpu": "processor"}
PYPI = "https://pypi.org/pypi/{name}/{version}/json"

kind: str | None = None  # the kind loaded in this process


def root() -> Path:
    from .models import model_dir

    return model_dir().parent / "runtime"


def candidates() -> list[str]:
    """The kinds to try on this computer, best first."""
    from .cuda import nvidia_present

    if sys.platform != "win32":
        return ["cpu"]
    out = ["directml", "cpu"]
    try:
        if nvidia_present():
            out.insert(0, "cuda")
    except Exception:
        pass
    return out


def _wheel(kind_: str) -> tuple[str, int]:
    """(URL, size) of the Windows wheel of this kind for the running Python."""
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    url = PYPI.format(name=PACKAGE[kind_], version=VERSION)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "FotoMedik"}), timeout=30) as r:
        info = json.load(r)
    for f in info.get("urls", []):
        fn = f.get("filename", "")
        if fn.endswith(".whl") and f"-{tag}-" in fn and "win_amd64" in fn:
            return f["url"], int(f.get("size") or 0)
    raise RuntimeError(f"no {PACKAGE[kind_]} {VERSION} package for Python {tag} on Windows")


def folder(kind_: str) -> Path:
    return root() / f"{kind_}-{VERSION}"


def fetch(kind_: str, progress=None, stop=None) -> Path:
    """The folder holding this kind's onnxruntime package, downloading it first if needed."""
    dest = folder(kind_)
    if (dest / "onnxruntime" / "__init__.py").exists():
        return dest
    url, size = _wheel(kind_)
    buf = io.BytesIO()
    req = urllib.request.Request(url, headers={"User-Agent": "FotoMedik"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or size or 0)
        while True:
            if stop is not None and stop.is_set():
                raise InterruptedError("download stopped")
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf.write(chunk)
            if progress:
                progress(buf.tell(), total, f"AI runtime ({NAMES[kind_]})")
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(buf) as z:
        for info in z.infolist():
            top = info.filename.split("/", 1)[0]
            if top == "onnxruntime":  # the package only (no .dist-info, no scripts)
                z.extract(info, tmp)
    os.replace(tmp, dest)
    return dest


def load(say=None, progress=None, stop=None):
    """The onnxruntime module for this computer, or None (the reason goes to say)."""
    global kind
    say = say or (lambda s: None)
    if "onnxruntime" in sys.modules:
        return sys.modules["onnxruntime"]
    try:  # installed alongside (run from source, or a build that carries it)
        ort = importlib.import_module("onnxruntime")
        kind = "installed"
        return ort
    except ImportError as e:
        if not getattr(sys, "frozen", False):
            say(f"AI steps are not available here (ImportError: {e})")
            return None
    errors = []
    for k in candidates():
        if k == "cuda":
            from . import cuda

            if not cuda.prepare(say, lambda d, t, p: progress(d, t, p) if progress else None, stop):
                errors.append("CUDA libraries")
                continue
        try:
            if not (folder(k) / "onnxruntime" / "__init__.py").exists():
                say(f"Downloading the AI runtime for {NAMES[k]} (only the first time)…")
            d = fetch(k, progress, stop)
        except InterruptedError:
            return None
        except Exception as e:
            errors.append(f"{NAMES[k]}: {type(e).__name__}: {e}")
            continue
        sys.path.insert(0, str(d))
        try:
            ort = importlib.import_module("onnxruntime")
        except Exception as e:
            sys.path.remove(str(d))
            sys.modules.pop("onnxruntime", None)
            errors.append(f"{NAMES[k]}: {type(e).__name__}: {e}")
            continue
        kind = k
        say(f"AI runtime: {NAMES[k]}")
        return ort
    say("The AI runtime could not be set up (" + "; ".join(errors) + ")")
    return None
