"""NVIDIA graphics cards: the CUDA libraries the CUDA build of the app runs its networks with.

The CUDA build of ONNX Runtime (FotoMedik-CUDA.exe) needs the CUDA 12 runtime, cuBLAS,
cuFFT and cuDNN 9 next to it - about 1.5 GB, too much to ship inside the program.
Most computers with an NVIDIA card that ever ran AI software have them somewhere
already: the CUDA Toolkit, cuDNN, a Python with PyTorch or the nvidia-* packages, apps
like ComfyUI. Those are looked for first; only what is missing is downloaded once (the
official NVIDIA packages from PyPI) into the app's own folder.
"""

from __future__ import annotations

import glob
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# DLL -> (PyPI package that carries it, major version wanted)
NEEDED = {
    "cudart64_12.dll": ("nvidia-cuda-runtime-cu12", 12),
    "cublasLt64_12.dll": ("nvidia-cublas-cu12", 12),
    "cublas64_12.dll": ("nvidia-cublas-cu12", 12),
    "cufft64_11.dll": ("nvidia-cufft-cu12", 11),
    "cudnn64_9.dll": ("nvidia-cudnn-cu12", 9),
}
# cuDNN loads these itself from its own folder; preloading them avoids "not found" later
CUDNN_PARTS = ("cudnn_graph64_9.dll", "cudnn_engines_precompiled64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
               "cudnn_heuristic64_9.dll", "cudnn_ops64_9.dll", "cudnn_adv64_9.dll", "cudnn_engines_tensor_ir64_9.dll")
LOAD_ORDER = ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cufft64_11.dll", "cudnn64_9.dll")
PYPI = "https://pypi.org/pypi/{name}/json"

_handles: list = []  # loaded DLLs stay loaded for the life of the process
_ready: bool | None = None


def nvidia_present() -> bool:
    """Whether an NVIDIA driver is installed (its CUDA driver library is in System32)."""
    if sys.platform != "win32":
        return Path("/proc/driver/nvidia").exists()
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return Path(root, "System32", "nvcuda.dll").exists()


def own_dir() -> Path:
    from .models import model_dir

    return model_dir().parent / "cuda"


def search_dirs() -> list[Path]:
    """Folders where CUDA 12 / cuDNN 9 libraries are usually found, own folder first."""
    out: list[Path] = [own_dir()]
    env = os.environ
    for k, v in env.items():
        if k.upper().startswith("CUDA_PATH") and v:
            out += [Path(v, "bin"), Path(v, "bin", "x64")]
    pf = env.get("ProgramFiles", r"C:\Program Files")
    pats = [
        rf"{pf}\NVIDIA GPU Computing Toolkit\CUDA\v12*\bin",
        rf"{pf}\NVIDIA\CUDNN\v9*\bin",
        rf"{pf}\NVIDIA\CUDNN\v9*\bin\12*",
    ]
    for base in {env.get("LOCALAPPDATA", ""), env.get("APPDATA", ""), env.get("USERPROFILE", ""), "C:\\"}:
        if not base:
            continue
        for sp in (r"Programs\Python\Python3*\Lib\site-packages", r"Python\Python3*\site-packages",
                   r"Python3*\Lib\site-packages", r"anaconda3\Lib\site-packages", r"miniconda3\Lib\site-packages",
                   r"anaconda3\envs\*\Lib\site-packages", r"miniconda3\envs\*\Lib\site-packages"):
            pats += [os.path.join(base, sp, "nvidia", "*", "bin"), os.path.join(base, sp, "torch", "lib")]
    for p in pats:
        out += [Path(x) for x in sorted(glob.glob(p), reverse=True)]
    out += [Path(p) for p in env.get("PATH", "").split(os.pathsep) if p]
    seen, uniq = set(), []
    for d in out:
        key = str(d).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    return uniq


def find(dirs: list[Path] | None = None) -> dict[str, Path]:
    """Where each needed DLL is on this computer (the first folder that has it)."""
    found: dict[str, Path] = {}
    for d in dirs if dirs is not None else search_dirs():
        try:
            if not d.is_dir():
                continue
        except OSError:
            continue
        for name in NEEDED:
            if name not in found and (d / name).is_file():
                found[name] = d / name
        if len(found) == len(NEEDED):
            break
    return found


def _wheel_url(name: str, major: int) -> tuple[str, int]:
    """(URL, size) of the newest Windows wheel of `name` with the given major version."""
    with urllib.request.urlopen(urllib.request.Request(PYPI.format(name=name), headers={"User-Agent": "FotoMedik"}),
                                timeout=30) as r:
        info = json.load(r)

    def key(v):
        parts = []
        for x in v.split("."):
            parts.append(int(x) if x.isdigit() else -1)
        return parts

    for ver in sorted(info.get("releases", {}), key=key, reverse=True):
        if not ver.split(".")[0].isdigit() or int(ver.split(".")[0]) != major or not all(p.isdigit() for p in ver.split(".")):
            continue
        for f in info["releases"][ver]:
            if f.get("filename", "").endswith("win_amd64.whl") and not f.get("yanked"):
                return f["url"], int(f.get("size") or 0)
    raise RuntimeError(f"no Windows package of {name} {major}.x found")


def download(names: list[str], progress=None, stop=None) -> None:
    """Fetch the packages carrying `names` (DLL file names) and keep their DLLs in own_dir()."""
    dest = own_dir()
    dest.mkdir(parents=True, exist_ok=True)
    pkgs = sorted({NEEDED[n] for n in names})
    for pkg, major in pkgs:
        url, size = _wheel_url(pkg, major)
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
                    progress(buf.tell(), total, pkg)
        with zipfile.ZipFile(buf) as z:
            for info in z.infolist():
                if info.filename.lower().endswith(".dll"):
                    tmp = dest / (Path(info.filename).name + ".part")
                    with z.open(info) as src, open(tmp, "wb") as out:
                        while True:
                            b = src.read(1 << 20)
                            if not b:
                                break
                            out.write(b)
                    os.replace(tmp, dest / Path(info.filename).name)


def load(found: dict[str, Path]) -> None:
    """Load the libraries (and cuDNN's parts) so ONNX Runtime's CUDA provider finds them."""
    import ctypes

    for d in {p.parent for p in found.values()}:
        if hasattr(os, "add_dll_directory"):
            _handles.append(os.add_dll_directory(str(d)))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    for name in LOAD_ORDER:
        _handles.append(ctypes.CDLL(str(found[name])))
    cudnn_dir = found["cudnn64_9.dll"].parent
    for part in CUDNN_PARTS:
        if (cudnn_dir / part).is_file():
            try:
                _handles.append(ctypes.CDLL(str(cudnn_dir / part)))
            except OSError:
                pass


def prepare(say=None, progress=None, stop=None) -> bool:
    """Make the CUDA libraries available to this process: found on the computer, else
    downloaded once. False (the reason told through say) when that cannot be done."""
    global _ready
    if _ready is not None:
        return _ready
    say = say or (lambda s: None)
    if not nvidia_present():
        _ready = False
        return False
    found = find()
    missing = [n for n in NEEDED if n not in found]
    if found:
        where = sorted({str(p.parent) for p in found.values()})
        say("CUDA libraries found on this computer: " + "; ".join(where))
    if missing:
        say(f"Downloading the NVIDIA CUDA libraries ({', '.join(sorted({NEEDED[n][0] for n in missing}))}), "
            "only the first time…")
        try:
            download(missing, progress, stop)
        except InterruptedError:
            _ready = None
            return False
        except Exception as e:
            say(f"The CUDA libraries could not be downloaded ({type(e).__name__}: {e}); the AI runs on the processor")
            _ready = False
            return False
        found.update({n: p for n, p in find([own_dir()]).items() if n in missing})
        if any(n not in found for n in NEEDED):
            say("The CUDA libraries are incomplete; the AI runs on the processor")
            _ready = False
            return False
    try:
        load(found)
    except OSError as e:
        say(f"The CUDA libraries could not be loaded ({e}); the AI runs on the processor")
        _ready = False
        return False
    _ready = True
    return True
