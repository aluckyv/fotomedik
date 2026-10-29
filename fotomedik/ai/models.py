"""Model files: downloaded once on first use, checked by SHA-256, kept per user."""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Model:
    name: str
    file: str
    urls: tuple[str, ...]
    sha256: str
    size_mb: int
    license: str


SR = Model(
    "Real-ESRGAN x4 (super-resolution)",
    "real_esrgan_x4.onnx",
    ("https://huggingface.co/SceneWorks/real-esrgan-onnx/resolve/main/real_esrgan_x4.onnx",),
    "5c586662929cbc686c1a5c38d9c060dbdb4ea5863a1f7672b8c0761e6b89c033",
    67,
    "BSD-3-Clause (Real-ESRGAN, Xintao Wang et al.)",
)
INPAINT = Model(
    "LaMa (inpainting)",
    "lama_fp32.onnx",
    ("https://huggingface.co/sapienkit/LaMa-ONNX/resolve/main/lama_fp32.onnx",
     "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"),
    "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
    208,
    "Apache-2.0 (LaMa, Suvorov et al.; trained on Places2)",
)
ALL = (SR, INPAINT)


def model_dir() -> Path:
    env = os.environ.get("FOTOMEDIK_MODELS")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "FotoMedik" / "models"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "fotomedik" / "models"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def missing() -> list[Model]:
    return [m for m in ALL if not (model_dir() / m.file).exists()]


def ensure(model: Model, progress=None, stop=None) -> Path:
    """Path of the model file, downloading it first if needed.
    progress(done_bytes, total_bytes, model) is called while downloading."""
    path = model_dir() / model.file
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in model.urls:
        tmp = path.with_suffix(".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FotoMedik"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or model.size_mb * 1 << 20)
                done = 0
                h = hashlib.sha256()
                while True:
                    if stop is not None and stop.is_set():
                        raise InterruptedError("download stopped")
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total, model)
            if h.hexdigest() != model.sha256:
                raise ValueError(f"checksum mismatch ({h.hexdigest()[:12]}…)")
            os.replace(tmp, path)
            return path
        except InterruptedError:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as e:  # try the next mirror
            tmp.unlink(missing_ok=True)
            errors.append(f"{url}: {e}")
    raise RuntimeError(f"could not download {model.name}: " + "; ".join(errors))
