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
    sha256: str | None  # None: not pinned yet (the computed hash is saved next to the file)
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
DEBLOCK = Model(
    "FBCNN (JPEG artifact removal)",
    "fbcnn-color.onnx",
    ("https://huggingface.co/Hawkynt/photomanager-models/resolve/main/fbcnn-color.onnx",),
    None,
    288,
    "Apache-2.0 (FBCNN, Jiang, Zhang, Timofte)",
)
FACE_DET = Model(
    "YuNet (face detection)",
    "face_detection_yunet_2023mar.onnx",
    ("https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
     "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/"
     "face_detection_yunet_2023mar.onnx"),
    None,
    1,
    "MIT (OpenCV Zoo, YuNet)",
)
FACE = Model(
    "GFPGAN 1.4 (face restoration)",
    "gfpgan_1.4.onnx",
    ("https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",),
    None,
    340,
    "Apache-2.0 (GFPGAN, Tencent ARC)",
)
_SD = "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/"
_SD_LICENSE = "CreativeML OpenRAIL-M (Stable Diffusion 1.5 inpainting, Runway/Stability AI)"
SD_UNET = Model("Stable Diffusion inpainting (UNet)", "sd15inp_unet.onnx",
                (_SD + "unet/model.onnx", "https://huggingface.co/sona-forge/sd15-inpainting-fp16/resolve/main/unet/model.onnx"),
                None, 1720, _SD_LICENSE)
SD_VAE_ENC = Model("Stable Diffusion (image encoder)", "sd15inp_vae_encoder.onnx", (_SD + "vae_encoder/model.onnx",),
                   None, 68, _SD_LICENSE)
SD_VAE_DEC = Model("Stable Diffusion (image decoder)", "sd15inp_vae_decoder.onnx", (_SD + "vae_decoder/model.onnx",),
                   None, 99, _SD_LICENSE)
SD_TEXT = Model("Stable Diffusion (text encoder)", "sd15inp_text_encoder.onnx", (_SD + "text_encoder/model.onnx",),
                None, 246, _SD_LICENSE)

STEP_MODELS = {
    "ai": (SR, INPAINT),
    "deblock": (DEBLOCK,),
    "faces": (FACE_DET, FACE),
    "diffusion": (SD_UNET, SD_VAE_ENC, SD_VAE_DEC, SD_TEXT),
}
ALL = tuple(m for ms in STEP_MODELS.values() for m in ms)


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


def needed(steps) -> list[Model]:
    out: list[Model] = []
    for st in steps:
        for m in STEP_MODELS.get(st, ()):
            if m not in out:
                out.append(m)
    return out


def missing(steps=("ai",)) -> list[Model]:
    return [m for m in needed(steps) if not (model_dir() / m.file).exists()]


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
            if model.sha256 is not None and h.hexdigest() != model.sha256:
                raise ValueError(f"checksum mismatch ({h.hexdigest()[:12]}…)")
            os.replace(tmp, path)
            if model.sha256 is None:  # remembered, to pin it in a later version
                path.with_name(path.name + ".sha256").write_text(h.hexdigest() + "\n", encoding="ascii")
            return path
        except InterruptedError:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as e:  # try the next mirror
            tmp.unlink(missing_ok=True)
            errors.append(f"{url}: {e}")
    raise RuntimeError(f"could not download {model.name}: " + "; ".join(errors))
