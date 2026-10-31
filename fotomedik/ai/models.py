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
_REL = "https://github.com/aluckyv/fotomedik/releases/download/models-sr/"
# enlargers of the thumbnail step that invent less than Real-ESRGAN (exported by the
# convert-models workflow from the authors' weights)
SR_NET = Model(
    "Real-ESRNet x4 (faithful super-resolution)",
    "RealESRNet_x4plus.onnx",
    (_REL + "RealESRNet_x4plus.onnx",),
    "7fbbe2b5676438c30482a3996591258f3e74413c0bbe819662c776fa0005f559",
    64,
    "BSD-3-Clause (Real-ESRGAN project, Xintao Wang et al.)",
)
SR_SWIN = Model(
    "SwinIR x4 (faithful super-resolution)",
    "SwinIR-M_classical_x4.onnx",
    (_REL + "SwinIR-M_classical_x4.onnx",),
    "a2cfaae74b606e7510983f5a76c1a09695bfa51582b1cb2c890ba19b89219800",
    55,
    "Apache-2.0 (SwinIR, Jingyun Liang et al.)",
)
# the thumbnail step's enlarger, by the key the user picks
ENLARGERS = {"esrgan": SR, "esrnet": SR_NET, "swinir": SR_SWIN}
DEFAULT_ENLARGER = "swinir"  # invents least; on the 2005 photos the best balance (Real-ESRGAN: sharper, adds lines)
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
    "b4fd55e0c260957955ae194cb85ea01b1dacae4f897936355920e61bd8c3ba16",
    288,
    "Apache-2.0 (FBCNN, Jiang, Zhang, Timofte)",
)
# FBCNN fine-tuned on the user's own photos (fotomedik.train.finetune --install puts it
# here); never downloaded: when the file is in the model folder, JPEG clean-up uses it
DEBLOCK_TUNED = Model(
    "FBCNN fine-tuned on your photos (JPEG clean-up)",
    "fbcnn-fotomedik.onnx",
    (),
    None,
    288,
    "Apache-2.0 (FBCNN, Jiang, Zhang, Timofte), fine-tuned locally",
)


# FBCNN fine-tuned with the thumbnail and the holes as inputs (finetune --guided --install)
DEBLOCK_GUIDED = Model(
    "FBCNN guided by the thumbnail, fine-tuned on your photos (JPEG clean-up)",
    "fbcnn-fotomedik-guided.onnx",
    (),
    None,
    288,
    "Apache-2.0 (FBCNN, Jiang, Zhang, Timofte), fine-tuned locally",
)


def deblocker() -> Model:
    """The JPEG clean-up network: the guided one, else the fine-tuned one, when installed."""
    for m in (DEBLOCK_GUIDED, DEBLOCK_TUNED):
        if (model_dir() / m.file).exists():
            return m
    return DEBLOCK


FACE_DET = Model(
    "YuNet (face detection)",
    "face_detection_yunet_2023mar.onnx",
    ("https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
     "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/"
     "face_detection_yunet_2023mar.onnx"),
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    1,
    "MIT (OpenCV Zoo, YuNet)",
)
FACE = Model(
    "GFPGAN 1.4 (face restoration)",
    "gfpgan_1.4.onnx",
    ("https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",),
    "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385",
    340,
    "Apache-2.0 (GFPGAN, Tencent ARC)",
)
_SD = "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/"
_CLIP = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/"
_CLIP_LICENSE = "MIT (OpenAI CLIP tokenizer files)"
# the text encoder's vocabulary: a prompt steers what Stable Diffusion paints into the holes
CLIP_VOCAB = Model("CLIP tokenizer (vocabulary)", "clip_vocab.json",
                   (_CLIP + "vocab.json", _SD + "tokenizer/vocab.json"),
                   "3f0c4f7d2086b61b38487075278ea9ed04edb53a03cbb045b86c27190fa8fb69", 1, _CLIP_LICENSE)
CLIP_MERGES = Model("CLIP tokenizer (merges)", "clip_merges.txt",
                    (_CLIP + "merges.txt", _SD + "tokenizer/merges.txt"),
                    "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a", 1, _CLIP_LICENSE)
_SD_LICENSE = "CreativeML OpenRAIL-M (Stable Diffusion 1.5 inpainting, Runway/Stability AI)"
SD_UNET = Model("Stable Diffusion inpainting (UNet)", "sd15inp_unet.onnx",
                (_SD + "unet/model.onnx", "https://huggingface.co/sona-forge/sd15-inpainting-fp16/resolve/main/unet/model.onnx"),
                "a3726e1944ec2fbc9596096cd001520aea789367f329360ee34eb520c751de16", 1720, _SD_LICENSE)
SD_VAE_ENC = Model("Stable Diffusion (image encoder)", "sd15inp_vae_encoder.onnx", (_SD + "vae_encoder/model.onnx",),
                   "f0da9070d007def0d6a4e7c10a21462bb6172e460ef2587c3fe91191397b4cea", 68, _SD_LICENSE)
SD_VAE_DEC = Model("Stable Diffusion (image decoder)", "sd15inp_vae_decoder.onnx", (_SD + "vae_decoder/model.onnx",),
                   "ce0ffca3fcfd0a2729a88d8d90dabf35c66d237404997e0461d52e46ac7a91bc", 99, _SD_LICENSE)
SD_TEXT = Model("Stable Diffusion (text encoder)", "sd15inp_text_encoder.onnx", (_SD + "text_encoder/model.onnx",),
                "8af516fb184866f8caed192ca3bf0636ef9d40d4f7799c05600cae431d18d8d1", 246, _SD_LICENSE)

STEP_MODELS = {
    "ai": (SR, INPAINT, FACE_DET),
    "deblock": (DEBLOCK,),
    "thumbsr": (SR, INPAINT, FACE_DET),
    "faces": (FACE_DET, FACE),
    "diffusion": (INPAINT, SR, SD_UNET, SD_VAE_ENC, SD_VAE_DEC, SD_TEXT, CLIP_VOCAB, CLIP_MERGES),  # SR enlarges the holes
}
ALL = tuple(m for ms in STEP_MODELS.values() for m in ms) + (SR_NET, SR_SWIN, DEBLOCK_TUNED, DEBLOCK_GUIDED)


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


def needed(steps, enlarger: str | None = None) -> list[Model]:
    """The models of `steps`; enlarger: the thumbnail step's enlarger (see ENLARGERS)."""
    out: list[Model] = []
    for st in steps:
        ms = STEP_MODELS.get(st, ())
        if st == "deblock":
            ms = (deblocker(),)
        if st == "thumbsr":
            ms = tuple(ENLARGERS.get(enlarger or DEFAULT_ENLARGER, SR) if m is SR else m for m in ms)
        for m in ms:
            if m not in out:
                out.append(m)
    return out


def missing(steps=("ai",), enlarger: str | None = None) -> list[Model]:
    return [m for m in needed(steps, enlarger) if not (model_dir() / m.file).exists()]


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
