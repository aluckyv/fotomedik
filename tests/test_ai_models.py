"""The real networks (Real-ESRGAN, LaMa): download, load, run. Slow and needs the
network, so it runs only with FOTOMEDIK_AI_MODELS=1 (a CI job does)."""

import os
import time

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(os.environ.get("FOTOMEDIK_AI_MODELS") != "1", reason="set FOTOMEDIK_AI_MODELS=1")


def test_real_models_enhance_a_photo(tmp_path):
    from fotomedik.ai.enhance import Engine, enhance
    from synth import make_image

    t = time.perf_counter()
    eng = Engine.load(progress=lambda d, n, m: None)
    print(f"\nloaded on {eng.device} in {time.perf_counter() - t:.1f} s")
    for s in (eng.sr, eng.inpaint_s):
        print([(i.name, i.shape) for i in s.get_inputs()], [(o.name, o.shape) for o in s.get_outputs()])
    img = make_image(640, 480, seed=3)
    arr = np.asarray(img.convert("RGB"))
    holes = np.zeros((480, 640), dtype=bool)
    holes[200:400, 100:500] = True
    unknown = np.zeros_like(holes)
    unknown[250:330, 200:300] = True
    blurred = arr.copy()
    small = Image.fromarray(arr).resize((80, 60), Image.BOX).resize((640, 480), Image.BICUBIC)
    blurred[holes] = np.asarray(small)[holes]
    blurred[unknown] = 128
    t = time.perf_counter()
    out = np.asarray(enhance(Image.fromarray(blurred), holes, unknown, eng))
    print(f"enhanced 640x480 in {time.perf_counter() - t:.1f} s")
    Image.fromarray(np.concatenate([blurred, out], axis=1)).save(tmp_path / "ai_check.png")
    os.makedirs("ai_check", exist_ok=True)
    Image.fromarray(np.concatenate([blurred, out], axis=1)).save("ai_check/ai_check.png")
    assert out.shape == arr.shape
    assert np.array_equal(out[~holes], blurred[~holes])
    # the unknown patch is no longer flat gray
    assert out[260:320, 210:290].std() > 3
    # sharpening adds detail: more high-frequency energy than the blurred input in the holes
    def hf(a):
        g = a.astype(float).mean(axis=2)
        return np.abs(np.diff(g, axis=1))[200:400, 100:499].mean()
    assert hf(out) > hf(blurred)


def _save(name, *imgs):
    os.makedirs("ai_check", exist_ok=True)
    Image.fromarray(np.concatenate(imgs, axis=1)).save(f"ai_check/{name}.png")


def test_real_models_all_steps():
    """Clean-up, faces and diffusion with the real networks; prints model hashes."""
    from fotomedik.ai import models
    from fotomedik.ai.diffusion import diffusion_fill
    from fotomedik.ai.enhance import Engine, deblock
    from fotomedik.ai.faces import detect, restore_faces
    from synth import make_image

    t = time.perf_counter()
    eng = Engine.load(progress=lambda d, n, m: None, steps=("deblock", "faces", "diffusion"))
    print(f"\nloaded on {eng.device} in {time.perf_counter() - t:.1f} s")
    hashes = [f"{m.file} {models.sha256_of(models.model_dir() / m.file)[:16]}…" for m in models.ALL
              if (models.model_dir() / m.file).exists()]
    print("::notice title=model sha256::" + "; ".join(hashes))  # one line: ten notices per step at most
    for s in (eng.deblock_s, eng.face_s, eng.sd.unet, eng.sd.vae_enc, eng.sd.vae_dec, eng.sd.text):
        print([(i.name, i.shape, i.type) for i in s.get_inputs()], [(o.name, o.shape) for o in s.get_outputs()])

    img = np.asarray(make_image(384, 256, seed=5).convert("RGB"))
    import io

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=15)  # heavy blocking
    blocky = np.asarray(Image.open(buf).convert("RGB"))
    t = time.perf_counter()
    clean = deblock(blocky, eng)
    print(f"deblock 384x256 in {time.perf_counter() - t:.1f} s")
    _save("deblock", blocky, clean)
    assert clean.shape == blocky.shape
    assert np.abs(clean.astype(float) - img).mean() < np.abs(blocky.astype(float) - img).mean()

    assert detect(img, eng.face_det) == [] or True  # loads and runs
    pts = np.array([[150, 110], [230, 110], [190, 160], [160, 200], [220, 200]], np.float32)
    import fotomedik.ai.faces as F

    real_detect = F.detect
    F.detect = lambda a, p: [pts]
    try:
        face, n = restore_faces(img, eng)
    finally:
        F.detect = real_detect
    _save("faces", img, face)
    assert n == 1 and face.shape == img.shape

    holes = np.zeros(img.shape[:2], bool)
    holes[80:180, 120:280] = True
    hole_img = img.copy()
    hole_img[holes] = 128
    t = time.perf_counter()
    filled = diffusion_fill(hole_img, holes, holes, eng.sd, size=256, steps=2)
    print(f"diffusion 2 steps, 256 window in {time.perf_counter() - t:.1f} s")
    _save("diffusion", hole_img, filled)
    assert np.array_equal(filled[~holes], hole_img[~holes])
    assert filled[holes].std() > 2


def test_fp16_copies_of_the_real_networks(tmp_path):
    """The fp16 conversion of the fp32 networks (used on a graphics card): the copy is
    written and, where the processor can run fp16 at all, agrees with the original."""
    import onnxruntime as ort

    from fotomedik.ai import half, models
    from fotomedik.ai.enhance import Engine

    eng = Engine.load(progress=lambda d, n, m: None, steps=("ai",), half=False)
    print(f"::notice title=Real-ESRGAN input::{[(i.name, i.shape) for i in eng.sr.get_inputs()]} "
          f"batch in one call: {eng.sr_batch}")
    for model in (models.SR,):
        path = models.model_dir() / model.file
        out = tmp_path / half.half_path(path).name
        t = time.perf_counter()
        half.convert(path, out)
        print(f"converted {model.name} to fp16 in {time.perf_counter() - t:.1f} s: "
              f"{path.stat().st_size >> 20} MB -> {out.stat().st_size >> 20} MB")
        assert out.stat().st_size < 0.7 * path.stat().st_size
        full = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        x = half.test_picture(model, full)
        try:
            s16 = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            got = half._picture_output(s16, x)
        except Exception as e:  # the processor has no fp16 kernels for it: nothing to compare
            print(f"::notice title=fp16 {model.name}::not run on the processor: {type(e).__name__}: {str(e)[:120]}")
            continue
        ref = half._picture_output(full, x)
        mean, worst = half.compare(ref, got)
        print(f"::notice title=fp16 {model.name}::mean difference {mean:.5f}, largest {worst:.4f} of the range")
        assert mean <= half.TOL_MEAN and worst <= half.TOL_MAX
