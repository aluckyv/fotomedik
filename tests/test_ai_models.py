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
