"""Enhanced-copy steps with stand-in networks (real models: test_ai_models, in CI)."""

import numpy as np
import pytest

from fotomedik.ai.enhance import Engine, enhance_steps
from test_ai import FakeLama, FakeSR, _In

cv2 = pytest.importorskip("cv2")


def _scene(h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8, w // 8, 3)).astype(np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(40):  # corners and edges for the feature matcher
        x, y = rng.integers(0, w - 40), rng.integers(0, h - 40)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(10, 40)), int(y + rng.integers(10, 40))),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def test_burst_fills_holes_with_the_neighbouring_shot():
    from fotomedik.ai.burst import Shot, burst_fill
    from datetime import datetime

    truth = _scene()
    h, w = truth.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 1.5, 1.0)
    m[:, 2] += (12, -7)
    donor_rgb = cv2.warpAffine(truth, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    holes = np.zeros((h, w), bool)
    holes[240:400, 160:480] = True
    blurred = cv2.resize(cv2.resize(truth, (w // 8, h // 8), interpolation=cv2.INTER_AREA), (w, h))
    target_rgb = np.where(holes[..., None], blurred, truth)
    t = datetime(2005, 8, 11, 11, 29, 17)
    target = Shot(target_rgb, ~holes, np.ones_like(holes), t, "a")
    donor = Shot(donor_rgb, np.ones_like(holes), np.ones_like(holes), t.replace(second=21), "b")
    rgb, filled = burst_fill(target, [donor], holes)
    assert filled.sum() > 0.6 * holes.sum()
    err = np.abs(rgb.astype(float) - truth)[filled].mean()
    assert err < 12, err
    assert np.array_equal(rgb[~holes], target_rgb[~holes])


def test_burst_rejects_a_different_scene():
    from fotomedik.ai.burst import Shot, burst_fill
    from datetime import datetime

    a, b = _scene(seed=1), _scene(seed=2)
    holes = np.zeros(a.shape[:2], bool)
    holes[200:300, 200:400] = True
    t = datetime(2005, 8, 11, 11, 29, 17)
    rgb, filled = burst_fill(Shot(a, ~holes, np.ones_like(holes), t, "a"),
                             [Shot(b, np.ones_like(holes), np.ones_like(holes), t, "b")], holes)
    assert not filled.any()


class FakeIdentity:
    def get_inputs(self):
        return [_In("input", [1, 3, "h", "w"])]

    def run(self, _o, feed):
        return [feed["input"]]


def test_steps_report_what_they_did():
    arr = _scene(96, 128)
    holes = np.zeros((96, 128), bool)
    holes[40:80, 30:100] = True
    unknown = np.zeros_like(holes)
    unknown[50:60, 40:60] = True
    eng = Engine(FakeSR(), FakeLama(), deblock=FakeIdentity())

    def burst(rgb, hm):
        f = np.zeros_like(hm)
        f[40:50, 30:100] = True
        return rgb, f & hm

    out, done = enhance_steps(arr, holes, unknown, eng, ["burst", "deblock", "ai"], burst)
    assert done[0].startswith("similar shots") and "JPEG" in done[1]
    assert any("healed" in d and "LaMa" in d for d in done)
    from fotomedik.ai.enhance import JUNK_NEAR, _grow

    far = ~_grow(holes, JUNK_NEAR + 8)  # next to a hole a wrongly coloured stripe may be healed too
    assert np.array_equal(out[far], arr[far])  # identity clean-up: real pixels as they were


def test_healing_fills_every_hole_window_by_window():
    from fotomedik.ai.enhance import heal_holes

    arr = _scene(1100, 1300)
    holes = np.zeros((1100, 1300), bool)
    holes[100:116, :] = True  # a thin stripe across the picture
    holes[500:1050, 200:1150] = True  # a wide hole: seen through a twice larger view
    calls = []

    class Lama(FakeLama):
        def run(self, names, feeds):
            calls.append(1)
            return super().run(names, feeds)

    st = {}
    out = heal_holes(arr, holes, Engine(None, Lama()), stats=st)
    assert st["tiles"] == len(calls) > 4
    assert np.array_equal(out[~holes], arr[~holes])


def test_faces_are_restored_where_found(monkeypatch):
    from fotomedik.ai import faces

    arr = np.full((300, 400, 3), 90, np.uint8)
    pts = np.array([[180, 120], [220, 120], [200, 145], [185, 170], [215, 170]], np.float32)
    monkeypatch.setattr(faces, "detect", lambda a, p: [pts])

    class FakeGFPGAN:
        def get_inputs(self):
            return [_In("input", [1, 3, 512, 512])]

        def run(self, _o, feed):
            return [np.ones_like(feed["input"])]  # a white "face"

    out, n = faces.restore_faces(arr, Engine(face=FakeGFPGAN(), face_det="x"))
    assert n == 1
    assert out[145, 200].mean() > 150  # the face centre took the restored face
    assert out[10, 10].mean() == 90  # far away nothing changed


def test_diffusion_fill_keeps_real_pixels():
    from fotomedik.ai.diffusion import Diffusion, diffusion_fill

    class Enc:
        def get_inputs(self):
            return [_In("sample", [1, 3, "h", "w"])]

        def run(self, _o, f):
            x = f["sample"]
            b, c, h, w = x.shape
            return [x.reshape(b, c, h // 8, 8, w // 8, 8).mean(axis=(3, 5))[:, [0, 1, 2, 0]]]

    class Dec:
        def get_inputs(self):
            return [_In("latent_sample", [1, 4, "h", "w"])]

        def run(self, _o, f):
            z = f["latent_sample"][:, :3]
            return [np.clip(z.repeat(8, axis=2).repeat(8, axis=3), -1, 1)]

    class Unet:
        def get_inputs(self):
            return [_In("sample", [1, 9, "h", "w"]), _In("timestep", [1]), _In("encoder_hidden_states", [1, 77, 768])]

        def run(self, _o, f):
            return [np.zeros_like(f["sample"][:, :4])]

    class Text:
        def get_inputs(self):
            return [_In("input_ids", [1, 77])]

        def run(self, _o, f):
            return [np.zeros((1, 77, 768), np.float32)]

    sd = Diffusion(Unet(), Enc(), Dec(), Text())
    arr = _scene(128, 192)
    holes = np.zeros((128, 192), bool)
    holes[40:90, 50:150] = True
    out = diffusion_fill(arr, holes, np.zeros_like(holes), sd, size=64, steps=3)
    assert out.shape == arr.shape
    assert np.array_equal(out[~holes], arr[~holes])


def test_wrongly_coloured_stripes_next_to_holes_are_healed_too():
    from fotomedik.ai.enhance import JUNK_TOL, _grow, colour_stripes

    arr = np.full((256, 256, 3), 128, np.uint8)
    arr[:, :, 0] = np.linspace(100, 160, 256).astype(np.uint8)[None, :]  # a gentle real picture
    holes = np.zeros((256, 256), bool)
    holes[96:128, :] = True
    arr[128:144, 40:200] = (255, 60, 170)  # a pink line of rotted data right under the hole
    arr[200:216, 20:60] = (200, 40, 40)  # a red object far from any hole stays
    j = colour_stripes(arr, _grow(holes, 16) & ~holes, JUNK_TOL)
    assert j[128:144, 60:180].mean() > 0.9
    assert not j[200:216, 20:60].any() and not j[:96].any()

    out, done = enhance_steps(arr, holes, holes.copy(), Engine(None, FakeLama()), ["ai"])
    assert "stripes" in done[0]
    assert np.abs(out[136, 100].astype(int) - (255, 60, 170)).sum() > 100  # the pink is gone
    assert np.array_equal(out[200:216, 20:60], arr[200:216, 20:60])
