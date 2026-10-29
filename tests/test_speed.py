"""The speed work on the enhanced copies: presets that repaint at a reduced size, tiles
enlarged in batches, stopping in the middle, copies made while repairs run, networks
kept between runs, fp16 copies only when accurate and faster (all with stand-ins)."""

import sys
import threading
import time
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from fotomedik.ai import half, models
from fotomedik.ai.diffusion import PRESETS, repaint_holes, working_divisor
from fotomedik.ai.enhance import Engine, Stopped, enhance_steps, enlarge_holes
from fotomedik.ai.run import Enhancer, engine_for, release_engine, run_extras
from fotomedik.report.report import FileReport
from test_ai import FakeLama, FakeSR, _In


class BatchSR(FakeSR):
    """Real-ESRGAN stand-in that takes a batch and remembers the batch sizes it saw."""

    def __init__(self):
        self.batches = []

    def get_inputs(self):
        return [_In("input", ["batch", 3, "h", "w"])]

    def run(self, _out, feed):
        self.batches.append(feed["input"].shape[0])
        return super().run(_out, feed)


def _fake_sd():
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
        calls = 0

        def get_inputs(self):
            return [_In("sample", [1, 9, "h", "w"]), _In("timestep", [1]), _In("encoder_hidden_states", [1, 77, 768])]

        def run(self, _o, f):
            Unet.calls += 1
            return [np.zeros_like(f["sample"][:, :4])]

    class Text:
        def get_inputs(self):
            return [_In("input_ids", [1, 77])]

        def run(self, _o, f):
            return [np.zeros((1, 77, 768), np.float32)]

    from fotomedik.ai.diffusion import Diffusion

    return Diffusion(Unet(), Enc(), Dec(), Text()), Unet


def _photo(h, w, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 // w), (yy * 255 // h), ((xx + yy) * 127 // (w + h))], axis=-1)
    return np.clip(base + rng.integers(0, 30, (h, w, 3)), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- presets


def test_working_divisor_spares_small_photos():
    assert working_divisor(4032, 4) == 4 and working_divisor(4032, 2) == 2 and working_divisor(4032, 1) == 1
    assert working_divisor(1600, 4) == 2  # a quarter would be 400 px: too small
    assert working_divisor(800, 4) == 1
    assert PRESETS["fast"][0] == 4 and PRESETS["balanced"][0] == 2 and PRESETS["full"][0] == 1
    assert PRESETS["fast"][1] < PRESETS["balanced"][1] < PRESETS["full"][1]


def test_balanced_preset_repaints_at_half_size_and_enlarges_back():
    sd, unet = _fake_sd()
    eng = Engine(FakeSR(), FakeLama(), sd=sd)
    arr = _photo(1200, 1600)
    holes = np.zeros((1200, 1600), bool)
    holes[400:560, 300:900] = True
    unet.calls = 0
    out, what = repaint_holes(arr, holes, np.zeros_like(holes), eng, "balanced")
    assert out.shape == arr.shape and out.dtype == np.uint8
    assert np.array_equal(out[~holes], arr[~holes])
    assert "1/2 size" in what and "Real-ESRGAN" in what
    half_calls = unet.calls
    unet.calls = 0
    repaint_holes(arr, holes, np.zeros_like(holes), eng, "full")
    assert unet.calls > 2 * half_calls  # more windows and more passes at full size


def test_enlarge_holes_doubles_with_the_4x_network():
    lo = _photo(300, 400)
    need = np.zeros((300, 400), bool)
    need[100:200, 100:300] = True
    hi = enlarge_holes(lo, need, Engine(FakeSR()), factor=2)
    assert hi.shape == (600, 800, 3) and hi.dtype == np.float32
    # the stand-in repeats pixels 4x, halved again: the enlarged hole equals the pixels repeated 2x
    inside = np.rint(hi[200:400, 200:600] * 255)
    assert np.array_equal(inside, lo[100:200, 100:300].repeat(2, 0).repeat(2, 1))


def test_tiles_go_in_batches_when_the_network_takes_them():
    sr = BatchSR()
    eng = Engine(sr)
    eng.sr_batch = 4
    lo = _photo(300, 700)
    need = np.ones((300, 700), bool)  # 3 x 6 = 18 tiles of 128
    enlarge_holes(lo, need, eng, factor=4)
    assert sr.batches == [4, 4, 4, 4, 2]
    eng.sr_batch = 1
    sr.batches.clear()
    enlarge_holes(lo, need, eng, factor=4)
    assert sr.batches == [1] * 18


def test_stop_interrupts_a_photo():
    stop = threading.Event()
    stop.set()
    arr = _photo(256, 256)
    holes = np.ones((256, 256), bool)
    with pytest.raises(Stopped):
        enhance_steps(arr, holes, np.zeros_like(holes), Engine(FakeSR(), FakeLama()), ["ai"], stop=stop)


def test_steps_report_times_and_progress_labels():
    arr = _photo(256, 384)
    holes = np.zeros((256, 384), bool)
    holes[64:192, 64:320] = True
    seen = []
    out, done = enhance_steps(arr, holes, np.zeros_like(holes), Engine(FakeSR(), FakeLama()), ["ai"],
                              progress=lambda i, n, what: seen.append((i, n, what)))
    assert any("Real-ESRGAN" in d and " s)" in d for d in done)
    assert seen and all(w == "Real-ESRGAN tile" for _i, _n, w in seen) and seen[-1][0] == seen[-1][1]


# ----------------------------------------------------------------------------- enhancer


def _report(name, holes=True):
    return FileReport(name, name, 1000, 1000, 80.0, findings=[{"kind": "scan_error"}],
                      holes=[{"mcu_first": 0, "mcu_count": 2, "reason": "x"}] if holes else [])


def _write_photo(path: Path, healthy_jpeg_rst: bytes):
    path.write_bytes(healthy_jpeg_rst)


def test_enhancer_makes_copies_while_photos_arrive(tmp_path, healthy_jpeg_rst):
    out = tmp_path / "out"
    out.mkdir()
    names = ["a.jpg", "b.jpg", "c.jpg"]
    for n in names:
        _write_photo(out / n, healthy_jpeg_rst)
    made = []
    enh = Enhancer([None] * 3, out, ["ai"], on_done=made.append, engine=Engine(FakeSR(), FakeLama()))
    enh.start()
    enh.add(0, _report("a.jpg"))
    t0 = time.perf_counter()
    while not made and time.perf_counter() - t0 < 30:
        time.sleep(0.05)
    assert [r.name for r in made] == ["a.jpg"]  # the first copy is made before the others are repaired
    enh.add(1, _report("b.jpg", holes=False))  # nothing to enhance
    enh.add(2, _report("c.jpg"))
    assert enh.finish() == "CPU"
    assert [r.name for r in made] == ["a.jpg", "c.jpg"]
    assert (out / "a_enhanced.jpg").exists() and (out / "c_enhanced.jpg").exists()
    assert not (out / "b_enhanced.jpg").exists()
    assert made[0].ai_output == "a_enhanced.jpg" and made[0].enhanced


def test_enhancer_waits_for_the_other_shots_of_a_scene(tmp_path):
    t = datetime(2017, 5, 1, 12, 0, 0)
    enh = Enhancer([Path(f"{i}.jpg") for i in range(5)], tmp_path, ["burst"])
    enh.whens = [t, t + timedelta(seconds=30), t + timedelta(minutes=10), None, None]
    enh.repaired = {0: _report("0.jpg")}
    assert not enh._eligible(0)  # 1 is the same scene and not repaired yet
    enh.repaired[1] = _report("1.jpg")
    assert not enh._eligible(0)  # 3 and 4 have no time: close in the list, could be the scene
    enh.repaired[3] = enh.repaired[4] = _report("x.jpg")
    assert enh._eligible(0)  # 2 is ten minutes later: another scene
    enh.repaired = {2: _report("2.jpg")}
    assert not enh._eligible(2)
    enh.repaired.update({3: _report("3.jpg"), 4: _report("4.jpg")})
    assert enh._eligible(2)
    enh.whens = None
    assert enh._eligible(0)


def test_run_extras_after_the_fact_still_works(tmp_path, healthy_jpeg_rst):
    out = tmp_path / "out"
    out.mkdir()
    _write_photo(out / "a.jpg", healthy_jpeg_rst)
    rep = _report("a.jpg")
    said = []
    assert run_extras([(rep, None)], out, ["ai"], said.append, engine=Engine(FakeSR(), FakeLama())) == "CPU"
    assert rep.ai_output == "a_enhanced.jpg"
    assert any("done" in s for s in said)


# ----------------------------------------------------------------------------- engine


class _FakeORT(types.ModuleType):
    """A stand-in onnxruntime: a graphics card that refuses one model."""

    class ExecutionMode:
        ORT_SEQUENTIAL = 0

    class SessionOptions:
        enable_mem_pattern = True
        execution_mode = 0

    refused = set()
    gpu = True

    def __init__(self):
        super().__init__("onnxruntime")
        ort = self

        class InferenceSession:
            def __init__(self, path, sess_options=None, providers=()):
                self.path = Path(path)
                if "DmlExecutionProvider" in providers and self.path.name in ort.refused:
                    raise RuntimeError("the card refused")
                self.providers = list(providers)

            def get_providers(self):
                return self.providers

            def get_inputs(self):
                return [_In("input", ["batch", 3, "h", "w"])]

            def run(self, _o, feed):
                x = next(iter(feed.values()))
                return [x]

        self.InferenceSession = InferenceSession

    def get_available_providers(self):
        return ["DmlExecutionProvider", "CPUExecutionProvider"] if self.gpu else ["CPUExecutionProvider"]


@pytest.fixture
def fake_ort(monkeypatch, tmp_path):
    ort = _FakeORT()
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setenv("FOTOMEDIK_MODELS", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    for m in models.ALL:
        (tmp_path / "models" / m.file).write_bytes(b"onnx")
    release_engine()
    yield ort
    release_engine()


def test_engine_reports_where_every_network_runs(fake_ort):
    fake_ort.refused = {models.INPAINT.file}
    eng = Engine.load(steps=("ai",), half=False)
    assert eng.device == "GPU (DirectML)"
    assert eng.devices[models.SR.name] == "GPU (DirectML)" and eng.devices[models.INPAINT.name] == "CPU"
    assert eng.notes == [f"{models.INPAINT.name} runs on the processor: the graphics card refused it"]
    assert eng.sr_batch == 4  # batch dimension is symbolic and the card runs it
    assert eng.has("ai") and not eng.has("faces")
    eng.ensure(("faces",), half=False)
    assert eng.has("faces") and eng.face_s is not None and eng.sr is not None


def test_engine_is_kept_between_runs(fake_ort):
    said = []
    a = engine_for(["ai"], said.append, half=False)
    b = engine_for(["ai", "deblock"], said.append, half=False)
    assert a is b and b.has("deblock") and b.has("ai")
    assert not any("Downloading" in s for s in said)  # the files were there
    release_engine()
    assert engine_for(["ai"], said.append, half=False) is not a


def test_engine_on_the_processor_alone(fake_ort):
    fake_ort.gpu = False
    eng = Engine.load(steps=("ai",))
    assert eng.device == "CPU" and eng.sr_batch == 1 and eng.notes == []


# ----------------------------------------------------------------------------- fp16


class _S:
    """A session stand-in for half.session: output = input * gain (+ a per-run delay)."""

    def __init__(self, path, provider="DmlExecutionProvider", gain=1.0, delay=0.0, nan=False):
        self.path, self.provider, self.gain, self.delay, self.nan = Path(path), provider, gain, delay, nan

    def get_providers(self):
        return [self.provider]

    def get_inputs(self):
        return [_In("input", [1, 3, "h", "w"])]

    def run(self, _o, feed):
        time.sleep(self.delay)
        x = next(iter(feed.values())) * self.gain
        if self.nan:
            x = x + np.nan
        return [x]


def _setup(tmp_path, monkeypatch, make_half):
    path = tmp_path / models.SR.file
    path.write_bytes(b"fp32")
    monkeypatch.setattr(half, "convert", lambda p, out: Path(out).write_bytes(b"fp16"))

    def make(p):
        p = Path(p)
        return make_half(p) if p.name.endswith(".fp16.onnx") else _S(p, delay=0.002)

    return path, make


def test_fp16_copy_is_used_when_accurate_and_faster(tmp_path, monkeypatch):
    path, make = _setup(tmp_path, monkeypatch, lambda p: _S(p, gain=1.001))
    notes = []
    s = half.session(models.SR, path, make, None, notes)
    assert s.path.name == "real_esrgan_x4.fp16.onnx" and half.marker(path, "ok").exists() and notes == []
    # next time: straight to the copy
    assert half.session(models.SR, path, make).path.name.endswith(".fp16.onnx")


def test_fp16_copy_is_dropped_when_wrong_or_slow(tmp_path, monkeypatch):
    path, make = _setup(tmp_path, monkeypatch, lambda p: _S(p, nan=True))
    notes = []
    s = half.session(models.SR, path, make, None, notes)
    assert s.path == path and half.marker(path, "bad").exists() and not half.half_path(path).exists()
    assert notes and "fp32 kept" in notes[0] and "not accurate" in notes[0]
    # tried before: no second attempt, no new note
    assert half.session(models.SR, path, make, None, notes).path == path and len(notes) == 1

    path2 = tmp_path / models.DEBLOCK.file
    path2.write_bytes(b"fp32")
    slow = lambda p: _S(p, delay=0.01)  # noqa: E731
    s = half.session(models.DEBLOCK, path2, lambda p: slow(p) if "fp16" in p.name else _S(p, delay=0.002))
    assert s.path == path2 and half.marker(path2, "bad").exists()


def test_fp16_is_not_tried_on_the_processor_or_for_other_networks(tmp_path, monkeypatch):
    path, _make = _setup(tmp_path, monkeypatch, lambda p: _S(p))
    cpu = lambda p: _S(p, provider="CPUExecutionProvider")  # noqa: E731
    assert half.session(models.SR, path, cpu).path == path and not half.half_path(path).exists()
    lama = tmp_path / models.INPAINT.file
    lama.write_bytes(b"x")
    assert half.session(models.INPAINT, lama, lambda p: _S(p)).path == lama


# ----------------------------------------------------------------------------- prompt


def test_clip_tokenizer_merges_and_pads():
    from fotomedik.ai.tokenizer import ClipTokenizer

    vocab = {"<|startoftext|>": 49406, "<|endoftext|>": 49407, "a</w>": 320, "p": 1, "h": 2, "o": 3, "t": 4,
             "o</w>": 5, "ph": 6, "ot": 7, "phot": 8, "photo</w>": 9, ",</w>": 10, "x": 11}
    merges = [("p", "h"), ("o", "t"), ("ph", "ot"), ("phot", "o</w>")]
    tok = ClipTokenizer(vocab, merges)
    ids = tok.encode("A photo, photo")
    assert ids[:6] == [49406, 320, 9, 10, 9, 49407] and len(ids) == 77 and set(ids[6:]) == {49407}
    # an unknown piece ("x</w>") is skipped, not crashed on; a long prompt is cut to the length
    assert tok.encode("xx a", length=5) == [49406, 11, 320, 49407, 49407]
    assert len(tok.encode(" ".join(["a"] * 200))) == 77


def test_prompt_guidance_runs_the_network_twice(tmp_path):
    from fotomedik.ai.tokenizer import ClipTokenizer

    sd, unet = _fake_sd()
    sd.tokenizer = ClipTokenizer({"<|startoftext|>": 49406, "<|endoftext|>": 49407, "a</w>": 320}, [])
    img = _photo(64, 64)
    mask = np.zeros((64, 64), bool)
    mask[16:48, 16:48] = True
    unet.calls = 0
    sd.inpaint(img, mask, steps=4, strength=1.0, prompt=None)
    plain = unet.calls
    unet.calls = 0
    out = sd.inpaint(img, mask, steps=4, strength=1.0, prompt="a")
    assert out.shape == img.shape and np.array_equal(out[~mask], img[~mask])
    assert unet.calls == plain  # the fake takes a batch: prompt and no-prompt in one call
    assert sd._pair_ok is True and set(sd._emb) == {"", "a"}


def test_melted_detail_fades_into_soft_shapes():
    from fotomedik.recipes.dcfield import build_field

    rows, cols = 30, 40
    arr = np.zeros((rows, cols))
    okm = np.zeros((rows, cols), bool)
    okm[0] = okm[-1] = True
    arr[0, ::2] = 40.0  # a real row with fine detail (every other block bright)
    base = np.full((rows, cols), 10.0)  # a flat thumbnail
    f = build_field(arr, okm, 8, base, 2.0)
    detail = lambda r: np.abs(np.diff(f.val[r])).mean()  # noqa: E731
    assert detail(1) > 15  # next to the real row the detail is carried in as it is
    assert detail(4) < detail(1) * 0.5  # a few rows in it has melted
    assert detail(8) < 0.5  # and further in it is gone, not a streak
