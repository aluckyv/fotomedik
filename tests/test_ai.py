"""AI copy with stand-in networks (the real models are tested in CI, see test_ai_models)."""

import io

import numpy as np
import pytest
from PIL import Image

from fotomedik.ai.enhance import Engine, enhance, enhance_file, mcu_geometry, mcu_mask
from fotomedik.ai.run import ai_name, run_ai
from fotomedik.cli import repair_bytes
from fotomedik.jpeg import markers as M
from synth import flip_bits
from test_decoder import breaking_flip


class _In:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class FakeSR:
    """Enlarges 4x by repeating pixels."""

    def get_inputs(self):
        return [_In("input", [1, 3, "h", "w"])]

    def run(self, _out, feed):
        x = feed["input"]
        return [x.repeat(4, axis=2).repeat(4, axis=3)]


class FakeLama:
    """Paints the masked area a fixed gray (output 0..255, like the real export)."""

    def get_inputs(self):
        return [_In("image", [1, 3, 512, 512]), _In("mask", [1, 1, 512, 512])]

    def run(self, _out, feed):
        img, m = feed["image"] * 255.0, feed["mask"]
        return [img * (1 - m) + 200.0 * m]


def test_enhance_changes_only_holes():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
    holes = np.zeros((96, 128), dtype=bool)
    holes[40:80, 30:100] = True
    unknown = np.zeros_like(holes)
    unknown[50:70, 40:90] = True
    out = np.asarray(enhance(Image.fromarray(arr), holes, unknown, Engine(FakeSR(), FakeLama())))
    assert np.array_equal(out[~holes], arr[~holes])
    assert abs(out[58:62, 60:70].astype(float).mean() - 200) < 25  # filled by the inpainter
    assert not np.array_equal(out[holes], arr[holes])


def test_mcu_mask_geometry(healthy_jpeg):
    geo = mcu_geometry(healthy_jpeg)
    mw, mh, per_row, w, h = geo
    m = mcu_mask([(per_row + 1, 2)], geo)
    assert m.shape == (h, w)
    assert m.sum() == 2 * mw * mh
    assert m[mh, mw] and m[mh, 3 * mw - 1] and not m[0, 0]


def _damaged(healthy_jpeg_rst):
    res = M.parse_segments(healthy_jpeg_rst)
    off, mask = breaking_flip(healthy_jpeg_rst, res.scan_start + 300, res.scan_start + 900)
    return flip_bits(healthy_jpeg_rst, off, mask)


def test_ai_copy_of_a_repaired_photo(tmp_path, healthy_jpeg_rst):
    ctx, rep = repair_bytes(_damaged(healthy_jpeg_rst), "x.jpg", {"time_budget": 1.0, "keep_damaged": False})
    if not rep.holes:  # the bit-flip search fixed it completely: nothing to enhance
        assert enhance_file(ctx.data, rep.holes, rep.unknown, Engine(FakeSR(), FakeLama())) is None
        return
    out = enhance_file(ctx.data, rep.holes, rep.unknown, Engine(FakeSR(), FakeLama()))
    im = Image.open(io.BytesIO(out))
    assert im.size == Image.open(io.BytesIO(ctx.data)).size


def test_run_ai_writes_copy_next_to_repair(tmp_path, healthy_jpeg_rst):
    ctx, rep = repair_bytes(_damaged(healthy_jpeg_rst), "x.jpg", {"time_budget": 0.0})
    rep.holes = rep.holes or [{"mcu_first": 3, "mcu_count": 5, "reason": "test"}]
    rep.output_name = "sub/x.jpg"
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.jpg").write_bytes(ctx.data)
    seen = []
    dev = run_ai([rep], tmp_path, on_done=seen.append, engine=Engine(FakeSR(), FakeLama(), "CPU"))
    assert dev == "CPU" and seen == [rep]
    assert rep.ai_output == ai_name("sub/x.jpg") == "sub/x_enhanced.jpg"
    assert rep.enhanced and "healed" in rep.enhanced[-1]
    Image.open(tmp_path / "sub" / "x_enhanced.jpg").load()


def test_block_means_are_kept_in_known_holes():
    from fotomedik.ai.enhance import keep_block_means

    rng = np.random.default_rng(2)
    ref = rng.integers(40, 200, (64, 64, 3)).astype(np.uint8)
    out = np.clip(ref.astype(int) + 30, 0, 255).astype(np.uint8)  # the "network" brightened everything
    holes = np.zeros((64, 64), dtype=bool)
    holes[16:48, 16:48] = True
    fixed = keep_block_means(out, ref, holes, holes)
    m = lambda a: a[24:40, 24:40].astype(float).reshape(2, 8, 2, 8, 3).mean(axis=(1, 3))  # noqa: E731
    assert np.abs(m(fixed) - m(ref)).max() < 4
    assert np.array_equal(fixed[~holes], out[~holes])


def test_lama_failing_on_the_graphics_card_moves_to_the_processor(monkeypatch, tmp_path):
    """DirectML can load LaMa and then fail while running it: the processor takes over."""
    ort = pytest.importorskip("onnxruntime")

    class GpuLama(FakeLama):
        def get_providers(self):
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

        def run(self, _out, feed):
            raise RuntimeError("Non-zero status code returned while running MatMul node. The parameter is incorrect.")

    made = []

    class CpuSession(FakeLama):
        def __init__(self, path, providers=None):
            made.append((path, providers))

    monkeypatch.setattr(ort, "InferenceSession", CpuSession)
    eng = Engine(FakeSR(), GpuLama(), "GPU (DirectML)")
    eng.paths["LaMa (inpainting)"] = tmp_path / "lama_fp32.onnx"
    rgb = np.zeros((512, 512, 3), np.uint8)
    m = np.zeros((512, 512), bool)
    m[100:200, 100:200] = True
    out = eng.fill(rgb, m)
    assert out[150, 150, 0] == 200 and made and made[0][1] == ["CPUExecutionProvider"]
    assert eng.devices["LaMa (inpainting)"] == "CPU" and any("processor" in n for n in eng.notes)
    eng.fill(rgb, m)  # stays on the processor
    assert len(made) == 1


def test_errors_go_to_a_file_in_the_output_folder(tmp_path):
    from fotomedik import errlog

    errlog.start(tmp_path)
    try:
        raise ValueError("boom")
    except ValueError as e:
        errlog.log_error(tmp_path, "enhanced copy of x.jpg", e)
    text = (tmp_path / errlog.LOG_NAME).read_text(encoding="utf-8")
    assert "enhanced copy of x.jpg" in text and "ValueError: boom" in text and "Traceback" in text
    assert errlog.count(tmp_path) == 1


def test_thumbnail_enlarged_fills_holes_and_keeps_real_pixels():
    from fotomedik.ai.enhance import enhance_steps

    rng = np.random.default_rng(0)
    arr = rng.integers(60, 200, (128, 160, 3), dtype=np.uint8)
    holes = np.zeros((128, 160), bool)
    holes[40:88, :] = True
    small = rng.integers(60, 200, (8, 10, 3), dtype=np.uint8)  # a tiny "thumbnail"
    eng = Engine(FakeSR(), FakeLama())
    out, done = enhance_steps(arr, holes, np.zeros_like(holes), eng, ["thumbsr"], thumb_small=small)
    assert np.array_equal(out[~holes], arr[~holes])
    assert any("Real-ESRGAN" in d for d in done)
    # without a thumbnail it heals as usual
    out2, done2 = enhance_steps(arr, holes, np.zeros_like(holes), eng, ["thumbsr"], thumb_small=None)
    assert any("LaMa" in d for d in done2) and np.array_equal(out2[~holes], arr[~holes])


def test_thumbnail_enlarged_leaves_faces_smooth():
    """On a face the thumbnail is only enlarged smoothly: Real-ESRGAN would invent eyes."""
    from fotomedik.ai.enhance import _smooth_thumbnail, heal_on_thumbnail

    class NoisySR(FakeSR):  # an "enlarger" that invents detail
        def run(self, _names, feeds):
            y = super().run(_names, feeds)[0]
            noise = np.random.default_rng(1).uniform(-0.3, 0.3, y.shape).astype(y.dtype)
            return [np.clip(y + noise, 0, 1)]

    rng = np.random.default_rng(0)
    arr = rng.integers(60, 200, (256, 256, 3), dtype=np.uint8)
    holes = np.zeros((256, 256), bool)
    holes[64:192, :] = True
    small = rng.integers(60, 200, (16, 16, 3), dtype=np.uint8)
    faces = np.zeros_like(holes)
    faces[96:160, 32:96] = True
    eng = Engine(NoisySR(), None)
    out, _ = heal_on_thumbnail(arr, holes, np.zeros_like(holes), small, eng, faces=faces)
    soft = _smooth_thumbnail(small, arr.shape)
    inner = np.zeros_like(holes)
    inner[112:144, 48:80] = True  # well inside the face
    other = np.zeros_like(holes)
    other[112:144, 160:192] = True  # a hole away from the face

    def rough(m):
        return float(np.abs(out[m].astype(np.float32) - soft[m]).mean())
    assert rough(inner) < 0.5 * rough(other)


def test_thumbnail_enlarged_takes_rotted_stripes_from_the_thumbnail():
    """A pink stripe of rotted data at a hole's edge is replaced from the thumbnail too."""
    from fotomedik.ai.enhance import enhance_steps

    arr = np.full((192, 256, 3), 120, np.uint8)
    holes = np.zeros((192, 256), bool)
    holes[96:160, :] = True
    arr[88:96, 32:224] = (250, 40, 200)  # decoded stripe right above the hole
    small = np.full((12, 16, 3), 120, np.uint8)
    thumb = np.full_like(arr, 120)
    eng = Engine(FakeSR(), FakeLama())
    out, done = enhance_steps(arr, holes, np.zeros_like(holes), eng, ["thumbsr"], thumb=thumb, thumb_small=small)
    got = out[88:96, 64:192].astype(int)
    assert np.abs(got[..., 0] - got[..., 1]).max() < 20  # gray again (the fake LaMa paints gray), not pink
    assert any("stripes" in d for d in done)
    assert np.array_equal(out[:72], arr[:72])  # the real picture above stays


def test_spread_correction_is_smooth():
    """Block averages dropped in a hole take a smooth correction from their neighbours,
    not a checkerboard."""
    from fotomedik.ai.enhance import _spread

    known = np.zeros((20, 20), bool)
    known[:, 0] = known[:, 19] = True
    diff = np.zeros((20, 20, 3))
    diff[:, 0], diff[:, 19] = 10, -10
    row = _spread(diff, known, ~known)[10, :, 0]
    assert np.all(np.diff(row) <= 1e-9)  # falls steadily from one side to the other
    assert abs(row[9] + row[10]) < 1e-6


def test_enlarger_choice():
    """The thumbnail step uses the enlarger picked for it; the other steps keep Real-ESRGAN."""
    from fotomedik.ai import models
    from fotomedik.ai.enhance import enhance_steps

    assert models.SR_SWIN in models.needed(["thumbsr"], "swinir")
    assert models.SR not in models.needed(["thumbsr"], "esrnet")
    assert models.SR in models.needed(["thumbsr", "ai"], "esrnet")
    assert models.needed(["thumbsr"])[0] is models.ENLARGERS[models.DEFAULT_ENLARGER]

    class Counting(FakeSR):
        calls = 0

        def run(self, out, feed):
            Counting.calls += 1
            return super().run(out, feed)

    rng = np.random.default_rng(0)
    arr = rng.integers(60, 200, (128, 160, 3), dtype=np.uint8)
    holes = np.zeros((128, 160), bool)
    holes[40:88, :] = True
    eng = Engine(FakeSR(), FakeLama())
    eng.thumb_sr, eng.enlarger = Counting(), "swinir"
    out, done = enhance_steps(arr, holes, np.zeros_like(holes), eng, ["thumbsr"],
                              thumb_small=rng.integers(60, 200, (8, 10, 3), dtype=np.uint8))
    assert Counting.calls > 0 and any("SwinIR" in d for d in done)


def test_batch_only_when_tiles_stay_apart():
    """A network whose batch mixes the tiles (SwinIR's export) enlarges one tile at a time."""
    from fotomedik.ai.enhance import _batch_ok

    class Mixing(FakeSR):
        def get_inputs(self):
            return [_In("input", ["n", 3, "h", "w"])]

        def run(self, out, feed):
            x = feed["input"]
            y = super().run(out, {"input": x})[0]
            return [y + y.mean(axis=0, keepdims=True) * (len(x) > 1)]  # the batch leaks into every tile

    class Clean(FakeSR):
        def get_inputs(self):
            return [_In("input", ["n", 3, "h", "w"])]

    assert not _batch_ok(Mixing())
    assert _batch_ok(Clean())
    assert not _batch_ok(FakeSR())  # a fixed batch of one


def test_thumbnail_rotted_in_part_is_rescued_from_the_photo():
    """A thumbnail whose bottom rotted: its real part stays, the rest comes from the photo
    where the photo is decoded; where neither knows, nothing is a witness."""
    from fotomedik.ai.enhance import rescue_small

    rng = np.random.default_rng(3)
    arr = rng.integers(60, 200, (160, 120, 3), dtype=np.uint8)
    holes = np.zeros((160, 120), bool)
    holes[120:, :60] = True  # bottom left lost in the photo
    rgb = np.full((40, 30, 3), 90, np.uint8)
    real = np.ones((40, 30), bool)
    real[30:] = False  # the bottom quarter of the thumbnail rotted
    small, seen = rescue_small(rgb, real, arr, holes)
    assert small[10, 10, 0] == 90 and seen[:29].all()  # the real part (one row in from the damage)
    assert seen[35, 25] and not seen[35, 5]  # bottom right from the photo; bottom left: no witness
    assert abs(int(small[35, 25, 0]) - int(arr[140:144, 100:104, 0].mean())) <= 2
    assert rescue_small(rgb, real, arr[:, :60], holes[:, :60]) == (None, None)  # another shape


def test_holes_the_rescued_thumbnail_never_saw_are_healed_from_the_picture():
    from fotomedik.ai.enhance import enhance_steps

    rng = np.random.default_rng(4)
    arr = rng.integers(60, 200, (128, 160, 3), dtype=np.uint8)
    holes = np.zeros((128, 160), bool)
    holes[40:56, :] = True  # seen by the thumbnail
    holes[96:120, :] = True  # where the thumbnail rotted and the photo is lost: no witness
    small = rng.integers(60, 200, (8, 10, 3), dtype=np.uint8)
    seen = np.ones((8, 10), bool)
    seen[6:] = False
    unknown = np.zeros_like(holes)
    unknown[96:120] = True  # the repair had no colours there either
    eng = Engine(FakeSR(), FakeLama())
    out, done = enhance_steps(arr, holes, unknown, eng, ["thumbsr"], thumb_small=small, thumb_seen=seen)
    assert np.array_equal(out[~holes], arr[~holes])
    assert any("rescued" in d for d in done)
    assert (out[100:116, 8:150] == 200).all()  # FakeLama's fill where the thumbnail saw nothing
    assert not (out[44:52, 8:150] == 200).all()  # the thumbnail's where it saw
