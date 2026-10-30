import pytest
"""AI copy with stand-in networks (the real models are tested in CI, see test_ai_models)."""

import io

import numpy as np
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
