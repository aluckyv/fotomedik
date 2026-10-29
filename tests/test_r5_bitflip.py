import time

from fotomedik.jpeg import decoder as D
from fotomedik.jpeg import markers as M
from fotomedik.jpeg.tables import FrameTables
from fotomedik.recipes.base import Context
from fotomedik.recipes.r5_bitflip import BitFlipRepair
from fotomedik.recipes.r9_holes import FillHoles
from fotomedik.recipes.registry import run_pipeline
from synth import flip_bits
from test_decoder import breaking_flip


def _decode(data, **kw):
    res = M.parse_segments(data)
    ft = FrameTables.from_parse(res)
    return D.decode_scan(data, res, ft, **kw), res, ft


def _run(data, **options):
    return run_pipeline(Context(data=data, name="x.jpg", options=options), recipes=[BitFlipRepair()])


def assert_equivalent(repaired, healthy, tol=4):
    """Repaired file decodes cleanly and every block's DC is within `tol` quantised units of the truth.

    A flip inside a magnitude field leaves the stream syntactically valid, so several
    different flips can 'complete' an interval; they differ only by tiny coefficient
    changes that are invisible. Byte equality is therefore the wrong test.
    """
    import numpy as np

    sr, _, _ = _decode(repaired)
    assert sr.error is None and all(iv.ok for iv in sr.intervals)
    ref, _, _ = _decode(healthy)
    for cid in ref.dc:
        a, b = sr.dc[cid], ref.dc[cid]
        assert not np.isnan(a).any()
        assert np.nanmax(np.abs(a - b)) <= tol, f"component {cid}: max DC diff {np.nanmax(np.abs(a - b))}"


def test_decoder_resumes_mid_interval_with_preds(healthy_jpeg):
    ref, res, ft = _decode(healthy_jpeg)
    m = ref.mcus[50]
    off, bit = divmod(m.bit_start, 8)
    preds = D.preds_before_mcu(ref, ft, 50)
    sr = D.decode_scan(healthy_jpeg, res, ft, start_byte=off, start_bit=bit, first_mcu=50, init_preds=preds)
    assert sr.error is None
    assert len(sr.mcus) == 192 - 50
    assert sr.mcus[0].bit_start == m.bit_start
    # DC values identical to the full decode for those MCUs
    import numpy as np

    # MCU 50 is row 3, col 2 -> block rows 8+ are fully covered by the resumed decode
    assert np.array_equal(sr.dc[1][8:], ref.dc[1][8:])
    assert np.array_equal(sr.dc[1][6:8, 4:], ref.dc[1][6:8, 4:])


def test_interval_records_resync_marker(healthy_jpeg_rst):
    ref, res, ft = _decode(healthy_jpeg_rst)
    iv = ref.intervals[4]
    off, mask = breaking_flip(healthy_jpeg_rst, iv.byte_start + 4, iv.byte_end - 4)
    sr, _, _ = _decode(flip_bits(healthy_jpeg_rst, off, mask))
    bad = sr.intervals[4]
    assert not bad.ok
    assert bad.resync_at == iv.byte_end  # the RST4 marker position


def test_single_flip_rst_restored(healthy_jpeg_rst):
    ref, res, ft = _decode(healthy_jpeg_rst)
    iv = ref.intervals[3]
    off, mask = breaking_flip(healthy_jpeg_rst, iv.byte_start + 4, iv.byte_end - 4)
    out = _run(flip_bits(healthy_jpeg_rst, off, mask))
    assert_equivalent(out.data, healthy_jpeg_rst)
    assert out.applied == ["bitflip_repair"]
    assert len(out.log[0].extra["flips"]) == 1
    assert out.log[0].extra["repaired"] == 1


def test_two_flips_same_interval_restored(healthy_jpeg_rst):
    ref, res, ft = _decode(healthy_jpeg_rst)
    iv = ref.intervals[6]
    span = iv.byte_end - iv.byte_start
    o1, m1 = breaking_flip(healthy_jpeg_rst, iv.byte_start + 4, iv.byte_start + span // 3)
    broken1 = flip_bits(healthy_jpeg_rst, o1, m1)
    o2, m2 = breaking_flip(broken1, iv.byte_start + 2 * span // 3, iv.byte_end - 4)
    broken = flip_bits(broken1, o2, m2)
    out = _run(broken)
    assert_equivalent(out.data, healthy_jpeg_rst)
    assert 2 <= len(out.log[0].extra["flips"]) <= 4
    assert out.log[0].extra["repaired"] == 1


def test_single_flip_without_rst_restored(healthy_jpeg):
    res = M.parse_segments(healthy_jpeg)
    mid = (res.scan_start + res.eoi_offset) // 2
    off, mask = breaking_flip(healthy_jpeg, mid, mid + 200)
    out = _run(flip_bits(healthy_jpeg, off, mask))
    assert_equivalent(out.data, healthy_jpeg)


def test_gives_up_on_dense_damage(healthy_jpeg_rst):
    ref, res, ft = _decode(healthy_jpeg_rst)
    iv = ref.intervals[9]
    data = healthy_jpeg_rst
    for i in range(20):
        data = flip_bits(data, iv.byte_start + 8 + i * 7, 0x10)
    t = time.perf_counter()
    out = _run(data)
    assert time.perf_counter() - t < 60
    assert out.data == data or out.data != healthy_jpeg_rst
    assert any("unrepaired" in e.message or "budget" in e.message for e in out.log)
    # the interval is left to R9
    ctx = run_pipeline(Context(data=out.data, name="x"), recipes=[FillHoles()])
    assert ctx.holes and ctx.holes[0].mcu_first // 16 == 9


def test_combined_pipeline_restores_fully(healthy_jpeg_rst):
    ref, res, ft = _decode(healthy_jpeg_rst)
    data = healthy_jpeg_rst
    for k in (2, 7, 10):
        iv = ref.intervals[k]
        off, mask = breaking_flip(data, iv.byte_start + 4, iv.byte_end - 4)
        data = flip_bits(data, off, mask)
    ctx = run_pipeline(Context(data=data, name="x"), recipes=[BitFlipRepair(), FillHoles()])
    assert ctx.holes == []
    assert_equivalent(ctx.data, healthy_jpeg_rst)


def test_time_budget_option_respected(healthy_jpeg_rst):
    ref, res, ft = _decode(healthy_jpeg_rst)
    data = healthy_jpeg_rst
    for k in range(12):
        iv = ref.intervals[k]
        data = flip_bits(data, iv.byte_start + 8, 0x10)
    t = time.perf_counter()
    _run(data, time_budget=0.5)
    assert time.perf_counter() - t < 5
