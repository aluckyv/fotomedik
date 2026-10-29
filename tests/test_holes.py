import io

import numpy as np
from PIL import Image

from fotomedik.jpeg import decoder as D
from fotomedik.jpeg import encoder as E
from fotomedik.jpeg import markers as M
from fotomedik.jpeg.tables import FrameTables
from fotomedik.jpeg.writer import assemble
from fotomedik.recipes.base import Context
from fotomedik.recipes.r6_garbage import GarbageTail
from fotomedik.recipes.r9_holes import FillHoles
from fotomedik.recipes.registry import run_pipeline
from synth import append_garbage, flip_bits, truncate
from test_decoder import breaking_flip


def _ft(data):
    res = M.parse_segments(data)
    return res, FrameTables.from_parse(res)


def test_bitwriter_stuffing_and_padding():
    w = E.BitWriter()
    w.put_bits(0xFF, 8)
    w.put_bits(0b101, 3)
    out = w.finish()
    assert out == bytes([0xFF, 0x00, 0b1011_1111])


def test_bitwriter_can_start_mid_byte():
    w = E.BitWriter()
    w.put_bits(0b11, 2)  # as if continuing a byte that already had 2 bits written
    w.put_bits(0x0, 6)
    assert w.finish() == bytes([0b1100_0000])


def test_huffman_codes_roundtrip_with_decoder(healthy_jpeg):
    res, ft = _ft(healthy_jpeg)
    t = ft.huff[(0, 0)]
    codes = E.codes_for(t)
    assert set(codes) == set(t.symbols)
    for sym, (ln, code) in codes.items():
        assert t.lookup[(ln, code)] == sym


def test_encode_dc_only_decodes_cleanly(healthy_jpeg_rst):
    res, ft = _ft(healthy_jpeg_rst)
    ref = D.decode_scan(healthy_jpeg_rst, res, ft)
    iv = ref.intervals[2]
    # targets: luma 100 (coef units), chroma 0
    targets = {1: 100, 2: 0, 3: 0}
    preds = {1: 0, 2: 0, 3: 0}
    blob = E.encode_flat_mcus(ft, iv.mcu_count, targets, preds)
    # splice into the file in place of interval 2 and decode
    data = healthy_jpeg_rst[: iv.byte_start] + blob + healthy_jpeg_rst[iv.byte_end :]
    res2, ft2 = _ft(data)
    sr = D.decode_scan(data, res2, ft2)
    assert sr.error is None
    assert all(i.ok for i in sr.intervals)
    # DC of the replaced interval's luma blocks equals the target
    rows = slice(iv.index * 2, iv.index * 2 + 2)  # 2 block rows per MCU row (v=2)
    assert np.all(sr.dc[1][rows] == 100)
    assert np.all(sr.dc[2][iv.index] == 0)


def test_assemble_roundtrip(healthy_jpeg):
    res, ft = _ft(healthy_jpeg)
    header = healthy_jpeg[: res.scan_start]
    scan = healthy_jpeg[res.scan_start : res.eoi_offset]
    assert assemble(header, scan) == healthy_jpeg


def test_r6_truncates_garbage_and_restores_eoi(healthy_jpeg):
    data = append_garbage(healthy_jpeg[:-2], 65536, seed=5)  # EOI lost under garbage
    ctx = run_pipeline(Context(data=data, name="g.jpg"), recipes=[GarbageTail()])
    assert ctx.data.endswith(b"\xff\xd9")
    assert len(ctx.data) <= len(healthy_jpeg) + 512 + 2
    assert ctx.applied == ["garbage_tail"]
    assert ctx.log[0].extra["removed_bytes"] >= 65536 - 512


def test_r6_leaves_clean_file(healthy_jpeg):
    ctx = run_pipeline(Context(data=healthy_jpeg, name="g.jpg"), recipes=[GarbageTail()])
    assert ctx.data == healthy_jpeg and ctx.log == []


def _opens_full_height(data, size):
    im = Image.open(io.BytesIO(data))
    im.load()
    assert im.size == size
    return im


def test_r9_fills_broken_interval_rst(healthy_jpeg_rst):
    res, ft = _ft(healthy_jpeg_rst)
    ref = D.decode_scan(healthy_jpeg_rst, res, ft)
    iv = ref.intervals[5]
    off, mask = breaking_flip(healthy_jpeg_rst, iv.byte_start + 4, iv.byte_end - 4)
    broken = flip_bits(healthy_jpeg_rst, off, mask)
    ctx = run_pipeline(Context(data=broken, name="h.jpg"), recipes=[FillHoles()])
    res2, ft2 = _ft(ctx.data)
    sr = D.decode_scan(ctx.data, res2, ft2)
    assert sr.error is None and all(i.ok for i in sr.intervals)
    assert len(ctx.holes) == 1
    h = ctx.holes[0]
    assert h.mcu_first // 16 == 5 and h.mcu_first + h.mcu_count <= 96
    assert ctx.recovered_fraction is not None and 0.85 <= ctx.recovered_fraction < 1.0
    _opens_full_height(ctx.data, (256, 192))
    # untouched intervals byte-identical
    assert ctx.data[: iv.byte_start] == healthy_jpeg_rst[: iv.byte_start]
    assert ctx.data[-(len(healthy_jpeg_rst) - iv.byte_end) :] == healthy_jpeg_rst[iv.byte_end :]


def test_r9_fills_tail_without_rst(healthy_jpeg):
    res, ft = _ft(healthy_jpeg)
    mid = (res.scan_start + res.eoi_offset) // 2
    off, mask = breaking_flip(healthy_jpeg, mid, mid + 64)
    broken = flip_bits(healthy_jpeg, off, mask)
    ctx = run_pipeline(Context(data=broken, name="n.jpg"), recipes=[FillHoles()])
    res2, ft2 = _ft(ctx.data)
    sr = D.decode_scan(ctx.data, res2, ft2)
    assert sr.error is None
    assert len(ctx.holes) == 1
    assert ctx.holes[0].mcu_first + ctx.holes[0].mcu_count == 192
    im = _opens_full_height(ctx.data, (256, 192))
    # top rows identical to the healthy image, bottom filled
    ref = Image.open(io.BytesIO(healthy_jpeg)).convert("RGB")
    a = np.asarray(im.convert("RGB"))[:16].astype(int)
    b = np.asarray(ref)[:16].astype(int)
    assert np.abs(a - b).mean() < 1.0


def test_r9_truncated_file(healthy_jpeg_rst):
    data = truncate(healthy_jpeg_rst, len(healthy_jpeg_rst) * 2 // 3)
    ctx = run_pipeline(Context(data=data, name="t.jpg"), recipes=[FillHoles()])
    res2, ft2 = _ft(ctx.data)
    sr = D.decode_scan(ctx.data, res2, ft2)
    assert sr.error is None and ctx.data.endswith(b"\xff\xd9")
    assert ctx.holes and ctx.holes[-1].mcu_first + ctx.holes[-1].mcu_count == 192
    _opens_full_height(ctx.data, (256, 192))


def test_r9_marks_holes_with_colour_option(healthy_jpeg_rst):
    res, ft = _ft(healthy_jpeg_rst)
    ref = D.decode_scan(healthy_jpeg_rst, res, ft)
    iv = ref.intervals[8]
    off, mask = breaking_flip(healthy_jpeg_rst, iv.byte_start + 4, iv.byte_end - 4)
    broken = flip_bits(healthy_jpeg_rst, off, mask)
    ctx = Context(data=broken, name="m.jpg", options={"mark_holes": True})
    run_pipeline(ctx, recipes=[FillHoles()])
    im = _opens_full_height(ctx.data, (256, 192)).convert("RGB")
    px = np.asarray(im)[8 * 16 + 8, 128]  # middle of hole row 8
    r, g, b = (int(v) for v in px)
    assert r > 150 and b > 150 and g < 110  # magenta-ish


def test_r9_nothing_to_do(healthy_jpeg_rst):
    ctx = run_pipeline(Context(data=healthy_jpeg_rst, name="ok.jpg"), recipes=[FillHoles()])
    assert ctx.data == healthy_jpeg_rst and ctx.log == [] and ctx.holes == []
