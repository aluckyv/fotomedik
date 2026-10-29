import io

import numpy as np
from PIL import Image

from fotomedik.manual import Editor
from fotomedik.jpeg import markers as M
from synth import make_image, make_jpeg


def _px(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=float)


def test_roundtrip_is_lossless(healthy_jpeg, healthy_jpeg_rst):
    for d in (healthy_jpeg, healthy_jpeg_rst):
        e = Editor(d)
        assert np.array_equal(_px(e.data()), _px(d))


def test_colour_shift_changes_from_the_chosen_mcu_on(healthy_jpeg):
    e = Editor(healthy_jpeg)
    before = _px(e.data())
    m = e.per_row * 2  # start of the third MCU row
    e.shift_colour(m, dy=30)
    after = _px(e.data())
    y0 = 2 * e.mcu_h
    assert np.array_equal(after[: y0 - 8], before[: y0 - 8])
    assert after[y0 + 4 :].mean() > before[y0 + 4 :].mean() + 10
    assert e.undo()
    assert np.array_equal(_px(e.data()), before)


def test_insert_then_delete_mcus_restores_the_picture(healthy_jpeg):
    e = Editor(healthy_jpeg)
    before = _px(e.data())
    e.insert_mcus(10, 3)
    shifted = _px(e.data())
    assert not np.array_equal(shifted, before)
    e.delete_mcus(10, 3)
    assert np.array_equal(_px(e.data()), before)


def test_deleting_junk_bits_brings_the_picture_back():
    data = make_jpeg(make_image(160, 128, seed=4))
    res = M.parse_segments(data)
    e0 = Editor(data)
    pos = e0.where(30)
    assert pos[2]
    # junk inserted at the start of MCU 30: the picture after it is garbage
    seg = e0.segments[pos[0]]
    junk = np.random.default_rng(1).integers(0, 2, 37).astype(np.uint8)
    seg.bits = np.concatenate([seg.bits[: pos[1]], junk, seg.bits[pos[1] :]])
    broken = e0.data()
    assert not np.array_equal(_px(broken), _px(data))
    e = Editor(broken)
    e.delete_bits(30, 37)
    assert np.array_equal(_px(e.data()), _px(data))
    assert res.scan_start > 0
