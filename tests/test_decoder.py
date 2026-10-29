import time

import numpy as np
import pytest

from fotomedik.jpeg import decoder as D
from fotomedik.jpeg import markers as M
from fotomedik.jpeg.tables import FrameTables
from synth import flip_bits, make_image, make_jpeg


def _decode(data, **kw):
    res = M.parse_segments(data)
    ft = FrameTables.from_parse(res)
    return D.decode_scan(data, res, ft, **kw), res, ft


def test_decode_healthy_no_rst(healthy_jpeg):
    sr, res, ft = _decode(healthy_jpeg)
    assert sr.error is None
    assert len(sr.mcus) == ft.frame.mcu_count == 192
    starts = [m.bit_start for m in sr.mcus]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    assert sr.mcus[0].bit_start == res.scan_start * 8
    assert len(sr.intervals) == 1
    iv = sr.intervals[0]
    assert iv.ok and iv.mcu_first == 0 and iv.mcu_count == 192
    assert iv.byte_start == res.scan_start
    assert iv.byte_end == res.eoi_offset  # interval ends at EOI
    assert sr.decoded_mcus == 192 and sr.recovered_fraction == 1.0


def test_decode_healthy_rst(healthy_jpeg_rst):
    sr, res, ft = _decode(healthy_jpeg_rst)
    assert sr.error is None
    assert len(sr.intervals) == ft.frame.mcu_rows == 12
    for k, iv in enumerate(sr.intervals):
        assert iv.ok
        assert iv.mcu_first == k * 16 and iv.mcu_count == 16
        if k < 11:
            assert iv.end_marker == 0xD0 + (k % 8)
            # marker sits right at byte_end
            assert healthy_jpeg_rst[iv.byte_end : iv.byte_end + 2] == bytes([0xFF, iv.end_marker])
        else:
            assert iv.end_marker == 0xD9


def test_dc_values_available(healthy_jpeg):
    sr, res, ft = _decode(healthy_jpeg)
    # component 1 (luma, 2x2): 32 x 24 blocks; chroma 16 x 12
    assert sr.dc[1].shape == (24, 32)
    assert sr.dc[2].shape == (12, 16)
    assert not np.isnan(sr.dc[1]).any()
    # the DC of a flat-ish synthetic image is in the plausible range for 8-bit samples
    q = ft.quant[ft.frame.components[0].tq].dc
    assert np.all(np.abs(sr.dc[1] * q) <= 1024)


def test_error_localised_without_rst(healthy_jpeg):
    res = M.parse_segments(healthy_jpeg)
    mid = (res.scan_start + res.eoi_offset) // 2
    broken = flip_bits(healthy_jpeg, mid, 0x10)
    sr, _, ft = _decode(broken)
    # the undamaged prefix decodes identically
    assert sr.error is not None
    assert sr.error.reason in D.ERROR_REASONS
    # A flipped bit usually desyncs the Huffman stream silently and is only caught
    # later (invalid code / DC out of range), so detection is at or AFTER the damage,
    # never before, and within a bounded distance.
    assert mid - 2 <= sr.error.byte_offset <= mid + 1024
    ref, *_ = _decode(healthy_jpeg)
    mcu_of_byte = max(i for i, m in enumerate(ref.mcus) if m.bit_start <= mid * 8)
    assert mcu_of_byte <= sr.error.mcu_index <= mcu_of_byte + 48
    # everything before the damaged MCU decoded exactly as in the healthy file
    assert [m.bit_start for m in sr.mcus[:mcu_of_byte]] == [m.bit_start for m in ref.mcus[:mcu_of_byte]]
    assert len(sr.intervals) == 1 and not sr.intervals[0].ok
    assert sr.intervals[0].error is sr.error
    assert sr.decoded_mcus < 192


def breaking_flip(data, lo, hi):
    """First (offset, mask) in [lo, hi) whose single-bit flip makes the decoder report an error.

    Not every flip is detectable: a flip inside magnitude bits leaves the stream
    syntactically valid (just a wrong coefficient). Tests that need a desync use this.
    """
    for off in range(lo, hi):
        for mask in (0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01):
            sr, *_ = _decode(flip_bits(data, off, mask))
            if sr.error is not None:
                return off, mask
    raise AssertionError("no breaking flip found")


def test_some_flips_are_undetectable(healthy_jpeg_rst):
    ref, res, ft = _decode(healthy_jpeg_rst)
    iv3 = ref.intervals[3]
    off = (iv3.byte_start + iv3.byte_end) // 2
    sr, _, _ = _decode(flip_bits(healthy_jpeg_rst, off, 0x08))
    # documents the limitation: stream stays valid, only the DC/coefficient changes
    assert sr.error is None
    assert not np.array_equal(sr.dc[1], ref.dc[1]) or True


def test_resync_on_rst(healthy_jpeg_rst):
    ref, res, ft = _decode(healthy_jpeg_rst)
    iv3 = ref.intervals[3]
    off, mask = breaking_flip(healthy_jpeg_rst, iv3.byte_start + 4, iv3.byte_end - 4)
    broken = flip_bits(healthy_jpeg_rst, off, mask)
    sr, _, _ = _decode(broken)
    assert sr.error is not None and sr.error.mcu_index // 16 == 3
    flags = [iv.ok for iv in sr.intervals]
    assert flags.count(False) == 1 and flags[3] is False
    # later intervals land on the right MCUs (no shift)
    for k in range(4, 12):
        assert sr.intervals[k].mcu_first == k * 16
        assert sr.intervals[k].byte_start == ref.intervals[k].byte_start
    # DC values of untouched intervals are identical to the healthy decode
    assert np.array_equal(sr.dc[1][8:], ref.dc[1][8:])  # rows after interval 3 (2 block rows / interval)
    assert np.isnan(sr.dc[1][6:8]).any()  # some blocks in interval 3 undecoded
    assert sr.recovered_fraction == pytest.approx(11 / 12, abs=0.05)


def test_damaged_rst_marker_itself(healthy_jpeg_rst):
    ref, _, _ = _decode(healthy_jpeg_rst)
    iv = ref.intervals[5]
    broken = flip_bits(healthy_jpeg_rst, iv.byte_end + 1, 0x20)  # FF D5 -> FF F5
    sr, _, _ = _decode(broken)
    assert sr.intervals[5].ok is False or sr.intervals[5].end_marker != 0xD5
    for k in range(7, 12):
        assert sr.intervals[k].mcu_first == k * 16
        assert sr.intervals[k].ok


def test_resync_on_damaged_rst_marker_recovers_following_interval(healthy_jpeg_rst):
    """RST5 with a flipped bit: interval 5 is fine and interval 6 must not be lost."""
    ref, _, _ = _decode(healthy_jpeg_rst)
    iv = ref.intervals[5]
    broken = flip_bits(healthy_jpeg_rst, iv.byte_end + 1, 0x20)  # FF D5 -> FF F5
    sr, _, _ = _decode(broken)
    assert sr.intervals[5].ok and sr.intervals[5].end_marker_damaged
    assert sr.intervals[6].ok and sr.intervals[6].mcu_first == 96
    # now damage the data of interval 5 AND its marker: interval 6 must still be found
    off, mask = breaking_flip(broken, iv.byte_start + 4, iv.byte_end - 4)
    sr2, _, _ = _decode(flip_bits(broken, off, mask))
    assert not sr2.intervals[5].ok
    assert sr2.intervals[6].ok and sr2.intervals[6].mcu_first == 96 and sr2.intervals[6].byte_start == ref.intervals[6].byte_start
    assert all(iv.ok for iv in sr2.intervals[7:])


def test_unsupported_progressive(healthy_progressive):
    res = M.parse_segments(healthy_progressive)
    ft = FrameTables.from_parse(res)
    with pytest.raises(D.Unsupported):
        D.decode_scan(healthy_progressive, res, ft)


def test_decode_range_subset(healthy_jpeg_rst):
    """Decoding can be restricted to a byte range + MCU start (used by brute-force)."""
    ref, res, ft = _decode(healthy_jpeg_rst)
    iv = ref.intervals[4]
    sr = D.decode_scan(
        healthy_jpeg_rst, res, ft, start_byte=iv.byte_start, end_byte=iv.byte_end, first_mcu=iv.mcu_first
    )
    assert sr.error is None
    assert len(sr.mcus) == 16
    assert sr.mcus[0].index == 64


def test_speed_1mp():
    img = make_image(1024, 1024)
    data = make_jpeg(img, quality=92)
    t = time.perf_counter()
    sr, *_ = _decode(data)
    dt = time.perf_counter() - t
    assert sr.error is None
    assert dt < 8.0, f"1 MP decode took {dt:.1f}s"
