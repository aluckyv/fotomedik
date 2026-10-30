"""An FF D9 spelled by rotted data in the middle of the scan is not the image's end."""

from fotomedik.jpeg import markers as M
from fotomedik.recipes.r6_garbage import GarbageTail
from fotomedik.recipes.base import Context
from synth import make_image, make_jpeg


def _rotted(healthy: bytes) -> bytes:
    res = M.parse_segments(healthy)
    b = bytearray(healthy)
    mid = res.scan_start + (res.eoi_offset - res.scan_start) // 3
    b[mid : mid + 2] = b"\xff\xd9"  # rot spells an end marker a third into the scan
    b[-1] ^= 0x10  # ... and the real one at the end lost a bit
    return bytes(b)


def test_rotted_end_marker_beats_one_spelled_by_rot():
    healthy = make_jpeg(make_image(512, 384), restart_rows=1)
    data = _rotted(healthy)
    assert M.parse_segments(data).eoi_offset == len(data) - 2


def test_garbage_tail_keeps_the_data_after_a_false_end():
    healthy = make_jpeg(make_image(512, 384), restart_rows=1)
    data = _rotted(healthy)
    assert GarbageTail().detect(Context(data=data, name="x.jpg")) is None


def test_a_real_end_with_slack_after_it_still_counts():
    healthy = make_jpeg(make_image(256, 192))
    data = healthy + bytes(9000)  # zeros after the real end (rest of the cluster)
    assert M.parse_segments(data).eoi_offset == len(healthy) - 2
