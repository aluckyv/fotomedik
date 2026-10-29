import io

from PIL import Image

from synth import (
    append_garbage,
    flip_bits,
    make_image,
    make_jpeg,
    truncate,
    xor_prefix,
)


def test_make_image_is_deterministic():
    a = make_image()
    b = make_image()
    assert a.size == (256, 192)
    assert a.tobytes() == b.tobytes()


def test_make_jpeg_is_valid_baseline():
    data = make_jpeg(make_image())
    assert data[:2] == b"\xff\xd8"
    assert data[-2:] == b"\xff\xd9"
    assert b"\xff\xc0" in data  # SOF0 baseline
    assert b"\xff\xdd" not in data  # no restart interval by default
    im = Image.open(io.BytesIO(data))
    im.load()
    assert im.size == (256, 192)


def test_make_jpeg_with_restart_markers():
    data = make_jpeg(make_image(), restart_rows=1)
    assert b"\xff\xdd" in data
    assert b"\xff\xd0" in data


def test_make_jpeg_progressive():
    data = make_jpeg(make_image(), progressive=True)
    assert b"\xff\xc2" in data


def test_flip_bits_changes_only_requested_byte():
    data = make_jpeg(make_image())
    out = flip_bits(data, 100, 0b0000_0100)
    assert len(out) == len(data)
    assert out[100] == data[100] ^ 0b0000_0100
    assert out[:100] == data[:100] and out[101:] == data[101:]


def test_truncate_and_xor_prefix_and_garbage():
    data = make_jpeg(make_image())
    assert len(truncate(data, 1000)) == 1000

    enc = xor_prefix(data)
    assert enc[:2] == bytes([0xFF ^ 0x76, 0xD8 ^ 0x76])
    assert enc[512:] == data[512:]
    assert xor_prefix(enc) == data  # involution

    g = append_garbage(data, 4096, seed=1)
    assert len(g) == len(data) + 4096
    assert g[: len(data)] == data
    assert append_garbage(data, 4096, seed=1) == g
