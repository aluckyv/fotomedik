from fotomedik.jpeg import markers as M
from fotomedik.jpeg.tables import FrameTables, parse_dht, standard_table_bytes
from fotomedik.recipes.base import Context
from fotomedik.recipes.r2_header import HeaderRepair
from fotomedik.recipes.registry import run_pipeline
from synth import flip_bits


def _run(data, name="x.jpg"):
    return run_pipeline(Context(data=data, name=name), recipes=[HeaderRepair()])


def test_leaves_healthy_alone(healthy_jpeg):
    out = _run(healthy_jpeg)
    assert out.data == healthy_jpeg and out.log == []


def test_fixes_soi(healthy_jpeg):
    broken = flip_bits(healthy_jpeg, 1, 0x04)  # FF D8 -> FF DC (the Milano case)
    out = _run(broken)
    assert out.data == healthy_jpeg
    assert any("SOI" in e.message for e in out.log)


def test_fixes_marker_byte(healthy_jpeg):
    res = M.parse_segments(healthy_jpeg)
    dqt = res.first(M.DQT)
    broken = flip_bits(healthy_jpeg, dqt.offset + 1, 0x20)  # FF DB -> FF FB
    out = _run(broken)
    assert out.data == healthy_jpeg
    assert M.parse_segments(out.data).ok


def test_fixes_missing_ff(healthy_jpeg):
    res = M.parse_segments(healthy_jpeg)
    dht = res.first(M.DHT)
    broken = flip_bits(healthy_jpeg, dht.offset, 0x01)  # FF -> FE
    out = _run(broken)
    assert out.data == healthy_jpeg


def test_fixes_length_field(healthy_jpeg):
    res = M.parse_segments(healthy_jpeg)
    dht = res.first(M.DHT)
    broken = flip_bits(healthy_jpeg, dht.offset + 2, 0x01)  # high byte of length (the Canon case: 0x11 -> 0x111)
    out = _run(broken)
    assert out.data == healthy_jpeg
    broken2 = flip_bits(healthy_jpeg, dht.offset + 3, 0x08)  # low byte
    assert _run(broken2).data == healthy_jpeg


def test_replaces_damaged_standard_dht(healthy_jpeg):
    # Build a file whose DHT is exactly the standard tables, then flip a count byte.
    res = M.parse_segments(healthy_jpeg)
    dht = res.first(M.DHT)
    std = b"".join(standard_table_bytes(c, i) for c in (0, 1) for i in (0, 1))
    seg = b"\xff\xc4" + (len(std) + 2).to_bytes(2, "big") + std
    data = healthy_jpeg[: dht.offset] + seg + healthy_jpeg[dht.end :]
    assert M.parse_segments(data).ok
    # flip a bit in the counts of the DC luma table so the sum exceeds the symbols
    broken = flip_bits(data, dht.offset + 4 + 1 + 2, 0x40)
    try:
        parse_dht(M.parse_segments(broken).first(M.DHT).payload)
        broke = False
    except Exception:
        broke = True
    assert broke
    out = _run(broken)
    assert out.data == data
    assert any("standard" in e.message for e in out.log)


def test_fixes_sof_precision_and_sos_spectral(healthy_jpeg):
    res = M.parse_segments(healthy_jpeg)
    sof = res.sof
    broken = flip_bits(healthy_jpeg, sof.offset + 4, 0x01)  # precision 8 -> 9
    out = _run(broken)
    assert out.data == healthy_jpeg
    sos = res.first(M.SOS)
    # Ss byte (should be 0) flipped
    broken = flip_bits(healthy_jpeg, sos.end - 3, 0x10)
    out = _run(broken)
    assert out.data == healthy_jpeg


def test_two_independent_flips(healthy_jpeg):
    res = M.parse_segments(healthy_jpeg)
    dqt, sof = res.first(M.DQT), res.sof
    broken = flip_bits(flip_bits(healthy_jpeg, dqt.offset + 1, 0x20), sof.offset + 3, 0x02)
    out = _run(broken)
    assert out.data == healthy_jpeg
    FrameTables.from_parse(M.parse_segments(out.data))


def test_gives_up_gracefully_on_random_header():
    data = b"\xff\xd8" + bytes(range(256)) * 4
    out = _run(data)
    # no crash; nothing claimed
    assert isinstance(out.data, bytes)
