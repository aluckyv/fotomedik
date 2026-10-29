from fotomedik.jpeg import markers as M
from synth import flip_bits


def test_parse_segments_healthy(healthy_jpeg):
    res = M.parse_segments(healthy_jpeg)
    assert res.errors == []
    kinds = [s.marker for s in res.segments]
    assert kinds[0] in (M.APP0, M.APP1)
    assert M.DQT in kinds and M.DHT in kinds and M.SOF0 in kinds
    assert kinds[-1] == M.SOS
    sos = res.segments[-1]
    # scan starts right after the SOS segment payload
    assert res.scan_start == sos.offset + 2 + sos.length
    assert healthy_jpeg[res.eoi_offset : res.eoi_offset + 2] == b"\xff\xd9"
    assert res.eoi_offset == len(healthy_jpeg) - 2
    # payload excludes the 2-byte length field
    for s in res.segments:
        assert len(s.payload) == s.length - 2


def test_parse_segments_reports_dri(healthy_jpeg_rst):
    res = M.parse_segments(healthy_jpeg_rst)
    assert M.DRI in [s.marker for s in res.segments]


def test_parse_segments_tolerates_broken_marker(healthy_jpeg):
    res0 = M.parse_segments(healthy_jpeg)
    dqt = next(s for s in res0.segments if s.marker == M.DQT)
    broken = flip_bits(healthy_jpeg, dqt.offset + 1, 0x20)  # FF DB -> FF FB (not a marker)
    res = M.parse_segments(broken)
    assert len(res.errors) == 1
    assert res.errors[0].offset == dqt.offset
    assert res.errors[0].reason == "unknown_marker"
    assert res.scan_start is None
    # segments before the damage are still returned
    assert [s.marker for s in res.segments] == [
        s.marker for s in res0.segments if s.offset < dqt.offset
    ]


def test_parse_segments_tolerates_missing_ff(healthy_jpeg):
    res0 = M.parse_segments(healthy_jpeg)
    dht = next(s for s in res0.segments if s.marker == M.DHT)
    broken = flip_bits(healthy_jpeg, dht.offset, 0x01)  # FF -> FE
    res = M.parse_segments(broken)
    assert res.errors[0].offset == dht.offset
    assert res.errors[0].reason == "no_marker"


def test_parse_segments_not_a_jpeg():
    res = M.parse_segments(b"\x00\x01\x02\x03" * 100)
    assert res.segments == []
    assert res.errors[0].reason == "no_soi"


def test_find_eoi_missing(healthy_jpeg):
    res = M.parse_segments(healthy_jpeg[:-2])
    assert res.eoi_offset is None
