import pytest

from fotomedik.jpeg import markers as M
from fotomedik.jpeg import tables as T


def _seg(data, marker):
    res = M.parse_segments(data)
    return [s for s in res.segments if s.marker == marker]


def test_parse_dqt(healthy_jpeg):
    tables = {}
    for seg in _seg(healthy_jpeg, M.DQT):
        tables.update(T.parse_dqt(seg.payload))
    assert set(tables) == {0, 1}
    q = tables[0]
    assert q.precision == 0
    assert len(q.values) == 64  # zig-zag order, as stored
    assert all(1 <= v <= 255 for v in q.values)
    assert q.values[0] == q.dc  # DC quantizer


def test_parse_dht(healthy_jpeg):
    tables = {}
    for seg in _seg(healthy_jpeg, M.DHT):
        tables.update(T.parse_dht(seg.payload))
    # Pillow writes luma/chroma DC+AC tables
    assert (0, 0) in tables and (1, 0) in tables  # (class, id): DC0, AC0
    dc = tables[(0, 0)]
    assert sum(dc.counts) == len(dc.symbols)
    # every symbol reachable through lookup
    assert len(dc.lookup) == len(dc.symbols)
    assert all(1 <= ln <= 16 for ln, _ in dc.lookup)
    # decode first code: lookup is a prefix code -> no code is a prefix of another
    codes = sorted(dc.lookup, key=lambda k: (k[0], k[1]))
    for (l1, c1) in codes:
        for (l2, c2) in codes:
            if l1 < l2:
                assert (c2 >> (l2 - l1)) != c1


def test_parse_dht_rejects_bad_counts():
    payload = bytes([0x00]) + bytes([20] * 16) + bytes(range(50))
    with pytest.raises(T.TableError):
        T.parse_dht(payload)


def test_parse_sof(healthy_jpeg):
    seg = M.parse_segments(healthy_jpeg).sof
    f = T.parse_sof(seg.marker, seg.payload)
    assert (f.width, f.height, f.precision) == (256, 192, 8)
    assert f.progressive is False
    assert [c.id for c in f.components] == [1, 2, 3]
    assert (f.components[0].h, f.components[0].v) == (2, 2)  # 4:2:0 luma
    assert (f.components[1].h, f.components[1].v) == (1, 1)
    assert f.hmax == 2 and f.vmax == 2
    assert f.mcu_width == 16 and f.mcu_height == 16
    assert f.mcus_per_row == 16 and f.mcu_rows == 12
    assert f.mcu_count == 192


def test_parse_sof_progressive(healthy_progressive):
    seg = M.parse_segments(healthy_progressive).sof
    assert T.parse_sof(seg.marker, seg.payload).progressive is True


def test_parse_sos_and_dri(healthy_jpeg_rst):
    res = M.parse_segments(healthy_jpeg_rst)
    s = T.parse_sos(res.first(M.SOS).payload)
    assert [(c.id, c.td, c.ta) for c in s.components] == [(1, 0, 0), (2, 1, 1), (3, 1, 1)]
    assert (s.ss, s.se, s.ah, s.al) == (0, 63, 0, 0)
    assert T.parse_dri(res.first(M.DRI).payload) == 16  # one MCU row = 16 MCUs


def test_standard_tables_build():
    dc = T.standard_table(0, 0)
    ac = T.standard_table(1, 0)
    assert len(dc.symbols) == 12 and len(ac.symbols) == 162
    assert len(T.standard_table(0, 1).symbols) == 12
    assert len(T.standard_table(1, 1).symbols) == 162


def test_frame_tables_bundle(healthy_jpeg):
    """Convenience: everything the decoder needs, from a ParseResult."""
    res = M.parse_segments(healthy_jpeg)
    ft = T.FrameTables.from_parse(res)
    assert ft.frame.width == 256
    assert set(ft.quant) == {0, 1}
    assert (0, 0) in ft.huff and (1, 1) in ft.huff
    assert ft.restart_interval == 0
    assert ft.scan.components[0].id == 1
