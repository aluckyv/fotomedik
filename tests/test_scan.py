from fotomedik.jpeg import markers as M
from fotomedik.scan import garbage as G
from fotomedik.scan.diagnosis import Diagnosis, Finding
from fotomedik.scan.scanner import scan_bytes
from synth import append_garbage, flip_bits, truncate, xor_prefix
from test_decoder import breaking_flip


def test_cluster_stats_marks_garbage(healthy_jpeg):
    # pad the JPEG to a cluster boundary so clusters are either pure image or pure garbage
    pad = 32768 - len(healthy_jpeg) % 32768
    data = append_garbage(healthy_jpeg + b"\xff\x00" * (pad // 2), 65536, seed=3)
    res = M.parse_segments(healthy_jpeg)
    stats = G.cluster_stats(data, res.scan_start, cluster=32768)
    kinds = [s.kind for s in stats]
    assert kinds == ["jpeg", "garbage", "garbage"]
    assert stats[-1].bad_ff_ratio > 0.5
    assert stats[0].bad_ff_ratio < 0.05


def test_garbage_tail_offset(healthy_jpeg):
    data = append_garbage(healthy_jpeg, 70000, seed=3)
    res = M.parse_segments(healthy_jpeg)
    tail = G.garbage_tail_start(data, res.scan_start)
    # sector precision: within one sector after the real end, at most a window before it
    assert tail is not None
    assert len(healthy_jpeg) - 8 * 512 <= tail <= len(healthy_jpeg) + 512


def test_no_garbage_tail_on_healthy(healthy_jpeg):
    res = M.parse_segments(healthy_jpeg)
    assert G.garbage_tail_start(healthy_jpeg, res.scan_start) is None


def test_garbage_tail_ignores_small_damage(healthy_jpeg):
    # 300 random bytes after EOI (slack) is not a garbage tail
    data = append_garbage(healthy_jpeg, 300, seed=9)
    res = M.parse_segments(healthy_jpeg)
    assert G.garbage_tail_start(data, res.scan_start) is None


def test_scan_healthy(healthy_jpeg):
    d = scan_bytes(healthy_jpeg, "a.jpg")
    assert isinstance(d, Diagnosis)
    assert d.findings == []
    assert d.recovered_fraction == 1.0
    assert d.openable is True
    dd = d.to_dict()
    assert dd["name"] == "a.jpg" and dd["findings"] == [] and dd["recovered_fraction"] == 1.0


def test_scan_cardrecovery(healthy_jpeg):
    d = scan_bytes(xor_prefix(healthy_jpeg), "Temporary_v5_File0001_JPG.~CR")
    assert [f.kind for f in d.findings] == ["cardrecovery_xor"]
    assert d.openable is False


def test_scan_bad_marker(healthy_jpeg):
    res = M.parse_segments(healthy_jpeg)
    dqt = res.first(M.DQT)
    d = scan_bytes(flip_bits(healthy_jpeg, dqt.offset + 1, 0x20), "b.jpg")
    kinds = [f.kind for f in d.findings]
    assert "header_error" in kinds
    f = next(f for f in d.findings if f.kind == "header_error")
    assert f.offset == dqt.offset and f.severity == "high"
    assert d.openable is False


def test_scan_scan_error(healthy_jpeg_rst):
    res = M.parse_segments(healthy_jpeg_rst)
    off, mask = breaking_flip(healthy_jpeg_rst, res.scan_start + 300, res.scan_start + 600)
    d = scan_bytes(flip_bits(healthy_jpeg_rst, off, mask), "c.jpg")
    kinds = [f.kind for f in d.findings]
    assert "scan_error" in kinds
    f = next(f for f in d.findings if f.kind == "scan_error")
    assert f.offset >= off
    assert 0 < d.recovered_fraction < 1.0
    assert d.openable is True  # header fine, Pillow would open it (partially)


def test_scan_truncated(healthy_jpeg):
    d = scan_bytes(truncate(healthy_jpeg, len(healthy_jpeg) // 2), "d.jpg")
    kinds = [f.kind for f in d.findings]
    assert "truncated" in kinds
    assert d.recovered_fraction < 0.7


def test_scan_garbage(healthy_jpeg):
    d = scan_bytes(append_garbage(healthy_jpeg, 65536, seed=5), "e.jpg")
    kinds = [f.kind for f in d.findings]
    assert "garbage_tail" in kinds
    f = next(f for f in d.findings if f.kind == "garbage_tail")
    assert len(healthy_jpeg) - 8 * 512 <= f.offset <= len(healthy_jpeg) + 512


def test_scan_not_a_jpeg():
    d = scan_bytes(b"\x00" * 5000, "x.jpg")
    assert [f.kind for f in d.findings] == ["not_jpeg"]
    assert d.recovered_fraction == 0.0


def test_scan_progressive(healthy_progressive):
    d = scan_bytes(healthy_progressive, "p.jpg")
    assert [f.kind for f in d.findings] == ["unsupported_progressive"]
    assert d.openable is True


def test_finding_fields():
    f = Finding("scan_error", "high", 123, "bad huffman")
    assert f.to_dict() == {"kind": "scan_error", "severity": "high", "offset": 123, "detail": "bad huffman"}
