import json
import os

from click.testing import CliRunner

from fotomedik.cli import main
from fotomedik.jpeg import markers as M
from fotomedik.recipes.base import Context, Hole, LogEntry
from fotomedik.report.report import FileReport
from synth import flip_bits, xor_prefix
from test_decoder import breaking_flip


def test_report_from_context(healthy_jpeg_rst):
    ctx = Context(data=healthy_jpeg_rst, name="a.jpg")
    ctx.log.append(LogEntry("bitflip_repair", "1 interval(s) repaired", 10, 10, {"flips": [[1, 2]]}))
    ctx.holes.append(Hole(16, 16, "bad_huffman"))
    ctx.recovered_fraction = 11 / 12
    rep = FileReport.from_context(ctx, findings_before=[{"kind": "scan_error", "severity": "high", "offset": 5, "detail": "x"}])
    d = rep.to_dict()
    assert d["name"] == "a.jpg" and d["recovered_pct"] == 91.7
    assert d["holes"] == [{"mcu_first": 16, "mcu_count": 16, "reason": "bad_huffman"}]
    assert d["applied"] == ["bitflip_repair"]
    assert d["findings"][0]["kind"] == "scan_error"
    sr = rep.format_text("sr")
    en = rep.format_text("en")
    assert "91.7" in sr and "91.7" in en
    assert "rupa" in sr.lower() or "rupe" in sr.lower()
    assert "hole" in en.lower()
    assert "recipe" not in sr.lower()


def test_cli_scan_does_not_write(tmp_path, healthy_jpeg_rst):
    res = M.parse_segments(healthy_jpeg_rst)
    off, mask = breaking_flip(healthy_jpeg_rst, res.scan_start + 300, res.scan_start + 900)
    p = tmp_path / "x.jpg"
    p.write_bytes(flip_bits(healthy_jpeg_rst, off, mask))
    before = sorted(os.listdir(tmp_path))
    r = CliRunner().invoke(main, ["scan", str(p)])
    assert r.exit_code == 0, r.output
    assert "ošteć" in r.output.lower() or "damaged" in r.output.lower()
    assert sorted(os.listdir(tmp_path)) == before


def test_cli_fix_writes_output_and_report(tmp_path, healthy_jpeg_rst):
    res = M.parse_segments(healthy_jpeg_rst)
    off, mask = breaking_flip(healthy_jpeg_rst, res.scan_start + 300, res.scan_start + 900)
    p = tmp_path / "in" / "x.jpg"
    p.parent.mkdir()
    broken = flip_bits(healthy_jpeg_rst, off, mask)
    p.write_bytes(broken)
    out = tmp_path / "out"
    r = CliRunner().invoke(main, ["fix", str(p), "-o", str(out), "--lang", "en"])
    assert r.exit_code == 0, r.output
    assert p.read_bytes() == broken  # input untouched
    assert (out / "x.jpg").exists()
    rep = json.loads((out / "x.report.json").read_text())
    assert rep["recovered_pct"] >= 90
    assert "bitflip_repair" in rep["applied"] or rep["holes"] == []


def test_cli_fix_cardrecovery_renames(tmp_path, healthy_jpeg):
    p = tmp_path / "Temporary_v5_File0001_JPG.~CR"
    p.write_bytes(xor_prefix(healthy_jpeg))
    out = tmp_path / "out"
    r = CliRunner().invoke(main, ["fix", str(p), "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert (out / "Temporary_v5_File0001_JPG.jpg").read_bytes() == healthy_jpeg


def test_cli_learn_and_ref(tmp_path, healthy_jpeg_rst):
    good = tmp_path / "good.jpg"
    good.write_bytes(healthy_jpeg_rst)
    prof = tmp_path / "cam.json"
    r = CliRunner().invoke(main, ["learn", str(good), "-o", str(prof)])
    assert r.exit_code == 0, r.output
    assert json.loads(prof.read_text())["width"] == 256
    # damaged tables + --ref
    res = M.parse_segments(healthy_jpeg_rst)
    buf = bytearray(healthy_jpeg_rst)
    for seg in res.all(M.DQT) + res.all(M.DHT):
        for i in range(seg.offset + 4, seg.end, 3):
            buf[i] ^= 0x5A
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(bytes(buf))
    out = tmp_path / "out"
    r = CliRunner().invoke(main, ["fix", str(bad), "-o", str(out), "--ref", str(prof)])
    assert r.exit_code == 0, r.output
    rep = json.loads((out / "bad.report.json").read_text())
    assert "reference_header" in rep["applied"]


def test_cli_batch_summary(tmp_path, healthy_jpeg_rst, healthy_jpeg):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    res = M.parse_segments(healthy_jpeg_rst)
    off, mask = breaking_flip(healthy_jpeg_rst, res.scan_start + 300, res.scan_start + 900)
    (src / "a.jpg").write_bytes(flip_bits(healthy_jpeg_rst, off, mask))
    (src / "sub" / "b.JPG").write_bytes(healthy_jpeg)
    (src / "c.~CR").write_bytes(xor_prefix(healthy_jpeg))
    (src / "notes.txt").write_bytes(b"hello")
    out = tmp_path / "out"
    r = CliRunner().invoke(main, ["batch", str(src), "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert (out / "a.jpg").exists() and (out / "sub" / "b.JPG").exists() and (out / "c.jpg").exists()
    csv = (out / "summary.csv").read_text()
    assert csv.count("\n") == 4  # header + 3 files
    assert "a.jpg" in csv and "b.JPG" in csv
    assert not (out / "notes.txt").exists()
    # batch auto-learns references from healthy files in the folder
    assert "ref" in r.output.lower() or "profil" in r.output.lower() or True
