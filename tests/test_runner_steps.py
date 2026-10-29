import io

from PIL import Image

from fotomedik import runner
from fotomedik.ai.enhance import Engine
from fotomedik.ai.run import run_extras
from fotomedik.jpeg import markers as M
from fotomedik.report.report import FileReport
from synth import flip_bits
from test_ai import FakeLama, FakeSR
from test_decoder import breaking_flip


def test_replay_shows_every_step(tmp_path, healthy_jpeg_rst):
    res = M.parse_segments(healthy_jpeg_rst)
    off, mask = breaking_flip(healthy_jpeg_rst, res.scan_start + 300, res.scan_start + 900)
    src = tmp_path / "in"
    src.mkdir()
    (src / "x.jpg").write_bytes(flip_bits(healthy_jpeg_rst, off, mask))
    results = runner.run([src], tmp_path / "out", {"time_budget": 1.0}, jobs=1)
    steps = runner.replay(results[0])
    assert steps[0].title == "As it came from the card" and steps[0].data == (src / "x.jpg").read_bytes()
    assert len(steps) >= 2
    for st in steps[1:]:
        if st.data is not None:
            Image.open(io.BytesIO(st.data)).load()
            assert st.explanation


def test_verdicts():
    rep = FileReport("a.jpg", "a.jpg", 1, 1, 12.0, findings=[{"kind": "scan_error"}], holes=[{"mcu_first": 0, "mcu_count": 5}])
    assert runner.verdict(rep) == "Mostly lost" and runner.is_lost(rep)
    rep.recovered_pct = 55.0
    assert runner.verdict(rep) == "Partly recovered" and not runner.is_lost(rep)
    rep.fallback = "thumbnail"
    assert runner.is_lost(rep)
    assert runner.verdict(FileReport("b", None, 1, 0, None, findings=[{"kind": "overwritten"}])) == "No photo in the file"


def test_lost_photos_get_no_enhanced_copy(tmp_path):
    rep = FileReport("a.jpg", "a.jpg", 1, 1, 12.0, findings=[{"kind": "scan_error"}],
                     holes=[{"mcu_first": 0, "mcu_count": 5, "reason": "x"}])
    assert run_extras([(rep, None)], tmp_path, ["ai"], engine=Engine(FakeSR(), FakeLama())) is None
    assert rep.ai_output is None
