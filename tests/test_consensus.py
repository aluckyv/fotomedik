import numpy as np

from fotomedik.jpeg import markers as M
from fotomedik.learn.consensus import consensus, table_block, vote
from fotomedik.learn.profile import CameraProfile
from synth import make_image, make_jpeg


def _rot_header(data: bytes, seed: int, rate: float = 0.04) -> bytes:
    """Flip bits all over the header (as a rotted first sector would), scan untouched."""
    scan = M.parse_segments(data).scan_start
    rng = np.random.default_rng(seed)
    buf = bytearray(data)
    for o in range(2, scan - 14):  # keep the SOS readable: it anchors the copies
        for b in range(8):
            if rng.random() < rate:
                buf[o] ^= 1 << b
    return bytes(buf)


def test_vote_is_bitwise_majority():
    assert vote([b"\x0f", b"\x0e", b"\x0d"]) == b"\x0f"


def test_tables_voted_clean_from_damaged_headers_without_any_healthy_one():
    photos = [make_jpeg(make_image(320, 240, seed=s)) for s in range(7)]
    truth = table_block(photos[0])
    damaged = [_rot_header(p, seed=i) for i, p in enumerate(photos)]
    assert sum(M.parse_segments(d).ok for d in damaged) == 0  # nothing parses
    profs = consensus([], damaged)
    assert profs
    p = profs[0]
    good = CameraProfile.from_jpeg(photos[0])
    assert (p.dqt, p.dht, p.sos) == (good.dqt, good.dht, good.sos)
    assert len(truth) > 0


def test_a_seed_with_flipped_table_bits_is_corrected_by_the_others():
    photos = [make_jpeg(make_image(320, 240, seed=s)) for s in range(6)]
    good = CameraProfile.from_jpeg(photos[0])
    block = bytearray(table_block(photos[0]))
    block[20] ^= 0x04  # a flipped bit inside DQT of the only parseable header
    bad = CameraProfile.from_jpeg(b"\xff\xd8" + bytes(block) + b"\xff\xd9")
    others = [_rot_header(p, seed=10 + i) for i, p in enumerate(photos[1:])]
    profs = consensus([(bad, bytes(block))], others)
    assert profs[0].dqt == good.dqt


def test_batch_repairs_a_set_where_every_header_is_damaged(tmp_path):
    import io
    import json

    from click.testing import CliRunner
    from PIL import Image

    from fotomedik.cli import main

    src = tmp_path / "src"
    src.mkdir()
    for i in range(6):
        w, h = (320, 240) if i % 3 else (240, 320)
        (src / f"p{i}.jpg").write_bytes(_rot_header(make_jpeg(make_image(w, h, seed=20 + i)), seed=30 + i))
    out = tmp_path / "out"
    r = CliRunner().invoke(main, ["batch", str(src), "-o", str(out), "--reports", "--jobs", "1"])
    assert r.exit_code == 0, r.output
    for i in range(6):
        rep = json.loads((out / f"p{i}.report.json").read_text())
        assert "reference_header" in rep["applied"], rep["log"]
        assert rep["recovered_pct"] >= 90
        im = Image.open(io.BytesIO((out / f"p{i}.jpg").read_bytes()))
        assert im.size == ((320, 240) if i % 3 else (240, 320))
