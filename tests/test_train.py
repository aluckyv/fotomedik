"""Training material for the deblocker, and the program picking up a fine-tuned one."""

from pathlib import Path

import numpy as np


def test_fine_tuned_deblocker_is_used_when_installed(tmp_path, monkeypatch):
    from fotomedik.ai import models

    monkeypatch.setenv("FOTOMEDIK_MODELS", str(tmp_path))
    assert models.needed(["deblock"]) == [models.DEBLOCK]
    (tmp_path / models.DEBLOCK_TUNED.file).write_bytes(b"x")
    assert models.needed(["deblock"]) == [models.DEBLOCK_TUNED]
    assert models.missing(["deblock"]) == []


def test_patches_come_from_the_damage_and_skip_bare_holes():
    from fotomedik.train.pairs import PATCH, cut_patches

    rng = np.random.default_rng(0)
    good = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    rep = good.copy()
    rep[160:320, 200:440] = 128  # a hole: one flat colour
    rep[96:112, :] = np.clip(good[96:112].astype(int) + 40, 0, 255)  # a band of wrong colour
    a, b = cut_patches(rep, good, 40, rng)
    assert a.shape == b.shape and a.shape[1:] == (PATCH, PATCH, 3) and 0 < len(a) <= 40
    for x, y in zip(a, b):
        right = (np.abs(x.astype(int) - y).max(axis=2) < 24).mean()
        assert right >= 0.25  # never a patch that is (nearly) all hole
    assert sum(not np.array_equal(x, y) for x, y in zip(a, b)) >= len(a) // 2  # mostly where it is damaged


def test_damage_after_a_cut_stays_inside_the_file():
    """Bit flips chosen after a fragment was cut out land in what is left of the file."""
    import io

    from PIL import Image

    from fotomedik.train.damage import damage

    rng = np.random.default_rng(0)
    b = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)).save(b, "JPEG", quality=90)
    good = b.getvalue()
    for seed in range(300):  # every pair of kinds, cuts before flips among them
        bad, what = damage(good, good, seed)
        assert bad[:2] == b"\xff\xd8" and what


def test_guided_deblocker_gets_thumbnail_and_holes_in_step_with_the_tiles(tmp_path, monkeypatch):
    """The guided network gets, for every tile, the thumbnail at a quarter and the holes per
    8x8 block, cut at the same place as the tile; and it is preferred when installed."""
    from fotomedik.ai import models
    from fotomedik.ai.enhance import Engine, deblock

    class In:
        def __init__(self, name):
            self.name = name

    seen = []

    class Guided:
        def get_inputs(self):
            return [In("input"), In("guide"), In("has_guide"), In("holes")]

        def run(self, _o, feeds):
            x, g, hv, m = feeds["input"], feeds["guide"], feeds["has_guide"], feeds["holes"]
            assert g.shape[2:] == (x.shape[2] // 4, x.shape[3] // 4)
            assert m.shape[2:] == (x.shape[2] // 8, x.shape[3] // 8)
            seen.append((float(hv[0, 0]), float(m.mean())))
            return [x, np.zeros((1, 1), np.float32)]

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (600, 700, 3), dtype=np.uint8)
    holes = np.zeros((600, 700), bool)
    holes[96:160] = True
    small = rng.integers(0, 255, (137, 160, 3), dtype=np.uint8)  # the photo's 7:6 shape
    out = deblock(arr, Engine(deblock=Guided()), holes=holes, thumb_small=small)
    assert np.abs(out.astype(int) - arr).max() <= 1  # the stand-in gives the picture back
    assert all(hv == 1.0 for hv, _m in seen) and any(m > 0 for _hv, m in seen)
    deblock(arr, Engine(deblock=Guided()), holes=holes, thumb_small=None)
    assert seen[-1][0] == 0.0  # no thumbnail: the network is told so

    monkeypatch.setenv("FOTOMEDIK_MODELS", str(tmp_path))
    (tmp_path / models.DEBLOCK_TUNED.file).write_bytes(b"x")
    (tmp_path / models.DEBLOCK_GUIDED.file).write_bytes(b"x")
    assert models.needed(["deblock"]) == [models.DEBLOCK_GUIDED]


def test_thumbnail_step_is_skipped_after_the_guided_network():
    """With a whole thumbnail the guided network rebuilds the holes itself (better than the
    enlarged thumbnail after it); with none, or one rescued in part, the step still runs."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_ai import FakeLama, FakeSR

    from fotomedik.ai import models
    from fotomedik.ai.enhance import Engine, enhance_steps

    class In:
        def __init__(self, name):
            self.name = name

    class Guided:
        def get_inputs(self):
            return [In("input"), In("guide"), In("has_guide"), In("holes")]

        def run(self, _o, feeds):
            return [feeds["input"], np.zeros((1, 1), np.float32)]

    rng = np.random.default_rng(1)
    arr = rng.integers(60, 200, (128, 160, 3), dtype=np.uint8)
    holes = np.zeros((128, 160), bool)
    holes[40:88] = True
    small = rng.integers(60, 200, (32, 40, 3), dtype=np.uint8)
    eng = Engine(FakeSR(), FakeLama(), deblock=Guided())
    eng.deblock_model = models.DEBLOCK_GUIDED
    _out, done = enhance_steps(arr, holes, np.zeros_like(holes), eng, ["deblock", "thumbsr"], thumb_small=small)
    assert any("skipped" in d for d in done) and not any("enlarged by" in d for d in done)
    big = rng.integers(60, 200, (2400, 3200, 3), dtype=np.uint8)  # 8 MP: larger than the network learnt on
    bh = np.zeros((2400, 3200), bool)
    bh[800:880] = True
    _out, done = enhance_steps(big, bh, np.zeros_like(bh), eng, ["deblock", "thumbsr"],
                               thumb_small=rng.integers(60, 200, (120, 160, 3), dtype=np.uint8))
    assert any("left out" in d for d in done) and any("enlarged by" in d for d in done)
