"""Training material for the deblocker, and the program picking up a fine-tuned one."""

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
