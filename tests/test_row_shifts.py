"""The jigsaw moves whole regions of rows the repair put 8, 16, 24 rows off (ai.puzzle.row_shifts)."""

import numpy as np

from fotomedik.ai.puzzle import row_shifts, shift_regions

ROWS, COLS = 60, 12


def _scene():
    """Block colours of a scene with a distinct colour per row (so a row tells its place)."""
    y = np.arange(ROWS, dtype=np.float32)[:, None, None]
    x = np.arange(COLS, dtype=np.float32)[None, :, None]
    return np.concatenate([y * 3 + x * 0, 200 - y * 2 + x, np.full((ROWS, COLS, 1), 90.0) + (y % 5) * 6], axis=-1)


def test_a_region_the_repair_put_16_rows_too_high_is_found():
    thumb = _scene()
    rep = thumb.copy()
    rep[10:30] = thumb[26:46]  # the repair lost two cycles of 8 intervals: rows 26..45 sit at 10..29
    real = np.ones((ROWS, COLS), bool)
    sh = row_shifts(rep, thumb, real)
    assert (sh[10:30] == 16).all()
    assert (sh[:10] == 0).all() and (sh[46:] == 0).all()


def test_a_flat_region_moves_with_the_rows_around_it_and_a_lone_row_does_not():
    thumb = _scene()
    rep = thumb.copy()
    rep[10:30] = thumb[26:46]
    rep[15:20] = rep[15:20].mean(axis=(0, 1))  # a flat sky inside: these rows fit anywhere
    real = np.ones((ROWS, COLS), bool)
    sh = row_shifts(rep, thumb, real)
    assert (sh[15:20] == 16).all()  # between rows that moved by 16: moved alike
    rep2 = thumb.copy()
    rep2[50] = thumb[58]  # one odd row only
    assert not row_shifts(rep2, thumb, real).any()


def test_shift_regions_moves_the_pixels_and_leaves_holes_behind():
    mh = 4
    rep = np.zeros((ROWS * mh, COLS, 3), np.uint8)
    rep[:, :, 0] = np.repeat(np.arange(ROWS), mh)[:, None]
    holes = np.zeros(rep.shape[:2], bool)
    ref = np.full_like(rep, 255)
    shift = np.zeros(ROWS, np.int64)
    shift[10:12] = 16
    real = np.ones((ROWS, COLS), bool)
    real[26:28] = False  # where they go: nothing decoded there
    out, hol, n = shift_regions(rep, holes, ref, shift, real, mh)
    assert n == 2
    assert (out[26 * mh : 28 * mh, :, 0] == np.repeat([10, 11], mh)[:, None]).all()
    assert hol[10 * mh : 12 * mh].all() and (out[10 * mh : 12 * mh] == 255).all()
    assert not hol[26 * mh : 28 * mh].any()
