"""Tail runs after a hole are set to the level of their seams (r12_smooth.tail_shifts)."""

from types import SimpleNamespace

import numpy as np

from fotomedik.recipes.r12_smooth import tail_shifts

COMP = SimpleNamespace(cid=1, h=1, v=1)
PER_ROW = 8  # 8 x 6 blocks of one component, one block per MCU


def _plane(rows=6, cols=8):
    y = np.arange(rows * 8, dtype=np.float32)[:, None]
    return np.repeat(y * 0.5 - 20.0, cols * 8, axis=1)  # a soft vertical gradient, like a sky


def test_a_run_off_by_a_shade_is_set_back_to_its_seams():
    plane = _plane()
    holes = np.zeros((6, 8), bool)
    holes[2, :2] = True  # the hole the run follows
    plane[16:24, 16:64] += 20.0  # the run (row 2, columns 2..7) decoded 20 levels too bright
    shifts, moved = tail_shifts({1: plane}, [COMP], PER_ROW, {1: holes}, [(18, 24)], {1: 4.0})
    assert moved == 1
    assert (shifts[1][2, 2:] == -40).all()  # -20 levels = -40 DC units at quantiser 4
    assert (shifts[1][holes] == 0).all() and (shifts[1][[0, 1, 3, 4, 5]] == 0).all()


def test_a_real_edge_along_one_side_moves_nothing():
    plane = _plane()
    holes = np.zeros((6, 8), bool)
    holes[2, :2] = True
    plane[:16] -= 30.0  # a real edge above the run (a roof against the sky): only one seam differs
    shifts, moved = tail_shifts({1: plane}, [COMP], PER_ROW, {1: holes}, [(18, 24)], {1: 4.0})
    assert moved == 0 and not shifts[1].any()


def test_a_run_between_holes_is_left_alone():
    plane = _plane()
    holes = np.zeros((6, 8), bool)
    holes[1] = holes[3] = True
    holes[2, :2] = True
    plane[16:24, 16:64] += 20.0
    shifts, moved = tail_shifts({1: plane}, [COMP], PER_ROW, {1: holes}, [(18, 24)], {1: 4.0})
    assert moved == 0 and not shifts[1].any()  # no seams to judge it by
