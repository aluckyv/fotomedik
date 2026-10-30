"""What the AI healing leaves alone: the date stamp, faces, and badly placed face landmarks."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fotomedik.ai.faces import TEMPLATE, agrees
from fotomedik.ai.guard import (
    date_stamp,
    drifted_blocks,
    landmarks_ok,
    outside_only,
)

YELLOW = (250, 240, 30)


def _scene(h=600, w=800):
    rng = np.random.default_rng(1)
    return rng.integers(40, 120, (h, w, 3), dtype=np.uint8)


def test_date_stamp_bottom_right():
    a = _scene()
    cv2.putText(a, "13/08/2005", (580, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
    m = date_stamp(a)
    ys, xs = np.nonzero(m)
    assert m.any()
    assert ys.min() > 500 and xs.min() > 540 and xs.max() > 700
    assert m.mean() < 0.06


def test_date_stamp_portrait_left_edge():
    a = _scene()
    cv2.putText(a, "13/08/2005", (580, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
    p = np.ascontiguousarray(np.rot90(a, -1))  # turned clockwise: the stamp runs down the left edge
    m = date_stamp(p)
    assert m.any()
    assert np.nonzero(m)[1].max() < 0.2 * p.shape[1]


def test_no_stamp_for_a_yellow_object_or_elsewhere():
    a = _scene()
    a[450:600, 300:800] = YELLOW  # a yellow bus across the bottom
    assert not date_stamp(a).any()
    b = _scene()
    cv2.putText(b, "13/08/2005", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)  # top left
    assert not date_stamp(b).any()


def test_landmarks():
    pts = TEMPLATE.copy()
    assert landmarks_ok(pts)
    assert landmarks_ok(pts * 0.3 + 100)  # smaller and elsewhere
    c = pts.mean(axis=0)
    rot = np.radians(20)
    r = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    assert landmarks_ok((pts - c) @ r.T + c)  # tilted head
    upside = pts.copy()
    upside[:, 1] = 512 - upside[:, 1]  # mouth above the eyes
    assert not landmarks_ok(upside)
    swapped = pts.copy()
    swapped[[0, 3]] = swapped[[3, 0]]  # an eye where a mouth corner is
    assert not landmarks_ok(swapped)
    assert not landmarks_ok(np.full((5, 2), np.nan))


def _face():
    f = np.full((512, 512, 3), 200, np.uint8)
    cv2.circle(f, (193, 240), 22, (40, 40, 40), -1)
    cv2.circle(f, (319, 240), 22, (40, 40, 40), -1)
    cv2.ellipse(f, (257, 371), (60, 18), 0, 0, 360, (90, 30, 30), -1)
    return f


def test_agrees_rejects_teeth_over_the_eyes():
    f = _face()
    assert agrees(f, f.astype(np.float32) * 0.9 + 10)
    bad = f.copy()
    bad[200:290, 150:370] = 250  # bright teeth where the eyes were
    cv2.circle(bad, (257, 371), 40, (250, 250, 250), -1)
    assert not agrees(f, bad)


def test_outside_only():
    faces = np.zeros((64, 64), bool)
    faces[8:40, 8:40] = True
    m = np.zeros_like(faces)
    m[16:24, 16:32] = True  # an eye, inside the face
    m[48:56, 0:64] = True  # a stripe below, outside
    m[32:40, 0:64] = True  # a stripe through the face and past it
    out = outside_only(m, faces)
    assert not out[16:24, 16:32].any()
    assert out[48:56].all() and out[32:40].all()


def test_drifted_blocks():
    ref = np.full((32, 32, 3), 180, np.uint8)
    out = ref.copy()
    out[8:12, 8:16] = 60  # a dark dash
    out[24:28, 24:28] = 175  # a small change
    where = np.ones((32, 32), bool)
    bad = drifted_blocks(out, ref, where)
    assert bad[8:16, 8:16].all()
    assert not bad[24:32, 24:32].any()
