"""What the AI healing must leave alone or treat with care.

* The camera's date stamp (yellow-orange digits in a corner). LaMa reads digits cut by
  a hole as texture and invents new ones ("13/08/2005" -> "18/118/02II05"); the stamp
  is therefore kept exactly as the repair left it and the healing sees it only as context.
* Faces. Eyes, brows and mouths are small dark marks on a light face, which the stripe
  checks take for rotted colour; on a face those checks are switched off (a stripe that
  runs on past the face is still treated).
* Face landmarks for GFPGAN: a detector fooled by a damaged face gives landmarks in the
  wrong places, and GFPGAN then draws a mouth over the eyes. Such faces are skipped.
"""

from __future__ import annotations

import numpy as np

# date stamp: the camera prints it in a corner - bottom right of the landscape frame, so
# along the left or right edge when the photo was turned to portrait
STAMP_CORNERS = (  # (top, bottom, left, right) shares of the picture, and the text direction
    ((0.78, 1.0, 0.5, 1.0), "h"),  # landscape: bottom right
    ((0.45, 1.0, 0.0, 0.16), "v"),  # turned clockwise: left edge, lower part
    ((0.0, 0.55, 0.84, 1.0), "v"),  # turned anticlockwise: right edge, upper part
)
STAMP_MIN_BLOCKS = 6  # 8x8 blocks with stamp yellow in them
STAMP_ASPECT = 1.6  # length / thickness of the yellow blocks' box
STAMP_FILL = 0.6  # at most this share of that box's pixels yellow: text is sparse, a yellow car is not
STAMP_LONG = 0.32  # the stamp's length at most this share of the picture side it runs along
STAMP_THICK = 0.1  # its thickness at most this share of the picture's short side


def yellow(arr: np.ndarray) -> np.ndarray:
    """The saturated yellow of a printed date stamp."""
    a = arr.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 150) & (g > 140) & (b * 2.2 < np.minimum(r, g)) & (np.abs(r - g) < 70)


def date_stamp(arr: np.ndarray) -> np.ndarray:
    """Pixel mask of the date stamp's box (8-pixel aligned, one block of margin), or all
    False when there is none."""
    h, w = arr.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    bh, bw = -(-h // 8), -(-w // 8)
    y = np.pad(yellow(arr), ((0, bh * 8 - h), (0, bw * 8 - w)))
    yb = y.reshape(bh, 8, bw, 8).sum(axis=(1, 3)) >= 2
    if h > w:  # only the corners the camera could have used for this orientation
        corners = [c for c in STAMP_CORNERS if c[1] == "v"]
    else:
        corners = [c for c in STAMP_CORNERS if c[1] == "h"]
    for (t, b, l, r), way in corners:
        y0, y1, x0, x1 = int(t * bh), int(np.ceil(b * bh)), int(l * bw), int(np.ceil(r * bw))
        part = yb[y0:y1, x0:x1]
        if part.sum() < STAMP_MIN_BLOCKS:
            continue
        rows, cols = np.nonzero(part.any(axis=1))[0], np.nonzero(part.any(axis=0))[0]
        ya, yz, xa, xz = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        bhgt, bwid = yz - ya, xz - xa
        length, thick = (bwid, bhgt) if way == "h" else (bhgt, bwid)
        along = bw if way == "h" else bh
        fill = y[(y0 + ya) * 8 : (y0 + yz) * 8, (x0 + xa) * 8 : (x0 + xz) * 8].mean()
        if (length < STAMP_ASPECT * thick or fill > STAMP_FILL
                or length > STAMP_LONG * along or thick > STAMP_THICK * min(bh, bw)):
            continue
        ya, yz = max(0, y0 + ya - 1), min(bh, y0 + yz + 1)
        xa, xz = max(0, x0 + xa - 1), min(bw, x0 + xz + 1)
        out[ya * 8 : yz * 8, xa * 8 : xz * 8] = True
    return out


# faces
FACE_GROW = 0.35  # the protected oval reaches this share of the face size past the detection


def landmarks_ok(pts: np.ndarray) -> bool:
    """Five YuNet landmarks (right eye, left eye, nose, right and left mouth corner, in the
    picture's x/y) that make a face: eyes side by side, nose below and between them,
    mouth below the nose, sizes in proportion. A face rotated up to ~35 degrees passes."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (5, 2) or not np.isfinite(pts).all():
        return False
    e1, e2, nose, m1, m2 = pts
    eye_v = e2 - e1
    d = float(np.hypot(*eye_v))
    if d < 4:
        return False
    ux = eye_v / d  # along the eyes
    uy = np.array([-ux[1], ux[0]])  # down the face (y grows downwards)
    if abs(np.degrees(np.arctan2(eye_v[1], eye_v[0]))) > 35:
        return False

    def local(p):
        q = p - (e1 + e2) / 2
        return float(q @ ux) / d, float(q @ uy) / d

    nx, ny = local(nose)
    (ax, ay), (bx, by) = local(m1), local(m2)
    mouth_w = bx - ax
    mouth_y = (ay + by) / 2
    # FFHQ template: nose at (0, 0.59), mouth at y 1.03, mouth width 0.89 eye distances
    return (abs(nx) < 0.45 and 0.25 < ny < 1.0 and ny + 0.1 < mouth_y < 1.7
            and 0.45 < mouth_w < 1.5 and abs(ay - by) < 0.35)


def face_mask(arr: np.ndarray, model_path) -> np.ndarray:
    """Pixel mask of ovals around the faces found (only well-formed ones), or all False."""
    h, w = arr.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    if not model_path:
        return out
    try:
        import cv2

        from .faces import detect
        faces = detect(arr, str(model_path))
    except Exception:  # noqa: BLE001 - no face guard is better than no healing
        return out
    m = np.zeros((h, w), np.uint8)
    for pts in faces:
        if not landmarks_ok(pts):
            continue
        e1, e2, _n, m1, m2 = pts
        d = float(np.hypot(*(e2 - e1)))
        c = (e1 + e2 + m1 + m2) / 4
        ax = int(d * (1.1 + FACE_GROW))
        ay = int(d * (1.4 + FACE_GROW))
        ang = float(np.degrees(np.arctan2(e2[1] - e1[1], e2[0] - e1[0])))
        cv2.ellipse(m, (int(c[0]), int(c[1] - 0.15 * d)), (ax, ay), ang, 0, 360, 1, -1)
    return m.astype(bool)


def outside_only(mask: np.ndarray, faces: np.ndarray, blocks: int = 3) -> np.ndarray:
    """`mask` without its pieces that lie (nearly) all inside `faces`: an eye or a mouth
    stays, a rotted stripe running across a face and on past it is kept whole."""
    if not mask.any() or not faces.any():
        return mask
    import cv2

    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    outside = np.bincount(lab[mask & ~faces], minlength=n)
    keep = outside >= blocks * 64
    keep[0] = False
    return keep[lab]
