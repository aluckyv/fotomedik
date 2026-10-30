"""Face restoration for the enhanced copy: YuNet finds faces and their five landmarks,
each face is aligned to the 512x512 FFHQ layout, GFPGAN restores it and the result is
put back with a soft oval edge.

GFPGAN draws a plausible face from very little; on a face rebuilt from a thumbnail it
can look like somebody else. That is why it only ever goes into the separate copy and
is blended (FIDELITY) rather than pasted.
"""

from __future__ import annotations

import numpy as np

MIN_FACE = 20  # pixels: smaller detections are noise
SCORE = 0.6
FIDELITY = 0.8  # share of the restored face in the result
AGREE = 0.8  # least correlation between the face before and after GFPGAN, seen coarsely
TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.19360], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)  # FFHQ 512


def detect(arr: np.ndarray, model_path: str) -> list[np.ndarray]:
    """Five landmarks (5, 2) of every face found (YuNet: eyes, nose, mouth corners)."""
    import cv2

    h, w = arr.shape[:2]
    det = cv2.FaceDetectorYN.create(model_path, "", (w, h), SCORE, 0.3, 5000)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    out = []
    # small faces are found better on an enlarged copy, large ones on the original
    for scale in (1.0, 2.0) if max(h, w) <= 2400 else (1.0,):
        img = bgr if scale == 1.0 else cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        det.setInputSize((img.shape[1], img.shape[0]))
        _n, faces = det.detect(img)
        if faces is None:
            continue
        for f in faces:
            if min(f[2], f[3]) / scale < MIN_FACE:
                continue
            pts = f[4:14].reshape(5, 2) / scale
            if all(np.linalg.norm(pts[0] - q[0]) > 0.5 * f[2] / scale for q in out):
                out.append(pts.astype(np.float32))
    return out


def _coarse(face: np.ndarray) -> np.ndarray:
    """Luma of the central face area at 16x16: where eyes, nose and mouth are, nothing finer."""
    import cv2

    f = face.astype(np.float32)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return cv2.resize(y[128:448, 96:416], (16, 16), interpolation=cv2.INTER_AREA).ravel()


def agrees(before: np.ndarray, after: np.ndarray) -> bool:
    """GFPGAN kept the face's layout: its coarse light and dark (eyes, mouth) follow the
    input. A face drawn from misplaced landmarks (teeth where the eyes were) does not."""
    a, b = _coarse(before), _coarse(after)
    if a.std() < 1.0 or b.std() < 1.0:
        return False
    return float(np.corrcoef(a, b)[0, 1]) >= AGREE


def restore_faces(arr: np.ndarray, engine) -> tuple[np.ndarray, int]:
    """Faces of `arr` restored by GFPGAN. Returns (rgb, number of faces)."""
    import cv2

    from .guard import landmarks_ok

    faces = [f for f in detect(arr, engine.face_det) if landmarks_ok(f)]
    if not faces:
        return arr, 0
    s = engine.face_s
    name = s.get_inputs()[0].name
    h, w = arr.shape[:2]
    out = arr.astype(np.float32)
    oval = np.zeros((512, 512), np.float32)
    cv2.ellipse(oval, (256, 280), (190, 230), 0, 0, 360, 1.0, -1)
    oval = cv2.GaussianBlur(oval, (0, 0), 24)
    done = 0
    for pts in faces:
        m, _ = cv2.estimateAffinePartial2D(pts, TEMPLATE, method=cv2.LMEDS)
        if m is None:
            continue
        crop = cv2.warpAffine(arr, m, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        x = (crop.astype(np.float32) / 255.0 - 0.5) / 0.5
        y = s.run(None, {name: np.ascontiguousarray(x.transpose(2, 0, 1)[None])})[0][0].transpose(1, 2, 0)
        face = np.clip((y + 1) / 2, 0, 1) * 255.0
        if not agrees(crop, face):  # GFPGAN drew a different face: leave this one alone
            continue
        inv = cv2.invertAffineTransform(m)
        back = cv2.warpAffine(face, inv, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        a = cv2.warpAffine(oval, inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
        a = a * FIDELITY
        out = out * (1 - a) + back * a
        done += 1
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), done
