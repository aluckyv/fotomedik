"""The jigsaw ("slagalica"): the pieces of a photo with restart markers put back where
they belong, by its thumbnail.

A camera that writes restart markers (most Kodaks write one after every row of blocks;
many phones too) starts the decoding afresh after each marker: the stretch of data
between two markers decodes on its own, colours included, and always begins at the
left edge of its interval. On a rotted card the markers are numbered only 0..7, so after
a damaged stretch the repair has to guess how many intervals were lost; a wrong guess
puts everything after it 8, 16 or 24 rows too high or too low, and the AI step then
smears the sharp pieces over the places the thumbnail shows (100_3130: the top of the
photo 128 px too high, heads twice).

Here every stretch between markers in the original file is decoded alone (also those the
repair dropped as unusable), and placed at the interval where its block colours agree
best with the thumbnail, among the intervals its marker number allows. Only the run of
blocks before its first decoding error counts. The result is used only when it covers
about as much as the repair's own decoded data and agrees with the thumbnail better.
Nothing is invented: every placed pixel is the file's own; the rest is a hole for the
following steps (the guided network, the thumbnail step, LaMa).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFile

MIN_BYTES = 64  # a stretch shorter than this between two markers holds no usable interval
MIN_RUN = 8  # blocks decoded before the first error: fewer say too little about the place
MAX_ERR = 22.0  # mean block-colour difference from the thumbnail above which a piece fits nowhere
DISTINCT = 0.75  # without a usable marker number the best place must beat the typical one by this
PRIOR_SLACK = 1.2  # the best place with the right marker number may be this much worse than the best of all
TAIL_TRIM = 2  # blocks dropped before a stretch's first decoding error
AMBIGUOUS = 0.85  # the best place must beat the second best with the same marker number by this
MERGE_ERR = 12.0  # a block the repair decoded stays (no piece replaces it) when this close to the thumbnail
KEEP_SHARE = 0.8  # the pieces must cover at least this share of what the repair decoded
RST = re.compile(rb"\xff[\xd0-\xd7]")


@dataclass
class Placed:
    rgb: np.ndarray  # (h, w, 3) uint8: the placed pieces, the enlarged thumbnail elsewhere
    holes: np.ndarray  # (h, w) bool: no piece there
    pieces: int  # stretches placed
    share: float  # share of the photo the pieces cover
    repair_share: float  # share the repair's own decoded data covered


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def chunk_header(repaired: bytes):
    """(header bytes for decoding one restart interval alone, frame info) from the repaired
    file's tables, or None. The frame is made one block row high and one interval wide,
    so the interval's blocks decode in their order."""
    from ..jpeg import markers as M
    from ..jpeg.tables import FrameTables

    res = M.parse_segments(repaired)
    ft = FrameTables.from_parse(res)
    ri = ft.restart_interval
    sof = res.sof
    if not ri or sof is None or sof.marker not in (0xC0, 0xC1):
        return None
    fr = ft.frame
    mw, mh = 8 * fr.hmax, 8 * fr.vmax
    out = bytearray(b"\xff\xd8")
    for s in res.segments:
        if s.marker in (0xDB, 0xC4):
            out += _segment(s.marker, s.payload)
    p = bytearray(sof.payload)
    p[1:3] = mh.to_bytes(2, "big")
    p[3:5] = (ri * mw).to_bytes(2, "big")
    out += _segment(sof.marker, bytes(p))
    out += _segment(0xDA, res.first(M.SOS).payload)
    return bytes(out), ri, mw, mh, fr.width, fr.height


def decode_stretch(head: bytes, chunk: bytes, ri: int, mw: int, mh: int):
    """(rgb (mh, run*mw, 3), run) of one interval decoded alone: the blocks before its first
    decoding error (run = 0: nothing usable)."""
    from ..jpeg import decoder as D
    from ..jpeg import markers as M
    from ..jpeg.tables import FrameTables

    j = head + chunk + b"\xff\xd9"
    try:
        r2 = M.parse_segments(j)
        sr = D.decode_scan(j, r2, FrameTables.from_parse(r2), resync=False)
        ok = np.asarray(sr.mcu_ok, bool)
        run = int(np.argmin(ok)) if not ok.all() else len(ok)
        if run < MIN_RUN:
            return None, 0
        px = _pixels(j)[:mh, : run * mw]
        if px.shape[1] < run * mw:
            return None, 0
        return px, run
    except Exception:  # noqa: BLE001 - a stretch that does not decode is no piece
        return None, 0


def _pixels(j: bytes) -> np.ndarray:
    """The pixels of a stretch decoded alone. It ends in damaged data, so Pillow must be told
    to load truncated pictures; other parts of the program switch that off again after
    their own use (the window's previews), so it is set here, each time, and put back."""
    for _try in range(2):
        old = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            return np.asarray(Image.open(io.BytesIO(j)).convert("RGB"))
        except OSError:
            continue  # switched off meanwhile by another thread: once more
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = old
    raise OSError("stretch does not decode")


def _means(img: np.ndarray, mw: int, mh: int, cols: int, rows: int) -> np.ndarray:
    """(rows*cols, 3) block colours of an image padded to whole blocks."""
    h, w = img.shape[:2]
    pad = np.pad(img.astype(np.float32), ((0, rows * mh - h), (0, cols * mw - w), (0, 0)), mode="edge")
    return pad.reshape(rows, mh, cols, mw, 3).mean(axis=(1, 3)).reshape(-1, 3)


def puzzle(original: bytes, repaired: bytes, small: np.ndarray | None,
           repaired_holes: np.ndarray | None = None) -> Placed | None:
    """The pieces of `original` placed by the thumbnail `small` (RGB), or None when the
    photo has no restart markers or thumbnail, or the pieces do no better than the repair
    (repaired_holes: the repair's holes at full size, for that comparison)."""
    if small is None:
        return None
    try:
        got = chunk_header(repaired)
    except Exception:  # noqa: BLE001
        return None
    if got is None:
        return None
    head, ri, mw, mh, w, h = got
    if abs((small.shape[1] / small.shape[0]) / (w / h) - 1.0) > 0.03:
        return None
    cols, rows = -(-w // mw), -(-h // mh)
    n = cols * rows
    n_iv = -(-n // ri)
    ref = np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC))
    rm = _means(ref, mw, mh, cols, rows)
    rm = np.concatenate([rm, np.repeat(rm[-1:], ri, axis=0)])  # the last interval may be short

    marks = [(m.start(), original[m.start() + 1] - 0xD0) for m in RST.finditer(original)]
    if len(marks) < 4:
        return None
    stretches = [(marks[i][0] + 2, marks[i + 1][0], marks[i][1]) for i in range(len(marks) - 1)]
    stretches.append((marks[-1][0] + 2, len(original), marks[-1][1]))

    cands = []
    starts = np.arange(n_iv) * ri
    for a, b, num in stretches:
        if b - a < MIN_BYTES:
            continue
        px, run = decode_stretch(head, original[a:b], ri, mw, mh)
        if px is None:
            continue
        if run < ri:  # it ended in an error: the last blocks before it are often wrong already
            run -= TAIL_TRIM
            if run < MIN_RUN:
                continue
            px = px[:, : run * mw]
        pm = px.reshape(mh, run, mw, 3).mean(axis=(0, 2)).astype(np.float32)
        errs = np.array([np.abs(pm - rm[s : s + run]).mean() for s in starts])
        best = int(np.argmin(errs))
        prior = (num + 1) % 8
        pick = None
        same = np.flatnonzero(np.arange(n_iv) % 8 == prior)
        if len(same):
            order = same[np.argsort(errs[same])]
            k = int(order[0])
            # a piece that fits several places with its marker number about as well (haze, sky)
            # says nothing about where it belongs: left out
            clear = len(order) < 2 or errs[k] <= AMBIGUOUS * errs[order[1]]
            if errs[k] <= errs[best] * PRIOR_SLACK and clear:
                pick = k
        if pick is None and errs[best] < DISTINCT * float(np.median(errs)):
            pick = best
        if pick is None or errs[pick] > MAX_ERR:
            continue
        cands.append((float(errs[pick]), pick, run, px))
    if not cands:
        return None

    cands.sort(key=lambda c: c[0])
    owner = np.full(n, np.inf, np.float32)
    canvas = np.zeros((rows * mh, cols * mw, 3), np.uint8)
    have = np.zeros(n, bool)
    placed = 0
    rep_px = None
    if repaired_holes is not None:
        # the repair's own decoded blocks that agree with the thumbnail stay where they are:
        # a piece goes only where the repair has nothing, or something that does not fit
        rep_px = np.asarray(Image.open(io.BytesIO(repaired)).convert("RGB"))
        if rep_px.shape[:2] == (h, w):
            real = _means(np.repeat((~repaired_holes)[..., None], 3, axis=2).astype(np.uint8) * 255,
                          mw, mh, cols, rows)[:, 0] > 254
            e_rep = np.abs(_means(rep_px, mw, mh, cols, rows) - rm[:n]).mean(axis=1)
            trusted = real & (e_rep < MERGE_ERR)
            pad = np.pad(rep_px, ((0, rows * mh - h), (0, cols * mw - w), (0, 0)), mode="edge")
            for g in np.flatnonzero(trusted):
                r, c = divmod(int(g), cols)
                canvas[r * mh : (r + 1) * mh, c * mw : (c + 1) * mw] = pad[r * mh : (r + 1) * mh, c * mw : (c + 1) * mw]
            owner[trusted] = -1.0  # never replaced
            have |= trusted
        else:
            rep_px = None
    for e, k, run, px in cands:
        idx = np.arange(k * ri, min(k * ri + run, n))
        free = owner[idx] > e
        if free.mean() < 0.5:
            continue
        for j in np.flatnonzero(free):
            g = idx[j]
            r, c = divmod(int(g), cols)
            canvas[r * mh : (r + 1) * mh, c * mw : (c + 1) * mw] = px[:, j * mw : (j + 1) * mw]
            owner[g] = e
            have[g] = True
        placed += 1

    if not placed:
        return None  # no piece found a better place than the repair gave it
    if placed == 0:
        return None  # nothing found a better place: the repair's own picture as it is
    if rep_px is not None:
        # blocks the repair decoded that neither fit the thumbnail well nor were replaced: kept
        # unless they fit nowhere near it (then they were put in the wrong place)
        keep = real & ~have & (e_rep < MAX_ERR)
        for g in np.flatnonzero(keep):
            r, c = divmod(int(g), cols)
            canvas[r * mh : (r + 1) * mh, c * mw : (c + 1) * mw] = pad[r * mh : (r + 1) * mh, c * mw : (c + 1) * mw]
        have |= keep
    share = float(have.mean())
    repair_share = 0.0
    if repaired_holes is not None:
        repair_share = float((~repaired_holes).mean())
        if share < KEEP_SHARE * repair_share:
            return None
        if rep_px is not None:
            real = ~repaired_holes
            rmask = _means(np.repeat(real[..., None], 3, axis=2).astype(np.uint8) * 255, mw, mh, cols, rows)[:, 0] > 127
            e_rep = np.abs(_means(rep_px, mw, mh, cols, rows) - rm[:n])[rmask].mean() if rmask.any() else np.inf
            e_new = np.abs(_means(canvas[:h, :w], mw, mh, cols, rows) - rm[:n])[have].mean()
            if e_new > e_rep * 1.05 and share < 1.2 * repair_share:
                return None  # the repair's own placement agrees with the thumbnail as well
    holes = ~np.repeat(np.repeat(have.reshape(rows, cols), mh, 0), mw, 1)[:h, :w]
    rgb = np.where(holes[..., None], ref, canvas[:h, :w]).astype(np.uint8)
    return Placed(rgb, holes, placed, share, repair_share)
