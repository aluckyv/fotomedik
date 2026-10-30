"""Compare a repaired photo with an intact copy of the same picture (ground truth).

An intact copy found later (another backup, a print scan, a copy somebody edited) tells
how good a repair really is: which "real" parts are right, which were put together
wrongly, how far the filled holes are from the truth. Such copies are rarely the same
file: editors re-save, resample by a pixel or two and sometimes brighten, and an
AI-restored copy drifts by a dozen pixels here and there. So the copy is first moved
onto the repaired picture with a smooth shift field (the one the reference-guided repair
uses, fotomedik.reference), then every row of blocks is compared at its best remaining
local shift, and the result is a map of the error per MCU rather than a bit-exact test.

Numbers: mean absolute luma error (0..255) over the whole picture, over the trusted
(decoded) MCUs and over the filled holes; the share of trusted MCUs that are right
(error below RIGHT) and wrong (above WRONG). The report JSON written next to a repaired
photo (--reports) supplies the holes; without it, only the overall error is known.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

RIGHT = 10.0  # mean |error| of an MCU that is as it should be (JPEG noise, small shifts)
WRONG = 20.0  # an MCU this far off is not the picture
SHIFT_Y, SHIFT_X = 4, 2  # pixels of local misalignment tolerated after the shift field (per row)
UNDO_ORIENTATION = {3: Image.ROTATE_180, 6: Image.ROTATE_90, 8: Image.ROTATE_270}


@dataclass
class Comparison:
    width: int
    height: int
    mcu_w: int
    mcu_h: int
    err: np.ndarray  # (rows, per_row) mean |luma error| per MCU
    holes: np.ndarray | None  # (rows, per_row) bool, when a report was given
    chroma: np.ndarray = field(default=None)  # (rows, per_row) mean |chroma error|
    drift: float = 0.0  # pixels: the largest shift the copy needed to lie on the repaired picture

    @property
    def trusted(self) -> np.ndarray | None:
        return None if self.holes is None else ~self.holes

    def summary(self) -> dict:
        out = {"error": float(self.err.mean()), "chroma_error": float(self.chroma.mean()), "mcus": int(self.err.size),
               "drift": round(float(self.drift), 1)}
        if self.holes is not None:
            t = self.trusted
            n = int(t.sum())
            out.update({
                "trusted_mcus": n, "hole_mcus": int(self.holes.sum()),
                "trusted_error": float(self.err[t].mean()) if n else 0.0,
                "hole_error": float(self.err[self.holes].mean()) if self.holes.any() else 0.0,
                "right": int((t & (self.err < RIGHT)).sum()), "wrong": int((t & (self.err >= WRONG)).sum()),
            })
        return out

    def format_text(self) -> str:
        s = self.summary()
        lines = [f"Mean error: {s['error']:.1f} of 255 (colour {s['chroma_error']:.1f}) over {s['mcus']} MCUs"
                 + (f"; the copy drifted up to {s['drift']:.0f} px" if s["drift"] >= 1 else "")]
        if self.holes is not None and s["trusted_mcus"]:
            n = s["trusted_mcus"]
            lines.append(f"Decoded (trusted) MCUs: {n} ({n / s['mcus']:.0%}), error {s['trusted_error']:.1f}: "
                         f"{s['right']} right ({s['right'] / n:.0%}), {s['wrong']} wrong ({s['wrong'] / n:.0%})")
            lines.append(f"Filled holes: {s['hole_mcus']} MCUs, error {s['hole_error']:.1f}")
        return "\n".join(lines)

    def error_map(self) -> Image.Image:
        """The error per MCU as a gray picture (4 gray levels per level of error)."""
        em = np.clip(self.err * 4, 0, 255).astype(np.uint8)
        em = np.repeat(np.repeat(em, self.mcu_h, 0), self.mcu_w, 1)[: self.height, : self.width]
        return Image.fromarray(em)

    def wrong_runs(self, min_len: int = 3) -> list[tuple[int, int, int, float]]:
        """(row, first column, end column, error) of runs of wrong trusted MCUs."""
        if self.holes is None:
            return []
        t = self.trusted
        runs = []
        for r in range(self.err.shape[0]):
            c = 0
            while c < self.err.shape[1]:
                if t[r, c] and self.err[r, c] >= WRONG:
                    c0 = c
                    while c < self.err.shape[1] and t[r, c] and self.err[r, c] >= (RIGHT + WRONG) / 2:
                        c += 1
                    if c - c0 >= min_len:
                        runs.append((r, c0, c, float(self.err[r, c0:c].mean())))
                else:
                    c += 1
        return runs


def frames(truth: Image.Image, repaired: Image.Image) -> list[Image.Image]:
    """The intact copy (as a viewer shows it) in the repaired file's stored frame and size.
    When the copy lies the other way round and the file does not say which way it was
    turned, both turns are returned; the caller keeps the one that fits."""
    o = repaired.getexif().get(0x0112, 1)
    t = truth
    if o in UNDO_ORIENTATION:
        t = t.transpose(UNDO_ORIENTATION[o])
    cands = [t]
    if (t.size[0] < t.size[1]) != (repaired.size[0] < repaired.size[1]):  # still the other way round
        cands = [t.transpose(Image.ROTATE_90), t.transpose(Image.ROTATE_270)]
    return [c.resize(repaired.size, Image.BICUBIC) if c.size != repaired.size else c for c in cands]


def into_frame(truth: Image.Image, repaired: Image.Image) -> Image.Image:
    """The best-fitting frame of `frames`."""
    cands = frames(truth, repaired)
    if len(cands) == 1:
        return cands[0]
    r = np.asarray(repaired.convert("L"), np.float32)
    return min(cands, key=lambda c: float(np.abs(r - np.asarray(c.convert("L"), np.float32)).mean()))


def holes_mask(report: dict, per_row: int, rows: int) -> np.ndarray:
    m = np.zeros(per_row * rows, bool)
    for h in report.get("holes", []):
        m[h["mcu_first"] : h["mcu_first"] + h["mcu_count"]] = True
    return m.reshape(rows, per_row)


def compare(repaired_path: Path, truth_path: Path, report: dict | None = None) -> Comparison:
    from .ai.enhance import mcu_geometry

    data = Path(repaired_path).read_bytes()
    mw, mh, per_row, W, H = mcu_geometry(data)
    rows = -(-H // mh)
    ri = Image.open(repaired_path)
    rep = np.asarray(ri.convert("YCbCr"), np.float32)
    tru = np.asarray(into_frame(Image.open(truth_path).convert("RGB"), ri).convert("YCbCr"), np.float32)
    holes = holes_mask(report, per_row, rows) if report else None
    tru, drift = aligned(tru, rep, holes, mw, mh)
    py, px = SHIFT_Y, SHIFT_X
    tp = np.pad(tru, ((py, py), (px, px), (0, 0)), mode="edge")
    err = np.zeros((rows, per_row), np.float32)
    chroma = np.zeros((rows, per_row), np.float32)
    for r in range(rows):
        y0, y1 = r * mh, min((r + 1) * mh, H)
        best = None
        for dy in range(-py, py + 1):
            for dx in range(-px, px + 1):
                d = np.abs(rep[y0:y1] - tp[y0 + py + dy : y1 + py + dy, px + dx : W + px + dx])
                m = float(d[..., 0].mean())
                if best is None or m < best[0]:
                    best = (m, d)
        d = np.pad(best[1], ((0, 0), (0, per_row * mw - W), (0, 0)), mode="edge")
        d = d.reshape(y1 - y0, per_row, mw, 3).mean(axis=(0, 2))
        err[r] = d[:, 0]
        chroma[r] = d[:, 1:].mean(axis=1)
    return Comparison(W, H, mw, mh, err, holes, chroma, drift)


def aligned(tru: np.ndarray, rep: np.ndarray, holes: np.ndarray | None, mw: int, mh: int) -> tuple[np.ndarray, float]:
    """The copy moved onto the repaired picture by a smooth shift field estimated where
    the picture is trusted (decoded) and textured; (copy, largest shift in pixels). The
    copy stays as it is when no shift can be told (flat picture, nothing trusted)."""
    from .reference import shift_field, warp

    H, W = rep.shape[:2]
    trusted = np.ones((H, W), dtype=bool)
    if holes is not None:
        trusted = ~np.repeat(np.repeat(holes, mh, 0), mw, 1)[:H, :W]
    try:
        dy, dx = shift_field(tru[..., 0], rep[..., 0], trusted)
    except Exception:
        return tru, 0.0
    if not (np.isfinite(dy).all() and np.isfinite(dx).all()):
        return tru, 0.0
    drift = float(np.hypot(dy, dx).max())
    if drift < 0.5:
        return tru, drift
    return np.stack([warp(tru[..., i], dy, dx) for i in range(tru.shape[2])], axis=-1), drift


def report_for(repaired_path: Path) -> dict | None:
    """The report JSON written next to a repaired photo, when there is one."""
    p = Path(repaired_path)
    rp = p.with_name(p.stem + ".report.json")
    if rp.exists():
        try:
            return json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def side_by_side(repaired_path: Path, truth_path: Path, cmp: Comparison, max_width: int = 2400) -> Image.Image:
    """truth | repaired | error map, in the repaired file's frame."""
    ri = Image.open(repaired_path)
    t = into_frame(Image.open(truth_path).convert("RGB"), ri)
    W, H = ri.size
    pic = Image.new("RGB", (W * 3, H))
    pic.paste(t, (0, 0))
    pic.paste(ri.convert("RGB"), (W, 0))
    pic.paste(cmp.error_map().convert("RGB"), (2 * W, 0))
    if pic.size[0] > max_width:
        pic.thumbnail((max_width, max_width))
    return pic
