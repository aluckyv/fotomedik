"""A quick look at a folder before the repair: what kind of damage it holds and which
quality preset fits. Nothing is written.

For every photo: whether it opens, how much of its scan decodes, whether its header is
damaged (the set's healthy headers mend it in the repair), whether it has a whole
thumbnail, a partly rescued one or none (the thumbnail decides how its holes are filled),
whether faces are in it (looked for on the thumbnail), when it was taken (similar shots
within a few minutes lend real pixels to each other). A few seconds for a folder: the
scan is the decoder's first pass, the thumbnails are tiny.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

WHOLE_THUMB = 0.95  # share of a rescued thumbnail that is real for it to count as whole
BURST_SECONDS = 180  # shots this close in time are similar shots (see ai.burst)


@dataclass
class PhotoFacts:
    name: str
    size: int = 0
    width: int | None = None
    height: int | None = None
    state: str = "unreadable"  # healthy | damaged | header | unreadable
    recovered: float | None = None  # share of the scan that decodes (None when the header is damaged)
    thumb: str = "none"  # whole | part | none
    thumb_real: float = 0.0
    thumb_damaged: bool = False  # the Exif preview rotted (the copy gets a fresh one)
    faces: int | None = None  # None: not looked for (no face finder here)
    taken: datetime | None = None
    error: str = ""

    @property
    def damaged(self) -> bool:
        return self.state in ("damaged", "header")


@dataclass
class Overview:
    photos: list[PhotoFacts] = field(default_factory=list)

    def count(self, pred) -> int:
        return sum(1 for p in self.photos if pred(p))

    @property
    def damaged(self) -> list[PhotoFacts]:
        return [p for p in self.photos if p.damaged]

    def bursts(self) -> int:
        """Groups of two or more shots taken within BURST_SECONDS of each other."""
        times = sorted(p.taken for p in self.photos if p.taken is not None)
        groups, i = 0, 0
        while i < len(times):
            j = i
            while j + 1 < len(times) and (times[j + 1] - times[j]).total_seconds() <= BURST_SECONDS:
                j += 1
            if j > i:
                groups += 1
            i = j + 1
        return groups

    def suggest(self, gpu: bool = False) -> tuple[str, list[str]]:
        """(preset key, the reasons in short lines). Presets: see gui.app.PRESETS."""
        d = self.damaged
        why: list[str] = []
        previews = self.count(lambda p: p.thumb_damaged)
        if not d:
            if previews:
                why.append(f"the pictures are whole; {previews} with a damaged preview (Exif thumbnail): "
                           "the copies get a fresh one")
            else:
                why.append("no damaged photo found: only the repaired copies are made" if self.photos
                           else "no JPEG photos found")
            return "faithful", why
        whole = sum(1 for p in d if p.thumb == "whole")
        part = sum(1 for p in d if p.thumb == "part")
        none = len(d) - whole - part
        header = sum(1 for p in d if p.state == "header")
        unread = self.count(lambda p: p.state == "unreadable")
        why.append(f"{len(d)} of {len(self.photos)} photos damaged" + (f", {header} with a damaged header" if header else ""))
        if whole:
            why.append(f"{whole} with a whole thumbnail: pieces put back and holes filled after it")
        if part:
            why.append(f"{part} with a partly rescued thumbnail: its real part guides the holes, faint bands evened")
        if none:
            why.append(f"{none} without a thumbnail: holes healed from the picture around them")
        faces = sum(1 for p in d if p.faces)
        if faces:
            why.append(f"faces in {faces} damaged photo(s)" + (": Maximum also restores faces" if gpu else ""))
        b = self.bursts()
        if b:
            why.append(f"{b} group(s) of similar shots: real pixels lent between them")
        if unread:
            why.append(f"{unread} file(s) cannot be read as JPEG at all")
        if previews:
            why.append(f"{previews} with a damaged preview (Exif thumbnail): the copies get a fresh one")
        return "standard", why

    def text(self, gpu: bool = False) -> str:
        preset, why = self.suggest(gpu)
        names = {"faithful": "Faithful", "standard": "Standard", "maximum": "Maximum"}
        return "; ".join(why) + f". Suggested: {names[preset]}."


def _facts(args) -> dict:
    """Worker: the facts of one photo (plain types for the process pool)."""
    path, rel, thumb_refs, face_model = args
    out = {"name": rel, "size": 0, "width": None, "height": None, "state": "unreadable", "recovered": None,
           "thumb": "none", "thumb_real": 0.0, "thumb_damaged": False, "faces": None, "taken": None, "error": ""}
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        out["error"] = str(e)
        return out
    out["size"] = len(data)
    try:
        from .scan.scanner import scan_bytes

        d = scan_bytes(data, Path(rel).name)
        out["width"], out["height"] = d.width, d.height
        out["thumb_damaged"] = d.has("thumbnail_damaged")
        if d.openable:
            out["recovered"] = float(d.recovered_fraction)
            out["state"] = "healthy" if d.bad_intervals == 0 and d.recovered_fraction >= 0.999 else "damaged"
        elif thumb_refs is not None or d.width:  # the set's headers can mend it
            out["state"] = "header"
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    try:
        import numpy as np

        from .recipes.dcfield import _crop_black_bars, thumbnail_image

        got = thumbnail_image(data, thumb_refs)
        if got is not None:
            img, mask = _crop_black_bars(got[0], got[1])
            real = 1.0 if mask is None else float(np.mean(mask))
            out["thumb"], out["thumb_real"] = ("whole" if real >= WHOLE_THUMB else "part"), real
            if face_model:
                out["faces"] = _faces_on(img, mask, face_model)
    except Exception:  # noqa: BLE001
        pass
    try:
        from .ai.burst import taken_at

        out["taken"] = taken_at(Path(path), data)
    except Exception:  # noqa: BLE001
        pass
    return out


def _faces_on(ycc, mask, face_model: str) -> int:
    """Faces on the thumbnail's real part, looked for on an enlarged copy (they are tiny)."""
    import numpy as np
    from PIL import Image

    from .ai.faces import detect

    rgb = np.asarray(Image.fromarray(np.ascontiguousarray(ycc), "YCbCr").convert("RGB"))
    if mask is not None:
        rgb = np.where(mask[..., None], rgb, 0).astype(np.uint8)
    h, w = rgb.shape[:2]
    k = max(1, 640 // max(h, w))
    big = np.asarray(Image.fromarray(rgb).resize((w * k, h * k), Image.BICUBIC))
    return len(detect(big, face_model))


def overview(files: list[tuple[Path, Path]], jobs: int = 0, say=None, faces: bool = True) -> Overview:
    """The facts of `files` ((path, name) pairs). say(text) reports the two stages."""
    from .cli import _learn_candidates

    if say:
        say("Looking at the photos: the set's headers…")
    thumb_refs = None
    try:
        _refs, thumb_refs = _learn_candidates(files, lambda s: None, jobs or (os.cpu_count() or 1))
    except Exception:  # noqa: BLE001
        thumb_refs = None
    face_model = None
    if faces:
        try:
            import cv2  # noqa: F401

            from .ai import models

            p = models.model_dir() / models.FACE_DET.file
            face_model = str(p) if p.exists() else str(models.ensure(models.FACE_DET))
        except Exception:  # noqa: BLE001
            face_model = None
    if say:
        say("Looking at the photos…")
    items = [(str(p), str(rel), thumb_refs, face_model) for p, rel in files]
    if jobs == 1 or len(items) <= 2:
        rows = [_facts(it) for it in items]
    else:
        with ProcessPoolExecutor(jobs or None) as ex:
            rows = list(ex.map(_facts, items))
    return Overview([PhotoFacts(**r) for r in rows])
