"""Twins: shots of the same scene taken seconds apart.

People often press the shutter twice. When a photo's own thumbnail is lost, the
thumbnail (or the real parts) of such a neighbouring shot tells what colour and
brightness its holes have - see recipes.dcfield.twin_base. Only the pairing lives
here: by the time the shots were taken, read from what is left of their headers.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

TWIN_GAP_S = 30.0  # shots further apart in time are not taken for the same scene
HEAD = 1 << 17  # bytes searched for a time stamp when the header no longer parses
_RAW = re.compile(rb"((?:19|20)\d\d):(\d\d):(\d\d) (\d\d):(\d\d):(\d\d)")


def taken_at(path: Path) -> datetime | None:
    """When the photo was taken: EXIF (or a time stamp in its name, see ai.burst), else
    the first EXIF-style date in the first bytes - a rotted header still holds it."""
    from .ai.burst import taken_at as exif_time

    try:
        when = exif_time(path)
    except Exception:
        when = None
    if when is not None:
        return when
    try:
        with open(path, "rb") as fh:
            head = fh.read(HEAD)
    except OSError:
        return None
    for m in _RAW.finditer(head):
        try:
            return datetime(*map(int, m.groups()))
        except ValueError:
            continue
    return None


def find_twins(paths: list[Path], gap_s: float = TWIN_GAP_S) -> dict[Path, Path]:
    """For each photo, the nearest other photo of the same folder taken within gap_s
    seconds of it (photos without a readable time get none)."""
    times = {p: taken_at(p) for p in paths}
    out: dict[Path, Path] = {}
    by_dir: dict[Path, list[tuple[datetime, Path]]] = {}
    for p, t in times.items():
        if t is not None:
            by_dir.setdefault(p.parent, []).append((t, p))
    for shots in by_dir.values():
        shots.sort(key=lambda s: s[0])
        for i, (t, p) in enumerate(shots):
            best = None
            for j in (i - 1, i + 1):
                if 0 <= j < len(shots):
                    gap = abs((shots[j][0] - t).total_seconds())
                    if gap <= gap_s and (best is None or gap < best[0]):
                        best = (gap, shots[j][1])
            if best is not None:
                out[p] = best[1]
    return out
