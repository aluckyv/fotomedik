"""Errors of a run, written out in full: FotoMedik-greske.txt in the output folder.

A message on the screen is short and cannot be copied; the file keeps the whole
story (what was being done, the error, where in the code) so it can be sent on.
The run itself goes on: one photo or one step failing never stops the others.
"""

from __future__ import annotations

import platform
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

LOG_NAME = "FotoMedik-greske.txt"

_lock = threading.Lock()
_counts: dict[str, int] = {}


def start(out: Path) -> None:
    """A new run into `out`: its error count starts at zero (the file is kept)."""
    with _lock:
        _counts[str(out)] = 0


def count(out: Path) -> int:
    with _lock:
        return _counts.get(str(out), 0)


def log_path(out: Path) -> Path:
    return Path(out) / LOG_NAME


def log_error(out: Path | None, what: str, exc: BaseException | None = None, text: str | None = None) -> None:
    """Append one error: `what` was being done, `exc` (with its traceback) or `text`."""
    if exc is not None:
        body = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    else:
        body = (text or "").rstrip() + "\n"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    from . import _build

    ver = f"build {_build.BUILD} {_build.COMMIT[:7]}".strip()
    entry = f"=== {stamp}  {what}\nFotoMedik {ver}, Python {sys.version.split()[0]}, {platform.platform()}\n{body}\n"
    with _lock:
        if out is not None:
            _counts[str(out)] = _counts.get(str(out), 0) + 1
            try:
                Path(out).mkdir(parents=True, exist_ok=True)
                with open(log_path(out), "a", encoding="utf-8") as fh:
                    fh.write(entry)
            except OSError:
                pass
