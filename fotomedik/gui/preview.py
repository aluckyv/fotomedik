"""Photo previews for the app: what a normal viewer shows, scaled to fit."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageFile


def load_preview(path: Path, max_side: int = 1200) -> Image.Image | None:
    """The photo as an ordinary viewer would show it (damaged data included), downscaled;
    None when it does not open at all."""
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        data = Path(path).read_bytes()
        im = Image.open(io.BytesIO(data))
        if im.format == "JPEG":
            im.draft("RGB", (max_side, max_side))
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        return im
    except Exception:
        return None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
