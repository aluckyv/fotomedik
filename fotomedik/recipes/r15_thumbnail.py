"""R15 — a fresh Exif thumbnail when the embedded one is damaged.

The preview in the header is what Explorer and galleries show. When it rotted (it sits in
the header's sectors) while the picture itself is whole or repaired, the copy gets a new
preview made from the picture; the picture's bytes are not touched. Last in the pipeline:
it previews the final file.
"""

from __future__ import annotations

import io

from PIL import Image

from ..jpeg.exifthumb import ThumbSlot, find_slot, renew, thumbnail_damaged
from .base import Context, Recipe
from .registry import register

PREVIEW_PX = 480  # the picture is decoded at about this size for the thumbnail


@register
class ThumbnailRenew(Recipe):
    name = "thumbnail_renew"
    priority = 98  # after everything, orientation included: the preview is of the final file
    description = "Write a fresh Exif thumbnail when the embedded one is damaged"

    def detect(self, ctx: Context) -> ThumbSlot | None:
        if ctx.fallback is not None:  # the thumbnail is all there is
            return None
        slot = find_slot(ctx.data)
        if slot is None or thumbnail_damaged(ctx.data, slot) is not True:
            return None
        return slot

    def fix(self, ctx: Context, slot: ThumbSlot) -> None:
        try:
            im = Image.open(io.BytesIO(ctx.data))
            im.draft("RGB", (PREVIEW_PX, PREVIEW_PX))
            im = im.convert("RGB")
        except Exception as e:  # noqa: BLE001 - a picture that does not open gets no preview
            ctx.note(self, f"damaged thumbnail left: the picture does not decode ({type(e).__name__})")
            return
        new = renew(ctx.data, im, slot)
        if new is None:
            ctx.note(self, "damaged thumbnail left: the Exif block cannot take a new one")
            return
        ctx.replace_data(new, self, "embedded thumbnail was damaged: a new one made from the picture",
                         old_thumbnail_bytes=slot.length)
