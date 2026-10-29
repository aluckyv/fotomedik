"""R1 — CardRecovery temporary files (*.~CR).

CardRecovery stores carved photos in TmpFiles/ with the first 512 bytes XORed
with 0x76 and tells users to buy a decode tool. Undoing it is one line.
"""

from __future__ import annotations

import re

from ..scan.scanner import CARDRECOVERY_KEY, CARDRECOVERY_PREFIX, looks_cardrecovery
from .base import Context, Recipe
from .registry import register

_NAME_RE = re.compile(r"\.~cr$", re.IGNORECASE)
_TYPE_RE = re.compile(r"_(JPG|JPEG)\.~cr$", re.IGNORECASE)


@register
class CardRecoveryXor(Recipe):
    name = "cardrecovery_xor"
    priority = 10
    description = "Decode CardRecovery .~CR temp file (first 512 bytes XOR 0x76)"

    def detect(self, ctx: Context):
        return True if looks_cardrecovery(ctx.data) else None

    def fix(self, ctx: Context, finding) -> None:
        d = ctx.data
        head = bytes(c ^ CARDRECOVERY_KEY for c in d[:CARDRECOVERY_PREFIX])
        ctx.replace_data(head + d[CARDRECOVERY_PREFIX:], self, "decoded CardRecovery XOR prefix")
        if _NAME_RE.search(ctx.name):
            ctx.suggested_name = _TYPE_RE.sub(r"_\1.jpg", ctx.name) if _TYPE_RE.search(ctx.name) else _NAME_RE.sub(".jpg", ctx.name)
