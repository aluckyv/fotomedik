"""Synthetic test material: a deterministic 'photo' and controlled corruptions.

Every corruption helper has a known answer, so tests can assert exactly what
the scanner must find and what the repair must restore.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageDraw


def make_image(w: int = 256, h: int = 192, seed: int = 7) -> Image.Image:
    """Photo-like RGB image: smooth gradients, a few shapes, mild noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 90 + 120 * (x / w)
    g = 60 + 140 * (y / h)
    b = 200 - 100 * ((x + y) / (w + h))
    arr = np.stack([r, g, b], axis=-1)
    arr += rng.normal(0, 3, arr.shape).astype(np.float32)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    d.ellipse((w * 0.15, h * 0.2, w * 0.45, h * 0.7), fill=(230, 200, 60))
    d.rectangle((w * 0.55, h * 0.35, w * 0.9, h * 0.9), fill=(40, 70, 160))
    d.line((0, h * 0.85, w, h * 0.6), fill=(250, 250, 250), width=3)
    return img


def make_jpeg(
    img: Image.Image,
    restart_rows: int | None = None,
    progressive: bool = False,
    quality: int = 90,
    subsampling: str | int = "4:2:0",
    exif: bytes | None = None,
) -> bytes:
    """Encode with Pillow. restart_rows=N puts a restart marker every N MCU rows."""
    buf = io.BytesIO()
    kw: dict = {"quality": quality, "subsampling": subsampling, "progressive": progressive}
    if restart_rows is not None:
        kw["restart_marker_rows"] = restart_rows
    if exif is not None:
        kw["exif"] = exif
    img.save(buf, format="JPEG", **kw)
    return buf.getvalue()


def flip_bits(data: bytes, byte_off: int, mask: int) -> bytes:
    """XOR one byte with mask (1 bit set in mask = 1 flipped bit)."""
    out = bytearray(data)
    out[byte_off] ^= mask
    return bytes(out)


def truncate(data: bytes, n: int) -> bytes:
    return data[:n]


def xor_prefix(data: bytes, key: int = 0x76, n: int = 512) -> bytes:
    """CardRecovery-style obfuscation: XOR the first n bytes with key."""
    head = bytes(c ^ key for c in data[:n])
    return head + data[n:]


def append_garbage(data: bytes, n: int, seed: int = 0) -> bytes:
    """Append n bytes of random (non-JPEG) content, deterministic by seed."""
    rnd = random.Random(seed)
    return data + bytes(rnd.getrandbits(8) for _ in range(n))
