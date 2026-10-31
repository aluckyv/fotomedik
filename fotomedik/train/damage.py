"""Damage as real cards and recoveries leave it, for measuring and for training.

The bench (scripts/bench.py) damages good photos with these to score the repair, and the
deblocker's training pairs (fotomedik.train.pairs) come from the same damage repaired.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

SECTOR = 512  # bytes: cards fail sector by sector


def scan_start(data: bytes) -> int:
    """Offset of the main image's entropy-coded data (after its SOS header)."""
    p = 2
    while p + 4 <= len(data):
        if data[p] != 0xFF:
            raise ValueError("lost in the header")
        m = data[p + 1]
        if m == 0xFF:
            p += 1
            continue
        n = int.from_bytes(data[p + 2 : p + 4], "big")
        if m == 0xDA:
            return p + 2 + n
        p += 2 + n
    raise ValueError("no SOS")


def _aligned(rng, lo: int, hi: int) -> int:
    return int(rng.integers(lo // SECTOR, hi // SECTOR + 1)) * SECTOR


def damage(data: bytes, other: bytes, seed: int, spans: list | None = None) -> tuple[bytes, list[str]]:
    """One or two kinds of damage seen on real photos, in the scan data. `spans` (when
    given) receives the destroyed byte ranges in the good file's offsets."""
    rng = np.random.default_rng(seed)
    s = scan_start(data)
    end = len(data) - 2
    kinds = ["rot", "flips", "foreign", "zeros", "missing", "truncate", "rot", "foreign"]
    chosen = list(rng.choice(kinds, size=int(rng.integers(1, 3)), replace=False))
    buf = bytearray(data)
    done = []
    deleted: list[tuple[int, int]] = []

    def mark(a: int, b: int) -> None:  # a range of buf, in the good file's offsets
        if spans is not None:
            shift = sum(n for p, n in deleted if p <= a)
            spans.append((a + shift, b + shift))

    for k in chosen:
        end = len(buf) - 2  # a fragment cut out or a truncation before shortens what is left
        span = end - s
        at = s + int(rng.uniform(0.1, 0.85) * span)
        if k == "rot":  # a stretch where ~30 % of the bytes have one bit wrong
            n = int(rng.integers(300, 3000))
            for o in range(at, min(at + n, len(buf) - 2)):
                if rng.random() < 0.3:
                    buf[o] ^= 1 << int(rng.integers(0, 8))
            mark(at, at + n)
            done.append(f"rot {n} B")
        elif k == "flips":
            n = int(rng.integers(1, 4))
            for _ in range(n):
                o = s + int(rng.integers(0, span))
                buf[o] ^= 1 << int(rng.integers(0, 8))
                mark(o, o + 1)
            done.append(f"{n} bit flip(s)")
        elif k == "foreign":  # sectors of another photo where this one's were
            n = _aligned(rng, 4096, 32768)
            at -= at % SECTOR
            os_ = scan_start(other)
            src = os_ + _aligned(rng, 0, max(SECTOR, len(other) - os_ - n - SECTOR))
            chunk = other[src : src + n]
            buf[at : at + len(chunk)] = chunk
            mark(at, at + len(chunk))
            done.append(f"foreign {n // 1024} KB")
        elif k == "zeros":
            n = _aligned(rng, 4096, 16384)
            at -= at % SECTOR
            buf[at : at + n] = bytes(min(n, len(buf) - 2 - at))
            mark(at, at + n)
            done.append(f"zeros {n // 1024} KB")
        elif k == "missing":  # a fragment the recovery lost: everything after it moves up
            n = _aligned(rng, 4096, 16384)
            at -= at % SECTOR
            mark(at, at + n)
            del buf[at : at + n]
            deleted.append((at, n))
            done.append(f"missing {n // 1024} KB")
        elif k == "truncate":
            cut = int(rng.uniform(0.55, 0.9) * (len(buf) - s)) + s
            mark(cut, len(buf))
            del buf[cut:]
            done.append(f"truncated at {100 * (cut - s) // (len(data) - s)} %")
    return bytes(buf), done


DENSE_SHARE = (0.15, 0.40)  # share of the scan's sectors rotted, as measured on photos from a dying card
DENSE_STAY = 0.5  # a rotted sector is followed by another one this often (runs of ~2 sectors)
DENSE_BIT = 0.08  # bits flipped in a rotted sector (half of its FF00 pairs break, as on the real photos)


def dense_rot(data: bytes, seed: int, spans: list | None = None, head: bool = False) -> tuple[bytes, list[str]]:
    """Rot spread over the whole scan the way a dying memory card leaves it: about a
    quarter of the 512-byte sectors, in runs of one to a few, every bit of them wrong
    with a probability of 8 % (measured on the 2005 photos: 26 % of the sectors show
    broken FF00 pairs, half of the pairs in them broken, runs of 2 rotted and 2 good
    sectors at the median)."""
    rng = np.random.default_rng(seed)
    s = scan_start(data)
    share = float(rng.uniform(*DENSE_SHARE))
    p_in = share * (1 - DENSE_STAY) / (1 - share)  # good -> rotted, for that share in the long run
    buf = bytearray(data)
    first = 0 if head else (s // SECTOR + 1) * SECTOR
    bad = False
    n_bad = 0
    for a in range(first, len(buf) - 2, SECTOR):
        bad = rng.random() < (DENSE_STAY if bad else p_in)
        if not bad:
            continue
        n_bad += 1
        b = min(a + SECTOR, len(buf) - 2)
        flips = rng.random((b - a, 8)) < DENSE_BIT
        mask = np.packbits(flips, axis=1).ravel()
        buf[a:b] = (np.frombuffer(bytes(buf[a:b]), np.uint8) ^ mask).tobytes()
        if spans is not None:
            spans.append((a, b))
    total = max(1, (len(buf) - 2 - first) // SECTOR)
    return bytes(buf), [f"dense rot: {n_bad} of {total} sectors ({100 * n_bad // total} %)"]


def exif_with_thumbnail(img: Image.Image) -> bytes:
    """A camera-like EXIF block (TIFF, IFD1) carrying a 160x120 JPEG thumbnail, the way
    cameras of the time wrote it - photos that went through other software often lost it."""
    import struct

    t = img.copy()
    t.thumbnail((160, 120), Image.LANCZOS)
    tb = io.BytesIO()
    t.save(tb, "JPEG", quality=85)
    thumb = tb.getvalue()
    ifd0 = struct.pack("<H", 0) + struct.pack("<I", 8 + 6)  # no entries, next IFD right after
    ifd1_at = 8 + len(ifd0)
    data_at = ifd1_at + 2 + 2 * 12 + 4
    ifd1 = struct.pack("<H", 2)
    ifd1 += struct.pack("<HHII", 0x0201, 4, 1, data_at)  # JPEGInterchangeFormat
    ifd1 += struct.pack("<HHII", 0x0202, 4, 1, len(thumb))  # JPEGInterchangeFormatLength
    ifd1 += struct.pack("<I", 0)
    return b"Exif\x00\x00" + b"II*\x00" + struct.pack("<I", 8) + ifd0 + ifd1 + thumb


def healthy(data: bytes, name: str) -> bool:
    from fotomedik.scan.scanner import scan_bytes

    d = scan_bytes(data, name)
    return not d.findings and d.recovered_fraction >= 0.999
