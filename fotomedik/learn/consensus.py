"""Header by vote: rebuild a camera's table block from many damaged copies.

All photos of one camera carry the same table block (DQT, DHT, SOF, SOS - up to the
image size inside SOF and, on some cameras, a second set of tables for portrait
shots). Rot hits every copy in different places, so a bit-by-bit majority over the
copies is clean even when no single photo has an intact header.

A copy is found in a damaged file next to the main image's SOS (the last close
match in the header region - the thumbnail's SOS comes earlier) and assigned to the
variant (seed) it differs least from; only copies within MAX_DIFF of a seed vote.
"""

from __future__ import annotations

import numpy as np

from ..jpeg import markers as M
from .profile import CameraProfile

HEAD = 128 * 1024  # header region searched in every file
CLOSE_SOS = 0.15
MAX_DIFF = 0.2  # a copy may differ from its seed in at most this share of bits
MIN_VOTES = 3

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)

TABLE_MARKERS = (M.DQT, M.DHT, M.DRI) + tuple(m for m in range(0xC0, 0xD0) if m not in (0xC4, 0xC8, 0xCC))


def table_block(data: bytes) -> bytes | None:
    """The bytes from the first table segment to the end of SOS, as the file has them."""
    res = M.parse_segments(data)
    if not res.ok:
        return None
    first = [s.offset for s in res.segments if s.marker in TABLE_MARKERS]
    if not first:
        return None
    return bytes(data[min(first) : res.scan_start])


def locate(head: bytes, block: bytes) -> tuple[int, float] | None:
    """(start, share of differing bits) of the best copy of `block` in head, or None."""
    sos = block[-14:] if len(block) >= 14 else block
    arr = np.frombuffer(head, dtype=np.uint8)
    L, S = len(block), len(sos)
    if len(arr) < L:
        return None
    dist = np.zeros(len(arr) - S + 1, dtype=np.int32)
    for j in range(S):
        dist += _POP[np.bitwise_xor(arr[j : len(arr) - S + 1 + j], sos[j])]
    ends = np.flatnonzero(dist <= CLOSE_SOS * 8 * S)[-3:] + S
    best = None
    ref = np.frombuffer(block, dtype=np.uint8)
    for e in ends:
        s = int(e) - L
        if s < 0:
            continue
        d = int(_POP[np.bitwise_xor(arr[s : s + L], ref)].sum()) / (8 * L)
        if best is None or d < best[1]:
            best = (s, d)
    return best


def vote(blocks: list[bytes]) -> bytes:
    """Bit-by-bit majority; ties go to the first copy."""
    bits = np.stack([np.unpackbits(np.frombuffer(b, dtype=np.uint8)) for b in blocks]).astype(np.int32)
    total = bits.sum(axis=0)
    n = len(blocks)
    out = np.where(2 * total > n, 1, np.where(2 * total < n, 0, bits[0]))
    return np.packbits(out.astype(np.uint8)).tobytes()


# SOS segments cameras write (3 components 4:2:0 / 4:2:2 with tables 0/1/1, grayscale)
GENERIC_SOS = (bytes.fromhex("ffda000c03010002110311003f00"), bytes.fromhex("ffda0008010100003f00"))
WINDOW = 2048  # bytes before the SOS that are voted on when no header parses at all


def _distance(a: bytes, b: bytes) -> int:
    return int(_POP[np.bitwise_xor(np.frombuffer(a, dtype=np.uint8), np.frombuffer(b, dtype=np.uint8))].sum())


def _block_in(window: bytes) -> bytes | None:
    """The table block at the end of a voted window: the earliest table marker from which
    the segment lengths chain exactly to the end (which is the SOS)."""
    n = len(window)
    for p in range(n - 4):
        if window[p] != 0xFF or window[p + 1] not in TABLE_MARKERS:
            continue
        q = p
        while q + 4 <= n and window[q] == 0xFF and window[q + 1] in TABLE_MARKERS + (M.SOS,):
            q += 2 + ((window[q + 2] << 8) | window[q + 3])
            if q == n:
                return bytes(window[p:])
    return None


def bootstrap(heads: list[bytes]) -> list[tuple[CameraProfile, bytes]]:
    """Seeds when no header of the set parses: align every file at its main SOS, vote
    the bytes before it (in up to two groups - some cameras use other tables for
    portrait shots) and read the table block out of the result."""
    ends = []
    for head in heads:
        head = head[:HEAD]
        for sos in GENERIC_SOS:
            got = locate(head, sos)
            if got is not None and got[1] <= CLOSE_SOS:
                ends.append((head, got[0] + len(sos)))
                break
    if len(ends) < MIN_VOTES:
        return []
    size = min(WINDOW, min(e for _h, e in ends))  # same length for all, aligned at the SOS end
    windows = [bytes(h[e - size : e]) for h, e in ends]
    # two-means on bit distance, medoid start
    d0 = [sum(_distance(w, v) for v in windows) for w in windows]
    c1 = windows[int(np.argmin(d0))]
    c2 = max(windows, key=lambda w: _distance(w, c1))
    groups = [windows]
    for _ in range(3):
        g1 = [w for w in windows if _distance(w, c1) <= _distance(w, c2)]
        g2 = [w for w in windows if _distance(w, c1) > _distance(w, c2)]
        groups = [g for g in (g1, g2) if len(g) >= MIN_VOTES]
        if not groups:
            groups = [windows]
            break
        c1 = vote(groups[0])
        c2 = vote(groups[1]) if len(groups) > 1 else c1
    seeds = []
    for g in groups:
        block = _block_in(vote(g))
        if block is None:
            continue
        try:
            prof = CameraProfile.from_jpeg(b"\xff\xd8" + block + b"\xff\xd9", source=f"vote of {len(g)} damaged headers")
        except Exception:
            continue
        prof.votes = len(g)
        # The image size is the one thing in the block that differs from photo to photo
        # (landscape/portrait): voting bits of 320 and 240 gives neither. Sizes read
        # exactly alike in several photos are real; rotted ones are not repeated.
        dims = _sizes(block, g)
        if dims:
            prof.width, prof.height = dims[0]
            prof.alt_dims = [list(d) for d in dims[1:]]
        seeds.append((prof, block))
    return seeds


def _sizes(block: bytes, windows: list[bytes]) -> list[tuple[int, int]]:
    """Image sizes carried at the SOF position of the block, most common first. Each copy
    has rot of its own, so the 4 size bytes are clustered (landscape / portrait differ in
    many bits, rot in few) and voted within each cluster."""
    res = M.parse_segments(b"\xff\xd8" + block)
    if res.sof is None:
        return []
    at = res.sof.offset - 2 + 5 - len(block)  # height, width follow length and precision
    raw = []
    for w in windows:
        pos = len(w) + at
        if 0 <= pos and pos + 4 <= len(w):
            raw.append(bytes(w[pos : pos + 4]))
    if not raw:
        return []
    c1 = min(raw, key=lambda r: sum(_distance(r, o) for o in raw))
    c2 = max(raw, key=lambda r: _distance(r, c1))
    groups = [raw]
    if _distance(c1, c2) > 4:  # two sizes in the set
        for _ in range(3):
            g1 = [r for r in raw if _distance(r, c1) <= _distance(r, c2)]
            g2 = [r for r in raw if _distance(r, c1) > _distance(r, c2)]
            groups = [g for g in (g1, g2) if g]
            c1 = vote(g1) if g1 else c1
            c2 = vote(g2) if g2 else c2
    out = []
    for g in sorted(groups, key=len, reverse=True):
        v = vote(g)
        h, wd = (v[0] << 8) | v[1], (v[2] << 8) | v[3]
        if 0 < wd <= 20000 and 0 < h <= 20000 and (wd, h) not in out:
            out.append((wd, h))
    return out


def consensus(seeds: list[tuple[CameraProfile, bytes]], heads: list[bytes]) -> list[CameraProfile]:
    """One profile per seed variant, its tables voted from every copy found in heads.
    Seeds: (profile, table block) of headers that parse, one per variant; without any,
    the seeds come from the damaged headers themselves (bootstrap)."""
    if not seeds:
        seeds = bootstrap(heads)
        if not seeds:
            return []
    found: list[list[bytes]] = [[] for _ in seeds]
    for head in heads:
        best = None
        for i, (_p, block) in enumerate(seeds):
            got = locate(head[:HEAD], block)
            if got is not None and got[1] <= MAX_DIFF and (best is None or got[1] < best[1]):
                best = (i, got[1], got[0])
        if best is not None:
            i, _d, s = best
            found[i].append(bytes(head[s : s + len(seeds[i][1])]))
    out = []
    for (prof, block), copies in zip(seeds, found):
        if len(copies) < MIN_VOTES:
            out.append(prof)
            continue
        voted = vote([block] + copies)
        try:
            p = CameraProfile.from_jpeg(b"\xff\xd8" + voted + b"\xff\xd9", source=f"vote of {len(copies)} headers")
        except Exception:
            out.append(prof)
            continue
        # tables from the vote; who made it and the image size(s) from the seed
        p.make, p.model, p.jfif = prof.make, prof.model, prof.jfif
        p.width, p.height, p.alt_dims = prof.width, prof.height, list(prof.alt_dims)
        p.votes = max(len(copies), prof.votes)
        out.append(p)
    return out
