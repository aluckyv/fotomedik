"""Tell JPEG entropy data from foreign content by the 0xFF rule.

In valid entropy-coded data every 0xFF is followed by 0x00 (stuffing), a
restart marker (D0-D7), EOI (D9) or another 0xFF (fill). Random data breaks
that rule ~96% of the time. Even heavily bit-rotted JPEG data keeps most of
its FF pairs valid, so windowed 'good vs bad FF' counts separate the two.
"""

from __future__ import annotations

from dataclasses import dataclass

SECTOR = 512
WINDOW = 8  # sectors (4 KB) — enough FFs in random data to vote reliably


def _ff_counts(chunk: bytes) -> tuple[int, int]:
    """(good, bad) FF pairs in chunk; a trailing lone FF counts as neither."""
    good = bad = 0
    p = chunk.find(b"\xff")
    n = len(chunk)
    while p >= 0:
        if p + 1 < n:
            nxt = chunk[p + 1]
            if nxt == 0x00 or 0xD0 <= nxt <= 0xD9 or nxt == 0xFF:
                good += 1
            else:
                bad += 1
        p = chunk.find(b"\xff", p + 1)
    return good, bad


def _sector_table(data: bytes, start: int, sector: int) -> tuple[list[int], list[int], list[int], list[bool]]:
    """Per-sector (offset, good, bad, mostly_zero) from `start` to end of data."""
    offs: list[int] = []
    goods: list[int] = []
    bads: list[int] = []
    zeros: list[bool] = []
    n = len(data)
    p = start
    while p < n:
        chunk = data[p : min(p + sector, n)]
        g, b = _ff_counts(chunk)
        offs.append(p)
        goods.append(g)
        bads.append(b)
        zeros.append(chunk.count(0) > 0.9 * len(chunk))
        p += sector
    return offs, goods, bads, zeros


def _jpeg_window(goods: list[int], bads: list[int], zeros: list[bool], i: int, window: int) -> bool:
    hi = min(len(goods), i + window)
    g = sum(goods[i:hi])
    b = sum(bads[i:hi])
    z = sum(zeros[i:hi])
    if z > (hi - i) // 2:
        return False
    if g == 0 and b == 0:
        return True  # no FF bytes at all: flat/synthetic data, not random garbage
    return g > b


@dataclass
class ClusterStat:
    offset: int
    length: int
    ff_good: int
    ff_bad: int
    zero_ratio: float
    kind: str  # jpeg | garbage | zeros

    @property
    def ff_total(self) -> int:
        return self.ff_good + self.ff_bad

    @property
    def bad_ff_ratio(self) -> float:
        return self.ff_bad / self.ff_total if self.ff_total else 0.0


def cluster_stats(data: bytes, scan_start: int, cluster: int = 32768) -> list[ClusterStat]:
    """Per-cluster classification (the header part of the first cluster is ignored)."""
    out: list[ClusterStat] = []
    n = len(data)
    for off in range(0, n, cluster):
        lo = max(off, scan_start)
        hi = min(off + cluster, n)
        if hi <= lo:
            continue
        chunk = data[lo:hi]
        good, bad = _ff_counts(chunk)
        zero_ratio = chunk.count(0) / len(chunk)
        if zero_ratio > 0.9:
            kind = "zeros"
        elif good + bad >= 4 and bad > good:
            kind = "garbage"
        else:
            kind = "jpeg"
        out.append(ClusterStat(off, hi - off, good, bad, zero_ratio, kind))
    return out


def garbage_tail_start(
    data: bytes, scan_start: int, sector: int = SECTOR, window: int = WINDOW, min_sectors: int = 4
) -> int | None:
    """Offset where the trailing run of non-JPEG data begins (sector precision), or None.

    The run must span at least `min_sectors` so a damaged last sector of a real
    image is not mistaken for foreign data.
    """
    offs, goods, bads, zeros = _sector_table(data, scan_start, sector)
    if not offs:
        return None
    n = len(offs)
    votes = [1 if _jpeg_window(goods, bads, zeros, i, window) else 0 for i in range(n)]
    # Earliest sector after which almost nothing votes JPEG any more (suffix rule),
    # snapped to a transition from a JPEG-voting sector.
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + votes[i]
    start_idx = None
    for i in range(n):
        remaining = n - i
        if remaining < min_sectors:
            break
        if votes[i] == 0 and suffix[i] <= 0.1 * remaining and (i == 0 or votes[i - 1] == 1):
            start_idx = i
            break
    if start_idx is None:
        return None
    if start_idx == 0:
        return scan_start
    # extend through individually clean sectors right after the transition
    j = start_idx - 1
    while j + 1 < n and bads[j + 1] == 0 and not zeros[j + 1] and goods[j + 1] >= 1:
        j += 1
    if n - 1 - j < min_sectors:
        return None
    return min(offs[j] + sector, len(data))


def garbage_ranges(stats: list[ClusterStat]) -> list[tuple[int, int]]:
    """Merged (start, end) byte ranges of non-JPEG clusters."""
    ranges: list[tuple[int, int]] = []
    for s in stats:
        if s.kind == "jpeg":
            continue
        if ranges and ranges[-1][1] == s.offset:
            ranges[-1] = (ranges[-1][0], s.offset + s.length)
        else:
            ranges.append((s.offset, s.offset + s.length))
    return ranges


def last_broken_stuffing(data: bytes, lo: int, hi: int) -> int | None:
    """Offset of the last FF in [lo, hi) not followed by 00, a marker byte or FF.

    In entropy-coded data every FF is followed by 00 (stuffing) or a marker, so a
    broken pair is a certain point of damage: intact data can only resume after it.
    """
    p = hi - 1
    while True:
        p = data.rfind(b"\xff", lo, p)
        if p < 0:
            return None
        n = data[p + 1]
        if not (n == 0x00 or 0xD0 <= n <= 0xD9 or n == 0xFF):
            return p
