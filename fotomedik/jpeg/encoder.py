"""Minimal baseline JPEG entropy *encoder* — only what hole filling needs.

We never re-encode real image data. We only synthesise DC-only blocks (flat
colour, EOB right after DC) with the file's own Huffman tables so that a
damaged restart interval can be replaced by bytes every decoder accepts.
"""

from __future__ import annotations

from .tables import FrameTables, HuffmanTable


class BitWriter:
    def __init__(self):
        self._out = bytearray()
        self._acc = 0
        self._n = 0

    def put_bits(self, value: int, n: int) -> None:
        if n == 0:
            return
        self._acc = (self._acc << n) | (value & ((1 << n) - 1))
        self._n += n
        while self._n >= 8:
            self._n -= 8
            b = (self._acc >> self._n) & 0xFF
            self._out.append(b)
            if b == 0xFF:
                self._out.append(0x00)
        self._acc &= (1 << self._n) - 1 if self._n else 0

    def finish(self) -> bytes:
        """Pad the last byte with 1-bits (as the standard prescribes) and return everything."""
        if self._n:
            pad = 8 - self._n
            self.put_bits((1 << pad) - 1, pad)
        return bytes(self._out)


def codes_for(t: HuffmanTable) -> dict[int, tuple[int, int]]:
    """symbol -> (length, code)"""
    return {sym: (ln, code) for (ln, code), sym in t.lookup.items()}


def _category(v: int) -> int:
    return abs(v).bit_length()


def _magnitude_bits(v: int, s: int) -> int:
    return v if v >= 0 else (v + (1 << s) - 1)


def _best_category(codes: dict[int, tuple[int, int]], want: int) -> int:
    """Largest encodable DC category <= want (tables may omit rare categories)."""
    while want > 0 and want not in codes:
        want -= 1
    if want == 0 and 0 not in codes:
        raise ValueError("DC table has no category-0 code")
    return want


def encode_flat_mcus(
    ft: FrameTables,
    mcu_count: int,
    targets: dict[int, int],
    preds: dict[int, int],
    per_block: dict[int, list[int]] | None = None,
    writer: "BitWriter | None" = None,
) -> bytes:
    """Entropy data for `mcu_count` MCUs of DC-only blocks.

    targets: component id -> DC coefficient (quantised units) for every block,
    unless per_block[cid] gives one value per block (consumed in scan order).
    preds: DC predictor state at the start (mutated to the state at the end),
    so the bytes can be glued after existing data without a restart marker.
    With `writer`, bits are appended to it unpadded (more data may follow) and b"" is returned.
    """
    frame, scan = ft.frame, ft.scan
    interleaved = len(scan.components) > 1
    w = writer if writer is not None else BitWriter()
    comps = []
    for sc in scan.components:
        fc = frame.component(sc.id)
        dc_codes = codes_for(ft.huff[(0, sc.td)])
        ac_codes = codes_for(ft.huff[(1, sc.ta)])
        if 0x00 not in ac_codes:
            raise ValueError("AC table has no EOB code")
        h, v = (fc.h, fc.v) if interleaved else (1, 1)
        comps.append((sc.id, h, v, dc_codes, ac_codes[0x00]))
    cursors = {cid: 0 for cid, *_ in comps}
    for _ in range(mcu_count):
        for cid, h, v, dc_codes, eob in comps:
            for _b in range(h * v):
                if per_block and cid in per_block:
                    seq = per_block[cid]
                    tgt = seq[min(cursors[cid], len(seq) - 1)]
                    cursors[cid] += 1
                else:
                    tgt = targets.get(cid, 0)
                diff = tgt - preds[cid]
                s = _best_category(dc_codes, _category(diff))
                if s < _category(diff):
                    # clamp the step to what the table can express
                    lim = (1 << s) - 1
                    diff = max(-lim, min(lim, diff))
                ln, code = dc_codes[s]
                w.put_bits(code, ln)
                if s:
                    w.put_bits(_magnitude_bits(diff, s), s)
                preds[cid] += diff
                ln, code = eob
                w.put_bits(code, ln)
    return b"" if writer is not None else w.finish()


def gray_dc(ft: FrameTables, cid: int, sample: int) -> int:
    """DC coefficient (quantised) that renders a flat block at `sample` (0..255)."""
    fc = ft.frame.component(cid)
    q = ft.quant[fc.tq].dc
    return round((sample - 128) * 8 / q)
