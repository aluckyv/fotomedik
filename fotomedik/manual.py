"""Manual repair: edit the entropy-coded data of one photo by hand, with an instant
preview, for photos where the automatic repair got something wrong.

Like the "patch" tools of professional JPEG repair software, but at the level of
MCUs (16x16 or 8x8 blocks of the picture):

* colour / brightness from here on - the DC difference of the first block of a
  component at the chosen MCU is changed, so this MCU and every one after it (up to
  the next restart marker) gets brighter/darker or changes colour;
* delete or insert MCUs - moves everything after the chosen place left or right
  (to undo a slip that shifted the picture); deleting keeps the colours after it;
* delete bits - removes junk bits at the chosen place (or where decoding stopped).

The scan is held as one bit array per restart interval with the FF00 stuffing
removed; edits happen on those arrays, the file is put together again for every
preview. Every edit can be undone.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFile

from .jpeg import decoder as D
from .jpeg import markers as M
from .jpeg.encoder import codes_for
from .jpeg.tables import FrameTables


@dataclass
class Segment:
    bits: np.ndarray  # uint8 0/1, unstuffed entropy-coded bits (no padding guarantee)
    marker: bytes  # what followed it in the file: an RST marker, or b"" for the last one


def _unstuff(data: bytes, start: int, end: int) -> tuple[bytes, list[int]]:
    """Entropy bytes of data[start:end] without FF00 stuffing, plus each byte's file offset."""
    out = bytearray()
    offs: list[int] = []
    p = start
    while p < end:
        b = data[p]
        out.append(b)
        offs.append(p)
        p += 2 if (b == 0xFF and p + 1 < end and data[p + 1] == 0x00) else 1
    return bytes(out), offs


def _stuff(bits: np.ndarray) -> bytes:
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.ones(pad, dtype=np.uint8)])
    raw = np.packbits(bits).tobytes()
    return raw.replace(b"\xff", b"\xff\x00")


class Editor:
    def __init__(self, data: bytes):
        self.history: list[tuple[bytes, list[Segment]]] = []
        self._load(data)

    # ------------------------------------------------------------------ model
    def _load(self, data: bytes) -> None:
        parse = M.parse_segments(data)
        if parse.scan_start is None:
            raise ValueError("no image data found (repair the header first)")
        self.ft = FrameTables.from_parse(parse)
        if self.ft.frame.progressive:
            raise ValueError("progressive JPEG: manual repair works on baseline photos")
        self.header = data[: parse.scan_start]
        end = parse.eoi_offset if parse.eoi_offset is not None else len(data)
        self.segments: list[Segment] = []
        self._offsets: list[list[int]] = []
        pos = parse.scan_start
        while True:
            hit = None
            p = pos
            while p + 1 < end:
                if data[p] == 0xFF and 0xD0 <= data[p + 1] <= 0xD7:
                    hit = p
                    break
                p += 1
            stop = hit if hit is not None else end
            raw, offs = _unstuff(data, pos, stop)
            self.segments.append(Segment(np.unpackbits(np.frombuffer(raw, dtype=np.uint8)),
                                         data[hit : hit + 2] if hit is not None else b""))
            self._offsets.append(offs)
            if hit is None:
                break
            pos = hit + 2
        comps, self.per_row, self.rows = D._scan_components(self.ft)
        self.comps = comps
        interleaved = len(comps) > 1
        fr = self.ft.frame
        self.mcu_w = 8 * (fr.hmax if interleaved else 1)
        self.mcu_h = 8 * (fr.vmax if interleaved else 1)
        self._dc = {}
        self._ac = {}
        for sc in self.ft.scan.components:
            self._dc[sc.id] = self.ft.huff[(0, sc.td)]
            self._ac[sc.id] = self.ft.huff[(1, sc.ta)]
        self._map = None

    def data(self) -> bytes:
        body = b"".join(_stuff(s.bits) + s.marker for s in self.segments)
        return self.header + body + b"\xff\xd9"

    def image(self, max_side: int | None = None) -> Image.Image:
        """The photo as a normal viewer shows it now."""
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            im = Image.open(io.BytesIO(self.data()))
            if max_side:
                im.draft("RGB", (max_side, max_side))
            return im.convert("RGB")
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = False

    # ------------------------------------------------------------------ where things are
    def mcu_at(self, x: float, y: float) -> int:
        """MCU index under a pixel of the full-size picture."""
        col = min(self.per_row - 1, max(0, int(x // self.mcu_w)))
        row = min(self.rows - 1, max(0, int(y // self.mcu_h)))
        return row * self.per_row + col

    def _positions(self) -> dict[int, tuple[int, int]]:
        """MCU index -> (segment, bit index in it) for every MCU that decodes; plus,
        per segment, where decoding stopped (key -1 - segment). Uses the fast decoder."""
        if self._map is not None:
            return self._map
        from .jpeg import kernel as K

        pk = K.packed_for(self.ft)
        out: dict[int, tuple[int, int]] = {}
        dri = self.ft.restart_interval
        total = self.per_row * self.rows
        first = 0
        for si, seg in enumerate(self.segments):
            count = min(dri if dri else total, total - first)
            if count <= 0:
                break
            stuffed = _stuff(seg.bits)
            blob = stuffed + b"\xff\xd9"
            # stuffed byte offset of every unstuffed byte (FF is followed by a stuffed 00)
            raw = np.packbits(np.concatenate([seg.bits, np.ones((-len(seg.bits)) % 8, np.uint8)]))
            step = np.where(raw == 0xFF, 2, 1)
            offs = np.concatenate([[0], np.cumsum(step)[:-1]])

            def unstuffed(byte, bit):
                i = int(np.searchsorted(offs, byte))
                return i * 8 + bit

            preds = np.zeros(pk.ncomp, dtype=np.int32)
            got = K.decode_mcus(blob, 0, 0, len(blob), pk, preds, count, check_dc=False)
            n, err, bits = got[0], got[1], got[7]
            for k in range(n):
                b = int(bits[k])
                out[first + k] = (si, unstuffed(b // 8, b % 8))
            if err is None or n == count:
                end = unstuffed(got[5], got[6])
            else:  # the clean end of the last MCU that decoded
                again = K.decode_mcus(blob, 0, 0, len(blob), pk, np.zeros(pk.ncomp, dtype=np.int32), n,
                                      check_dc=False) if n else None
                end = unstuffed(again[5], again[6]) if again else 0
            out[-1 - si] = (si, min(end, len(seg.bits)))
            first += count
        self._map = out
        return out

    def _code(self, bits: np.ndarray, pos: int, table) -> tuple[int, int]:
        code = 0
        for ln in range(1, 17):
            code = (code << 1) | int(bits[pos + ln - 1])
            sym = table.lookup.get((ln, code))
            if sym is not None:
                return sym, pos + ln
        raise ValueError("bad Huffman code")

    def _walk(self, bits: np.ndarray, pos: int, preds: dict, blocks: list | None = None) -> tuple[int, dict]:
        """Decode one MCU from bit `pos`; returns the end position. With `blocks`, appends
        (cid, dc_start, dc_end, diff) of every block."""
        for c in self.comps:
            for _ in range(c.h * c.v):
                start = pos
                s, pos = self._code(bits, pos, self._dc[c.cid])
                if s > 11:
                    raise ValueError("bad DC category")
                v = 0
                for i in range(s):
                    v = (v << 1) | int(bits[pos + i])
                pos += s
                if s and v < (1 << (s - 1)):
                    v -= (1 << s) - 1
                preds[c.cid] += v
                if blocks is not None:
                    blocks.append((c.cid, start, pos, v))
                k = 1
                while k < 64:
                    rs, pos = self._code(bits, pos, self._ac[c.cid])
                    r, sz = rs >> 4, rs & 15
                    if sz == 0:
                        if r == 15:
                            k += 16
                            continue
                        break
                    k += r + 1
                    pos += sz
                if k > 64:
                    raise ValueError("coefficient overflow")
        return pos, preds

    def where(self, mcu: int) -> tuple[int, int, bool]:
        """(segment, bit, exact): the bit where `mcu` starts, or where decoding of its
        segment stopped (exact False) when it does not decode."""
        pos = self._positions()
        if mcu in pos:
            si, b = pos[mcu]
            return si, b, True
        dri = self.ft.restart_interval
        si = min(len(self.segments) - 1, mcu // dri) if dri else 0
        return si, pos.get(-1 - si, (si, 0))[1], False

    # ------------------------------------------------------------------ edits
    def _change(self) -> None:
        self.history.append((self.header, [Segment(s.bits.copy(), s.marker) for s in self.segments]))
        self._map = None

    def undo(self) -> bool:
        if not self.history:
            return False
        self.header, self.segments = self.history.pop()
        self._map = None
        return True

    def delete_bits(self, mcu: int, n: int) -> None:
        si, b, _exact = self.where(mcu)
        self._change()
        seg = self.segments[si]
        seg.bits = np.delete(seg.bits, np.s_[b : b + n])

    def _flat_mcu_bits(self) -> np.ndarray:
        out: list[int] = []
        for c in self.comps:
            dc = codes_for(self._dc[c.cid])
            ac = codes_for(self._ac[c.cid])
            ln, code = dc[0]
            eln, ecode = ac[0x00]
            for _ in range(c.h * c.v):
                out += [(code >> (ln - 1 - i)) & 1 for i in range(ln)]
                out += [(ecode >> (eln - 1 - i)) & 1 for i in range(eln)]
        return np.array(out, dtype=np.uint8)

    def insert_mcus(self, mcu: int, k: int = 1) -> None:
        """k flat MCUs (no colour change) before `mcu`: everything after moves right."""
        si, b, _ = self.where(mcu)
        self._change()
        seg = self.segments[si]
        seg.bits = np.concatenate([seg.bits[:b], np.tile(self._flat_mcu_bits(), k), seg.bits[b:]])

    def delete_mcus(self, mcu: int, k: int = 1, keep_colour: bool = True) -> None:
        """Remove k MCUs at `mcu`: everything after moves left. With keep_colour the DC
        steps of the removed MCUs are carried over, so the colours after stay."""
        si, b, exact = self.where(mcu)
        if not exact:
            raise ValueError("this place does not decode; delete bits there instead")
        seg = self.segments[si]
        preds = {c.cid: 0 for c in self.comps}
        pos = b
        for _ in range(k):
            pos, preds = self._walk(seg.bits, pos, preds)
        self._change()
        seg.bits = np.delete(seg.bits, np.s_[b:pos])
        if keep_colour and any(preds.values()):
            for i, c in enumerate(self.comps):
                if preds[c.cid]:
                    self._shift(si, b, i, preds[c.cid])

    def shift_colour(self, mcu: int, dy: int = 0, dcb: int = 0, dcr: int = 0) -> None:
        """Brightness (dy) and colour (dcb: blue-yellow, dcr: red-green) of `mcu` and
        everything after it in its restart interval, in gray levels (0..255 scale)."""
        si, b, exact = self.where(mcu)
        if not exact:
            raise ValueError("this place does not decode")
        steps = []
        for i, d in enumerate((dy, dcb, dcr)[: len(self.comps)]):
            q = self.ft.quant[self.ft.frame.component(self.comps[i].cid).tq].dc
            steps.append(int(round(d * 8 / q)))
        if not any(steps):
            return
        self._change()
        for i, d in enumerate(steps):
            if d:
                self._shift(si, b, i, d)

    def _shift(self, si: int, b: int, comp_index: int, delta: int) -> None:
        """Change the DC difference of component comp_index's first block at bit b by delta."""
        seg = self.segments[si]
        blocks: list = []
        self._walk(seg.bits, b, {c.cid: 0 for c in self.comps}, blocks)
        cid = self.comps[comp_index].cid
        _cid, start, end, diff = next(x for x in blocks if x[0] == cid)
        new = diff + delta
        s = abs(new).bit_length()
        codes = codes_for(self._dc[cid])
        while s not in codes and s > 0:  # the table lacks this size: the biggest step it has
            s -= 1
            lim = (1 << s) - 1
            new = max(-lim, min(lim, new))
        ln, code = codes[s]
        mag = new if new >= 0 else new + (1 << s) - 1
        rep = [(code >> (ln - 1 - i)) & 1 for i in range(ln)] + [(mag >> (s - 1 - i)) & 1 for i in range(s)]
        seg.bits = np.concatenate([seg.bits[:start], np.array(rep, dtype=np.uint8), seg.bits[end:]])
        self._map = None
