"""Bit-level reader for JPEG entropy-coded data.

Handles byte stuffing (FF 00 -> FF), stops at markers (FF xx, xx != 0) and at
the end bound, and knows the exact stream position of the next unread bit so
the decoder can record where every MCU starts and where decoding broke.
"""

from __future__ import annotations


class MarkerHit(Exception):
    """Raised when real bits are needed but a marker sits at the read position."""

    def __init__(self, marker: int, offset: int):
        super().__init__(f"marker 0xFF{marker:02X} at {offset}")
        self.marker = marker
        self.offset = offset


class BitReader:
    __slots__ = ("data", "pos", "end", "buf", "nbits", "real", "_hit", "_eof", "_bytepos")

    def __init__(self, data: bytes, start: int, end: int | None = None):
        self.data = data
        self.pos = start  # next byte to load
        self.end = len(data) if end is None else end
        self.buf = 0  # bit buffer, valid bits are the low `nbits`
        self.nbits = 0  # bits in buffer (real + padding)
        self.real = 0  # how many of those are real data bits (rest is padding after marker/EOF)
        self._hit: tuple[int, int] | None = None
        self._eof = False
        self._bytepos: list[int] = []  # stream offsets of the last loaded bytes (newest last)

    # ------------------------------------------------------------------ filling

    def _fill(self, need: int) -> None:
        """Ensure at least `need` bits in buffer, padding with zeros past a marker/EOF."""
        data = self.data
        while self.nbits < need:
            if self._hit is not None or self._eof:
                self.buf = (self.buf << 8) & 0xFFFFFFFFFFFF
                self.nbits += 8
                continue
            p = self.pos
            if p >= self.end:
                self._eof = True
                continue
            b = data[p]
            if b == 0xFF:
                q = p + 1
                while q < self.end and data[q] == 0xFF:
                    q += 1
                if q >= self.end:
                    self._eof = True
                    continue
                nxt = data[q]
                if nxt == 0x00 and q == p + 1:
                    # stuffed FF
                    self.pos = p + 2
                elif nxt == 0x00:
                    # FF FF ... 00: treat trailing FF 00 as stuffed, earlier FFs as fill
                    self.pos = q + 1
                else:
                    self._hit = (nxt, p)
                    continue
            else:
                self.pos = p + 1
            self.buf = ((self.buf << 8) | b) & 0xFFFFFFFFFFFF
            self.nbits += 8
            self.real += 8
            bp = self._bytepos
            bp.append(p)
            if len(bp) > 6:
                del bp[0]

    def _raise_end(self) -> None:
        if self._hit is not None:
            raise MarkerHit(self._hit[0], self._hit[1])
        raise EOFError("entropy data ended")

    # ------------------------------------------------------------------ reading

    def read_bit(self) -> int:
        if self.nbits < 1:
            self._fill(1)
        if self.real < 1:
            self._raise_end()
        self.nbits -= 1
        self.real -= 1
        return (self.buf >> self.nbits) & 1

    def read_bits(self, n: int) -> int:
        if n == 0:
            return 0
        if self.nbits < n:
            self._fill(n)
        if self.real < n:
            self._raise_end()
        self.nbits -= n
        self.real -= n
        return (self.buf >> self.nbits) & ((1 << n) - 1)

    def peek_bits(self, n: int) -> int:
        """Look at the next n bits (zero-padded past a marker/EOF); does not advance."""
        if self.nbits < n:
            self._fill(n)
        return (self.buf >> (self.nbits - n)) & ((1 << n) - 1)

    def skip_bits(self, n: int) -> None:
        if self.nbits < n:
            self._fill(n)
        if self.real < n:
            self._raise_end()
        self.nbits -= n
        self.real -= n

    def receive_extend(self, s: int) -> int:
        """Read an s-bit magnitude and sign-extend per JPEG F.2.2.1."""
        if s == 0:
            return 0
        v = self.read_bits(s)
        if v < (1 << (s - 1)):
            v -= (1 << s) - 1
        return v

    # ------------------------------------------------------------------ positions

    @property
    def byte_pos(self) -> int:
        """Stream offset of the next byte not yet loaded (stuffing included)."""
        return self.pos

    @property
    def bit_pos(self) -> int:
        """Approximate absolute bit position of the next unread bit."""
        off, bit = self.tell()
        return off * 8 + bit

    def tell(self) -> tuple[int, int]:
        """Exact (byte_offset, bit_in_byte) of the next unread real bit.

        If the buffer holds only padding, returns the position of the marker/end.
        """
        real = self.real
        if real <= 0:
            return (self.pos if self._hit is None else self._hit[1], 0)
        # real bits are the oldest of the buffered real bytes; padding is newest.
        # number of real bytes still partially unread:
        k = -(-real // 8)  # ceil
        bp = self._bytepos
        if k > len(bp):
            return (bp[0] if bp else self.pos, 0)
        off = bp[len(bp) - k]
        bit = 8 - (real - 8 * (k - 1))
        return (off, bit)

    def align_to_byte(self) -> None:
        """Discard remaining bits of the current byte (padding before a marker)."""
        drop = self.nbits % 8
        self.nbits -= drop
        self.real = max(0, self.real - drop)
        # Any whole bytes still buffered are rewound so pos points at them again.
        while self.nbits >= 8 and self.real >= 8:
            self.nbits -= 8
            self.real -= 8
            self.pos = self._bytepos.pop()
        self.nbits = 0
        self.real = 0
        self.buf = 0
        self._hit = None
        self._eof = False
        self._bytepos.clear()

    def marker_at(self) -> tuple[int, int] | None:
        """(marker, offset) if a marker (after optional FF fill) is at the read position."""
        p = self.pos
        data = self.data
        # 1-bit padding can form a 0xFF that encoders stuff as FF 00 right before the marker
        hops = 0
        while p + 2 < self.end and data[p] == 0xFF and data[p + 1] == 0x00 and hops < 4:
            p += 2
            hops += 1
        if p >= self.end or data[p] != 0xFF:
            return None
        q = p + 1
        while q < self.end and data[q] == 0xFF:
            q += 1
        if q >= self.end or data[q] == 0x00:
            return None
        return (data[q], p)

    def skip_marker(self) -> None:
        m = self.marker_at()
        if m is None:
            raise ValueError("no marker at read position")
        p = m[1] + 1
        while self.data[p] == 0xFF:
            p += 1
        self.pos = p + 1
        self._hit = None
        self._eof = False

    def seek(self, byte_offset: int) -> None:
        self.pos = byte_offset
        self.buf = 0
        self.nbits = 0
        self.real = 0
        self._hit = None
        self._eof = False
        self._bytepos.clear()
