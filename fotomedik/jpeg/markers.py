"""Tolerant JPEG marker-segment parser.

Walks the header segments up to SOS. Never raises on damaged input: it returns
what it could parse plus a list of `ParseError`s describing where and why it
stopped. Repair recipes use those errors to decide what to fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SOI = 0xD8
EOI = 0xD9
SOS = 0xDA
DQT = 0xDB
DNL = 0xDC
DRI = 0xDD
DHP = 0xDE
EXP = 0xDF
DHT = 0xC4
JPG = 0xC8
DAC = 0xCC
SOF0 = 0xC0  # baseline
SOF1 = 0xC1  # extended sequential
SOF2 = 0xC2  # progressive
SOF3 = 0xC3  # lossless
COM = 0xFE
APP0 = 0xE0
APP1 = 0xE1
APP2 = 0xE2
APP15 = 0xEF
RST0 = 0xD0
RST7 = 0xD7

SOF_MARKERS = frozenset(
    [0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF]
)
# Markers that carry a length-prefixed payload.
SEGMENT_MARKERS = frozenset(
    list(SOF_MARKERS) + [DHT, DAC, DQT, DNL, DRI, DHP, EXP, COM, SOS] + list(range(0xE0, 0xF0))
)
# Markers that stand alone (no payload).
STANDALONE_MARKERS = frozenset([SOI, EOI, 0x01] + list(range(RST0, RST7 + 1)))
KNOWN_MARKERS = SEGMENT_MARKERS | STANDALONE_MARKERS

MARKER_NAMES = {
    SOI: "SOI", EOI: "EOI", SOS: "SOS", DQT: "DQT", DNL: "DNL", DRI: "DRI",
    DHT: "DHT", DAC: "DAC", SOF0: "SOF0", SOF1: "SOF1", SOF2: "SOF2", SOF3: "SOF3",
    COM: "COM",
}
for _i in range(16):
    MARKER_NAMES[0xE0 + _i] = f"APP{_i}"
for _i in range(8):
    MARKER_NAMES[RST0 + _i] = f"RST{_i}"


def marker_name(m: int) -> str:
    return MARKER_NAMES.get(m, f"0x{m:02X}")


@dataclass
class Segment:
    marker: int
    offset: int  # offset of the 0xFF byte
    length: int  # length field value (includes its own 2 bytes)
    payload: bytes  # segment data without the length field

    @property
    def end(self) -> int:
        return self.offset + 2 + self.length

    @property
    def name(self) -> str:
        return marker_name(self.marker)


@dataclass
class ParseError:
    offset: int
    reason: str  # no_soi | no_marker | unknown_marker | bad_length | eof
    detail: str = ""


@dataclass
class ParseResult:
    segments: list[Segment] = field(default_factory=list)
    scan_start: int | None = None  # first byte of entropy-coded data
    eoi_offset: int | None = None  # offset of the final FF D9, if present
    errors: list[ParseError] = field(default_factory=list)

    def first(self, marker: int) -> Segment | None:
        for s in self.segments:
            if s.marker == marker:
                return s
        return None

    def all(self, marker: int) -> list[Segment]:
        return [s for s in self.segments if s.marker == marker]

    @property
    def sof(self) -> Segment | None:
        for s in self.segments:
            if s.marker in SOF_MARKERS:
                return s
        return None

    @property
    def ok(self) -> bool:
        return not self.errors and self.scan_start is not None


def find_eoi(data: bytes) -> int | None:
    """Offset of the trailing EOI, ignoring zero/garbage padding after it."""
    pos = data.rfind(b"\xff\xd9")
    return pos if pos >= 0 else None


def parse_segments(data: bytes) -> ParseResult:
    res = ParseResult()
    n = len(data)
    if n < 4 or data[0] != 0xFF or data[1] != SOI:
        res.errors.append(ParseError(0, "no_soi"))
        return res
    res.eoi_offset = find_eoi(data)
    p = 2
    while True:
        if p + 4 > n:
            res.errors.append(ParseError(p, "eof", "header ends before SOS"))
            return res
        if data[p] != 0xFF:
            res.errors.append(ParseError(p, "no_marker", f"byte 0x{data[p]:02X} where 0xFF expected"))
            return res
        # Fill bytes: any number of FF before a marker is legal.
        while p < n and data[p] == 0xFF and p + 1 < n and data[p + 1] == 0xFF:
            p += 1
        m = data[p + 1]
        if m in STANDALONE_MARKERS:
            if m == SOI:
                p += 2
                continue
            res.errors.append(ParseError(p, "unknown_marker", f"standalone {marker_name(m)} inside header"))
            return res
        if m not in SEGMENT_MARKERS:
            res.errors.append(ParseError(p, "unknown_marker", f"0xFF{m:02X} is not a segment marker"))
            return res
        length = (data[p + 2] << 8) | data[p + 3]
        if length < 2:
            res.errors.append(ParseError(p, "bad_length", f"length {length} < 2"))
            return res
        if p + 2 + length > n:
            res.errors.append(ParseError(p, "bad_length", f"length {length} runs past end of file"))
            return res
        seg = Segment(m, p, length, bytes(data[p + 4 : p + 2 + length]))
        res.segments.append(seg)
        p = seg.end
        if m == SOS:
            res.scan_start = p
            return res
