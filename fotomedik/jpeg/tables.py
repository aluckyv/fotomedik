"""JPEG table and header parsing: DQT, DHT, SOF, SOS, DRI, plus Annex K standard tables."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import markers as M


class TableError(ValueError):
    """Raised when a table segment is internally inconsistent."""


# --------------------------------------------------------------------------- DQT


@dataclass
class QuantTable:
    id: int
    precision: int  # 0 = 8-bit, 1 = 16-bit entries
    values: tuple[int, ...]  # 64 values in zig-zag order

    @property
    def dc(self) -> int:
        return self.values[0]


def parse_dqt(payload: bytes) -> dict[int, QuantTable]:
    out: dict[int, QuantTable] = {}
    p = 0
    while p < len(payload):
        pq = payload[p] >> 4
        tq = payload[p] & 0x0F
        p += 1
        if pq not in (0, 1):
            raise TableError(f"DQT precision {pq} invalid")
        if tq > 3:
            raise TableError(f"DQT id {tq} invalid")
        size = 64 * (2 if pq else 1)
        if p + size > len(payload):
            raise TableError("DQT payload too short")
        if pq:
            vals = tuple((payload[p + 2 * i] << 8) | payload[p + 2 * i + 1] for i in range(64))
        else:
            vals = tuple(payload[p : p + 64])
        if any(v == 0 for v in vals):
            raise TableError("DQT contains zero quantizer")
        out[tq] = QuantTable(tq, pq, vals)
        p += size
    return out


# --------------------------------------------------------------------------- DHT


@dataclass
class HuffmanTable:
    cls: int  # 0 = DC, 1 = AC
    id: int
    counts: tuple[int, ...]  # 16 counts (codes of length 1..16)
    symbols: tuple[int, ...]
    lookup: dict[tuple[int, int], int] = field(default_factory=dict)  # (length, code) -> symbol
    # Canonical decode tables (Annex F.2.2.3), indexed by code length 1..16
    maxcode: list[int] = field(default_factory=list)
    valptr: list[int] = field(default_factory=list)
    mincode: list[int] = field(default_factory=list)

    def __post_init__(self):
        if len(self.counts) != 16:
            raise TableError("DHT needs 16 counts")
        if sum(self.counts) != len(self.symbols):
            raise TableError("DHT counts do not match symbol count")
        if sum(self.counts) > 256:
            raise TableError("DHT has more than 256 symbols")
        self.lookup = {}
        self.maxcode = [-1] * 17
        self.valptr = [0] * 17
        self.mincode = [0] * 17
        code = 0
        k = 0
        for length in range(1, 17):
            n = self.counts[length - 1]
            if n:
                self.valptr[length] = k
                self.mincode[length] = code
                for _ in range(n):
                    if code >= (1 << length):
                        raise TableError("DHT code space overflow")
                    self.lookup[(length, code)] = self.symbols[k]
                    code += 1
                    k += 1
                self.maxcode[length] = code - 1
            code <<= 1

    @property
    def key(self) -> tuple[int, int]:
        return (self.cls, self.id)


def parse_dht(payload: bytes) -> dict[tuple[int, int], HuffmanTable]:
    out: dict[tuple[int, int], HuffmanTable] = {}
    p = 0
    while p < len(payload):
        if p + 17 > len(payload):
            raise TableError("DHT payload too short")
        tc = payload[p] >> 4
        th = payload[p] & 0x0F
        if tc not in (0, 1) or th > 3:
            raise TableError(f"DHT class/id {tc}/{th} invalid")
        counts = tuple(payload[p + 1 : p + 17])
        total = sum(counts)
        if total > 256:
            raise TableError(f"DHT symbol count {total} > 256")
        p += 17
        if p + total > len(payload):
            raise TableError("DHT symbols run past payload")
        symbols = tuple(payload[p : p + total])
        p += total
        out[(tc, th)] = HuffmanTable(tc, th, counts, symbols)
    return out


# --------------------------------------------------------------------------- SOF


@dataclass
class Component:
    id: int
    h: int
    v: int
    tq: int


@dataclass
class FrameInfo:
    marker: int
    precision: int
    height: int
    width: int
    components: list[Component]

    @property
    def progressive(self) -> bool:
        return self.marker in (0xC2, 0xC6, 0xCA, 0xCE)

    @property
    def baseline(self) -> bool:
        return self.marker in (0xC0, 0xC1)

    @property
    def hmax(self) -> int:
        return max(c.h for c in self.components)

    @property
    def vmax(self) -> int:
        return max(c.v for c in self.components)

    @property
    def mcu_width(self) -> int:
        return 8 * self.hmax

    @property
    def mcu_height(self) -> int:
        return 8 * self.vmax

    @property
    def mcus_per_row(self) -> int:
        return -(-self.width // self.mcu_width)

    @property
    def mcu_rows(self) -> int:
        return -(-self.height // self.mcu_height)

    @property
    def mcu_count(self) -> int:
        return self.mcus_per_row * self.mcu_rows

    def component(self, cid: int) -> Component:
        for c in self.components:
            if c.id == cid:
                return c
        raise TableError(f"component id {cid} not in frame")


def parse_sof(marker: int, payload: bytes) -> FrameInfo:
    if len(payload) < 6:
        raise TableError("SOF too short")
    precision = payload[0]
    height = (payload[1] << 8) | payload[2]
    width = (payload[3] << 8) | payload[4]
    nc = payload[5]
    if precision not in (8, 12):
        raise TableError(f"SOF precision {precision} invalid")
    if width == 0:
        raise TableError("SOF width 0")
    if nc not in (1, 3, 4):
        raise TableError(f"SOF component count {nc} invalid")
    if len(payload) < 6 + 3 * nc:
        raise TableError("SOF component list truncated")
    comps = []
    for i in range(nc):
        cid = payload[6 + 3 * i]
        hv = payload[7 + 3 * i]
        tq = payload[8 + 3 * i]
        h, v = hv >> 4, hv & 0x0F
        if not (1 <= h <= 4 and 1 <= v <= 4) or tq > 3:
            raise TableError(f"SOF component {cid} sampling {h}x{v} / tq {tq} invalid")
        comps.append(Component(cid, h, v, tq))
    return FrameInfo(marker, precision, height, width, comps)


# --------------------------------------------------------------------------- SOS / DRI


@dataclass
class ScanComponent:
    id: int
    td: int  # DC table id
    ta: int  # AC table id


@dataclass
class ScanInfo:
    components: list[ScanComponent]
    ss: int
    se: int
    ah: int
    al: int


def parse_sos(payload: bytes) -> ScanInfo:
    if len(payload) < 1:
        raise TableError("SOS too short")
    ns = payload[0]
    if ns not in (1, 2, 3, 4) or len(payload) < 1 + 2 * ns + 3:
        raise TableError(f"SOS component count {ns} invalid")
    comps = []
    for i in range(ns):
        cid = payload[1 + 2 * i]
        t = payload[2 + 2 * i]
        td, ta = t >> 4, t & 0x0F
        if td > 3 or ta > 3:
            raise TableError(f"SOS table ids {td}/{ta} invalid")
        comps.append(ScanComponent(cid, td, ta))
    p = 1 + 2 * ns
    ss, se = payload[p], payload[p + 1]
    ah, al = payload[p + 2] >> 4, payload[p + 2] & 0x0F
    return ScanInfo(comps, ss, se, ah, al)


def parse_dri(payload: bytes) -> int:
    if len(payload) < 2:
        raise TableError("DRI too short")
    return (payload[0] << 8) | payload[1]


# --------------------------------------------------------------------------- Annex K standard tables

_STD_DC_LUMA_COUNTS = (0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
_STD_DC_LUMA_SYMBOLS = tuple(range(12))
_STD_DC_CHROMA_COUNTS = (0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
_STD_DC_CHROMA_SYMBOLS = tuple(range(12))
_STD_AC_LUMA_COUNTS = (0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D)
_STD_AC_LUMA_SYMBOLS = (
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61,
    0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52,
    0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25,
    0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3,
    0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8,
    0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
)
_STD_AC_CHROMA_COUNTS = (0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77)
_STD_AC_CHROMA_SYMBOLS = (
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61,
    0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33,
    0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18,
    0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44,
    0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63,
    0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A,
    0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA,
    0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7,
    0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
)

_STANDARD = {
    (0, 0): (_STD_DC_LUMA_COUNTS, _STD_DC_LUMA_SYMBOLS),
    (0, 1): (_STD_DC_CHROMA_COUNTS, _STD_DC_CHROMA_SYMBOLS),
    (1, 0): (_STD_AC_LUMA_COUNTS, _STD_AC_LUMA_SYMBOLS),
    (1, 1): (_STD_AC_CHROMA_COUNTS, _STD_AC_CHROMA_SYMBOLS),
}


def standard_table(cls: int, id: int) -> HuffmanTable:
    """Annex K.3 table: cls 0/1 = DC/AC, id 0/1 = luma/chroma."""
    counts, symbols = _STANDARD[(cls, id)]
    return HuffmanTable(cls, id, counts, symbols)


def standard_table_bytes(cls: int, id: int) -> bytes:
    """The same table as a DHT payload fragment (Tc/Th byte + counts + symbols)."""
    counts, symbols = _STANDARD[(cls, id)]
    return bytes([(cls << 4) | id]) + bytes(counts) + bytes(symbols)


# --------------------------------------------------------------------------- bundle


@dataclass
class FrameTables:
    """Everything the entropy decoder needs, gathered from parsed segments."""

    frame: FrameInfo
    scan: ScanInfo
    quant: dict[int, QuantTable]
    huff: dict[tuple[int, int], HuffmanTable]
    restart_interval: int = 0

    @classmethod
    def from_parse(cls, res: M.ParseResult) -> "FrameTables":
        sof = res.sof
        if sof is None:
            raise TableError("no SOF segment")
        sos = res.first(M.SOS)
        if sos is None:
            raise TableError("no SOS segment")
        quant: dict[int, QuantTable] = {}
        for s in res.all(M.DQT):
            quant.update(parse_dqt(s.payload))
        huff: dict[tuple[int, int], HuffmanTable] = {}
        for s in res.all(M.DHT):
            huff.update(parse_dht(s.payload))
        dri = res.first(M.DRI)
        return cls(
            frame=parse_sof(sof.marker, sof.payload),
            scan=parse_sos(sos.payload),
            quant=quant,
            huff=huff,
            restart_interval=parse_dri(dri.payload) if dri else 0,
        )
