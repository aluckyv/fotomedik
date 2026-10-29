"""Baseline JPEG entropy decoder built for diagnosis, not for pixels.

It walks the scan MCU by MCU, records where every MCU starts (bit-exact),
decodes DC values (for plausibility checks and smoothness scoring), detects the
first inconsistency inside each restart interval and resynchronises on the
next restart marker. No IDCT, no colour conversion: rendering is Pillow's job
once the bytes are repaired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import markers as M
from .bitreader import BitReader, MarkerHit
from .tables import FrameTables, HuffmanTable

ERROR_REASONS = frozenset(
    ["bad_huffman", "coef_overflow", "dc_range", "unexpected_marker", "eof", "bad_rst", "skipped"]
)


class Unsupported(Exception):
    """Scan type the v0.1 decoder does not handle (progressive, 12-bit, arithmetic)."""


class MissingTables(Exception):
    """Header lacks a table the scan references; repair the header first."""


@dataclass
class DecodeError:
    reason: str
    mcu_index: int
    byte_offset: int
    bit: int = 0
    detail: str = ""


@dataclass
class MCUInfo:
    index: int
    bit_start: int  # absolute bit position in the file


@dataclass
class Interval:
    index: int
    mcu_first: int
    mcu_count: int  # expected MCUs in this interval
    byte_start: int
    byte_end: int | None = None  # offset of the marker (FF) that ends it, or end of data
    end_marker: int | None = None  # 0xD0..0xD7, 0xD9, or None (range end)
    end_marker_damaged: bool = False
    decoded: int = 0  # MCUs decoded cleanly
    ok: bool = False
    error: DecodeError | None = None


@dataclass
class ScanResult:
    mcus: list[MCUInfo]
    intervals: list[Interval]
    error: DecodeError | None
    dc: dict[int, np.ndarray]  # component id -> (block_rows, block_cols) float, NaN = not decoded
    mcu_ok: np.ndarray  # bool per MCU in the frame
    total_mcus: int
    trailing_bytes: int = 0  # bytes between last decoded data and EOI (garbage)

    @property
    def decoded_mcus(self) -> int:
        return int(self.mcu_ok.sum())

    @property
    def recovered_fraction(self) -> float:
        return self.decoded_mcus / self.total_mcus if self.total_mcus else 0.0

    @property
    def errors(self) -> list[DecodeError]:
        return [iv.error for iv in self.intervals if iv.error is not None]


# --------------------------------------------------------------------------- Huffman fast tables

_FAST_BITS = 9
_fast_cache: dict[int, tuple[list[int], HuffmanTable]] = {}


def _fast_table(t: HuffmanTable) -> list[int]:
    key = id(t)
    hit = _fast_cache.get(key)
    if hit is not None and hit[1] is t:
        return hit[0]
    fast = [0] * (1 << _FAST_BITS)
    for (length, code), sym in t.lookup.items():
        if length <= _FAST_BITS:
            shift = _FAST_BITS - length
            base = code << shift
            entry = (length << 8) | sym
            for i in range(1 << shift):
                fast[base + i] = entry
    _fast_cache[key] = (fast, t)
    return fast


class _Bad(Exception):
    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail


def _decode_symbol(br: BitReader, t: HuffmanTable, fast: list[int]) -> int:
    code16 = br.peek_bits(16)
    e = fast[code16 >> (16 - _FAST_BITS)]
    if e:
        br.skip_bits(e >> 8)
        return e & 0xFF
    maxcode = t.maxcode
    for length in range(_FAST_BITS + 1, 17):
        c = code16 >> (16 - length)
        if c <= maxcode[length]:
            br.skip_bits(length)
            return t.symbols[t.valptr[length] + c - t.mincode[length]]
    raise _Bad("bad_huffman", f"no code matches 0x{code16:04X}")


# --------------------------------------------------------------------------- geometry


@dataclass
class _Comp:
    cid: int
    h: int
    v: int
    qdc: int
    dc_t: HuffmanTable
    ac_t: HuffmanTable
    dc_fast: list[int]
    ac_fast: list[int]
    blocks_w: int
    blocks_h: int


def _scan_components(ft: FrameTables) -> tuple[list[_Comp], int, int]:
    """Components in scan order plus MCU grid (per row, rows) for this scan."""
    frame, scan = ft.frame, ft.scan
    comps: list[_Comp] = []
    interleaved = len(scan.components) > 1
    for sc in scan.components:
        fc = frame.component(sc.id)
        if fc.tq not in ft.quant:
            raise MissingTables(f"quant table {fc.tq} missing")
        if (0, sc.td) not in ft.huff or (1, sc.ta) not in ft.huff:
            raise MissingTables(f"huffman tables {sc.td}/{sc.ta} missing")
        dc_t, ac_t = ft.huff[(0, sc.td)], ft.huff[(1, sc.ta)]
        if interleaved:
            h, v = fc.h, fc.v
            bw, bh = frame.mcus_per_row * h, frame.mcu_rows * v
        else:
            h = v = 1
            cw = -(-frame.width * fc.h // frame.hmax)
            ch = -(-frame.height * fc.v // frame.vmax)
            bw, bh = -(-cw // 8), -(-ch // 8)
        comps.append(
            _Comp(sc.id, h, v, ft.quant[fc.tq].dc, dc_t, ac_t, _fast_table(dc_t), _fast_table(ac_t), bw, bh)
        )
    if interleaved:
        return comps, frame.mcus_per_row, frame.mcu_rows
    return comps, comps[0].blocks_w, comps[0].blocks_h


def _find_rst(data: bytes, start: int, end: int) -> tuple[int, int] | None:
    """Next FF Dn (n in 0..7) at or after start; returns (n, offset of FF)."""
    p = start
    while True:
        p = data.find(b"\xff", p, end - 1)
        if p < 0:
            return None
        n = data[p + 1]
        if 0xD0 <= n <= 0xD7:
            return (n - 0xD0, p)
        p += 1


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- main entry


def decode_scan(
    data: bytes,
    parse: M.ParseResult,
    ft: FrameTables,
    start_byte: int | None = None,
    end_byte: int | None = None,
    first_mcu: int = 0,
    max_mcus: int | None = None,
) -> ScanResult:
    frame = ft.frame
    if frame.progressive or frame.marker not in (0xC0, 0xC1):
        raise Unsupported(f"SOF marker 0x{frame.marker:02X} (progressive/lossless/arithmetic) not supported")
    if frame.precision != 8:
        raise Unsupported(f"{frame.precision}-bit precision not supported")
    if parse.scan_start is None:
        raise MissingTables("no SOS / scan start")

    comps, mcus_per_row, mcu_rows = _scan_components(ft)
    total = mcus_per_row * mcu_rows
    dri = ft.restart_interval

    start = parse.scan_start if start_byte is None else start_byte
    end = len(data) if end_byte is None else end_byte
    if end_byte is None and parse.eoi_offset is not None and parse.eoi_offset >= start:
        end = parse.eoi_offset + 2  # keep EOI inside the range so the reader sees it as a marker
    partial = start_byte is not None or end_byte is not None or max_mcus is not None
    limit = total if max_mcus is None else min(total, first_mcu + max_mcus)

    dc_arrays = {c.cid: np.full((c.blocks_h, c.blocks_w), np.nan) for c in comps}
    mcu_ok = np.zeros(total, dtype=bool)
    mcus: list[MCUInfo] = []
    intervals: list[Interval] = []
    first_error: DecodeError | None = None

    br = BitReader(data, start, end)
    mcu = first_mcu
    iv_index = first_mcu // dri if dri else 0
    pred = {c.cid: 0 for c in comps}

    def block(c: _Comp, brow: int, bcol: int) -> None:
        t = _decode_symbol(br, c.dc_t, c.dc_fast)
        if t > 11:
            raise _Bad("bad_huffman", f"DC category {t}")
        diff = br.receive_extend(t)
        pred[c.cid] += diff
        dcv = pred[c.cid]
        if abs(dcv * c.qdc) > 1024 + c.qdc:
            raise _Bad("dc_range", f"DC {dcv} * q{c.qdc}")
        k = 1
        while k < 64:
            rs = _decode_symbol(br, c.ac_t, c.ac_fast)
            s = rs & 15
            r = rs >> 4
            if s == 0:
                if r == 15:
                    k += 16
                    continue
                break
            k += r
            if k > 63:
                raise _Bad("coef_overflow", f"k={k}")
            if s > 10:
                raise _Bad("bad_huffman", f"AC category {s}")
            br.skip_bits(s)
            k += 1
        dc_arrays[c.cid][brow, bcol] = dcv

    while mcu < limit:
        iv_first = mcu
        iv_count = min(dri, limit - mcu) if dri else (limit - mcu)
        if dri and (iv_first // dri + 1) * dri < total:
            iv_count = min(iv_count, (iv_first // dri + 1) * dri - iv_first)
        iv = Interval(iv_index, iv_first, iv_count, br.byte_pos)
        for cid in pred:
            pred[cid] = 0
        err: DecodeError | None = None
        stop = False
        try:
            for _ in range(iv_count):
                off, bit = br.tell()
                mcus.append(MCUInfo(mcu, off * 8 + bit))
                mrow, mcol = divmod(mcu, mcus_per_row)
                for c in comps:
                    for by in range(c.v):
                        for bx in range(c.h):
                            block(c, mrow * c.v + by, mcol * c.h + bx)
                mcu_ok[mcu] = True
                iv.decoded += 1
                mcu += 1
        except _Bad as e:
            off, bit = br.tell()
            err = DecodeError(e.reason, mcu, off, bit, e.detail)
        except MarkerHit as e:
            err = DecodeError("unexpected_marker", mcu, e.offset, 0, f"marker 0xFF{e.marker:02X} inside MCU")
        except EOFError:
            off, bit = br.tell()
            err = DecodeError("eof", mcu, off, bit, "data ended inside interval")

        if err is None:
            # Interval decoded; now the boundary: marker, range end, or EOI.
            br.align_to_byte()
            iv.byte_end = br.byte_pos
            m = br.marker_at()
            last_in_scan = mcu >= total
            if m is None:
                if br.byte_pos >= end or (partial and mcu >= limit):
                    iv.end_marker = None
                    iv.ok = True
                    stop = True
                elif last_in_scan:
                    iv.ok = True  # data after the last MCU is just trailing bytes
                else:
                    err = DecodeError("bad_rst", mcu, br.byte_pos, 0, "no restart marker where one is expected")
            else:
                marker, moff = m
                iv.byte_end = moff
                iv.end_marker = marker
                expected = 0xD0 + (iv_index % 8)
                if last_in_scan:
                    iv.ok = True
                elif not dri:
                    err = DecodeError("unexpected_marker", mcu, moff, 0, f"marker 0xFF{marker:02X} before end of scan")
                elif marker == expected:
                    iv.ok = True
                    br.skip_marker()
                elif _hamming(marker, expected) <= 2 and not (0xD0 <= marker <= 0xD7):
                    iv.ok = True
                    iv.end_marker_damaged = True
                    br.skip_marker()
                else:
                    err = DecodeError("bad_rst", mcu, moff, 0, f"marker 0xFF{marker:02X}, expected 0xFF{expected:02X}")

        if err is not None:
            iv.ok = False
            iv.error = err
            if first_error is None:
                first_error = err
            intervals.append(iv)
            if not dri or partial and end_byte is not None and err.reason != "bad_rst":
                # Without restart markers there is nothing to resync on.
                if not dri:
                    iv.byte_end = end if parse.eoi_offset is None or end_byte is not None else parse.eoi_offset
                    break
            # Resync: find next RST marker and work out which interval it closes.
            hit = _find_rst(data, max(err.byte_offset, iv.byte_start + 1), end)
            if hit is None:
                iv.byte_end = end if end_byte is not None else (parse.eoi_offset or end)
                break
            n, moff = hit
            iv.byte_end = moff if iv.byte_end is None else iv.byte_end
            k = iv_index
            j = k + ((n - k) % 8)
            # choose among j, j+8, j+16 by byte distance plausibility
            done = [x for x in intervals if x.ok and x.byte_end is not None]
            if done:
                avg = sum(x.byte_end - x.byte_start for x in done) / len(done)
            else:
                avg = max(1.0, (end - start) / max(1, -(-total // dri)))
            best_j, best_d = j, abs((moff - iv.byte_start) - (j - k + 1) * avg)
            for cand in (j + 8, j + 16):
                d = abs((moff - iv.byte_start) - (cand - k + 1) * avg)
                if d < best_d:
                    best_j, best_d = cand, d
            j = best_j
            # intervals k+1..j are lost
            for skipped in range(k + 1, j + 1):
                sf = skipped * dri
                if sf >= total:
                    break
                sc = min(dri, total - sf)
                intervals.append(
                    Interval(skipped, sf, sc, iv.byte_end or moff, moff, None, False, 0, False,
                             DecodeError("skipped", sf, moff, 0, "lost in resync"))
                )
            iv_index = j + 1
            mcu = iv_index * dri
            br.seek(moff + 2)
            continue

        intervals.append(iv)
        iv_index += 1
        if stop:
            break

    # Trailing bytes between the last interval and EOI (or end).
    trailing = 0
    if intervals and not partial:
        last = intervals[-1]
        if last.byte_end is not None and parse.eoi_offset is not None and last.ok:
            trailing = max(0, parse.eoi_offset - last.byte_end)
    return ScanResult(mcus, intervals, first_error, dc_arrays, mcu_ok, total, trailing)
