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
    resync_at: int | None = None  # for a failed interval: offset of the RST marker decoding resumed at


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
    def clean_fraction(self) -> float:
        """Share of MCUs lying in intervals that decoded cleanly end to end."""
        if not self.total_mcus:
            return 0.0
        return sum(iv.mcu_count for iv in self.intervals if iv.ok) / self.total_mcus

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


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int8)


def _marker_candidates(data: bytes, start: int, end: int, expected: int, max_ham: int = 1) -> list[tuple[int, int, bool]]:
    """RST-like byte pairs in [start, end): (n, offset, fuzzy), sorted by offset.

    Exact FF Dn markers are always listed; pairs within `max_ham` flipped bits of
    FF D<expected> are listed as fuzzy candidates with n = expected.
    """
    out: list[tuple[int, int, bool]] = []
    p = start
    while True:
        p = data.find(b"\xff", p, end - 1)
        if p < 0:
            break
        n = data[p + 1]
        if 0xD0 <= n <= 0xD7:
            out.append((n - 0xD0, p, False))
        p += 1
    if end - start > 2:
        arr = np.frombuffer(data, dtype=np.uint8)[start:end]
        h = _POP[arr[:-1] ^ 0xFF].astype(np.int16) + _POP[arr[1:] ^ (0xD0 + expected)]
        for i in np.flatnonzero(h <= max_ham):
            off = start + int(i)
            if data[off] == 0xFF and 0xD0 <= data[off + 1] <= 0xD7:
                continue  # exact marker, already listed
            out.append((expected, off, True))
    out.sort(key=lambda t: t[1])
    return out


# --------------------------------------------------------------------------- main entry


def preds_before_mcu(sr: ScanResult, ft: FrameTables, mcu_index: int) -> dict[int, int]:
    """DC predictor state right before `mcu_index`, from a previous decode of the same scan.

    Within a restart interval the predictor is the DC of the last block of the
    previous MCU per component; at an interval start it is zero.
    """
    frame, scan = ft.frame, ft.scan
    dri = ft.restart_interval
    comps = [frame.component(sc.id) for sc in scan.components]
    interleaved = len(comps) > 1
    if mcu_index == 0 or (dri and mcu_index % dri == 0):
        return {c.id: 0 for c in comps}
    prev = mcu_index - 1
    out: dict[int, int] = {}
    for c in comps:
        h, v = (c.h, c.v) if interleaved else (1, 1)
        arr = sr.dc[c.id]
        per_row = arr.shape[1] // h
        mrow, mcol = divmod(prev, per_row)
        val = arr[mrow * v + v - 1, mcol * h + h - 1]
        out[c.id] = 0 if np.isnan(val) else int(val)
    return out


def decode_scan(
    data: bytes,
    parse: M.ParseResult,
    ft: FrameTables,
    start_byte: int | None = None,
    end_byte: int | None = None,
    first_mcu: int = 0,
    max_mcus: int | None = None,
    start_bit: int = 0,
    init_preds: dict[int, int] | None = None,
    resync: bool = True,
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
    if start_bit:
        br.read_bits(start_bit)
    mcu = first_mcu
    iv_index = first_mcu // dri if dri else 0
    pred = {c.cid: 0 for c in comps}
    resume_preds = dict(init_preds) if init_preds else None

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
        iv_start_off, iv_start_bit = br.tell()
        iv = Interval(iv_index, iv_first, iv_count, iv_start_off)
        if resume_preds is not None:
            pred.update(resume_preds)  # resuming mid-interval: predictors carried in
            resume_preds = None
        else:
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
            if not dri or not resync:
                # Without restart markers there is nothing to resync on (and probes never resync).
                iv.byte_end = end if parse.eoi_offset is None or end_byte is not None else parse.eoi_offset
                break
            # Resync: find the next RST marker and work out which interval it closes.
            # A flipped bit can also *create* a fake FF Dn; a marker whose sequence
            # number does not fit the byte distance is skipped.
            done = [x for x in intervals if x.ok and x.byte_end is not None and x.mcu_count == dri]
            if done:
                avg = sum(x.byte_end - x.byte_start for x in done) / len(done)
            else:
                avg = max(1.0, (end - start) / max(1, -(-total // dri)))
            k = iv_index
            search_from = max(err.byte_offset, iv.byte_start + 1)
            hit = None
            j = k
            fuzzy_hit = False
            for n, moff, fuzzy in _marker_candidates(data, search_from, end, k % 8):
                dist = moff - iv.byte_start
                spans_est = max(1.0, dist / avg)
                if fuzzy:
                    # a damaged copy of the very marker this interval should end with
                    if not (0.3 * avg <= dist <= 2.5 * avg):
                        continue
                    cand = k
                else:
                    base = k + ((n - k) % 8)
                    cands = sorted(
                        (abs((c - k + 1) - spans_est) / spans_est, c) for c in (base, base + 8, base + 16, base + 24)
                    )
                    rel_err, cand = cands[0]
                    if not (rel_err <= 0.6 or (cand == k and dist <= 3 * avg)):
                        continue
                if (cand + 1) * dri < total:
                    # A flipped bit can create a fake FF Dn inside the data: the bytes after a
                    # real marker decode cleanly for a few MCUs, those after a fake one rarely do.
                    # A fuzzy (damaged-looking) marker must prove itself over the whole next interval.
                    probe = decode_scan(data, parse, ft, start_byte=moff + 2, end_byte=end,
                                        first_mcu=(cand + 1) * dri, max_mcus=dri if fuzzy else min(3, dri),
                                        resync=False)
                    if probe.error is not None and probe.error.reason != "eof":
                        continue
                    if fuzzy and not (probe.intervals and probe.intervals[0].ok and probe.intervals[0].end_marker is not None):
                        continue
                hit, j, fuzzy_hit = (n, moff), cand, fuzzy
                break
            if hit is None:
                iv.byte_end = end if end_byte is not None else (parse.eoi_offset or end)
                break
            n, moff = hit
            iv.byte_end = moff if iv.byte_end is None else iv.byte_end
            iv.resync_at = moff
            if fuzzy_hit:
                iv.end_marker_damaged = True
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
