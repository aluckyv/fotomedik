"""The two hot loops, with one interface and two implementations.

decode_mcus()  decode up to N MCUs of an interval from a bit position.
eval_flips()   try a list of (offset, mask) bit flips in place and score each.

The C implementation (_fastdec.c via ctypes) is used when the compiled library
is present; the pure-Python one below mirrors it statement by statement and is
the reference the tests also run against (FOTOMEDIK_PURE=1).
"""

from __future__ import annotations

import ctypes
import math

import numpy as np

from . import fastdec
from .bitreader import BitReader, MarkerHit
from .tables import FrameTables, HuffmanTable

FAST_BITS = 9
ERR_NAMES = {0: None, 1: "bad_huffman", 2: "coef_overflow", 3: "dc_range", 4: "marker", 5: "eof"}


class MissingTables(Exception):
    """Header lacks a table the scan references; repair the header first."""


# --------------------------------------------------------------------------- tables


def fast_table(t: HuffmanTable) -> list[int]:
    fast = [0] * (1 << FAST_BITS)
    for (length, code), sym in t.lookup.items():
        if length <= FAST_BITS:
            shift = FAST_BITS - length
            base = code << shift
            entry = (length << 8) | sym
            for i in range(1 << shift):
                fast[base + i] = entry
    return fast


class Packed:
    """Scan layout + Huffman tables in both the C layout and the Python layout."""

    def __init__(self, ft: FrameTables):
        frame, scan = ft.frame, ft.scan
        interleaved = len(scan.components) > 1
        self.fast = np.zeros((8, 1 << FAST_BITS), dtype=np.int32)
        self.maxcode = np.full((8, 17), -1, dtype=np.int32)
        self.valptr = np.zeros((8, 17), dtype=np.int32)
        self.mincode = np.zeros((8, 17), dtype=np.int32)
        self.symbols = np.zeros((8, 256), dtype=np.uint8)
        self.py: list[tuple] = []  # (h, v, dc_table, ac_table, dc_fast, ac_fast, qdc)
        self.cids: list[int] = []
        self.hv: list[tuple[int, int]] = []
        self.qdc: list[int] = []
        self.blocks_per_mcu = 0
        comps = []
        for sc in scan.components:
            fc = frame.component(sc.id)
            if fc.tq not in ft.quant:
                raise MissingTables(f"quant table {fc.tq} missing")
            if (0, sc.td) not in ft.huff or (1, sc.ta) not in ft.huff:
                raise MissingTables(f"huffman tables {sc.td}/{sc.ta} missing")
            h, v = (fc.h, fc.v) if interleaved else (1, 1)
            dc_t, ac_t = ft.huff[(0, sc.td)], ft.huff[(1, sc.ta)]
            q = ft.quant[fc.tq].dc
            self.py.append((h, v, dc_t, ac_t, fast_table(dc_t), fast_table(ac_t), q))
            comps.append((h, v, sc.td, sc.ta, q, sc.id))
            self.cids.append(sc.id)
            self.hv.append((h, v))
            self.qdc.append(q)
            self.blocks_per_mcu += h * v
        for (cls, tid), t in ft.huff.items():
            if tid > 3:
                continue
            slot = cls * 4 + tid
            for (length, code), sym in t.lookup.items():
                if length <= FAST_BITS:
                    shift = FAST_BITS - length
                    b = code << shift
                    self.fast[slot, b : b + (1 << shift)] = (length << 8) | sym
            self.maxcode[slot, :] = t.maxcode
            self.valptr[slot, :] = t.valptr
            self.mincode[slot, :] = t.mincode
            self.symbols[slot, : len(t.symbols)] = list(t.symbols)
        self.ncomp = len(comps)
        self.c_comps = (fastdec.Comp * len(comps))(*[fastdec.Comp(*c) for c in comps])


_packed_cache: dict[int, tuple[FrameTables, Packed]] = {}


def packed_for(ft: FrameTables) -> Packed:
    hit = _packed_cache.get(id(ft))
    if hit is not None and hit[0] is ft:
        return hit[1]
    p = Packed(ft)
    if len(_packed_cache) > 64:
        _packed_cache.clear()
    _packed_cache[id(ft)] = (ft, p)
    return p


def _ptr(a) -> int:
    return a.ctypes.data


def as_array(data) -> np.ndarray:
    """uint8 view of bytes/bytearray/ndarray without copying (writable if the source is)."""
    if isinstance(data, np.ndarray):
        return data
    return np.frombuffer(data, dtype=np.uint8)


# --------------------------------------------------------------------------- Python reference


class _Bad(Exception):
    def __init__(self, code: int):
        self.code = code


def _py_symbol(br: BitReader, t: HuffmanTable, fast: list[int]) -> int:
    code16 = br.peek_bits(16)
    e = fast[code16 >> (16 - FAST_BITS)]
    if e:
        br.skip_bits(e >> 8)
        return e & 0xFF
    maxcode = t.maxcode
    for length in range(FAST_BITS + 1, 17):
        c = code16 >> (16 - length)
        if c <= maxcode[length]:
            br.skip_bits(length)
            return t.symbols[t.valptr[length] + c - t.mincode[length]]
    raise _Bad(1)


def py_run_decode(data, start_byte: int, start_bit: int, end: int, pk: Packed, max_mcus: int,
                  preds: list[int], want_bits: bool = True):
    """Returns (n, err, err_byte, err_bit, marker, end_byte, end_bit, bitstarts, dc_flat)."""
    br = BitReader(data, start_byte, end)
    bitstarts: list[int] = []
    dc: list[int] = []
    err = 0
    marker = 0
    eb_err = eb_bit = 0
    n = 0
    try:
        if start_bit:
            br.read_bits(start_bit)
        while n < max_mcus:
            if want_bits:
                o, b = br.tell()
                bitstarts.append(o * 8 + b)
            for ci, (h, v, dc_t, ac_t, dc_fast, ac_fast, q) in enumerate(pk.py):
                for _ in range(h * v):
                    s0 = _py_symbol(br, dc_t, dc_fast)
                    if s0 > 11:
                        raise _Bad(1)
                    diff = br.receive_extend(s0)
                    preds[ci] += diff
                    dcv = preds[ci]
                    if abs(dcv * q) > 1024 + q:
                        raise _Bad(3)
                    k = 1
                    while k < 64:
                        rs = _py_symbol(br, ac_t, ac_fast)
                        s = rs & 15
                        r = rs >> 4
                        if s == 0:
                            if r == 15:
                                k += 16
                                continue
                            break
                        k += r
                        if k > 63:
                            raise _Bad(2)
                        if s > 10:
                            raise _Bad(1)
                        br.skip_bits(s)
                        k += 1
                    dc.append(dcv)
            n += 1
    except _Bad as e:
        err = e.code
    except MarkerHit as e:
        err, marker, eb_err = 4, e.marker, e.offset
    except EOFError:
        err = 5
    end_byte, end_bit = br.tell()
    if err != 4:
        eb_err, eb_bit = end_byte, end_bit
    # drop DC values of a partially decoded MCU (the C kernel overwrites them later)
    dc = dc[: n * pk.blocks_per_mcu]
    return n, err, eb_err, eb_bit, marker, end_byte, end_bit, bitstarts[:n], dc


def py_aligned_end(data, length: int, eb: int, ebit: int) -> tuple[int, bool]:
    p = eb
    pad_ok = True
    if ebit:
        mask = (1 << (8 - ebit)) - 1
        pad_ok = (data[eb] & mask) == mask
        p = eb + 1
        if data[eb] == 0xFF and p < length and data[p] == 0x00:
            p += 1
    hops = 0
    while hops < 4 and p + 2 < length and data[p] == 0xFF and data[p + 1] == 0x00:
        p += 2
        hops += 1
    return p, pad_ok


def py_roughness(dc, nblocks: int, up_v, dn_v, left_k, w) -> float:
    total = 0.0
    n = 0
    for k in range(nblocks):
        a = float(dc[k])
        lk = int(left_k[k])
        if 0 <= lk < k:
            total += 2.0 * abs(a - dc[lk]) * w[k]
            n += 1
        if not math.isnan(up_v[k]):
            total += abs(a - up_v[k]) * w[k]
            n += 1
        if not math.isnan(dn_v[k]):
            total += abs(a - dn_v[k]) * w[k]
            n += 1
    return total / n if n else 0.0


# --------------------------------------------------------------------------- public API


def decode_mcus(data, start_byte: int, start_bit: int, end: int, pk: Packed, preds: np.ndarray, max_mcus: int):
    """Decode up to max_mcus MCUs. `preds` (int32[ncomp]) is updated in place.

    Returns (n, err_name, err_byte, err_bit, marker, end_byte, end_bit, bitstarts[n], dc[n, bpm]).
    """
    lib = fastdec.load()
    bpm = pk.blocks_per_mcu
    if lib is not None:
        arr = as_array(data)
        cap = max(1, max_mcus)
        bitstarts = np.zeros(cap, dtype=np.int64)
        dc_vals = np.zeros(cap * bpm, dtype=np.int32)
        out = np.zeros(6, dtype=np.int64)
        n = int(lib.decode_mcus(
            _ptr(arr), start_byte, start_bit, end, pk.c_comps, pk.ncomp,
            _ptr(pk.fast), _ptr(pk.maxcode), _ptr(pk.valptr), _ptr(pk.mincode), _ptr(pk.symbols),
            max_mcus, _ptr(preds), _ptr(bitstarts), _ptr(dc_vals), _ptr(out),
        ))
        return (n, ERR_NAMES[int(out[0])], int(out[1]), int(out[2]), int(out[3]), int(out[4]), int(out[5]),
                bitstarts[:n], dc_vals[: n * bpm].reshape(n, bpm))
    p = [int(x) for x in preds]
    n, err, eb, ebit, marker, end_b, end_bit, bits, dc = py_run_decode(data, start_byte, start_bit, end, pk, max_mcus, p)
    preds[:] = p
    return (n, ERR_NAMES[err], eb, ebit, marker, end_b, end_bit,
            np.asarray(bits, dtype=np.int64), np.asarray(dc, dtype=np.int32).reshape(n, bpm))


def eval_flips(buf: np.ndarray, start_byte: int, start_bit: int, end: int, pk: Packed, preds0: np.ndarray,
               needed: int, stage1_n: int, marker_pos: int, min_gain: int,
               offs: np.ndarray, masks: np.ndarray, geom) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate bit-flip candidates. buf must be a writable uint8 array (bytes are flipped and restored).

    geom = (up_v, dn_v, left_k, w) per block of the `needed` MCUs (see r5_bitflip.geometry).
    Returns (dec int32, complete bool, pad_ok bool, rough float64), one entry per candidate.
    """
    ncand = len(offs)
    res_dec = np.zeros(ncand, dtype=np.int32)
    res_complete = np.zeros(ncand, dtype=np.int8)
    res_pad = np.zeros(ncand, dtype=np.int8)
    res_rough = np.zeros(ncand, dtype=np.float64)
    if ncand == 0:
        return res_dec, res_complete.astype(bool), res_pad.astype(bool), res_rough
    up_v, dn_v, left_k, w = geom
    lib = fastdec.load()
    if lib is not None:
        offs = np.ascontiguousarray(offs, dtype=np.int64)
        masks = np.ascontiguousarray(masks, dtype=np.uint8)
        scratch = np.zeros(max(1, needed) * pk.blocks_per_mcu, dtype=np.int32)
        p0 = np.ascontiguousarray(preds0, dtype=np.int32)
        lib.eval_flips(
            _ptr(buf), len(buf), start_byte, start_bit, end, pk.c_comps, pk.ncomp,
            _ptr(pk.fast), _ptr(pk.maxcode), _ptr(pk.valptr), _ptr(pk.mincode), _ptr(pk.symbols),
            _ptr(p0), needed, stage1_n, marker_pos, min_gain,
            _ptr(offs), _ptr(masks), ncand,
            _ptr(up_v), _ptr(dn_v), _ptr(left_k), _ptr(w),
            _ptr(res_dec), _ptr(res_complete), _ptr(res_pad), _ptr(res_rough), _ptr(scratch),
        )
        return res_dec, res_complete.astype(bool), res_pad.astype(bool), res_rough
    length = len(buf)
    buf = memoryview(buf).cast("B")  # int indexing (numpy scalars would overflow in BitReader)
    for i in range(ncand):
        off, m = int(offs[i]), int(masks[i])
        buf[off] ^= m
        lim = min(stage1_n, needed)
        p = [int(x) for x in preds0]
        r = py_run_decode(buf, start_byte, start_bit, end, pk, lim, p, want_bits=False)
        if r[1] == 0 and r[0] == lim and lim < needed:
            p = [int(x) for x in preds0]
            r = py_run_decode(buf, start_byte, start_bit, end, pk, needed, p, want_bits=False)
        n, err = r[0], r[1]
        complete = pad_ok = False
        if err == 0 and n == needed:
            pos, pad_ok = py_aligned_end(buf, length, r[5], r[6])
            complete = marker_pos < 0 or pos == marker_pos
        buf[off] ^= m
        res_dec[i] = n
        res_complete[i] = complete
        res_pad[i] = pad_ok
        if complete:
            nb = n * pk.blocks_per_mcu
        else:
            nb = min(max(n - min_gain, 1), n) * pk.blocks_per_mcu
        res_rough[i] = py_roughness(r[8], nb, up_v, dn_v, left_k, w) if nb > 0 else math.inf
    return res_dec, res_complete.astype(bool), res_pad.astype(bool), res_rough
