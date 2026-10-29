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
        # raw table pointers, computed once (getting a pointer per call is slow)
        self.tp = tuple(a.ctypes.data for a in (self.fast, self.maxcode, self.valptr, self.mincode, self.symbols))
        self._enc = None

    def enc_tables(self, ft: FrameTables):
        """Huffman codes for the DC-only encoder: dc_len/dc_code int32[ncomp, 16],
        eob_len/eob_code int32[ncomp] (length 0 = symbol absent)."""
        if self._enc is None:
            dc_len = np.zeros((self.ncomp, 16), dtype=np.int32)
            dc_code = np.zeros((self.ncomp, 16), dtype=np.int32)
            eob_len = np.zeros(self.ncomp, dtype=np.int32)
            eob_code = np.zeros(self.ncomp, dtype=np.int32)
            for i, sc in enumerate(ft.scan.components):
                for (ln, code), sym in ft.huff[(0, sc.td)].lookup.items():
                    if sym < 16:
                        dc_len[i, sym], dc_code[i, sym] = ln, code
                for (ln, code), sym in ft.huff[(1, sc.ta)].lookup.items():
                    if sym == 0:
                        eob_len[i], eob_code[i] = ln, code
            self._enc = (dc_len, dc_code, eob_len, eob_code)
        return self._enc


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


_data_cache: dict[int, tuple] = {}


def _data_ptr(data) -> tuple[np.ndarray, int]:
    """uint8 array over data and its address; cached for immutable bytes."""
    if isinstance(data, bytes):
        hit = _data_cache.get(id(data))
        if hit is not None and hit[0] is data:
            return hit[1], hit[2]
        arr = np.frombuffer(data, dtype=np.uint8)
        if len(_data_cache) >= 16:
            _data_cache.clear()
        ptr = arr.ctypes.data
        _data_cache[id(data)] = (data, arr, ptr)
        return arr, ptr
    arr = as_array(data)
    return arr, arr.ctypes.data


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


def _py_mcu(br: BitReader, pk: Packed, preds: list[int], dc: list[int], check_dc: bool = True) -> None:
    """Decode one MCU, appending its DC values; raises _Bad / MarkerHit / EOFError."""
    for ci, (h, v, dc_t, ac_t, dc_fast, ac_fast, q) in enumerate(pk.py):
        for _ in range(h * v):
            s0 = _py_symbol(br, dc_t, dc_fast)
            if s0 > 11:
                raise _Bad(1)
            diff = br.receive_extend(s0)
            preds[ci] += diff
            dcv = preds[ci]
            if check_dc and abs(dcv * q) > 1024 + q:
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


def py_run_decode(data, start_byte: int, start_bit: int, end: int, pk: Packed, max_mcus: int,
                  preds: list[int], want_bits: bool = True, check_dc: bool = True):
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
            _py_mcu(br, pk, preds, dc, check_dc)
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


def near_marker(b0: int, b1: int, expect: int, max_bits: int = 3) -> bool:
    """True when the byte pair is within max_bits flipped bits of FF <expect>."""
    return bin(b0 ^ 0xFF).count("1") + bin(b1 ^ expect).count("1") <= max_bits


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


def decode_mcus(data, start_byte: int, start_bit: int, end: int, pk: Packed, preds: np.ndarray, max_mcus: int,
                check_dc: bool = True):
    """Decode up to max_mcus MCUs. `preds` (int32[ncomp]) is updated in place.
    check_dc=False skips the DC range test (for data whose predictor is unknown).

    Returns (n, err_name, err_byte, err_bit, marker, end_byte, end_bit, bitstarts[n], dc[n, bpm]).
    """
    lib = fastdec.load()
    bpm = pk.blocks_per_mcu
    if lib is not None:
        _arr, dptr = _data_ptr(data)
        cap = max(1, max_mcus)
        bitstarts = np.empty(cap, dtype=np.int64)
        dc_vals = np.empty(cap * bpm, dtype=np.int32)
        out = np.zeros(6, dtype=np.int64)
        n = int(lib.decode_mcus(
            dptr, start_byte, start_bit, end, pk.c_comps, pk.ncomp,
            *pk.tp,
            max_mcus, _ptr(preds), _ptr(bitstarts), _ptr(dc_vals), _ptr(out), 1 if check_dc else 0,
        ))
        return (n, ERR_NAMES[int(out[0])], int(out[1]), int(out[2]), int(out[3]), int(out[4]), int(out[5]),
                bitstarts[:n], dc_vals[: n * bpm].reshape(n, bpm))
    p = [int(x) for x in preds]
    n, err, eb, ebit, marker, end_b, end_bit, bits, dc = py_run_decode(data, start_byte, start_bit, end, pk, max_mcus, p,
                                                                       check_dc=check_dc)
    preds[:] = p
    return (n, ERR_NAMES[err], eb, ebit, marker, end_b, end_bit,
            np.asarray(bits, dtype=np.int64), np.asarray(dc, dtype=np.int32).reshape(n, bpm))


def eval_flips(buf: np.ndarray, start_byte: int, start_bit: int, end: int, pk: Packed, preds0: np.ndarray,
               needed: int, stage1_n: int, marker_pos: int, min_gain: int,
               offs: np.ndarray, masks: np.ndarray, geom,
               mcu_bits: np.ndarray | None = None, cur_dc: np.ndarray | None = None,
               expect_marker: int = -1,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate bit-flip candidates. buf must be a writable uint8 array (bytes are flipped and restored).

    geom = (up_v, dn_v, left_k, w) per block of the `needed` MCUs (see r5_bitflip.IntervalGeom).
    mcu_bits / cur_dc: bit starts and DC values (scan order) of the MCUs already decoded
    from the resume point; each candidate then restarts at the MCU holding its byte.
    Returns (dec int32, complete bool, pad_ok bool, rough float64), one entry per candidate.
    """
    if mcu_bits is None or cur_dc is None:
        mcu_bits = np.zeros(0, dtype=np.int64)
        cur_dc = np.zeros(0, dtype=np.int32)
    mcu_bits = np.ascontiguousarray(mcu_bits, dtype=np.int64)
    cur_dc = np.ascontiguousarray(cur_dc, dtype=np.int32)
    nknown = len(mcu_bits)
    bpm = pk.blocks_per_mcu
    last_block = np.cumsum([h * v for h, v in pk.hv]) - 1
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
            *pk.tp,
            _ptr(p0), needed, stage1_n, marker_pos, expect_marker, min_gain,
            _ptr(mcu_bits), nknown, _ptr(cur_dc),
            _ptr(offs), _ptr(masks), ncand,
            _ptr(up_v), _ptr(dn_v), _ptr(left_k), _ptr(w),
            _ptr(res_dec), _ptr(res_complete), _ptr(res_pad), _ptr(res_rough), _ptr(scratch),
        )
        return res_dec, res_complete.astype(bool), res_pad.astype(bool), res_rough
    length = len(buf)
    buf = memoryview(buf).cast("B")  # int indexing (numpy scalars would overflow in BitReader)
    for i in range(ncand):
        off, m = int(offs[i]), int(masks[i])
        j = 0
        if nknown and start_byte <= off:
            j = int(np.searchsorted(mcu_bits, off * 8, side="right")) - 1
            j = max(j, 0)
        if j > 0:
            sb, sbit = divmod(int(mcu_bits[j]), 8)
            jpreds = [int(cur_dc[(j - 1) * bpm + lb]) for lb in last_block]
            prefix = [int(x) for x in cur_dc[: j * bpm]]
        else:
            sb, sbit = start_byte, start_bit
            jpreds = [int(x) for x in preds0]
            prefix = []
        buf[off] ^= m
        full = needed - j
        lim = max(1, min(stage1_n, needed) - j)
        p = list(jpreds)
        r = py_run_decode(buf, sb, sbit, end, pk, lim, p, want_bits=False)
        if r[1] == 0 and r[0] == lim and lim < full:
            p = list(jpreds)
            r = py_run_decode(buf, sb, sbit, end, pk, full, p, want_bits=False)
        n, err = r[0], r[1]
        complete = pad_ok = False
        if err == 0 and n == full:
            pos, pad_ok = py_aligned_end(buf, length, r[5], r[6])
            if marker_pos >= 0:
                complete = pos == marker_pos
            elif expect_marker >= 0:
                complete = pos + 1 < length and near_marker(buf[pos], buf[pos + 1], expect_marker)
            else:
                complete = True
        buf[off] ^= m
        n += j
        res_dec[i] = n
        res_complete[i] = complete
        res_pad[i] = pad_ok
        dc = prefix + list(r[8])
        if complete:
            nb = n * bpm
        else:
            nb = min(max(n - min_gain, 1), n) * bpm
        res_rough[i] = py_roughness(dc, nb, up_v, dn_v, left_k, w) if nb > 0 else math.inf
    return res_dec, res_complete.astype(bool), res_pad.astype(bool), res_rough


def py_tail_flow(data, lo: int, marker_pos: int, pk: Packed, step: int, max_nodes: int, cap: int):
    """Reference for the C tail_flow (see _fastdec.c): (mcu_start_bits, flows)."""
    length = len(data)
    nxt: dict[int, int | None] = {}  # position -> next position (None: dead end, -2: ends on the marker)
    starts: dict[int, int] = {}
    full = False
    p = marker_pos - 1
    while p >= lo and not full:
        for ph in (0, 4):
            sbit = p * 8 + ph
            if sbit in nxt:
                starts[sbit] = starts.get(sbit, 0) + 1
                continue
            if len(nxt) >= max_nodes:
                full = True
                break
            nxt[sbit] = None
            starts[sbit] = starts.get(sbit, 0) + 1
            cur = sbit
            br = BitReader(data, p, marker_pos + 2)
            try:
                if ph:
                    br.read_bits(ph)
                while True:
                    _py_mcu(br, pk, [0] * pk.ncomp, [], check_dc=False)
                    eb, ebit = br.tell()
                    a, pad = py_aligned_end(data, length, eb, ebit)
                    if a == marker_pos and pad:
                        nxt[cur] = -2
                        break
                    if eb > marker_pos:
                        break
                    e = eb * 8 + ebit
                    if e in nxt:
                        nxt[cur] = e
                        break
                    if len(nxt) >= max_nodes:
                        full = True
                        break
                    nxt[cur] = e
                    nxt[e] = None
                    cur = e
            except (_Bad, MarkerHit, EOFError):
                pass
        p -= step
    flow = {v: starts.get(v, 0) for v in nxt}
    best: dict[int, int] = {}
    bestf: dict[int, int] = {}
    term = None
    for v in sorted(nxt):
        n = nxt[v]
        if n == -2:
            if term is None or flow[v] > flow[term]:
                term = v
            continue
        if n is None:
            continue
        flow[n] += flow[v]
        if flow[v] > bestf.get(n, 0):
            bestf[n], best[n] = flow[v], v
    if term is None:
        return [], []
    chain = []
    v = term
    while v is not None and len(chain) < cap:
        chain.append(v)
        v = best.get(v)
    chain.reverse()
    return chain, [flow[v] for v in chain]


def _pydata(data):
    return memoryview(as_array(data)).cast("B") if isinstance(data, np.ndarray) else data


TAIL_MAX_NODES = 1 << 20
TAIL_MAX_TRIALS = 1 << 17


def tail_flow(data, lo: int, marker_pos: int, pk: Packed, cap: int, step: int | None = None):
    """MCU start bits of the tail that ends exactly at marker_pos, found by parse-graph
    flow (see _fastdec.c): (mcu_start_bits ascending, flow per start). At most `cap`
    starts, the last being the MCU right before the marker; empty when none ends there."""
    if step is None:
        step = max(8, (marker_pos - lo) * 2 // TAIL_MAX_TRIALS + 1)
    lib = fastdec.load()
    if lib is None:
        return py_tail_flow(_pydata(data), lo, marker_pos, pk, step, TAIL_MAX_NODES, cap)
    arr, dptr = _data_ptr(data)
    bits = np.zeros(max(1, cap), dtype=np.int64)
    flows = np.zeros(max(1, cap), dtype=np.int32)
    trials = 2 * ((marker_pos - lo) // step + 1)
    max_nodes = min(TAIL_MAX_NODES, 16 * trials + 4 * cap + 4096)  # memory sized to the job
    n = int(lib.tail_flow(
        dptr, len(arr), lo, marker_pos, pk.c_comps, pk.ncomp,
        *pk.tp,
        step, max_nodes, cap, _ptr(bits), _ptr(flows),
    ))
    if n < 0:
        raise MemoryError("tail_flow: out of memory")
    return [int(x) for x in bits[:n]], [int(x) for x in flows[:n]]


def py_find_interval_start(data, lo: int, hi: int, mpos: int, pk: Packed, count: int, known: list[int],
                           prev_marker: int, probe: int) -> int:
    """Reference for the C find_interval_start (see _fastdec.c)."""
    rem = {b: len(known) - i for i, b in enumerate(known)}
    best, best_ham = -1, 1 << 20
    length = len(data)
    for s in range(lo, hi):
        p = [0] * pk.ncomp
        n, err, _eb, _ebit, _mk, end_b, end_bit, bits, _dc = py_run_decode(data, s, 0, mpos + 2, pk, probe, p)
        total = None
        for i in range(1, len(bits)):
            if bits[i] in rem:
                total = i + rem[bits[i]]
                break
        if total is None and err == 0:
            e = end_b * 8 + end_bit
            if e in rem:
                total = n + rem[e]
        if total != count:
            continue
        p = [0] * pk.ncomp
        n, err, _eb, _ebit, _mk, end_b, end_bit, _bits, _dc = py_run_decode(data, s, 0, mpos + 2, pk, count, p,
                                                                            want_bits=False)
        if n != count or err:
            continue
        a, pad = py_aligned_end(data, length, end_b, end_bit)
        if a != mpos or not pad:
            continue
        ham = 16
        if s >= 2 and prev_marker >= 0:
            ham = bin(data[s - 2] ^ 0xFF).count("1") + bin(data[s - 1] ^ prev_marker).count("1")
        if ham < best_ham:
            best, best_ham = s, ham
    return best


def find_interval_start(data, lo: int, hi: int, mpos: int, pk: Packed, count: int, known: list[int],
                        prev_marker: int | None = None, probe: int = 8) -> int | None:
    """Byte offset where a whole interval of `count` MCUs ending at the marker at mpos
    starts, or None. Each candidate offset in [lo, hi) is decoded a few MCUs (fresh
    predictors) until it lands on the reference parse `known` (MCU start bits of the
    tail), which fixes its MCU count; those with the right count must then decode in
    full and end exactly on the marker. Junk starting inside the first true MCU can
    pass too, so the one preceded by the most marker-like byte pair (the previous
    interval's damaged RST) wins, then the earliest."""
    if not known or hi <= lo:
        return None
    pm = -1 if prev_marker is None else prev_marker
    lib = fastdec.load()
    if lib is None:
        s = py_find_interval_start(_pydata(data), lo, hi, mpos, pk, count, list(known), pm, probe)
    else:
        arr, dptr = _data_ptr(data)
        kn = np.asarray(known, dtype=np.int64)
        s = int(lib.find_interval_start(dptr, len(arr), lo, hi, mpos, pk.c_comps, pk.ncomp, *pk.tp,
                                        count, _ptr(kn), len(kn), pm, probe))
    return None if s < 0 else s
