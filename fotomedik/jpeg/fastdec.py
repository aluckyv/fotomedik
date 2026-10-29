"""ctypes bridge to the C decoding kernel (_fastdec.c). Optional: the pure-Python
decoder is used when the compiled library is missing or FOTOMEDIK_PURE=1 is set."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from .tables import FrameTables, HuffmanTable

_HERE = Path(__file__).parent
_LIB = None
_TRIED = False


class _Comp(ctypes.Structure):
    _fields_ = [("h", ctypes.c_int32), ("v", ctypes.c_int32), ("dc_t", ctypes.c_int32),
                ("ac_t", ctypes.c_int32), ("qdc", ctypes.c_int32), ("cid", ctypes.c_int32)]


def _candidates() -> list[Path]:
    names = ["_fastdec.dll", "_fastdec.pyd"] if sys.platform == "win32" else ["_fastdec.so", "_fastdec.dylib"]
    return [_HERE / n for n in names]


def load() -> ctypes.CDLL | None:
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    if os.environ.get("FOTOMEDIK_PURE"):
        return None
    for p in _candidates():
        if p.exists():
            try:
                lib = ctypes.CDLL(str(p))
            except OSError:
                continue
            f = lib.decode_mcus
            f.restype = ctypes.c_int64
            f.argtypes = [
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int64,
                ctypes.POINTER(_Comp), ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ]
            _LIB = lib
            break
    return _LIB


def available() -> bool:
    return load() is not None


# --------------------------------------------------------------------------- table packing

_FAST_BITS = 9


def _pack_table(t: HuffmanTable, fast: np.ndarray, maxcode: np.ndarray, valptr: np.ndarray,
                mincode: np.ndarray, symbols: np.ndarray, slot: int) -> None:
    for (length, code), sym in t.lookup.items():
        if length <= _FAST_BITS:
            shift = _FAST_BITS - length
            base = code << shift
            fast[slot, base : base + (1 << shift)] = (length << 8) | sym
    maxcode[slot, :] = t.maxcode
    valptr[slot, :] = t.valptr
    mincode[slot, :] = t.mincode
    symbols[slot, : len(t.symbols)] = list(t.symbols)


class Packed:
    """Huffman tables and component layout in the kernel's memory layout, cached per FrameTables."""

    def __init__(self, ft: FrameTables):
        self.fast = np.zeros((8, 512), dtype=np.int32)
        self.maxcode = np.full((8, 17), -1, dtype=np.int32)
        self.valptr = np.zeros((8, 17), dtype=np.int32)
        self.mincode = np.zeros((8, 17), dtype=np.int32)
        self.symbols = np.zeros((8, 256), dtype=np.uint8)
        for (cls, tid), t in ft.huff.items():
            if tid < 4:
                _pack_table(t, self.fast, self.maxcode, self.valptr, self.mincode, self.symbols, cls * 4 + tid)
        frame, scan = ft.frame, ft.scan
        interleaved = len(scan.components) > 1
        self.comps = (_Comp * len(scan.components))()
        self.cids: list[int] = []
        self.blocks_per_mcu = 0
        for i, sc in enumerate(scan.components):
            fc = frame.component(sc.id)
            h, v = (fc.h, fc.v) if interleaved else (1, 1)
            self.comps[i] = _Comp(h, v, sc.td, sc.ta, ft.quant[fc.tq].dc, sc.id)
            self.cids.append(sc.id)
            self.blocks_per_mcu += h * v


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


ERR_NAMES = {0: None, 1: "bad_huffman", 2: "coef_overflow", 3: "dc_range", 4: "marker", 5: "eof"}


def decode_mcus(arr: np.ndarray, start_byte: int, start_bit: int, end: int, pk: Packed,
                preds: np.ndarray, max_mcus: int):
    """Run the kernel. Returns (n, err_name, err_byte, err_bit, marker, end_byte, end_bit, bitstarts, dc_vals)."""
    lib = load()
    assert lib is not None
    bitstarts = np.zeros(max(1, max_mcus), dtype=np.int64)
    dc_vals = np.zeros(max(1, max_mcus) * pk.blocks_per_mcu, dtype=np.int32)
    out = np.zeros(6, dtype=np.int64)
    n = lib.decode_mcus(
        arr.ctypes.data, start_byte, start_bit, end,
        pk.comps, len(pk.cids),
        pk.fast.ctypes.data, pk.maxcode.ctypes.data, pk.valptr.ctypes.data, pk.mincode.ctypes.data, pk.symbols.ctypes.data,
        max_mcus, preds.ctypes.data, bitstarts.ctypes.data, dc_vals.ctypes.data, out.ctypes.data,
    )
    return (int(n), ERR_NAMES[int(out[0])], int(out[1]), int(out[2]), int(out[3]), int(out[4]), int(out[5]),
            bitstarts[:n], dc_vals[: n * pk.blocks_per_mcu].reshape(n, pk.blocks_per_mcu) if n else dc_vals[:0])
