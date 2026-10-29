"""ctypes loader for the compiled kernel (_fastdec.c). Use fotomedik.jpeg.kernel, not this module.

The library is optional: without it (or with FOTOMEDIK_PURE=1) kernel.py runs
the pure-Python reference implementation.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent
ABI = 3  # must match FASTDEC_ABI in _fastdec.c
_LIB = None
_TRIED = False

_P = ctypes.c_void_p
_I64 = ctypes.c_int64
_INT = ctypes.c_int


class Comp(ctypes.Structure):
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
        if not p.exists():
            continue
        try:
            lib = ctypes.CDLL(str(p))
        except OSError:
            continue
        try:
            lib.fastdec_abi.restype = ctypes.c_int
            if lib.fastdec_abi() != ABI:
                continue  # stale library from an older build: use the Python reference
            f = lib.decode_mcus
            f.restype = _I64
            f.argtypes = [_P, _I64, _INT, _I64, ctypes.POINTER(Comp), _INT,
                          _P, _P, _P, _P, _P, _I64, _P, _P, _P, _P]
            g = lib.eval_flips
            g.restype = _I64
            g.argtypes = [_P, _I64, _I64, _INT, _I64, ctypes.POINTER(Comp), _INT,
                          _P, _P, _P, _P, _P,
                          _P, _I64, _I64, _I64, _INT, _INT,
                          _P, _I64, _P,
                          _P, _P, _I64,
                          _P, _P, _P, _P,
                          _P, _P, _P, _P, _P]
        except AttributeError:
            continue  # stale library from an older build: fall back to Python
        _LIB = lib
        break
    return _LIB


def available() -> bool:
    return load() is not None


def reset() -> None:
    """Forget the loaded library (tests toggle FOTOMEDIK_PURE)."""
    global _LIB, _TRIED
    _LIB = None
    _TRIED = False
