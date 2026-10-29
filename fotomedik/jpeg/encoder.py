"""Minimal baseline JPEG entropy *encoder* — only what hole filling needs.

We never re-encode real image data (it is copied bit for bit). We only synthesise
blocks for holes with the file's own Huffman tables: DC-only blocks (flat colour,
EOB right after DC), or full blocks carrying a smooth surface (encode_blocks), so
that a damaged restart interval can be replaced by bytes every decoder accepts.
"""

from __future__ import annotations

import numpy as np

from . import fastdec
from .tables import FrameTables, HuffmanTable


class BitWriter:
    """Entropy-coded output with FF00 stuffing. Uses the compiled kernel when present
    (CBitWriter), the Python reference otherwise."""

    def __new__(cls, *a, **kw):
        if cls is BitWriter and fastdec.load() is not None:
            return super().__new__(CBitWriter)
        return super().__new__(cls)

    def __init__(self):
        self._out = bytearray()
        self._acc = 0
        self._n = 0

    def copy_bits(self, data: bytes, start_bit: int, end_bit: int) -> None:
        """Append entropy-coded bits [start_bit, end_bit) of data (positions in the
        stuffed stream, as the decoder reports them), unstuffed."""
        p = start_bit // 8
        first = True
        n_data = len(data)
        while p * 8 < end_bit and p < n_data:
            b = data[p]
            lo_bit = start_bit % 8 if first else 0
            hi_bit = 8 if (p + 1) * 8 <= end_bit else end_bit - p * 8
            n = hi_bit - lo_bit
            if n > 0:
                self.put_bits((b >> (8 - hi_bit)) & ((1 << n) - 1), n)
            first = False
            if b == 0xFF and p + 1 < n_data and data[p + 1] == 0x00:
                p += 2
            else:
                p += 1

    def put_bits(self, value: int, n: int) -> None:
        if n == 0:
            return
        self._acc = (self._acc << n) | (value & ((1 << n) - 1))
        self._n += n
        while self._n >= 8:
            self._n -= 8
            b = (self._acc >> self._n) & 0xFF
            self._out.append(b)
            if b == 0xFF:
                self._out.append(0x00)
        self._acc &= (1 << self._n) - 1 if self._n else 0

    def finish(self) -> bytes:
        """Pad the last byte with 1-bits (as the standard prescribes) and return everything."""
        if self._n:
            pad = 8 - self._n
            self.put_bits((1 << pad) - 1, pad)
        return bytes(self._out)


class CBitWriter(BitWriter):
    """BitWriter backed by the C kernel: state lives in a numpy buffer."""

    def __init__(self):
        self._lib = fastdec.load()
        self._buf = np.empty(1 << 16, dtype=np.uint8)
        self._st = np.zeros(3, dtype=np.int64)  # bytes written, accumulator, bits in it

    def _reserve(self, extra: int) -> None:
        need = int(self._st[0]) + extra + 16
        if need > len(self._buf):
            nb = np.empty(max(need, 2 * len(self._buf)), dtype=np.uint8)
            nb[: int(self._st[0])] = self._buf[: int(self._st[0])]
            self._buf = nb

    def _call(self, fn, *args) -> int:
        return fn(self._buf.ctypes.data, len(self._buf), self._st.ctypes.data, *args)

    def put_bits(self, value: int, n: int) -> None:
        if n == 0:
            return
        self._reserve(8)
        if self._call(self._lib.bw_put, value & ((1 << n) - 1), n):
            raise RuntimeError("bit writer overflow")

    def copy_bits(self, data: bytes, start_bit: int, end_bit: int) -> None:
        from .kernel import _data_ptr

        self._reserve(2 * ((end_bit - start_bit) // 8 + 2))
        arr, ptr = _data_ptr(data)
        if self._call(self._lib.bw_copy, ptr, len(arr), start_bit, end_bit):
            raise RuntimeError("bit writer overflow")

    def finish(self) -> bytes:
        self._reserve(4)
        self._call(self._lib.bw_finish)
        return self._buf[: int(self._st[0])].tobytes()


def codes_for(t: HuffmanTable) -> dict[int, tuple[int, int]]:
    """symbol -> (length, code)"""
    return {sym: (ln, code) for (ln, code), sym in t.lookup.items()}


def _category(v: int) -> int:
    return abs(v).bit_length()


def _magnitude_bits(v: int, s: int) -> int:
    return v if v >= 0 else (v + (1 << s) - 1)


def _best_category(codes: dict[int, tuple[int, int]], want: int) -> int:
    """Largest encodable DC category <= want (tables may omit rare categories)."""
    while want > 0 and want not in codes:
        want -= 1
    if want == 0 and 0 not in codes:
        raise ValueError("DC table has no category-0 code")
    return want


def encode_flat_mcus(
    ft: FrameTables,
    mcu_count: int,
    targets: dict[int, int],
    preds: dict[int, int],
    per_block: dict[int, list[int]] | None = None,
    writer: "BitWriter | None" = None,
) -> bytes:
    """Entropy data for `mcu_count` MCUs of DC-only blocks.

    targets: component id -> DC coefficient (quantised units) for every block,
    unless per_block[cid] gives one value per block (consumed in scan order).
    preds: DC predictor state at the start (mutated to the state at the end),
    so the bytes can be glued after existing data without a restart marker.
    With `writer`, bits are appended to it unpadded (more data may follow) and b"" is returned.
    """
    w = writer if writer is not None else BitWriter()
    if isinstance(w, CBitWriter):
        _encode_flat_c(ft, mcu_count, targets, preds, per_block, w)
        return b"" if writer is not None else w.finish()
    frame, scan = ft.frame, ft.scan
    interleaved = len(scan.components) > 1
    if per_block:
        per_block = {cid: [int(x) for x in seq] for cid, seq in per_block.items()}
    comps = []
    for sc in scan.components:
        fc = frame.component(sc.id)
        dc_codes = codes_for(ft.huff[(0, sc.td)])
        ac_codes = codes_for(ft.huff[(1, sc.ta)])
        if 0x00 not in ac_codes:
            raise ValueError("AC table has no EOB code")
        h, v = (fc.h, fc.v) if interleaved else (1, 1)
        comps.append((sc.id, h, v, dc_codes, ac_codes[0x00]))
    cursors = {cid: 0 for cid, *_ in comps}
    for _ in range(mcu_count):
        for cid, h, v, dc_codes, eob in comps:
            for _b in range(h * v):
                if per_block and cid in per_block:
                    seq = per_block[cid]
                    tgt = seq[min(cursors[cid], len(seq) - 1)]
                    cursors[cid] += 1
                else:
                    tgt = targets.get(cid, 0)
                diff = tgt - preds[cid]
                s = _best_category(dc_codes, _category(diff))
                if s < _category(diff):
                    # clamp the step to what the table can express
                    lim = (1 << s) - 1
                    diff = max(-lim, min(lim, diff))
                ln, code = dc_codes[s]
                w.put_bits(code, ln)
                if s:
                    w.put_bits(_magnitude_bits(diff, s), s)
                preds[cid] += diff
                ln, code = eob
                w.put_bits(code, ln)
    return b"" if writer is not None else w.finish()


def _encode_flat_c(ft: FrameTables, mcu_count: int, targets: dict, preds: dict, per_block: dict | None,
                   w: CBitWriter) -> None:
    from .kernel import packed_for

    pk = packed_for(ft)
    dc_len, dc_code, eob_len, eob_code = pk.enc_tables(ft)
    if (eob_len == 0).any():
        raise ValueError("AC table has no EOB code")
    cols = []
    for cid, (h, v) in zip(pk.cids, pk.hv):
        nb = h * v
        need = mcu_count * nb
        if per_block and cid in per_block:
            seq = np.asarray(per_block[cid], dtype=np.int32)
            if len(seq) < need:  # the Python encoder repeats the last value
                seq = np.concatenate([seq, np.full(need - len(seq), seq[-1] if len(seq) else 0, dtype=np.int32)])
            seq = seq[:need]
        else:
            seq = np.full(need, int(targets.get(cid, 0)), dtype=np.int32)
        cols.append(seq.reshape(mcu_count, nb))
    tg = np.ascontiguousarray(np.concatenate(cols, axis=1).reshape(-1), dtype=np.int32)
    pr = np.array([int(preds[cid]) for cid in pk.cids], dtype=np.int32)
    w._reserve(len(tg) * 12)
    rc = w._call(w._lib.encode_dc_fill, pk.c_comps, pk.ncomp, dc_len.ctypes.data, dc_code.ctypes.data,
                 eob_len.ctypes.data, eob_code.ctypes.data, mcu_count, tg.ctypes.data, pr.ctypes.data)
    if rc == -2:
        raise ValueError("DC table has no category-0 code")
    if rc:
        raise RuntimeError("bit writer overflow")
    for i, cid in enumerate(pk.cids):
        preds[cid] = int(pr[i])


def encode_blocks(ft: FrameTables, coefs: np.ndarray, preds: dict[int, int], writer: BitWriter) -> None:
    """Append full blocks (DC + AC) for len(coefs) MCUs to `writer`, unpadded.

    coefs: int[n_mcu, blocks_per_mcu, 64] in zigzag order, DC absolute (quantised).
    preds: DC predictor state at the start, updated to the state at the end. A
    coefficient whose run/size symbol the AC table lacks is shrunk or dropped."""
    from .kernel import packed_for

    pk = packed_for(ft)
    dc_len, dc_code, _eob_len, _eob_code = pk.enc_tables(ft)
    ac_len, ac_code = pk.ac_tables(ft)
    n = len(coefs)
    flat = np.ascontiguousarray(np.asarray(coefs).reshape(-1), dtype=np.int32)
    if flat.size != n * pk.blocks_per_mcu * 64:
        raise ValueError("coefficient array does not match the scan layout")
    if isinstance(writer, CBitWriter):
        pr = np.array([int(preds[cid]) for cid in pk.cids], dtype=np.int32)
        writer._reserve(flat.size * 4 + 64)
        rc = writer._call(writer._lib.encode_blocks, pk.c_comps, pk.ncomp, dc_len.ctypes.data, dc_code.ctypes.data,
                          ac_len.ctypes.data, ac_code.ctypes.data, n, flat.ctypes.data, pr.ctypes.data)
        if rc == -1:
            raise RuntimeError("bit writer overflow")
        if rc == -2:
            raise ValueError("DC table has no category-0 code")
        if rc == -3:
            raise ValueError("AC table has no EOB or ZRL code")
        for i, cid in enumerate(pk.cids):
            preds[cid] = int(pr[i])
        return
    _py_encode_blocks(pk, dc_len, dc_code, ac_len, ac_code, flat.tolist(), n, preds, writer)


def _py_encode_blocks(pk, dc_len, dc_code, ac_len, ac_code, flat: list, n: int, preds: dict, w: BitWriter) -> None:
    """Python mirror of encode_blocks in _fastdec.c (statement by statement)."""
    k = 0
    for _m in range(n):
        for c, (cid, (h, v)) in enumerate(zip(pk.cids, pk.hv)):
            ln, cd = dc_len[c], dc_code[c]
            al, ac = ac_len[c], ac_code[c]
            if al[0x00] == 0 or al[0xF0] == 0:
                raise ValueError("AC table has no EOB or ZRL code")
            for _b in range(h * v):
                blk = flat[k * 64 : k * 64 + 64]
                k += 1
                diff = blk[0] - preds[cid]
                want = abs(diff).bit_length()
                s = want
                while s > 0 and ln[s] == 0:
                    s -= 1
                if s == 0 and ln[0] == 0:
                    raise ValueError("DC table has no category-0 code")
                if s < want:
                    lim = (1 << s) - 1
                    diff = max(-lim, min(lim, diff))
                w.put_bits(int(cd[s]), int(ln[s]))
                if s:
                    w.put_bits(_magnitude_bits(diff, s), s)
                preds[cid] += diff
                run = 0
                for z in range(1, 64):
                    val = blk[z]
                    if val == 0:
                        run += 1
                        continue
                    sz = min(abs(val).bit_length(), 10)
                    while run > 15:
                        w.put_bits(int(ac[0xF0]), int(al[0xF0]))
                        run -= 16
                    sym = (run << 4) | sz
                    while sz > 0 and al[sym] == 0:
                        sz -= 1
                        sym = (run << 4) | sz
                    if sz == 0:
                        run += 1
                        continue
                    lim = (1 << sz) - 1
                    val = max(-lim, min(lim, val))
                    w.put_bits(int(ac[sym]), int(al[sym]))
                    w.put_bits(_magnitude_bits(val, sz), sz)
                    run = 0
                if run > 0:
                    w.put_bits(int(ac[0x00]), int(al[0x00]))


def gray_dc(ft: FrameTables, cid: int, sample: int) -> int:
    """DC coefficient (quantised) that renders a flat block at `sample` (0..255)."""
    fc = ft.frame.component(cid)
    q = ft.quant[fc.tq].dc
    return round((sample - 128) * 8 / q)
