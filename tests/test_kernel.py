"""The C kernel and the pure-Python reference must agree exactly."""

import numpy as np
import pytest

from fotomedik.jpeg import decoder as D
from fotomedik.jpeg import fastdec, kernel
from fotomedik.jpeg import markers as M
from fotomedik.jpeg.tables import FrameTables
from synth import flip_bits, make_image, make_jpeg

needs_c = pytest.mark.skipif(not fastdec.available(), reason="compiled kernel not built")


def _setup(data):
    res = M.parse_segments(data)
    ft = FrameTables.from_parse(res)
    return res, ft, kernel.packed_for(ft)


def _both(fn):
    """Run fn() with the C library, then with the Python reference."""
    c = fn()
    saved = fastdec._LIB, fastdec._TRIED
    fastdec._LIB, fastdec._TRIED = None, True
    try:
        py = fn()
    finally:
        fastdec._LIB, fastdec._TRIED = saved
    return c, py


@needs_c
@pytest.mark.parametrize("rst", [None, 1])
def test_decode_mcus_parity_on_damage(rst):
    data = make_jpeg(make_image(), restart_rows=rst)
    res, ft, pk = _setup(data)
    rng = np.random.default_rng(1)
    for _ in range(60):
        off = int(rng.integers(res.scan_start, res.eoi_offset))
        broken = flip_bits(data, off, 1 << int(rng.integers(0, 8)))
        start = int(rng.integers(res.scan_start, off))
        bit = int(rng.integers(0, 8))

        def run():
            preds = np.zeros(pk.ncomp, dtype=np.int32)
            out = kernel.decode_mcus(broken, start, bit, len(broken), pk, preds, 40)
            return out[:7], out[7].tolist(), out[8].tolist(), preds.tolist()

        c, py = _both(run)
        assert c == py


@needs_c
def test_eval_flips_parity():
    data = make_jpeg(make_image(), restart_rows=1)
    res, ft, pk = _setup(data)
    ref = D.decode_scan(data, res, ft)
    iv = ref.intervals[4]
    broken = bytearray(flip_bits(data, (iv.byte_start + iv.byte_end) // 2, 0x10))
    needed = iv.mcu_count
    nb = needed * pk.blocks_per_mcu
    rng = np.random.default_rng(2)
    geom = (
        np.where(rng.random(nb) < 0.5, rng.normal(0, 50, nb), np.nan),
        np.where(rng.random(nb) < 0.5, rng.normal(0, 50, nb), np.nan),
        np.array([k - 1 if k % 3 else -1 for k in range(nb)], dtype=np.int32),
        np.full(nb, 0.5),
    )
    offs = np.repeat(np.arange(iv.byte_start + 1, iv.byte_end - 1), 8).astype(np.int64)
    masks = np.tile(np.array([1 << b for b in range(8)], dtype=np.uint8), len(offs) // 8)
    arr = np.frombuffer(broken, dtype=np.uint8)
    before = bytes(broken)

    def run():
        out = kernel.eval_flips(arr, iv.byte_start, 0, iv.byte_end + 2, pk, np.zeros(pk.ncomp, np.int32),
                                needed, 4, iv.byte_end, 3, offs, masks, geom)
        return [o.tolist() for o in out]

    c, py = _both(run)
    assert bytes(broken) == before  # every flip was undone
    assert c[0] == py[0] and c[1] == py[1] and c[2] == py[2]
    assert np.allclose(c[3], py[3], equal_nan=True)
    assert any(c[1])  # the true repair is among the candidates and is complete


@needs_c
def test_tail_flow_parity_and_recovery():
    """Rot the first half of interval 6: the tail must be found, identically in C and Python."""
    data = make_jpeg(make_image(512, 256), restart_rows=1)
    res, ft, pk = _setup(data)
    ref = D.decode_scan(data, res, ft)
    iv = ref.intervals[6]
    mid = (iv.byte_start + iv.byte_end) // 2
    rng = np.random.default_rng(5)
    buf = bytearray(data)
    for o in range(iv.byte_start, mid):
        if rng.random() < 0.08:
            buf[o] ^= 1 << int(rng.integers(0, 8))
    broken = bytes(buf)

    def run():
        return kernel.tail_flow(broken, iv.byte_start, iv.byte_end, pk, iv.mcu_count)

    c, py = _both(run)
    assert c == py
    bits, flows = c
    n = len(bits)
    true_starts = [m.bit_start for m in ref.mcus if iv.mcu_first <= m.index < iv.mcu_first + iv.mcu_count]
    # after a few junk MCUs (a parse that met the true one late) the tail's boundaries are
    # exactly the true ones, and that true part is a good share of the interval
    k = next(i for i in range(n) if bits[i:] == true_starts[-(n - i):])
    assert k <= 6
    assert n - k >= iv.mcu_count // 3
    assert flows[-1] == max(flows)  # every start flows into the last MCU


@needs_c
def test_encoder_and_bit_copy_parity():
    """DC-only fill and bit copying give the same bytes in C and Python."""
    from fotomedik.jpeg.encoder import BitWriter, encode_flat_mcus

    data = make_jpeg(make_image(), restart_rows=1)
    res, ft, pk = _setup(data)
    rng = np.random.default_rng(3)
    count = 37
    per_block = {cid: rng.integers(-120, 120, count * h * v) for cid, (h, v) in zip(pk.cids, pk.hv)}

    def run():
        w = BitWriter()
        w.put_bits(0b101, 3)
        preds = {cid: 5 for cid in pk.cids}
        encode_flat_mcus(ft, count, {}, preds, per_block, writer=w)
        w.copy_bits(data, res.scan_start * 8 + 3, res.scan_start * 8 + 3 + 4000)
        return w.finish(), preds

    (c_bytes, c_preds), (py_bytes, py_preds) = _both(run)
    assert c_bytes == py_bytes and c_preds == py_preds


@needs_c
def test_find_interval_start_parity():
    data = make_jpeg(make_image(512, 256), restart_rows=1)
    res, ft, pk = _setup(data)
    ref = D.decode_scan(data, res, ft)
    iv = ref.intervals[5]
    known = [m.bit_start for m in ref.mcus if iv.mcu_first + 5 <= m.index < iv.mcu_first + iv.mcu_count]

    def run():
        return kernel.find_interval_start(data, iv.byte_start - 40, iv.byte_start + 40, iv.byte_end, pk,
                                          iv.mcu_count, known, 0xD0 + 4)

    c, py = _both(run)
    assert c == py == iv.byte_start


def test_parallel_decode_matches_serial():
    from fotomedik import parallel

    data = make_jpeg(make_image(512, 256), restart_rows=1)
    broken = flip_bits(data, len(data) // 2, 0x10)
    res, ft, _pk = _setup(broken)
    out = []
    for t in (1, 4):
        parallel.set_threads(t)
        D._full_cache.clear()
        try:
            out.append(D.decode_scan(broken, res, ft))
        finally:
            parallel.set_threads(None)
    a, b = out
    assert [(i.ok, i.decoded, i.byte_start, i.byte_end) for i in a.intervals] == \
        [(i.ok, i.decoded, i.byte_start, i.byte_end) for i in b.intervals]
    assert (a.mcu_ok == b.mcu_ok).all()
    for cid in a.dc:
        assert np.array_equal(a.dc[cid], b.dc[cid], equal_nan=True)
