"""Junk blocks (R14): decoded garbage is found and refilled, real blocks are left alone."""

import io

import numpy as np
from PIL import Image

from fotomedik.jpeg import decoder as D
from fotomedik.jpeg import kernel as K
from fotomedik.jpeg import markers as M
from fotomedik.jpeg.encoder import BitWriter, encode_blocks
from fotomedik.jpeg.tables import FrameTables
from fotomedik.jpeg.writer import assemble, rst_marker
from fotomedik.recipes import r14_junk as R
from fotomedik.recipes.base import Context, Hole
from fotomedik.recipes.r12_smooth import SmoothHoles, component_planes
from fotomedik.recipes.r14_junk import JunkBlocks
from synth import make_image, make_jpeg


def _decode(data):
    parse = M.parse_segments(data)
    ft = FrameTables.from_parse(parse)
    return parse, ft, D.decode_scan(data, parse, ft)


def junked(data: bytes, first: int, count: int, seed: int = 3) -> bytes:
    """`data` with MCUs first..first+count re-encoded as garbage: random AC coefficients
    (the DC values stay, so the chain and everything after decode as before)."""
    parse, ft, sr = _decode(data)
    comps, per_row, _rows = D._scan_components(ft)
    pk = K.packed_for(ft)
    bit_start = {mi.index: mi.bit_start for mi in sr.mcus}
    rng = np.random.default_rng(seed)
    pieces = []
    n_iv = len(sr.intervals)
    for k, iv in enumerate(sr.intervals):
        a, b = iv.mcu_first, iv.mcu_first + iv.mcu_count
        end_byte = iv.byte_end if iv.byte_end is not None else len(data) - 2
        if not (a <= first and first + count <= b):
            pieces.append(data[iv.byte_start : end_byte])
        else:
            last = bit_start[b - 1]
            got = K.decode_mcus(data, last // 8, last % 8, end_byte, pk, np.zeros(pk.ncomp, dtype=np.int32), 1, check_dc=False)
            end_bit = got[5] * 8 + got[6]
            w = BitWriter()
            w.copy_bits(data, iv.byte_start * 8, bit_start[first])
            coefs = np.zeros((count, pk.blocks_per_mcu, 64), dtype=np.int32)
            preds = {}
            col = 0
            for c in comps:
                for m in range(count):
                    row, cc = divmod(first + m, per_row)
                    for i, (yy, xx) in enumerate((yy, xx) for yy in range(c.v) for xx in range(c.h)):
                        coefs[m, col + i, 0] = int(sr.dc[c.cid][row * c.v + yy, cc * c.h + xx])
                        pos = rng.choice(np.arange(1, 64), 12, replace=False)
                        coefs[m, col + i, pos] = rng.integers(-30, 31, 12)
                if first == a:
                    preds[c.cid] = 0
                else:
                    row, cc = divmod(first - 1, per_row)
                    preds[c.cid] = int(sr.dc[c.cid][row * c.v + c.v - 1, cc * c.h + c.h - 1])
                col += c.h * c.v
            encode_blocks(ft, coefs, preds, w)
            w.copy_bits(data, bit_start[first + count], end_bit)
            pieces.append(w.finish())
        if ft.restart_interval and k < n_iv - 1:
            pieces.append(rst_marker(iv.index))
    return assemble(data[: parse.scan_start], b"".join(pieces))


def test_saturated_jumps_need_unsaturated_rows_on_both_sides():
    sat = np.zeros((5, 4))
    level = np.full((5, 4), 20.0)
    sat[2, 1] = 1.0
    level[2, 1] = -128.0  # a black MCU between mid-gray rows
    sat[2, 3] = 1.0
    level[2, 3] = -128.0
    sat[1, 3] = sat[3, 3] = 1.0  # ... and one between black rows (a dark doorway)
    level[1, 3] = level[3, 3] = -128.0
    trusted = np.ones((5, 4), dtype=bool)
    both, either = R.saturated_jumps({"sat": sat, "level": level}, trusted)
    assert both[2, 1] and either[2, 1]
    assert not both[2, 3] and not either[2, 3]
    # at the picture's edge one side suffices
    sat[0, 0] = 1.0
    level[0, 0] = 127.0
    both, _ = R.saturated_jumps({"sat": sat, "level": level}, trusted)
    assert both[0, 0]


def test_junk_runs_rules():
    rows, cols = 1, 40
    trusted = np.ones((rows, cols), dtype=bool)
    trusted[0, 20] = False  # a hole in the middle of the row
    strong = np.zeros_like(trusted)
    weak = np.zeros_like(trusted)
    isolated = np.zeros_like(trusted)
    # a dense stretch: strong at 5 of 9 places, weak around them
    strong[0, [3, 5, 6, 8, 10]] = True
    weak[0, 2:12] = True
    # a tail next to the hole: weak MCUs with a strong one among them
    weak[0, 17:20] = True
    strong[0, 18] = True
    # a head after the hole with weak MCUs only: not junk
    weak[0, 21:24] = True
    # an isolated strong MCU far from everything: not junk; a very strong one: junk
    strong[0, 30] = weak[0, 30] = True
    strong[0, 36] = weak[0, 36] = isolated[0, 36] = True
    out = R.junk_runs(strong, weak, isolated, trusted)
    assert out[0, 2:12].all()
    assert out[0, 17:20].all()
    assert not out[0, 21:24].any()
    assert not out[0, 30]
    assert out[0, 36]
    assert not out[0, 12:17].any() and not out[0, 24:30].any()
    assert not out[0, 20]  # the hole itself is not trusted
    # a run that is junk for the most part keeps its clean head (a kept prefix may end in junk)
    trusted2 = np.ones((1, 20), dtype=bool)
    strong2 = np.zeros_like(trusted2)
    weak2 = np.zeros_like(trusted2)
    strong2[0, 6:20:2] = True
    weak2[0, 6:20] = True
    out2 = R.junk_runs(strong2, weak2, np.zeros_like(trusted2), trusted2)
    assert out2[0, 6:20].all() and not out2[0, :5].any()


def test_junk_blocks_found_and_refilled():
    img = make_image(512, 384, seed=11)
    data = make_jpeg(img, restart_rows=1, quality=92)
    parse, ft, sr = _decode(data)
    comps, per_row, _rows = D._scan_components(ft)
    total = sr.total_mcus
    first, count = 9 * per_row + 5, 12
    bad = junked(data, first, count)
    parse2, ft2, sr2 = _decode(bad)
    assert sr2.error is None and all(iv.ok for iv in sr2.intervals)
    for cid in sr.dc:  # the chain is intact: every DC as before
        assert np.array_equal(sr.dc[cid], sr2.dc[cid])
    ctx = Context(data=bad, name="junk.jpg", options={})
    ctx.holes = [Hole(first + count, 1, "damaged")]  # the garbage ends at a hole, as a spliced piece does
    ctx.recovered_fraction = 1.0 - 1 / total
    assert JunkBlocks().detect(ctx)
    JunkBlocks().fix(ctx, True)
    junk = ctx.memo.get("junk_mcus")
    assert junk is not None and len(junk) == total
    found = junk[first : first + count]
    assert found.sum() >= 0.6 * count, found
    outside = junk.copy()
    outside[first : first + count] = False
    assert outside.sum() == 0, np.flatnonzero(outside)
    assert any(h.reason == "junk" for h in ctx.holes)
    assert ctx.recovered_fraction < 1.0 - 1 / total
    assert ctx.memo.get("hole_dc") is not None
    assert "junk_blocks" in [e.recipe for e in ctx.log]
    # smooth_holes writes the junk anew
    SmoothHoles().fix(ctx, True)
    smooth = [e for e in ctx.log if e.recipe == "smooth_holes"]
    assert smooth and "junk MCU(s) refilled" in smooth[-1].message
    Image.open(io.BytesIO(ctx.data)).load()
    _p3, _f3, sr3 = _decode(ctx.data)
    assert sr3.error is None and all(iv.ok for iv in sr3.intervals)
    # the garbage is gone: the refilled MCUs are close to the original picture again
    planes_orig = component_planes(data, ft, comps)
    planes_fixed = component_planes(ctx.data, ft2, comps)
    planes_bad = component_planes(bad, ft2, comps)
    y = comps[0]
    r0 = (first // per_row) * y.v * 8
    c0, c1 = (first % per_row) * y.h * 8, ((first + count) % per_row) * y.h * 8
    sl = (slice(r0, r0 + y.v * 8), slice(c0, c1))
    err_bad = np.abs(planes_bad[y.cid][sl] - planes_orig[y.cid][sl]).mean()
    err_fixed = np.abs(planes_fixed[y.cid][sl] - planes_orig[y.cid][sl]).mean()
    assert err_fixed < 0.5 * err_bad, (err_bad, err_fixed)


def test_junk_blocks_leaves_a_clean_picture_alone(healthy_jpeg_rst):
    _parse, _ft, sr = _decode(healthy_jpeg_rst)
    ctx = Context(data=healthy_jpeg_rst, name="ok.jpg", options={})
    ctx.holes = [Hole(sr.total_mcus // 2, 2, "damaged")]
    JunkBlocks().fix(ctx, True)
    assert "junk_mcus" not in ctx.memo
    assert all(h.reason != "junk" for h in ctx.holes)
    notes = [e for e in ctx.log if e.recipe == "junk_blocks"]
    assert notes and "no junk blocks" in notes[0].message
