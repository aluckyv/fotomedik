"""R11 — put intact stretches from the middle of damaged data back in place.

Between damaged spots the data is often intact for kilobytes, but after the first
damage nothing tells where it belongs: without restart markers (or with several
damaged spots inside one interval) the MCU count is lost, and the DC predictor
with it. Huffman decoding resynchronises by itself, so the MCUs of an intact
stretch are found as the parse that decoding trials from many start points agree
on. What is missing is their position and their DC offset.

Both come from the picture: the reference field (the Exif thumbnail mapped onto
the block grid, corrected by the nearest clean rows - see dcfield) says what DC
every MCU should roughly have. The chain is cut into blocks of a few dozen MCUs;
each block slides along the expected stretch of the image, with its own offset,
and is placed where it fits clearly best. Later blocks of a chain must continue
where the previous one ended (a desync inside the chain shows up as a block that
fits nowhere and is left out). Placed runs become pieces that R9 splices
bit-exactly, like the tails of R10.
"""

from __future__ import annotations

import numpy as np

from .. import parallel
from ..jpeg import decoder as D
from ..jpeg import kernel as K
from .base import Context, Recipe
from .dcfield import block_mask, build_field, thumbnail_base
from .r9_holes import decode_for
from .registry import register

BLOCK = 32  # MCUs placed together
MIN_SEG = 512  # bytes: shorter intact stretches are not worth it
PROBE = 256  # bytes searched for a converged start at the beginning of a stretch
ACCEPT = 3.0  # mean squared misfit per component (in units of the reference's expected error)
ACCEPT_ROW = 12.0  # inside one restart interval (a short search) a clearly best place may fit worse
DISTINCT = 0.6  # a first placement must beat every other position (more than 3 MCUs away) by this factor
FIRST_WINDOW = 0.25  # search +-25 % of the MCU distance to the nearest anchor (at least MIN_WINDOW)
MIN_WINDOW = 48
MAX_WINDOW = 3000
SHIFT = 6  # MCUs a block may move against its predecessor (a lost or extra MCU after a flipped bit)
SHORT = 10  # with restart markers the search stays inside one interval: chains this short are placed too


def damage_points(data: bytes, lo: int, hi: int) -> np.ndarray:
    """Offsets of FF bytes in [lo, hi) not followed by 00, FF or a marker byte."""
    arr = np.frombuffer(data, dtype=np.uint8, count=min(hi + 1, len(data)) - lo, offset=lo)
    ff = np.flatnonzero(arr[:-1] == 0xFF)
    nxt = arr[ff + 1]
    bad = ~((nxt == 0x00) | (nxt == 0xFF) | ((nxt >= 0xD0) & (nxt <= 0xD9)))
    return ff[bad] + lo


def intact_segments(data: bytes, lo: int, hi: int, min_len: int = MIN_SEG) -> list[tuple[int, int]]:
    """Byte ranges between damage points that are long enough to carry MCUs."""
    pts = damage_points(data, lo, hi)
    edges = [lo] + [int(p) + 2 for p in pts]
    ends = [int(p) for p in pts] + [hi]
    return [(a, c) for a, c in zip(edges, ends) if c - a >= min_len]


def chains(data: bytes, pk: K.Packed, a: int, c: int, min_len: int = BLOCK) -> list[np.ndarray]:
    """MCU start bits of the parses found in the intact stretch [a, c); a decoding error
    (a damaged spot the FF rule did not reveal) ends one chain and starts the next."""
    out = []
    pos = a
    while pos < c - PROBE // 2:
        s = K.converged_start(data, pos, c, pk, PROBE)
        if s is None:
            pos += PROBE
            continue
        preds = np.zeros(pk.ncomp, dtype=np.int32)
        n, err, eb, _ebit, *_r, bits, _dc = K.decode_mcus(data, s // 8, s % 8, c, pk, preds, 1 << 20, check_dc=False)
        if n >= min_len:
            out.append(np.asarray(bits[:n], dtype=np.int64))
        nxt = max(eb, s // 8) + 16
        if nxt <= pos:
            nxt = pos + PROBE
        pos = nxt
    return out


def _win_sum(a: np.ndarray, n: int) -> np.ndarray:
    """Sums of every window of n consecutive values."""
    cs = np.concatenate([[0.0], np.cumsum(a)])
    return cs[n:] - cs[:-n]


class Placer:
    """Per-MCU reference DC (first block of every component, gray levels) and its
    expected error, for sliding chains of relative DC values along the image."""

    def __init__(self, ft, sr: D.ScanResult, thumb, clean: np.ndarray):
        comps, per_row, _rows = D._scan_components(ft)
        total = sr.total_mcus
        mrow, mcol = np.divmod(np.arange(total), per_row)
        refs, sigs, qs = [], [], []
        # every block of the MCU (4 luma blocks in 4:2:0): shifts of a single MCU show
        for c in comps:
            q = ft.quant[ft.frame.component(c.cid).tq].dc
            f = build_field(sr.dc[c.cid], block_mask(clean, sr.dc[c.cid], c.h, c.v, per_row), q,
                            thumb.dc[c.cid] if thumb else None, thumb.sigma[c.cid] if thumb else None)
            for by in range(c.v):
                for bx in range(c.h):
                    refs.append(f.val[mrow * c.v + by, mcol * c.h + bx] * q / 8.0)
                    sigs.append(f.sigma[mrow * c.v + by, mcol * c.h + bx])
                    qs.append(q / 8.0)
        self.ref = np.stack(refs, axis=1)  # (total, blocks per MCU) gray levels, NaN unknown
        self.sig = np.stack(sigs, axis=1)
        self.q = np.array(qs)
        self.total = total

    def costs(self, rel: np.ndarray, m_lo: int, m_hi: int) -> tuple[np.ndarray, int]:
        """Misfit of a block of relative DC (n, blocks per MCU, gray levels) at every start m in
        [m_lo, m_hi - n]: per block column, mean squared residual after the best offset, in
        units of the reference's expected error (averaged over the window). Sliding sums
        and correlations keep it O(window * n) in compiled loops. Returns (costs, m_lo)."""
        n = len(rel)
        m_lo, m_hi = max(0, m_lo), min(self.total, m_hi)
        if m_hi - m_lo < n:
            return np.zeros(0), m_lo
        L = m_hi - m_lo - n + 1
        tot = np.zeros(L)
        for ci in range(rel.shape[1]):
            r = self.ref[m_lo:m_hi, ci]
            sg = self.sig[m_lo:m_hi, ci]
            ok = np.isfinite(r) & np.isfinite(sg)
            r0 = np.where(ok, r, 0.0)
            x = rel[:, ci]
            cnt = _win_sum(ok.astype(float), n)
            s_r = _win_sum(r0, n)
            s_rr = _win_sum(r0 * r0, n)
            s_sg = _win_sum(np.where(ok, sg, 0.0), n)
            s_rx = np.correlate(r0, x, "valid")
            if ok.all():
                s_x, s_xx = float(x.sum()), float((x * x).sum())
            else:
                w = ok.astype(float)
                s_x, s_xx = np.correlate(w, x, "valid"), np.correlate(w, x * x, "valid")
            c = np.maximum(cnt, 1.0)
            sum_d = s_r - s_x
            sum_dd = s_rr - 2 * s_rx + s_xx
            var = np.maximum(sum_dd - sum_d * sum_d / c, 0.0) / c
            sbar = np.maximum(s_sg / c, 1e-6)
            tot += np.where(cnt >= n // 2, var / (sbar * sbar), np.inf)
        return tot / rel.shape[1], m_lo


class Anchors:
    """Known (byte offset, MCU index) pairs of one damaged range; every placed block adds
    one, so later searches get narrower."""

    def __init__(self, pairs):
        self.pairs = sorted(pairs)

    def add(self, byte: float, m: int) -> None:
        self.pairs.append((byte, m))
        self.pairs.sort()

    def expect(self, byte: float) -> tuple[int, int]:
        """(expected MCU index, search half-width) for data at this byte offset."""
        ab = np.array([p[0] for p in self.pairs], dtype=float)
        am = np.array([p[1] for p in self.pairs], dtype=float)
        e = float(np.interp(byte, ab, am))
        i = int(np.searchsorted(ab, byte))
        near = min(abs(e - am[j]) for j in (i - 1, i) if 0 <= j < len(am))
        return int(e), int(min(MAX_WINDOW, max(MIN_WINDOW, FIRST_WINDOW * near)))


def _blocks(n: int, min_len: int) -> list[tuple[int, int]]:
    """(start, length) of the blocks a chain of n MCUs is placed in: BLOCK-sized, the
    remainder joined to the last one; a chain shorter than BLOCK is one block."""
    if n < min_len:
        return []
    if n < BLOCK:
        return [(0, n)]
    starts = list(range(0, n - BLOCK + 1, BLOCK))
    out = [(k, BLOCK) for k in starts]
    rest = n - (starts[-1] + BLOCK)
    if rest and min_len < BLOCK:  # short pieces allowed: the remainder is not wasted
        k, ln = out[-1]
        out[-1] = (k, ln + rest)
    return out


def place_chain(placer: Placer, rel: np.ndarray, byte_of: np.ndarray, anchors: Anchors,
                m_lo: int, m_hi: int, min_len: int = BLOCK, accept: float = ACCEPT) -> list[tuple[int, int, int]]:
    """Blocks of the chain that fit: (chain index, MCU index, count), in chain order."""
    out: list[tuple[int, int, int]] = []
    n = len(rel)
    prev_end = None  # MCU index right after the last placed block
    floor = m_lo
    for k, bl in _blocks(n, min_len):
        blk = rel[k : k + bl]
        if prev_end is not None:
            cs, base = placer.costs(blk, prev_end - SHIFT, prev_end + SHIFT + bl)
            if len(cs):
                i = int(np.argmin(cs))
                m = base + i
                if cs[i] <= accept and m >= floor:
                    out.append((k, m, bl))
                    prev_end, floor = m + bl, m + bl
                    continue
            prev_end = None  # lost the thread: search afresh
        e, w = anchors.expect(float(byte_of[k]))
        cs, base = placer.costs(blk, max(floor, e - w), min(m_hi, e + w + bl))
        if not len(cs) or not np.isfinite(cs).any():
            continue
        i = int(np.argmin(cs))
        far = np.abs(np.arange(len(cs)) - i) > 3
        second = float(cs[far].min()) if far.any() else np.inf
        if cs[i] <= accept and cs[i] <= DISTINCT * second:
            m = base + i
            out.append((k, m, bl))
            anchors.add(float(byte_of[k]), m)
            prev_end, floor = m + bl, m + bl
    return out


def merge_blocks(blocks: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Runs of blocks contiguous in the chain and in the image: (chain index, MCU index, count)."""
    runs: list[list[int]] = []
    for k, m, n in blocks:
        if runs and runs[-1][0] + runs[-1][2] == k and runs[-1][1] + runs[-1][2] == m:
            runs[-1][2] += n
        else:
            runs.append([k, m, n])
    return [tuple(r) for r in runs]


@register
class SegmentPlacement(Recipe):
    name = "segment_placement"
    priority = 46  # after tails (their positions are anchors), before bit-flip repair
    description = "Place intact stretches from the middle of damaged data by matching them to the thumbnail"

    def detect(self, ctx: Context):
        if ctx.memo.get("segments_done"):
            return None
        got = decode_for(ctx)
        if got is None:
            return None
        parse, ft, sr = got
        return got if any(not iv.ok and iv.error is not None and iv.error.reason != "skipped"
                          for iv in sr.intervals) else None

    def _jobs(self, sr: D.ScanResult, parse, ft, tails: dict) -> list[dict]:
        """Damaged stretches to search: byte range, MCU range and anchor points (byte, MCU)."""
        dri = ft.restart_interval
        total = sr.total_mcus
        jobs = []
        ivs = sr.intervals
        for k, iv in enumerate(ivs):
            if iv.ok or iv.error is None or iv.error.reason == "skipped":
                continue
            last = iv.index + (iv.resync_skipped or 0) if iv.resync_at is not None else iv.index
            b_hi = iv.resync_at if iv.resync_at is not None else (parse.eoi_offset or iv.byte_end or 0)
            m_end = min(total, (last + 1) * dri) if dri else total
            b_lo = max(iv.byte_start, iv.error.byte_offset)
            m_lo = iv.mcu_first + iv.decoded
            anchors = [(iv.byte_start, iv.mcu_first), (b_lo, m_lo)]
            m_hi = m_end
            # a recovered tail closes the range; whole intervals recovered after it too
            for j in range(iv.index, last + 1):
                t = tails.get(j)
                if t is None:
                    continue
                if t["full"]:
                    m_hi = min(m_hi, t["first_mcu"])
                    b_hi = min(b_hi, t["start_bit"] // 8)
                else:
                    m_hi = min(m_hi, t["first_mcu"])
                    b_hi = min(b_hi, t["start_bit"] // 8)
                    anchors.append((t["start_bit"] // 8, t["first_mcu"]))
            anchors.append((b_hi, m_hi))
            if b_hi - b_lo >= (MIN_SEG // 2 if dri else MIN_SEG) and m_hi - m_lo >= (SHORT if dri else BLOCK):
                jobs.append({"iv": iv, "b_lo": b_lo, "b_hi": b_hi, "m_lo": m_lo, "m_hi": m_hi,
                             "anchors": sorted(anchors), "dri": dri})
        return jobs

    def _run(self, data: bytes, pk: K.Packed, placer: Placer, job: dict) -> list[dict]:
        anchors = Anchors(job["anchors"])
        pieces = []
        floor = job["m_lo"]
        # inside one restart interval the search is narrow, so short chains are safe to place
        min_len = SHORT if job["dri"] else BLOCK
        segs = intact_segments(data, job["b_lo"], job["b_hi"], MIN_SEG if not job["dri"] else MIN_SEG // 2)
        # finding the parses is independent per stretch (compiled code, parallel threads);
        # placing them is sequential, every piece must come after the previous one
        found = parallel.pmap(lambda sc: chains(data, pk, sc[0], sc[1], min_len), segs)
        for (a, c), seg_chains in zip(segs, found):
            for bits in seg_chains:
                if len(bits) < min_len:
                    continue
                preds = np.zeros(pk.ncomp, dtype=np.int32)
                s = int(bits[0])
                dn, _err, *_r, dc = K.decode_mcus(data, s // 8, s % 8, c, pk, preds, len(bits), check_dc=False)
                rel = dc[:dn].astype(float) * placer.q[None, :]
                blocks = place_chain(placer, rel, bits[:dn] / 8.0, anchors, floor, job["m_hi"], min_len,
                                     ACCEPT_ROW if job["dri"] else ACCEPT)
                for k, m, n in merge_blocks(blocks):
                    pieces.append({"chain_bits": bits, "k": k, "m": m, "n": n, "end": c})
                    floor = max(floor, m + n)
        return pieces

    def fix(self, ctx: Context, got) -> None:
        parse, ft, sr = got
        ctx.memo["segments_done"] = True
        data = ctx.data
        pk = K.packed_for(ft)
        tails = ctx.memo.get("tails", {})
        clean = sr.mcu_ok.copy()
        for iv in sr.intervals:
            if not iv.ok:
                clean[iv.mcu_first : iv.mcu_first + iv.mcu_count] = False
        thumb = ctx.memo["thumb_base"] if "thumb_base" in ctx.memo else thumbnail_base(data, ft, sr, clean, ctx.options.get("thumb_refs"))
        ctx.memo["thumb_base"] = thumb
        placer = Placer(ft, sr, thumb, clean)
        jobs = self._jobs(sr, parse, ft, tails)
        if not jobs:
            return
        found = parallel.pmap(lambda jb: self._run(data, pk, placer, jb), jobs)
        dri = ft.restart_interval
        out: dict[int, list[dict]] = {}
        placed = 0
        for job, pieces in zip(jobs, found):
            for p in pieces:
                bits, k, m, n = p["chain_bits"], p["k"], p["m"], p["n"]
                # a piece must stay inside one restart interval (its own decoding context)
                iv_idx = m // dri if dri else job["iv"].index
                if dri:
                    n = min(n, (iv_idx + 1) * dri - m)
                if n < 2:
                    continue
                s = int(bits[k])
                preds = np.zeros(pk.ncomp, dtype=np.int32)
                dn, err, *_r, end_b, end_bit, _bs, dc_rel = K.decode_mcus(data, s // 8, s % 8, p["end"], pk, preds, n,
                                                                         check_dc=False)
                if dn != n:
                    continue
                out.setdefault(iv_idx, []).append({
                    "start_bit": s, "end_bit": int(end_b * 8 + end_bit), "n": int(n), "first_mcu": int(m),
                    "delta": {cid: 0 for cid in pk.cids}, "full": False, "dc_rel": dc_rel.astype(np.int32),
                    "piece": True,
                })
                placed += n
        if out:
            for v in out.values():
                v.sort(key=lambda t: t["first_mcu"])
            ctx.memo["pieces"] = out
            ctx.note(self, f"placed {placed} MCU(s) from intact stretches in {len(out)} interval(s)",
                     placed_mcus=placed, intervals=sorted(out))
