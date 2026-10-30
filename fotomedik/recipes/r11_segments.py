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

Without any reference (no thumbnail, nothing clean nearby - a photo whose header was
rebuilt and whose first rows are the only clean ones) a long chain can still tell its
own horizontal position: laid out from the wrong column, the end of every picture row
lands beside the start of the next one, and the pixels across that seam do not match,
while every other MCU boundary in the chain is a real one (Two-ACIR, CVPR 2023, does
the same with its row alignment). The chain is decoded on its own, the pixel
disagreement across each MCU boundary (see seam_phase) is averaged per column phase,
and the phase that stands out clearly is the seam. On healthy photos cut into chains
of 2 to 9 rows this places 87 to 98 percent of them in the right column and none in
a wrong one; a star field or a black background can fool it (a few percent wrong).

The row comes from the byte position, counted from the last thing placed (at first,
the rows decoded before the damage): every chain on the way, placed or not, adds the
MCUs it decoded, and only the bytes between chains - the rotted sectors - are turned
into MCUs at the rate of the chains around them. Interpolating bytes over the whole
damaged range instead put pieces 4 to 8 rows off (a sky packs into fewer bytes than a
crowd); counted chain by chain they land within a row or two, most of them exactly.
The count must also end about where the range ends (a recovered tail, the last MCU):
it is stretched to do so exactly, and when it is far off (a tail placed wrongly, a
foreign file in the range) nothing is counted and the rows are interpolated as before.
On photos with a rotted sector every few KB this puts 61 percent of the placed blocks
at their place, against 18 percent before. The DC offset comes from R9, as for every
piece.
"""

from __future__ import annotations

import numpy as np

from .. import parallel
from ..jpeg import decoder as D
from ..jpeg import kernel as K
from .base import Context, Recipe
from .dcfield import block_mask, build_field, thumbnail_base
from ..reference import for_context as reference_for
from .r9_holes import decode_for
from .registry import register

BLOCK = 32  # MCUs placed together
STEP_LEAD = 2  # MCUs before the thumbnail's step where the search starts (the damage lies just before it)
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
BLIND_ROWS = 2.0  # rows of MCUs a chain must span to tell its own column (no reference)
BLIND_LONG_ROWS = 6.0  # ... from here on the seam is clear enough at a smaller lead
BLIND_LEAD, BLIND_LEAD_SHORT = 2.0, 2.5  # standard deviations the seam must lead the runner-up by


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


def chain_pixels(data: bytes, parse, ft, pk: K.Packed, start_bit: int, end_bit: int,
                 dc_rel: np.ndarray) -> np.ndarray | None:
    """YCbCr pixels (float32, the frame's size) of a chain decoded on its own: laid out
    from MCU 1, after a flat MCU whose DC puts the chain's values near mid-gray (its
    DC chain is relative to an unknown predictor)."""
    import io

    from PIL import Image, ImageFile

    from ..jpeg.encoder import BitWriter, encode_flat_mcus
    from ..jpeg.writer import assemble

    comps, _per_row, _rows = D._scan_components(ft)
    targets = {}
    col = 0
    for c in comps:
        q = ft.quant[ft.frame.component(c.cid).tq].dc
        lim = (1024 + q) // q - 1
        med = float(np.median(dc_rel[:, col : col + c.h * c.v]))
        targets[c.cid] = int(np.clip(round(-med), -lim, lim))
        col += c.h * c.v
    w = BitWriter()
    encode_flat_mcus(ft, 1, targets, {c.cid: 0 for c in comps}, writer=w)
    w.copy_bits(data, start_bit, end_bit)
    blob = assemble(data[: parse.scan_start], w.finish())
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        im = Image.open(io.BytesIO(blob))
        im.draft("YCbCr", im.size)
        im = im.convert("YCbCr") if im.mode != "YCbCr" else im
        return np.asarray(im, dtype=np.float32)
    except Exception:
        return None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def _local_norm(d: np.ndarray, radius: int = 8) -> np.ndarray:
    """Every boundary's difference relative to the median of its neighbours along the
    chain (the boundary itself left out): a seam in flat sky stands out as much as one
    between textured walls."""
    n = len(d)
    pad = np.pad(d, radius, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * radius + 1)[:n]
    neigh = np.concatenate([win[:, :radius], win[:, radius + 1 :]], axis=1)
    return d / np.maximum(np.median(neigh, axis=1), 0.5)


def seam_phase(ycc: np.ndarray, n: int, per_row: int, mw: int, mh: int, first: int = 1) -> tuple[int, float]:
    """For a chain of n MCUs laid out from image MCU `first`: the phase (chain index mod
    per_row) of the boundary between consecutive chain MCUs where the pixels disagree
    most, and how clearly it stands out (its lead over the runner-up in standard
    deviations of all other phases). Four measures of a boundary are averaged per phase,
    each normalised to its largest phase mean and added: the mean absolute pixel
    difference across it, the same relative to the neighbouring boundaries, its excess
    over the gradients just inside either side (a seam jumps where the texture does
    not), and the share of pixel rows with such a clear jump. ycc: the layout."""
    H, W = ycc.shape[:2]
    rows = -(-H // mh)
    total = rows * per_row
    img = np.pad(ycc, ((0, rows * mh - H), (0, per_row * mw - W), (0, 0)), mode="edge")
    grid = img.reshape(rows, mh, per_row, mw, 3)

    def column(i):
        return grid[:, :, :, i, :].transpose(0, 2, 1, 3).reshape(total, mh, 3)

    last = min(first + n, total)  # chain MCUs at image indices first .. last-1
    a, b = slice(first, last - 1), slice(first + 1, last)  # boundary k | k+1 of the chain
    l0, l1, r0, r1 = column(0), column(1), column(-1), column(-2)
    plain = np.abs(r0[a] - l0[b])
    inner = 0.5 * (np.abs(r0[a] - r1[a]) + np.abs(l0[b] - l1[b]))
    excess = np.maximum(0.0, plain - inner)

    def weigh(d):
        return d[..., 0].mean(1) + 0.5 * d[..., 1:].mean((1, 2))

    score = weigh(plain)
    measures = (score, np.minimum(_local_norm(score), 10.0), weigh(excess),
                (plain[..., 0] > 2.0 * inner[..., 0] + 4.0).mean(1))
    phase = np.arange(len(score)) % per_row
    cnt = np.bincount(phase, minlength=per_row)
    have = cnt >= 1
    combined = np.full(per_row, -np.inf)
    combined[have] = 0.0
    for v in measures:
        mean = np.bincount(phase, weights=v, minlength=per_row) / np.maximum(cnt, 1)
        combined[have] += mean[have] / max(float(mean[have].max()), 1e-6)
    j = int(np.argmax(combined))
    rest = np.sort(combined[have])[:-1]
    if len(rest) < 3:
        return j, 0.0
    lead = (float(combined[j]) - float(rest[-1])) / max(float(np.std(rest)), 1e-6)
    return j, lead


def blind_offset(data: bytes, parse, ft, pk: K.Packed, start_bit: int, end_bit: int, dc_rel: np.ndarray,
                 per_row: int) -> list[tuple[int, int, int]]:
    """(chain index, column the piece starts at, MCUs) of the pieces of a chain placed by
    its own seam: the whole chain when its seam is clear and its halves agree; the halves
    on their own when each is sure of a different column (the decoder desynced inside);
    nothing when the seam does not stand out or the chain is too short."""
    n = len(dc_rel)
    span = n / per_row
    if span < BLIND_ROWS:
        return []
    ycc = chain_pixels(data, parse, ft, pk, start_bit, end_bit, dc_rel)
    if ycc is None:
        return []
    fr = ft.frame
    mw, mh = fr.mcu_width, fr.mcu_height

    def column_of(k0: int, count: int) -> int | None:
        """Column where chain MCU k0 sits, from the seam of chain MCUs k0 .. k0+count-1."""
        j, lead = seam_phase(ycc, count, per_row, mw, mh, k0 + 1)
        if lead < (BLIND_LEAD if count >= BLIND_LONG_ROWS * per_row else BLIND_LEAD_SHORT):
            return None
        return (per_row - 1 - j) % per_row  # the phase counts from k0

    if span >= 2 * BLIND_ROWS:
        half = n // 2
        a, b = column_of(0, half), column_of(half, n - half)
        if a is not None and b is not None and (a + half) % per_row != b:
            return [(0, a, half), (half, b, n - half)]
    whole = column_of(0, n)
    return [(0, whole, n)] if whole is not None else []


def dc_in_range(dc_rel: np.ndarray, lim: np.ndarray) -> bool:
    """Whether the relative DC values of a piece (n, blocks per MCU) can be offset into
    the legal range of every block column (a desynced stretch drifts out of it)."""
    if not len(dc_rel):
        return True
    span = dc_rel.max(axis=0) - dc_rel.min(axis=0)
    return bool((span <= 2 * lim).all())


EXT_SUB = 16  # MCUs: an extension is checked against the reference in steps of this many
EXT_TOL = 3.0  # ... and stops where its level differs from the placed blocks' by this many expected errors


def verify_extension(placer: "Placer", rel: np.ndarray, placed: list[tuple[int, int, int]],
                     seg: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """An extended segment (chain index, MCU index, count) as the pieces that agree with
    the reference. A bit error inside a chain can keep the decoder in step and still
    change one DC difference: from there on a component sits at another level (a green
    band through a white facade). So the placed blocks of the segment are grouped by
    level (the reference minus their relative DC, per block column), every group grows
    outwards only while the reference agrees with its level, and each group becomes a
    piece of its own, with its own DC offset. Where the reference knows nothing the
    extension stands."""
    k0, m0, c = seg
    off = m0 - k0
    own = sorted((k, m, n) for k, m, n in placed if m - k == off and k0 <= k < k0 + c)
    if not own:
        return []
    nb = rel.shape[1]

    def level_of(a: int, b: int) -> np.ndarray:
        r = placer.ref[a + off : b + off] - rel[a:b]
        out = np.full(nb, np.nan)
        for j in range(nb):
            f = np.isfinite(r[:, j])
            if f.sum() >= max(2, (b - a) // 2):
                out[j] = np.median(r[f, j])
        return out

    lo, hi = own[0][0], own[-1][0] + own[-1][2]
    sig = placer.sig[lo + off : hi + off]
    tol = EXT_TOL * np.array([np.nanmedian(sig[:, j]) if np.isfinite(sig[:, j]).any() else np.inf for j in range(nb)])

    def differ(a: np.ndarray, b: np.ndarray) -> bool:
        d = np.abs(a - b)
        return bool((np.isfinite(d) & (d > tol)).any())

    # the placed blocks in groups of one level, checked in steps of EXT_SUB
    steps = [(a, min(a + EXT_SUB, k + n)) for k, _m, n in own for a in range(k, k + n, EXT_SUB)]
    groups: list[list] = []  # [start, end, level]
    for a, b in steps:
        lv = level_of(a, b)
        g = groups[-1] if groups else None
        if g is not None and a - g[1] <= 2 * BLOCK and not differ(lv, g[2]):
            g[1] = b
            g[2] = np.where(np.isfinite(g[2]), g[2], lv)
        else:
            groups.append([a, b, lv])
    out = []
    for i, (a, b, lv) in enumerate(groups):
        left = k0 if i == 0 else groups[i - 1][1]
        right = k0 + c if i == len(groups) - 1 else groups[i + 1][0]
        while a > left:
            a2 = max(left, a - EXT_SUB)
            if differ(level_of(a2, a), lv):
                break
            a = a2
        while b < right:
            b2 = min(right, b + EXT_SUB)
            if differ(level_of(b, b2), lv):
                break
            b = b2
        if out and out[-1][0] + out[-1][2] >= a:  # two groups meet: one MCU between them carries
            a = out[-1][0] + out[-1][2] + 1      # the predictor, so they stay two pieces
        if b - a >= 2:
            out.append((a, a + off, b - a))
    return out


def extend_blocks(blocks: list[tuple[int, int, int]], n: int, per_row: int) -> list[tuple[int, int, int]]:
    """Blocks placed by the reference (chain index, MCU index, count) stretched over the
    rest of the chain: a chain is one run of MCUs, so the blocks between two placed ones
    that agree on the chain's offset (MCU index minus chain index) belong at that
    offset too - they only fitted the reference nowhere clearly (flat sky). Where the
    offset changes (a desync inside the chain, or MCUs lost), the gap is split halfway."""
    if not blocks:
        return []
    runs: list[list[int]] = []  # [chain start, chain end, offset]
    for k, m, c in sorted(blocks):
        off = m - k
        if runs and runs[-1][2] == off and runs[-1][1] == k:
            runs[-1][1] = k + c
        else:
            runs.append([k, k + c, off])
    out = []
    for i, (ka, kb, off) in enumerate(runs):
        lo = 0 if i == 0 else (runs[i - 1][1] + ka) // 2
        hi = n if i == len(runs) - 1 else (kb + runs[i + 1][0]) // 2
        if i > 0 and runs[i - 1][2] == off:  # the same offset again: everything between is ours too
            lo = ka
        if i + 1 < len(runs) and runs[i + 1][2] == off:
            hi = kb
        out.append((lo, lo + off, hi - lo))
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
        self.lim = np.array([(1024 + 8 * q) // (8 * q) - 1 for q in qs], dtype=float)  # legal |DC| per block column
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


MIN_LEAD_MCUS = 50  # MCUs decoded before the damage needed to count lost bytes from there


RATE_RANGE = (0.5, 2.0)  # a gap's bytes per MCU stays within this share of the median chain's
COUNT_AGREE = (0.7, 1.4)  # the range's MCUs over the chain-by-chain count: within this, the count is used (stretched)


def chain_rates(chains_: list[tuple[float, float, int, bool]]) -> float | None:
    """Median bytes per MCU over the credible chains (garbage that decoded can pack
    hundreds of MCUs into a few bytes and must not set the rate)."""
    r = [(eb - sb) / n for sb, eb, n, ok in chains_ if ok and n > 0]
    return float(np.median(r)) if r else None


def local_rate(prev_rate: float | None, here: float, med: float | None) -> float:
    """Bytes per MCU for the bytes between two chains: the mean of their rates, kept
    within RATE_RANGE of the median."""
    r = here if prev_rate is None else (prev_rate + here) / 2.0
    if med is not None:
        r = float(np.clip(r, RATE_RANGE[0] * med, RATE_RANGE[1] * med))
    return max(r, 1e-6)


def count_mcus(chains_: list[tuple[float, float, int, bool]], b0: float, m0: int, rate0: float | None, b_end: float) -> float:
    """The MCU index reached at byte b_end counting from (b0, m0): every credible chain
    (start byte, end byte, MCUs, credible) adds its MCUs, the bytes between chains and
    the bytes of garbage chains count at the rate of the chains around them."""
    med = chain_rates(chains_)
    m, b, rate = float(m0), b0, rate0
    for sb, eb, n, ok in chains_:
        here = (eb - sb) / max(n, 1)
        if ok:
            m += max(0.0, sb - b) / local_rate(rate, here, med) + n
            rate = here
        else:
            m += max(0.0, eb - b) / local_rate(rate, med if med else here, med)
        b = eb
    m += max(0.0, b_end - b) / local_rate(rate, med if med else 1.0, med)
    return m


def blind_position(expected: int, offset: int, n: int, floor: int, m_hi: int, per_row: int) -> int | None:
    """The MCU index nearest to `expected` that starts at column `offset`, with room for
    n MCUs between floor and m_hi; None when there is none."""
    base = expected - ((expected - offset) % per_row)  # the start at or before expected
    cands = sorted((base, base + per_row), key=lambda m: abs(m - expected))
    for m in cands:
        while m < floor:
            m += per_row
        if m + n <= m_hi:
            return m
    return None


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

    def _jobs(self, sr: D.ScanResult, parse, ft, tails: dict, step: int | None = None) -> list[dict]:
        """Damaged stretches to search: byte range, MCU range and anchor points (byte, MCU).

        step: the MCU where the decode starts to disagree with the thumbnail (see
        r5_bitflip.thumbnail_divergence). Without restart markers a lost or foreign stretch
        is often noticed thousands of MCUs later; what decoded in between is real data of
        the photo at the wrong place, so the search starts at the step, not at the error."""
        dri = ft.restart_interval
        total = sr.total_mcus
        _comps, per_row, _rows = D._scan_components(ft)
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
            if step is not None and iv.mcu_first + STEP_LEAD < step < m_lo:
                k = step - STEP_LEAD - iv.mcu_first
                if 0 <= k < len(sr.mcus) and sr.mcus[k].index == step - STEP_LEAD:
                    m_lo, b_lo = step - STEP_LEAD, sr.mcus[k].bit_start // 8
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
                             "anchors": sorted(anchors), "dri": dri, "per_row": per_row})
        return jobs

    def _run(self, data: bytes, pk: K.Packed, placer: Placer, job: dict, parse_ft=(None, None)) -> list[dict]:
        anchors = Anchors(job["anchors"])
        pieces = []
        floor = job["m_lo"]
        # inside one restart interval the search is narrow, so short chains are safe to place
        min_len = SHORT if job["dri"] else BLOCK
        segs = intact_segments(data, job["b_lo"], job["b_hi"], MIN_SEG if not job["dri"] else MIN_SEG // 2)
        # finding the parses is independent per stretch (compiled code, parallel threads);
        # placing them is sequential, every piece must come after the previous one
        found = parallel.pmap(lambda sc: chains(data, pk, sc[0], sc[1], min_len), segs)
        per_row = job["per_row"]
        # (end byte, end MCU, bytes per MCU) of the last thing placed, for the blind estimate: at first the
        # stretch decoded before the damage (counting the lost bytes from where the picture is known beats
        # interpolating over the whole range: a sky packs into fewer bytes than a crowd)
        iv0 = job["iv"]
        lead_rate = None
        if job["m_lo"] - iv0.mcu_first >= MIN_LEAD_MCUS and job["b_lo"] > iv0.byte_start:
            lead_rate = (job["b_lo"] - iv0.byte_start) / (job["m_lo"] - iv0.mcu_first)
        decoded = []  # (bits, segment end, MCUs, end bit, DC) per chain, in byte order
        for (a, c), seg_chains in zip(segs, found):
            for bits in seg_chains:
                if len(bits) < min_len:
                    continue
                preds = np.zeros(pk.ncomp, dtype=np.int32)
                s = int(bits[0])
                dn, _err, *_r, end_b, end_bit, _bs, dc = K.decode_mcus(data, s // 8, s % 8, c, pk, preds, len(bits),
                                                                       check_dc=False)
                decoded.append((bits, c, dn, end_b * 8 + end_bit, dc[:dn]))
        # counting MCUs chain by chain from the start must end about where the range ends; when it
        # does not (a tail placed wrongly, a foreign file in between) nothing is counted and the
        # rows are interpolated between the known points as before
        counting = False
        scale = 1.0
        if not job["dri"] and decoded:
            spans = [(int(b[0]) / 8.0, eb / 8.0, n, True) for b, _c, n, eb, _d in decoded]
            e_end = count_mcus(spans, float(job["b_lo"]), job["m_lo"], lead_rate, float(job["b_hi"]))
            if e_end > job["m_lo"]:
                scale = (job["m_hi"] - job["m_lo"]) / (e_end - job["m_lo"])
                counting = COUNT_AGREE[0] <= scale <= COUNT_AGREE[1]
        if not counting:
            scale = 1.0
        med_rate = chain_rates([(int(b[0]) / 8.0, eb / 8.0, n, True) for b, _c, n, eb, _d in decoded])
        prev = (float(job["b_lo"]), job["m_lo"], lead_rate) if counting else None  # the rate may be unknown at first
        for bits, c, dn, end_bit_abs, dc in decoded:
            s = int(bits[0])
            end_b, end_bit = divmod(end_bit_abs, 8)
            rel = dc[:dn].astype(float) * placer.q[None, :]
            blocks = place_chain(placer, rel, bits[:dn] / 8.0, anchors, floor, job["m_hi"], min_len,
                                 ACCEPT_ROW if job["dri"] else ACCEPT)
            if not job["dri"]:
                if blocks:
                    # the reference placed parts of the chain: the rest goes with them,
                    # unless the DC runs away in it (a desync: the stretched piece would be lost whole)
                    ext = [(k, m, c) for k, m, c in extend_blocks(blocks, dn, per_row)
                           if m >= floor and m + c <= job["m_hi"]]
                    ext = [piece for e in ext for piece in verify_extension(placer, rel, blocks, e)]
                    if ext and all(dc_in_range(dc[k : k + c], placer.lim) for k, _m, c in ext):
                        blocks = ext
                elif dn >= BLIND_ROWS * per_row:
                    # the reference said nothing: the chain's own seam
                    pieces_b = blind_offset(data, parse_ft[0], parse_ft[1], pk, s, end_b * 8 + end_bit, dc[:dn], per_row)
                    bytes_here = (end_b * 8 + end_bit - s) / 8.0
                    start_e = None
                    for k0, o, count in pieces_b:
                        if not dc_in_range(dc[k0 : k0 + count], placer.lim):
                            continue
                        if start_e is None:
                            if prev is not None:
                                rate = local_rate(prev[2], bytes_here / dn, med_rate)
                                start_e = int(round(prev[1] + scale * max(0.0, s / 8.0 - prev[0]) / rate))
                            else:
                                start_e = anchors.expect(s / 8.0)[0]
                        m = blind_position(start_e + k0, o, count, floor, job["m_hi"], per_row)
                        if m is not None:
                            blocks.append((k0, m, count))
                            anchors.add(float(bits[k0]) / 8.0, m)
                            floor = m + count
                            start_e = m - k0  # the rest of the chain follows
            for k, m, n in merge_blocks(blocks):
                pieces.append({"chain_bits": bits, "k": k, "m": m, "n": n, "end": c})
                floor = max(floor, m + n)
            here_rate = (end_b * 8 + end_bit - s) / 8.0 / max(dn, 1)
            if blocks:
                k, m, n = blocks[-1]
                prev = ((end_b * 8 + end_bit) / 8.0, m + n, here_rate)
            elif prev is not None and counting:
                # not placed, but its MCUs are known: only the bytes between chains are counted at a rate
                e = prev[1] + scale * (max(0.0, s / 8.0 - prev[0]) / local_rate(prev[2], here_rate, med_rate) + dn)
                prev = ((end_b * 8 + end_bit) / 8.0, int(round(e)), here_rate)
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
        thumb = ctx.memo["thumb_base"] if "thumb_base" in ctx.memo else thumbnail_base(
            data, ft, sr, clean, ctx.options.get("thumb_refs"), reference_for(ctx, ft))
        ctx.memo["thumb_base"] = thumb
        placer = Placer(ft, sr, thumb, clean)
        step = None
        if not ft.restart_interval and thumb is not None:
            from .r5_bitflip import thumbnail_divergence

            dv = thumbnail_divergence(data, ft, sr, ctx.options.get("thumb_refs"))
            step = dv(sr) if dv is not None else None
        jobs = self._jobs(sr, parse, ft, tails, step)
        if not jobs:
            return
        found = parallel.pmap(lambda jb: self._run(data, pk, placer, jb, (parse, ft)), jobs)
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
