"""R5 — undo flipped bits in the entropy-coded data by trial decoding.

A flipped bit desyncs the Huffman stream and the decoder notices a little
later (invalid code, coefficient overflow, implausible DC, marker in the wrong
place). We try flipping every bit in a window before the detection point and
keep the flip that lets the interval decode all the way to its restart marker.
Observed damage on dying flash is mostly 1->0, so setting bits is tried first.
Several flips per interval are handled greedily; dense damage is given up on
and left to R9.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..jpeg import decoder as D
from ..jpeg import markers as M
from ..jpeg.tables import FrameTables
from .base import Context, Recipe
from .r9_holes import decode_for
from .registry import register

W_BACK = 96  # bytes before the detected error to search
W_FWD = 2  # bytes at/after the detection point
TWO_BIT_WINDOW = 24
MAX_ROUNDS = 8  # flips per interval
MIN_GAIN = 3  # MCUs of extra progress needed to accept a partial fix
BEAM = 3  # distinct partial outcomes explored per round when nothing completes
PER_OUTCOME = 3  # equivalent-looking flips explored per outcome
MAX_EXPANSIONS = 40  # search nodes per interval
ENOUGH_COMPLETE = 8  # stop scanning once this many *distinct* complete outcomes were found
PER_INTERVAL_MIN = 3.0  # seconds; each failed interval gets at least this much of the budget
PER_INTERVAL_MAX = 30.0
DEFAULT_TIME_BUDGET = 120.0  # seconds per file
ANOMALY_LEVELS = 64

_ONE_BIT = [1 << b for b in range(8)]
_TWO_BIT = [(1 << a) | (1 << b) for a in range(8) for b in range(a + 1, 8)]

REPAIRABLE = ("bad_huffman", "coef_overflow", "dc_range", "unexpected_marker", "bad_rst")


def _complete(r: D.ScanResult, needed: int, marker_pos: int | None) -> bool:
    if r.error is not None or not r.intervals or not all(iv.ok for iv in r.intervals):
        return False
    if sum(iv.decoded for iv in r.intervals) != needed:
        return False
    if marker_pos is not None:
        end = r.intervals[-1].byte_end
        if end is None or marker_pos - end > 2 or end > marker_pos + 1:
            return False
    return True


def _decoded(r: D.ScanResult) -> int:
    return sum(iv.decoded for iv in r.intervals)


def _first_anomaly(cur: D.ScanResult, base: D.ScanResult, ft: FrameTables, first: int, count: int) -> int | None:
    """MCU index in [first, first+count) whose luma DC jumps vs. the row above, or None."""
    frame = ft.frame
    luma = ft.scan.components[0].id
    fc = frame.component(luma)
    interleaved = len(ft.scan.components) > 1
    h, v = (fc.h, fc.v) if interleaved else (1, 1)
    per_row = cur.dc[luma].shape[1] // h
    q = ft.quant[fc.tq].dc
    for mi in range(first, first + count):
        mrow, mcol = divmod(mi, per_row)
        brow, bcol = mrow * v, mcol * h
        if brow == 0:
            continue
        a = cur.dc[luma][brow, bcol]
        b = base.dc[luma][brow - 1, bcol]
        if math.isnan(a) or math.isnan(b):
            continue
        if abs(a - b) * q / 8 > ANOMALY_LEVELS:
            return mi
    return None


def _roughness(r: D.ScanResult, base: D.ScanResult, ft: FrameTables, first: int, count: int) -> float:
    """How jumpy the DC is inside the interval vs. neighbours (gray levels, lower = smoother).

    Plain mean of neighbour differences (a pooled tail/percentile metric proved
    non-monotonic: candidates can swap which blocks land in the tail).
    All scan components count (a wrong flip often shifts only a chroma plane).
    Vertical differences use the full-file decode `base` for the rows above/below
    (only where those MCUs decoded cleanly), horizontal ones the candidate itself.
    """
    frame = ft.frame
    interleaved = len(ft.scan.components) > 1
    diffs: list[float] = []
    for sc in ft.scan.components:
        fc = frame.component(sc.id)
        h, v = (fc.h, fc.v) if interleaved else (1, 1)
        arr = r.dc[sc.id]
        barr = base.dc[sc.id]
        per_row = arr.shape[1] // h
        q = ft.quant[fc.tq].dc / 8.0
        for mi in range(first, first + count):
            mrow, mcol = divmod(mi, per_row)
            for by in range(v):
                for bx in range(h):
                    brow, bcol = mrow * v + by, mcol * h + bx
                    a = arr[brow, bcol]
                    if math.isnan(a):
                        continue
                    if bcol > 0 and not math.isnan(arr[brow, bcol - 1]):
                        # horizontal jumps inside the candidate weigh double: a wrong flip
                        # starts/ends a shifted run with a step the picture never had
                        diffs.append(2.0 * abs(a - arr[brow, bcol - 1]) * q)
                    if by == 0 and brow > 0:
                        up_m = ((brow - 1) // v) * per_row + bcol // h
                        if base.mcu_ok[up_m] and not math.isnan(barr[brow - 1, bcol]):
                            diffs.append(abs(a - barr[brow - 1, bcol]) * q)
                    if by == v - 1 and brow + 1 < barr.shape[0]:
                        dn_m = ((brow + 1) // v) * per_row + bcol // h
                        if dn_m < len(base.mcu_ok) and base.mcu_ok[dn_m] and not math.isnan(barr[brow + 1, bcol]):
                            diffs.append(abs(a - barr[brow + 1, bcol]) * q)
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


class _Interval:
    def __init__(self, iv: D.Interval, parse: M.ParseResult, data_len: int):
        self.first = iv.mcu_first
        self.count = iv.mcu_count
        self.start = iv.byte_start
        if iv.resync_at is not None:
            self.marker_pos: int | None = iv.resync_at
        elif parse.eoi_offset is not None and parse.eoi_offset > iv.byte_start:
            self.marker_pos = parse.eoi_offset
        else:
            self.marker_pos = None
        self.decode_end = (self.marker_pos + 2) if self.marker_pos is not None else data_len


def repair_interval(
    buf: bytearray,
    parse: M.ParseResult,
    ft: FrameTables,
    base: D.ScanResult,
    iv: D.Interval,
    deadline: float,
    baseline: float = 0.0,
) -> tuple[bool, list[tuple[int, int]], str]:
    """Try to make one failed interval decode cleanly. Mutates buf. Returns (ok, flips, note).

    `baseline` is the typical DC roughness of the undamaged part of this image; a
    'complete' candidate much rougher than that is a wrong flip that happens to decode.
    """
    spec = _Interval(iv, parse, len(buf))

    def decode_from(start_byte, start_bit, first_mcu, preds, max_mcus=None):
        n = spec.count - (first_mcu - spec.first)
        return D.decode_scan(
            bytes(buf),
            parse,
            ft,
            start_byte=start_byte,
            start_bit=start_bit,
            end_byte=spec.decode_end,
            first_mcu=first_mcu,
            max_mcus=n if max_mcus is None else min(n, max_mcus),
            init_preds=preds,
        )

    expansions = [0]

    def search(depth: int, flips: list[tuple[int, int]]):
        """Depth-limited beam search. Returns (complete, progress, roughness, flips)."""
        cur = decode_from(spec.start, 0, spec.first, None)
        if _complete(cur, spec.count, spec.marker_pos):
            return (True, spec.count, _roughness(cur, base, ft, spec.first, spec.count), list(flips))
        progress = _decoded(cur)
        if depth >= MAX_ROUNDS or time.monotonic() > deadline or expansions[0] >= MAX_EXPANSIONS:
            return (False, progress, float("inf"), list(flips))
        expansions[0] += 1
        # Where to look: the detection point, and the first visibly implausible MCU before
        # it (a desync is often noticed hundreds of bytes after the flipped bit).
        if cur.error is not None and cur.error.reason in ("bad_huffman", "coef_overflow", "dc_range", "eof"):
            e_byte = cur.error.byte_offset
        elif cur.error is not None and cur.error.reason == "unexpected_marker":
            e_byte = cur.error.byte_offset + 1
        else:
            e_byte = cur.intervals[-1].byte_end if cur.intervals and cur.intervals[-1].byte_end else spec.decode_end
        mi = _first_anomaly(cur, base, ft, spec.first, spec.count)
        hint = e_byte
        if mi is not None and mi - spec.first < len(cur.mcus):
            # an image edge can look like an anomaly too, so never trust it further back
            # than a few windows before the detection point
            hint = max(min(e_byte, cur.mcus[mi - spec.first].bit_start // 8 + 2), e_byte - 4 * W_BACK)
        hi = min(e_byte + W_FWD, spec.decode_end - (2 if spec.marker_pos is not None else 0))
        lo = max(spec.start, hint - W_BACK)
        if hi <= lo:
            return (False, progress, float("inf"), list(flips))
        # Resume point: last MCU that starts before the window (its predictors are known)
        m0 = spec.first
        for m in cur.mcus:
            if m.bit_start // 8 < lo:
                m0 = m.index
            else:
                break
        if m0 > spec.first:
            start_off, start_bit = divmod(cur.mcus[m0 - spec.first].bit_start, 8)
            preds0 = D.preds_before_mcu(cur, ft, m0)
        else:
            start_off, start_bit, preds0 = spec.start, 0, None

        partials: dict[tuple[int, float], list[tuple[int, int, int]]] = {}  # (dec, rough) -> [(dir, off, mask)]
        complete: list[tuple[float, int, int, int]] = []  # (roughness, direction_penalty, off, mask)

        # Stage 1 decodes only a little past the detection point (cheap, kills most
        # candidates); stage 2 decodes to the marker for the survivors.
        err_mcu = cur.error.mcu_index if cur.error is not None else spec.first + spec.count
        stage1_n = max(1, err_mcu - m0 + MIN_GAIN + 1)

        def consider(off: int, mask: int) -> None:
            was_zero = not (buf[off] & mask)
            buf[off] ^= mask
            r = decode_from(start_off, start_bit, m0, dict(preds0) if preds0 else None, max_mcus=stage1_n)
            if r.error is None and _decoded(r) >= stage1_n and stage1_n < spec.count - (m0 - spec.first):
                r = decode_from(start_off, start_bit, m0, dict(preds0) if preds0 else None)
            buf[off] ^= mask
            if _complete(r, spec.count - (m0 - spec.first), spec.marker_pos):
                # Several different flips can decode "completely"; rank them by how
                # smoothly the result joins its neighbours, then prefer 0->1 repairs.
                complete.append((_roughness(r, base, ft, spec.first, spec.count), 0 if was_zero else 1, off, mask))
                return
            dec = (m0 - spec.first) + _decoded(r)
            if dec < progress + MIN_GAIN:
                return
            # judge smoothness on the prefix before the next error (the last few MCUs
            # before a detection are usually already garbage)
            n_ok = max(1, dec - (m0 - spec.first) - MIN_GAIN)
            key = (dec, round(_roughness(r, base, ft, m0, n_ok), 3))
            partials.setdefault(key, []).append((0 if was_zero else 1, off, mask))

        limit_rough = max(1.5 * baseline, baseline + 6.0)

        def acceptable() -> list:
            return [c for c in complete if c[0] <= limit_rough]

        def enough() -> bool:
            # many flips are equivalent (same effect, different bit); count distinct outcomes
            return len({round(c[0], 2) for c in acceptable()}) >= ENOUGH_COMPLETE

        # Single-bit flips, nearest to the hint first (the anomaly start is the best guess
        # for where the flip is); widen the window while nothing acceptable turns up and
        # stop once a few complete candidates exist (the smoothest wins).
        scanned: set[int] = set()
        for si, span in enumerate((W_BACK, 3 * W_BACK, 8 * W_BACK)):
            new_lo = max(spec.start, hint - span)
            new_hi = hi if si == 0 else min(hi, hint + span)  # first window always reaches the detection point
            offs = sorted((o for o in range(new_lo, new_hi) if o not in scanned), key=lambda o: abs(o - hint))
            for off in offs:
                scanned.add(off)
                for mask in _ONE_BIT:
                    consider(off, mask)
                # the first window is always scanned fully (quality); wider ones stop early
                if (si > 0 and enough()) or time.monotonic() > deadline:
                    break
            if acceptable() or time.monotonic() > deadline or (new_lo == spec.start and new_hi == hi):
                break
        if not acceptable() and time.monotonic() <= deadline:
            for off in range(hi - 1, max(lo, e_byte - TWO_BIT_WINDOW) - 1, -1):
                for mask in _TWO_BIT:
                    consider(off, mask)
                if time.monotonic() > deadline:
                    break
        good = acceptable()
        if good:
            good.sort()
            _, _, off, mask = good[0]
            buf[off] ^= mask
            res = search(depth + 1, flips + [(off, mask)])  # re-decode from the start to confirm
            buf[off] ^= mask
            if res[0]:
                return res
        # No clean completion: try the most promising partial fixes (distinct outcomes),
        # deeper. Equivalent-looking flips can differ later in the interval, so a small
        # beam beats a greedy choice.
        best = (False, progress, float("inf"), list(flips))
        ranked = sorted(partials.items(), key=lambda kv: (-kv[0][0], kv[0][1]))[:BEAM]
        branches: list[tuple[int, int]] = []
        for (dec, _rough), lst in ranked:
            # flips with an identical prefix outcome can still differ later in the
            # interval: explore a couple of each, 0->1 repairs first (the likelier damage)
            lst.sort()
            branches += [(off, mask) for _d, off, mask in lst[:PER_OUTCOME]]
        for off, mask in branches:
            if time.monotonic() > deadline:
                break
            buf[off] ^= mask
            res = search(depth + 1, flips + [(off, mask)])
            buf[off] ^= mask
            if res[0] and (not best[0] or res[2] < best[2]):
                best = res
            elif not best[0] and not res[0] and res[1] > best[1]:
                best = res
        return best

    ok, progress, _rough, chosen = search(0, [])
    for off, mask in chosen:
        buf[off] ^= mask
    if ok:
        return True, chosen, "repaired"
    if time.monotonic() > deadline:
        return False, chosen, "budget"
    return False, chosen, "unrepaired" if chosen or expansions[0] else "unrepaired"


@register
class BitFlipRepair(Recipe):
    name = "bitflip_repair"
    priority = 50
    description = "Undo flipped bits in scan data by trial decoding up to the next restart marker"

    def _candidates(self, sr: D.ScanResult, attempted: set[int]) -> list[D.Interval]:
        out = []
        for iv in sr.intervals:
            if iv.ok or iv.error is None or iv.error.reason not in REPAIRABLE:
                continue
            if iv.index in attempted:
                continue
            if iv.byte_end is not None and iv.byte_end <= iv.byte_start and iv.resync_at is None:
                continue
            out.append(iv)
        return out

    def detect(self, ctx: Context):
        got = decode_for(ctx)
        if got is None:
            return None
        attempted = ctx.memo.setdefault("bitflip_attempted", set())
        return got if self._candidates(got[2], attempted) else None

    def fix(self, ctx: Context, got) -> None:
        parse, ft, sr = got
        attempted: set[int] = ctx.memo.setdefault("bitflip_attempted", set())
        budget = float(ctx.options.get("time_budget", DEFAULT_TIME_BUDGET))
        deadline = time.monotonic() + budget
        buf = bytearray(ctx.data)
        baseline = _roughness(sr, sr, ft, 0, sr.total_mcus)
        all_flips: list[list[int]] = []
        repaired = unrepaired = 0
        notes: dict[str, int] = {}
        skipped_for_time = 0
        cands = self._candidates(sr, attempted)
        per_interval = min(PER_INTERVAL_MAX, max(PER_INTERVAL_MIN, budget / max(1, len(cands))))
        for iv in cands:
            if time.monotonic() > deadline:
                skipped_for_time += 1
                continue
            attempted.add(iv.index)
            iv_deadline = min(deadline, time.monotonic() + per_interval)
            ok, flips, note = repair_interval(buf, parse, ft, sr, iv, iv_deadline, baseline)
            all_flips += [[o, m] for o, m in flips]
            if ok:
                repaired += 1
            else:
                unrepaired += 1
                notes[note] = notes.get(note, 0) + 1
        msg = f"{repaired} interval(s) repaired with {len(all_flips)} bit flip(s); {unrepaired} unrepaired"
        if notes:
            msg += " (" + ", ".join(f"{k}: {v}" for k, v in notes.items()) + ")"
        if skipped_for_time:
            msg += f"; {skipped_for_time} skipped: time budget {budget:.0f}s exhausted"
        if all_flips:
            ctx.replace_data(bytes(buf), self, msg, flips=all_flips, repaired=repaired, unrepaired=unrepaired)
        else:
            ctx.note(self, msg, repaired=0, unrepaired=unrepaired)
