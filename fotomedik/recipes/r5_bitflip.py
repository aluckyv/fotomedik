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
DEFAULT_TIME_BUDGET = 120.0  # seconds per file
ANOMALY_LEVELS = 80

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

    All scan components count (a wrong flip often shifts only a chroma plane).
    Vertical differences use the full-file decode `base` for the rows above/below
    (only where those MCUs decoded cleanly), horizontal ones the candidate itself.
    """
    frame = ft.frame
    interleaved = len(ft.scan.components) > 1
    total = 0.0
    n = 0
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
                        total += abs(a - arr[brow, bcol - 1]) * q
                        n += 1
                    if by == 0 and brow > 0:
                        up_m = ((brow - 1) // v) * per_row + bcol // h
                        if base.mcu_ok[up_m] and not math.isnan(barr[brow - 1, bcol]):
                            total += abs(a - barr[brow - 1, bcol]) * q
                            n += 1
                    if by == v - 1 and brow + 1 < barr.shape[0]:
                        dn_m = ((brow + 1) // v) * per_row + bcol // h
                        if dn_m < len(base.mcu_ok) and base.mcu_ok[dn_m] and not math.isnan(barr[brow + 1, bcol]):
                            total += abs(a - barr[brow + 1, bcol]) * q
                            n += 1
    return total / n if n else 0.0


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
    flips: list[tuple[int, int]] = []

    def decode_from(start_byte, start_bit, first_mcu, preds):
        return D.decode_scan(
            bytes(buf),
            parse,
            ft,
            start_byte=start_byte,
            start_bit=start_bit,
            end_byte=spec.decode_end,
            first_mcu=first_mcu,
            max_mcus=spec.count - (first_mcu - spec.first),
            init_preds=preds,
        )

    for _round in range(MAX_ROUNDS):
        cur = decode_from(spec.start, 0, spec.first, None)
        if _complete(cur, spec.count, spec.marker_pos):
            return True, flips, "repaired"
        if time.monotonic() > deadline:
            return False, flips, "budget"
        progress = _decoded(cur)
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
            hint = min(e_byte, cur.mcus[mi - spec.first].bit_start // 8 + 2)
        hi = min(e_byte + W_FWD, spec.decode_end - (2 if spec.marker_pos is not None else 0))
        lo = max(spec.start, hint - W_BACK)
        if hi <= lo:
            return False, flips, "unrepaired"
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

        best_partial: tuple[int, float, int, int] | None = None  # (decoded, roughness, off, mask)
        complete: list[tuple[float, int, int, int]] = []  # (roughness, direction_penalty, off, mask)

        def consider(off: int, mask: int) -> None:
            nonlocal best_partial
            was_zero = not (buf[off] & mask)
            buf[off] ^= mask
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
            # Among partial fixes prefer more progress, then the smoother decoded prefix
            # (a wrong flip that happens to decode further usually leaves a DC step behind).
            if best_partial is None or dec > best_partial[0] + MIN_GAIN or abs(dec - best_partial[0]) <= MIN_GAIN:
                # judge smoothness on the prefix before the next error (the last few MCUs
                # before a detection are usually already garbage)
                n_ok = max(1, dec - (m0 - spec.first) - MIN_GAIN)
                rough = _roughness(r, base, ft, m0, n_ok)
                if best_partial is None or dec > best_partial[0] + MIN_GAIN or rough < best_partial[1]:
                    best_partial = (dec, rough, off, mask)

        limit_rough = max(1.5 * baseline, baseline + 6.0)

        def acceptable() -> list:
            return [c for c in complete if c[0] <= limit_rough]

        # Single-bit flips in the window; widen the window while nothing acceptable turns up.
        scanned_lo = hi
        for span in (W_BACK, 3 * W_BACK, 8 * W_BACK):
            new_lo = max(spec.start, hint - span)
            for off in range(scanned_lo - 1, new_lo - 1, -1):
                for mask in _ONE_BIT:
                    consider(off, mask)
                if time.monotonic() > deadline:
                    break
            scanned_lo = new_lo
            if acceptable() or time.monotonic() > deadline or new_lo == spec.start:
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
            flips.append((off, mask))
            continue  # re-decode from the start to confirm (and catch leftovers)
        if best_partial is not None and best_partial[0] >= progress + MIN_GAIN:
            _, _, off, mask = best_partial
            buf[off] ^= mask
            flips.append((off, mask))
            continue
        return False, flips, "unrepaired" if time.monotonic() <= deadline else "budget"
    return False, flips, "too many errors"


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
        for iv in self._candidates(sr, attempted):
            if time.monotonic() > deadline:
                skipped_for_time += 1
                continue
            attempted.add(iv.index)
            ok, flips, note = repair_interval(buf, parse, ft, sr, iv, deadline, baseline)
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
