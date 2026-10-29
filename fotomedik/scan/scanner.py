"""Run every detector over a file and produce a Diagnosis (read-only)."""

from __future__ import annotations

from ..jpeg import decoder as D
from ..jpeg import markers as M
from ..jpeg.tables import FrameTables, TableError
from . import garbage as G
from .diagnosis import Diagnosis

CARDRECOVERY_KEY = 0x76
CARDRECOVERY_PREFIX = 512


def looks_cardrecovery(data: bytes) -> bool:
    return len(data) > 4 and data[:2] != b"\xff\xd8" and (data[0] ^ CARDRECOVERY_KEY, data[1] ^ CARDRECOVERY_KEY) == (
        0xFF,
        0xD8,
    )


def scan_bytes(data: bytes, name: str = "") -> Diagnosis:
    d = Diagnosis(name=name, size=len(data))

    if looks_cardrecovery(data):
        d.add("cardrecovery_xor", "high", 0, "CardRecovery temp file: first 512 bytes XOR 0x76")
        return d

    parse = M.parse_segments(data)
    if parse.errors and parse.errors[0].reason == "no_soi":
        d.add("not_jpeg", "fatal", 0, "no JPEG start marker")
        return d

    for e in parse.errors:
        d.add("header_error", "high", e.offset, f"{e.reason}: {e.detail}", reason=e.reason)

    sof = parse.sof
    if sof is not None:
        try:
            from ..jpeg.tables import parse_sof

            fi = parse_sof(sof.marker, sof.payload)
            d.width, d.height = fi.width, fi.height
            if fi.progressive:
                d.add("unsupported_progressive", "medium", sof.offset, "progressive JPEG: scan repair not supported in v0.1")
                d.openable = not parse.errors
                d.recovered_fraction = 1.0 if not parse.errors else 0.0
                return d
        except TableError as te:
            d.add("header_error", "high", sof.offset, f"SOF invalid: {te}", reason="bad_sof")

    if parse.eoi_offset is None:
        d.add("no_eoi", "low", len(data), "no end-of-image marker")

    if parse.scan_start is None:
        return d

    try:
        ft = FrameTables.from_parse(parse)
    except TableError as te:
        d.add("header_error", "high", None, f"table invalid: {te}", reason="bad_table")
        return d
    d.restart_interval = ft.restart_interval

    try:
        sr = D.decode_scan(data, parse, ft)
    except D.MissingTables as me:
        d.add("missing_tables", "high", None, str(me))
        return d
    except D.Unsupported as ue:
        d.add("unsupported_progressive", "medium", None, str(ue))
        d.openable = True
        return d

    d.openable = True
    d.total_mcus = sr.total_mcus
    d.total_intervals = len(sr.intervals)
    d.recovered_fraction = sr.recovered_fraction
    bad = [iv for iv in sr.intervals if not iv.ok]
    d.bad_intervals = len(bad)

    errs = [iv.error for iv in bad if iv.error is not None and iv.error.reason != "skipped"]
    if errs:
        first = errs[0]
        d.add(
            "scan_error",
            "high" if sr.recovered_fraction < 0.5 else "medium",
            first.byte_offset,
            f"{len(errs)} damaged interval(s); first: {first.reason} at MCU {first.mcu_index}",
            count=len(errs),
            first_reason=first.reason,
            intervals=[(iv.index, iv.error.reason, iv.error.byte_offset) for iv in bad[:50] if iv.error],
        )
    if any(e.reason == "eof" for e in errs):
        d.add("truncated", "high", len(data), "file ends before the image is complete")

    damaged_rst = sum(1 for iv in sr.intervals if iv.end_marker_damaged)
    if damaged_rst:
        d.add("damaged_rst", "low", None, f"{damaged_rst} restart marker(s) with flipped bits", count=damaged_rst)

    tail = G.garbage_tail_start(data, parse.scan_start)
    if tail is not None:
        d.add("garbage_tail", "medium", tail, f"{len(data) - tail} bytes of non-image data at the end", bytes=len(data) - tail)

    return d
