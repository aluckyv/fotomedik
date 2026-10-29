"""Assemble a JPEG file from header bytes and entropy-coded scan bytes."""

from __future__ import annotations

EOI = b"\xff\xd9"


def assemble(header: bytes, scan: bytes) -> bytes:
    """header (SOI .. end of SOS segment) + scan data + EOI."""
    return header + scan + EOI


def rst_marker(index: int) -> bytes:
    """Restart marker closing interval `index` (RST0 follows interval 0)."""
    return bytes([0xFF, 0xD0 + (index % 8)])
