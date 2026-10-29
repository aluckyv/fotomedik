"""CameraProfile: the header a camera writes, learned from one healthy photo.

Cameras write identical quantisation/Huffman tables, sampling factors and
restart intervals for every shot at a given quality setting, so a healthy
photo from the same camera is a complete spare header for a damaged one.
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import asdict, dataclass, field

from ..jpeg import markers as M
from ..jpeg.tables import TableError, parse_sof


def exif_model(data: bytes) -> tuple[str, str] | None:
    """(Make, Model) from the Exif APP1 segment, if present and readable."""
    res = M.parse_segments(data)
    for seg in res.all(M.APP1):
        p = seg.payload
        if not p.startswith(b"Exif\x00\x00") or len(p) < 16:
            continue
        t = p[6:]
        try:
            endian = "<" if t[:2] == b"II" else ">" if t[:2] == b"MM" else None
            if endian is None:
                return None
            ifd = struct.unpack(endian + "I", t[4:8])[0]
            n = struct.unpack(endian + "H", t[ifd : ifd + 2])[0]
            out = {}
            for i in range(min(n, 200)):
                e = ifd + 2 + 12 * i
                tag, typ, cnt = struct.unpack(endian + "HHI", t[e : e + 8])
                if tag in (0x010F, 0x0110) and typ == 2:
                    off = struct.unpack(endian + "I", t[e + 8 : e + 12])[0] if cnt > 4 else e + 8
                    s = t[off : off + cnt].split(b"\x00")[0].decode("ascii", "replace").strip()
                    out[tag] = s
            if out:
                return (out.get(0x010F, ""), out.get(0x0110, ""))
        except (struct.error, IndexError):
            return None
    return None


@dataclass
class CameraProfile:
    make: str = ""
    model: str = ""
    width: int = 0
    height: int = 0
    restart_interval: int = 0
    sof_marker: int = 0xC0
    dqt: list[bytes] = field(default_factory=list)  # DQT payloads
    dht: list[bytes] = field(default_factory=list)  # DHT payloads
    sof: bytes = b""  # SOF payload (dimensions are patched on use)
    sos: bytes = b""  # SOS payload
    jfif: bytes | None = None  # APP0 payload, if the camera writes one
    source: str = ""
    votes: int = 1  # photos of the set that carry exactly this header

    @classmethod
    def from_jpeg(cls, data: bytes, source: str = "") -> "CameraProfile":
        res = M.parse_segments(data)
        if not res.ok or res.sof is None:
            raise ValueError("reference image header does not parse")
        sof = res.sof
        fi = parse_sof(sof.marker, sof.payload)
        mm = exif_model(data) or ("", "")
        dri = res.first(M.DRI)
        app0 = res.first(M.APP0)
        return cls(
            make=mm[0],
            model=mm[1],
            width=fi.width,
            height=fi.height,
            restart_interval=((dri.payload[0] << 8) | dri.payload[1]) if dri else 0,
            sof_marker=sof.marker,
            dqt=[s.payload for s in res.all(M.DQT)],
            dht=[s.payload for s in res.all(M.DHT)],
            sof=sof.payload,
            sos=res.first(M.SOS).payload,
            jfif=app0.payload if app0 and app0.payload.startswith(b"JFIF") else None,
            source=source,
        )

    # ---------------------------------------------------------------- serialisation

    def to_json(self) -> str:
        d = asdict(self)
        for k in ("dqt", "dht"):
            d[k] = [base64.b64encode(x).decode() for x in d[k]]
        for k in ("sof", "sos", "jfif"):
            d[k] = base64.b64encode(d[k]).decode() if d[k] is not None else None
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "CameraProfile":
        d = json.loads(text)
        for k in ("dqt", "dht"):
            d[k] = [base64.b64decode(x) for x in d[k]]
        for k in ("sof", "sos", "jfif"):
            d[k] = base64.b64decode(d[k]) if d[k] is not None else None
        return cls(**d)

    # ---------------------------------------------------------------- use

    def header_bytes(self, width: int | None = None, height: int | None = None, extra_segments: list[bytes] | None = None) -> bytes:
        """A complete header (SOI .. SOS) for an image of the given size."""
        w = width or self.width
        h = height or self.height
        sof = bytearray(self.sof)
        sof[1:3] = h.to_bytes(2, "big")
        sof[3:5] = w.to_bytes(2, "big")
        out = bytearray(b"\xff\xd8")
        if self.jfif is not None:
            out += _segment(M.APP0, self.jfif)
        for seg in extra_segments or []:
            out += seg
        for p in self.dqt:
            out += _segment(M.DQT, p)
        out += _segment(self.sof_marker, bytes(sof))
        for p in self.dht:
            out += _segment(M.DHT, p)
        if self.restart_interval:
            out += _segment(M.DRI, self.restart_interval.to_bytes(2, "big"))
        out += _segment(M.SOS, self.sos)
        return bytes(out)

    @property
    def sos_bytes(self) -> bytes:
        return _segment(M.SOS, self.sos)

    def matches(self, data: bytes) -> int:
        """Crude affinity score of this profile for a damaged file (higher = better)."""
        score = 0
        mm = exif_model(data)
        if mm and self.model and mm[1] == self.model:
            score += 10
        if mm and self.make and mm[0] == self.make:
            score += 2
        try:
            res = M.parse_segments(data)
            if res.sof is not None:
                fi = parse_sof(res.sof.marker, res.sof.payload)
                if (fi.width, fi.height) == (self.width, self.height):
                    score += 3
        except TableError:
            pass
        return score


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload
