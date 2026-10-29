"""FileReport: what was wrong, what was done, how much came back."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..recipes.base import Context

STRINGS = {
    "sr": {
        "title": "FotoMedik izveštaj",
        "file": "Fajl",
        "recovered": "Vraćeno slike",
        "findings": "Nađena oštećenja",
        "none": "nema",
        "applied": "Primenjene popravke",
        "holes": "Rupe (nepopravljeni delovi)",
        "hole_line": "MCU {first}–{last} ({count} blokova): {reason}",
        "fallback_thumb": "Glavna slika nije spasiva — sačuvan je ugrađeni thumbnail.",
        "thumb_guided": "Rupe su popunjene prema ugrađenoj sličici (thumbnail): tačne boje i grubi oblici, bez sitnih detalja.",
        "output": "Izlazni fajl",
        "no_output": "nije napravljen (nema šta da se sačuva)",
        "kind": {
            "not_jpeg": "fajl nije JPEG (ili je početak uništen)",
            "overwritten": "u fajlu nema fotografije — sadržaj je prepisan drugim podacima",
            "cardrecovery_xor": "CardRecovery privremeni fajl (XOR kodiran)",
            "header_error": "oštećen header (markeri/dužine/tabele)",
            "missing_tables": "nedostaju tabele u headeru",
            "unsupported_progressive": "progressive JPEG — popravka skena nije podržana",
            "scan_error": "oštećeni podaci slike (izvrnuti bitovi)",
            "truncated": "fajl odsečen pre kraja slike",
            "garbage_tail": "tuđi sadržaj na kraju fajla",
            "no_eoi": "nedostaje završni marker",
            "damaged_rst": "oštećeni restart markeri",
            "internal_error": "greška u programu pri obradi ovog fajla",
        },
        "recipe": {
            "cardrecovery_xor": "dekodovan CardRecovery fajl",
            "header_repair": "popravljen header",
            "reference_header": "header zamenjen iz referentne slike",
            "garbage_tail": "odsečen tuđi sadržaj",
            "bitflip_repair": "vraćeni izvrnuti bitovi",
            "fill_holes": "popunjene rupe",
            "thumbnail_fallback": "iskorišćen ugrađeni thumbnail",
            "smooth_holes": "rupe izglađene (bez kockica)",
            "foreign_sectors": "izbačeni sektori koji nisu deo slike (nule, tekst, drugi fajlovi)",
            "orientation": "vraćena orijentacija (uspravna slika)",
        },
    },
    "en": {
        "title": "FotoMedik report",
        "file": "File",
        "recovered": "Image recovered",
        "findings": "Damage found",
        "none": "none",
        "applied": "Repairs applied",
        "holes": "Holes (unrecovered parts)",
        "hole_line": "MCU {first}–{last} ({count} blocks): {reason}",
        "fallback_thumb": "Main image unrecoverable — the embedded thumbnail was saved instead.",
        "thumb_guided": "Holes were filled following the embedded thumbnail: right colours and rough shapes, no fine detail.",
        "output": "Output file",
        "no_output": "none (nothing to save)",
        "kind": {
            "not_jpeg": "not a JPEG (or the start is destroyed)",
            "overwritten": "no photo in the file — its content was overwritten by other data",
            "cardrecovery_xor": "CardRecovery temp file (XOR encoded)",
            "header_error": "damaged header (markers/lengths/tables)",
            "missing_tables": "tables missing from header",
            "unsupported_progressive": "progressive JPEG — scan repair not supported",
            "scan_error": "damaged image data (flipped bits)",
            "truncated": "file cut off before the end of the image",
            "garbage_tail": "foreign data at the end of the file",
            "no_eoi": "end marker missing",
            "damaged_rst": "damaged restart markers",
            "internal_error": "program error while processing this file",
        },
        "recipe": {
            "cardrecovery_xor": "CardRecovery file decoded",
            "header_repair": "header repaired",
            "reference_header": "header replaced from reference photo",
            "garbage_tail": "foreign data cut off",
            "bitflip_repair": "flipped bits restored",
            "fill_holes": "holes filled",
            "thumbnail_fallback": "embedded thumbnail used",
            "smooth_holes": "holes smoothed (no squares)",
            "foreign_sectors": "foreign sectors blanked (zeros, text, other files)",
            "orientation": "orientation restored (photo shows upright)",
        },
    },
}


@dataclass
class FileReport:
    name: str
    output_name: str | None
    size_in: int
    size_out: int
    recovered_pct: float | None
    findings: list[dict] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    holes: list[dict] = field(default_factory=list)
    fallback: str | None = None
    width: int | None = None
    height: int | None = None
    # holes nothing is known about (no thumbnail there): MCU ranges, for the AI copy
    unknown: list = field(default_factory=list, repr=False)
    ai_output: str | None = None  # the enhanced copy, when one was made
    enhanced: list = field(default_factory=list)  # what went into it

    @classmethod
    def from_context(cls, ctx: Context, findings_before: list[dict] | None = None, width=None, height=None) -> "FileReport":
        pct = None if ctx.recovered_fraction is None else round(100 * ctx.recovered_fraction, 1)
        return cls(
            name=ctx.name,
            output_name=ctx.suggested_name or ctx.name,
            size_in=len(ctx.original),
            size_out=len(ctx.data),
            recovered_pct=pct,
            findings=list(findings_before or []),
            applied=list(ctx.applied),
            log=[e.to_dict() for e in ctx.log],
            holes=[h.to_dict() for h in ctx.holes],
            fallback=ctx.fallback,
            width=width,
            height=height,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "output_name": self.output_name,
            "size_in": self.size_in,
            "size_out": self.size_out,
            "width": self.width,
            "height": self.height,
            "recovered_pct": self.recovered_pct,
            "findings": self.findings,
            "applied": self.applied,
            "holes": self.holes,
            "fallback": self.fallback,
            "enhanced_output": self.ai_output,
            "enhanced": self.enhanced,
            "log": self.log,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def format_text(self, lang: str = "sr") -> str:
        s = STRINGS.get(lang, STRINGS["en"])
        lines = [f"{s['title']}: {self.name}"]
        if self.recovered_pct is not None:
            lines.append(f"{s['recovered']}: {self.recovered_pct:.1f}%")
        if self.findings:
            lines.append(f"{s['findings']}:")
            for f in self.findings:
                desc = s["kind"].get(f["kind"], f["kind"])
                where = f" @ {f['offset']}" if f.get("offset") is not None else ""
                lines.append(f"  - {desc}{where}")
        else:
            lines.append(f"{s['findings']}: {s['none']}")
        if self.applied:
            lines.append(f"{s['applied']}:")
            for a in self.applied:
                lines.append(f"  - {s['recipe'].get(a, a)}")
        if self.holes:
            lines.append(f"{s['holes']}: {len(self.holes)}")
            for h in self.holes[:20]:
                lines.append(
                    "  - " + s["hole_line"].format(first=h["mcu_first"], last=h["mcu_first"] + h["mcu_count"] - 1, count=h["mcu_count"], reason=h["reason"])
                )
            if len(self.holes) > 20:
                lines.append(f"  … +{len(self.holes) - 20}")
        if any(e.get("recipe") == "fill_holes" and (e.get("extra") or {}).get("thumbnail_guided") for e in self.log):
            lines.append(s["thumb_guided"])
        if self.fallback == "thumbnail":
            lines.append(s["fallback_thumb"])
        lines.append(f"{s['output']}: {self.output_name or s['no_output']}")
        return "\n".join(lines)
