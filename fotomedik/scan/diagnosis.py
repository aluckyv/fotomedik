"""Diagnosis data model shared by the scanner, the recipes and the report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

SEVERITIES = ("low", "medium", "high", "fatal")

# Finding kinds (stable identifiers; the report maps them to sr/en text):
#   not_jpeg, cardrecovery_xor, header_error, missing_tables, unsupported_progressive,
#   scan_error, truncated, garbage_tail, no_eoi, damaged_rst


@dataclass
class Finding:
    kind: str
    severity: str
    offset: int | None = None
    detail: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "severity": self.severity, "offset": self.offset, "detail": self.detail}
        if self.extra:
            d["extra"] = dict(self.extra)
        return d


@dataclass
class Diagnosis:
    name: str
    size: int
    findings: list[Finding] = field(default_factory=list)
    recovered_fraction: float = 0.0  # share of MCUs that decode cleanly as-is
    openable: bool = False  # would a normal viewer open it at all (header parses, SOF present)
    width: int | None = None
    height: int | None = None
    restart_interval: int = 0
    total_mcus: int = 0
    bad_intervals: int = 0
    total_intervals: int = 0

    def add(self, kind: str, severity: str, offset: int | None = None, detail: str = "", **extra) -> Finding:
        f = Finding(kind, severity, offset, detail, dict(extra))
        self.findings.append(f)
        return f

    def has(self, kind: str) -> bool:
        return any(f.kind == kind for f in self.findings)

    @property
    def worst(self) -> str:
        if not self.findings:
            return "none"
        return max((f.severity for f in self.findings), key=SEVERITIES.index)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [f.to_dict() for f in self.findings]
        d["worst"] = self.worst
        return d
