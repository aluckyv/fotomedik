"""Compare window: a repaired photo against an intact copy (see fotomedik.compare)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout

from .. import compare as C
from .. import runner


class CompareDialog(QDialog):
    def __init__(self, parent, result: runner.Result, truth: Path):
        super().__init__(parent)
        self.setWindowTitle(f"Compared with the original — {result.source.name}")
        self.resize(1400, 700)
        report = {"holes": result.report.holes}
        self.cmp = C.compare(result.output, truth, report)
        self.pic = C.side_by_side(result.output, truth, self.cmp)
        lay = QVBoxLayout(self)
        head = QLabel("Left: the intact copy.  Middle: the repaired copy.  Right: error per block "
                      "(brighter = further from the truth; filled holes count too).")
        head.setWordWrap(True)
        lay.addWidget(head)
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setMinimumSize(600, 200)
        lay.addWidget(self.view, 1)
        text = QLabel(self.cmp.format_text())
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(text)
        row = QHBoxLayout()
        row.addStretch(1)
        b = QPushButton("Save picture")
        b.clicked.connect(self._save)
        row.addWidget(b)
        b = QPushButton("Close")
        b.clicked.connect(self.accept)
        row.addWidget(b)
        lay.addLayout(row)
        self.result = result
        self._rescale()

    def _rescale(self) -> None:
        from .app import _to_pixmap

        pm = _to_pixmap(self.pic)
        self.view.setPixmap(pm.scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self._rescale()

    def _save(self) -> None:
        target = self.result.output.with_name(self.result.output.stem + "_compared.jpg")
        self.pic.save(target, quality=90)
        self.setWindowTitle(f"Compared with the original — saved: {target.name}")
