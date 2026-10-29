"""Manual repair window: click a place in the photo, then shift colours, move the
picture or cut junk bits from there on, with an instant preview (see fotomedik.manual)."""

from __future__ import annotations

from pathlib import Path

from PIL import ImageDraw
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from ..manual import Editor

PREVIEW = 1400


class ClickImage(QLabel):
    clicked = Signal(float, float)  # full-size picture coordinates

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._pix = None
        self._full = (1, 1)

    def show_pixmap(self, pix, full_size) -> None:
        self._pix = pix
        self._full = full_size
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self._rescale()

    def mousePressEvent(self, e):  # noqa: N802
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return
        ox = (self.width() - pm.width()) / 2
        oy = (self.height() - pm.height()) / 2
        x, y = e.position().x() - ox, e.position().y() - oy
        if 0 <= x < pm.width() and 0 <= y < pm.height():
            self.clicked.emit(x * self._full[0] / pm.width(), y * self._full[1] / pm.height())


class ManualDialog(QDialog):
    def __init__(self, parent, source: Path, original: Path | None, out_dir: Path):
        super().__init__(parent)
        self.setWindowTitle(f"Fix by hand — {source.name}")
        self.resize(1300, 860)
        self.source, self.original, self.out_dir = source, original, out_dir
        self.editor = Editor(source.read_bytes())
        self.mcu = 0

        root = QHBoxLayout(self)
        self.view = ClickImage()
        self.view.clicked.connect(self._pick)
        root.addWidget(self.view, 1)

        side = QVBoxLayout()
        help_ = QLabel("Click where the damage starts (the first wrong block).\n"
                       "Changes apply from that block onwards. Undo any step.")
        help_.setWordWrap(True)
        side.addWidget(help_)
        self.where = QLabel("No place chosen yet.")
        self.where.setWordWrap(True)
        side.addWidget(self.where)

        col = QGroupBox("Colour from here on")
        cl = QVBoxLayout(col)
        self.sliders = {}
        for key, label in (("dy", "Brightness"), ("dcb", "Blue ↔ yellow"), ("dcr", "Red ↔ green")):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(-120, 120)
            s.setValue(0)
            val = QLabel("0")
            s.valueChanged.connect(lambda v, lab=val: lab.setText(str(v)))
            row.addWidget(s, 1)
            row.addWidget(val)
            cl.addLayout(row)
            self.sliders[key] = s
        b = QPushButton("Apply colour change")
        b.clicked.connect(self._colour)
        cl.addWidget(b)
        side.addWidget(col)

        mv = QGroupBox("Move the picture after this place")
        ml = QVBoxLayout(mv)
        row = QHBoxLayout()
        row.addWidget(QLabel("Blocks:"))
        self.count = QSpinBox()
        self.count.setRange(1, 500)
        row.addWidget(self.count)
        ml.addLayout(row)
        row = QHBoxLayout()
        b = QPushButton("◀ Delete blocks")
        b.setToolTip("Removes blocks here: everything after moves left (colours stay)")
        b.clicked.connect(lambda: self._do(lambda: self.editor.delete_mcus(self.mcu, self.count.value())))
        row.addWidget(b)
        b = QPushButton("Insert blocks ▶")
        b.setToolTip("Inserts plain blocks here: everything after moves right")
        b.clicked.connect(lambda: self._do(lambda: self.editor.insert_mcus(self.mcu, self.count.value())))
        row.addWidget(b)
        ml.addLayout(row)
        side.addWidget(mv)

        jk = QGroupBox("Junk data")
        jl = QVBoxLayout(jk)
        row = QHBoxLayout()
        row.addWidget(QLabel("Bits:"))
        self.bits = QSpinBox()
        self.bits.setRange(1, 1_000_000)
        self.bits.setValue(8)
        row.addWidget(self.bits)
        jl.addLayout(row)
        b = QPushButton("Delete bits here")
        b.setToolTip("Cuts bits where this block starts (or where decoding stopped, in a grey area).\n"
                     "8 bits = 1 byte; a disk sector is 4096 bits.")
        b.clicked.connect(lambda: self._do(lambda: self.editor.delete_bits(self.mcu, self.bits.value())))
        jl.addWidget(b)
        side.addWidget(jk)

        side.addStretch(1)
        row = QHBoxLayout()
        b = QPushButton("Undo")
        b.clicked.connect(lambda: self.editor.undo() and self._render())
        row.addWidget(b)
        if original is not None and original != source:
            b = QPushButton("Start from the original")
            b.setToolTip("Work on the damaged file as it came from the card instead of the repaired copy")
            b.clicked.connect(self._from_original)
            row.addWidget(b)
        side.addLayout(row)
        row = QHBoxLayout()
        b = QPushButton("Save copy")
        b.setObjectName("primary")
        b.clicked.connect(self._save)
        row.addWidget(b)
        b = QPushButton("Close")
        b.clicked.connect(self.reject)
        row.addWidget(b)
        side.addLayout(row)
        w = QVBoxLayout()
        w.addLayout(side)
        holder = QGroupBox()
        holder.setLayout(w)
        holder.setFixedWidth(330)
        root.addWidget(holder)
        self._render()

    # ------------------------------------------------------------------
    def _render(self) -> None:
        from .app import _to_pixmap

        im = self.editor.image(PREVIEW)
        full = (self.editor.ft.frame.width, self.editor.ft.frame.height)
        sx, sy = im.size[0] / full[0], im.size[1] / full[1]
        d = ImageDraw.Draw(im)
        e = self.editor
        r, c = divmod(self.mcu, e.per_row)
        x0, y0 = c * e.mcu_w * sx, r * e.mcu_h * sy
        d.rectangle([x0, y0, x0 + e.mcu_w * sx, y0 + e.mcu_h * sy], outline=(255, 0, 255), width=2)
        d.line([0, y0, x0, y0], fill=(255, 0, 255), width=1)
        self.view.show_pixmap(_to_pixmap(im), full)
        si, bit, exact = e.where(self.mcu)
        state = "decodes" if exact else "does not decode: edits act where decoding stopped"
        self.where.setText(f"Block {self.mcu} (row {r + 1}, column {c + 1}) — {state}.\n"
                           f"Position: segment {si + 1}, bit {bit}.  Steps to undo: {len(e.history)}")

    def _pick(self, x: float, y: float) -> None:
        self.mcu = self.editor.mcu_at(x, y)
        self._render()

    def _do(self, fn) -> None:
        try:
            fn()
        except Exception as e:
            QMessageBox.warning(self, "Fix by hand", str(e))
            return
        self._render()

    def _colour(self) -> None:
        v = {k: s.value() for k, s in self.sliders.items()}
        self._do(lambda: self.editor.shift_colour(self.mcu, v["dy"], v["dcb"], v["dcr"]))
        for s in self.sliders.values():
            s.setValue(0)

    def _from_original(self) -> None:
        try:
            self.editor = Editor(self.original.read_bytes())
        except Exception as e:
            QMessageBox.warning(self, "Fix by hand", f"The original cannot be edited: {e}")
            return
        self._render()

    def _save(self) -> None:
        stem = self.source.stem
        target = self.out_dir / f"{stem}_manual.jpg"
        target.write_bytes(self.editor.data())
        QMessageBox.information(self, "Fix by hand", f"Saved: {target}")
