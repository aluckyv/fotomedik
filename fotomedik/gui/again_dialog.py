"""Another enhanced copy of one repaired photo, with other steps: the person picks the steps,
the photo is run again (its repair is quick) and the new copy lands next to the first one as
<name>_enhanced_2.jpg, _3.jpg... Each try is a row of its own in the results, so they can be
looked at side by side. The first copy is never overwritten."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from ..ai.run import STEP_NAMES, enhanced_name, free_suffix as next_suffix

STEPS = ("puzzle", "burst", "deblock", "ai", "thumbsr", "patch", "faces", "diffusion")


class AgainDialog(QDialog):
    def __init__(self, parent, result):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle(f"Enhance again — {result.source.name}")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        prev = (result.options or {}).get("enhance") or []
        hint = QLabel("Another enhanced copy of this photo, with the steps below. The first copy stays; "
                      "the new one is saved as " + Path(enhanced_name(result.output.name, next_suffix(result.output))).name
                      + " and shown as a new row.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.steps: dict[str, QCheckBox] = {}
        for key in STEPS:
            cb = QCheckBox(STEP_NAMES[key].capitalize())
            cb.setChecked(key in prev)
            lay.addWidget(cb)
            self.steps[key] = cb
        form = QFormLayout()
        self.enlarger = QComboBox()
        for key, label in (("swinir", "SwinIR (faithful)"), ("esrgan", "Real-ESRGAN (sharp)"),
                           ("esrnet", "Real-ESRNet (soft)")):
            self.enlarger.addItem(label, key)
        cur = (result.options or {}).get("enlarger") or "swinir"
        self.enlarger.setCurrentIndex(max(0, self.enlarger.findData(cur)))
        form.addRow("Thumbnail enlarger:", self.enlarger)
        self.quality = QComboBox()
        for key, label in (("fast", "Fast"), ("balanced", "Balanced"), ("full", "Full size")):
            self.quality.addItem(label, key)
        self.quality.setCurrentIndex(max(0, self.quality.findData((result.options or {}).get("quality") or "balanced")))
        form.addRow("Diffusion:", self.quality)
        self.scene = QLineEdit((result.options or {}).get("prompt") or "")
        self.scene.setPlaceholderText("what the photo shows, for the diffusion (optional)")
        form.addRow("Scene:", self.scene)
        lay.addLayout(form)
        row = QHBoxLayout()
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Enhance again")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        lay.addLayout(row)

    def options(self) -> dict:
        """The run's options: the first run's, with the new steps and a fresh copy name."""
        r = self.result
        options = dict(r.options or {})
        options.pop("reference_dir", None)
        options["enhance"] = [k for k, cb in self.steps.items() if cb.isChecked()]
        options["enlarger"] = self.enlarger.currentData()
        options["quality"] = self.quality.currentData()
        options["prompt"] = self.scene.text().strip() or None
        options["ai_suffix"] = next_suffix(r.output)
        return options
