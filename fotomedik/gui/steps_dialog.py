"""Step by step: how one photo was recovered - the picture before and after every
repair step, with what the step does (the repair is replayed for this)."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from .. import runner
from .preview import preview_bytes


class _Replay(QObject):
    done = Signal(object, str)


class StepsDialog(QDialog):
    def __init__(self, parent, result: runner.Result):
        super().__init__(parent)
        from .app import ImageBox

        self.setWindowTitle(f"Step by step — {result.source.name}")
        self.resize(1250, 800)
        self.result = result
        self.steps: list[runner.Step] = []
        self._previews: dict[int, object] = {}

        root = QVBoxLayout(self)
        self.head = QLabel("Replaying the repair of this photo…")
        root.addWidget(self.head)
        body = QHBoxLayout()
        self.list = QListWidget()
        self.list.setFixedWidth(280)
        self.list.currentRowChanged.connect(self._show)
        body.addWidget(self.list)
        right = QVBoxLayout()
        pics = QHBoxLayout()
        self.before = ImageBox("Before this step")
        self.after = ImageBox("After this step")
        pics.addWidget(self.before, 1)
        pics.addWidget(self.after, 1)
        right.addLayout(pics, 4)
        self.expl = QLabel("")
        self.expl.setWordWrap(True)
        right.addWidget(self.expl)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMaximumHeight(110)
        right.addWidget(self.detail)
        nav = QHBoxLayout()
        self.prevb = QPushButton("◀ Previous step")
        self.prevb.clicked.connect(lambda: self.list.setCurrentRow(max(0, self.list.currentRow() - 1)))
        self.nextb = QPushButton("Next step ▶")
        self.nextb.clicked.connect(lambda: self.list.setCurrentRow(min(self.list.count() - 1, self.list.currentRow() + 1)))
        nav.addWidget(self.prevb)
        nav.addWidget(self.nextb)
        nav.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        nav.addWidget(close)
        right.addLayout(nav)
        body.addLayout(right, 1)
        root.addLayout(body, 1)

        self._sig = _Replay()
        self._sig.done.connect(self._loaded)
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        try:
            steps, err = runner.replay(self.result), ""
        except Exception as e:  # shown in the window
            steps, err = [], f"{type(e).__name__}: {e}"
        self._sig.done.emit(steps, err)

    def _loaded(self, steps, err: str) -> None:
        self.steps = steps
        if err:
            self.head.setText(f"The repair could not be replayed: {err}")
            return
        rep = self.result.report
        pct = "" if rep.recovered_pct is None else f" — {rep.recovered_pct:.0f} % of the picture is real data"
        self.head.setText(f"How {self.result.source.name} was recovered: {len(steps)} step(s){pct}. "
                          "Pick a step to see the photo before and after it.")
        for i, st in enumerate(steps):
            self.list.addItem(f"{i}. {st.title}" if i else st.title)
        if steps:
            self.list.setCurrentRow(min(1, len(steps) - 1))

    def _preview(self, i: int):
        if i not in self._previews:
            d = self.steps[i].data
            self._previews[i] = preview_bytes(d) if d is not None else None
        return self._previews[i]

    def _show(self, i: int) -> None:
        if not 0 <= i < len(self.steps):
            return
        st = self.steps[i]
        prev = next((j for j in range(i - 1, -1, -1) if self.steps[j].data is not None), None)
        self.before.show_image(self._preview(prev) if prev is not None and i > 0 else None,
                               "—" if i == 0 else "Nothing to compare")
        self.after.show_image(self._preview(i) if st.data is not None else None,
                              "No picture for this step" if st.data is None else "Does not open in a normal viewer")
        self.expl.setText(f"<b>{st.title}</b><br>{st.explanation}")
        self.detail.setPlainText(st.detail)
        self.prevb.setEnabled(i > 0)
        self.nextb.setEnabled(i < len(self.steps) - 1)
