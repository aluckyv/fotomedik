"""FotoMedik window: pick or drop photos, repair them, compare before and after."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QProgressDialog,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import runner
from . import updater
from .preview import load_preview

APP_NAME = "FotoMedik"
ACCENT = "#2f6fde"
STYLE = f"""
QMainWindow, QWidget {{ background: #f6f7f9; color: #1d2330; font-size: 10pt; }}
QLineEdit {{ background: white; border: 1px solid #cfd5df; border-radius: 6px; padding: 6px 8px; }}
QPushButton {{ background: white; border: 1px solid #cfd5df; border-radius: 6px; padding: 7px 14px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #9aa3b2; background: #eef0f3; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border: none; font-weight: 600; padding: 8px 22px; }}
QPushButton#primary:disabled {{ background: #9db6e8; }}
QTableWidget {{ background: white; border: 1px solid #dde2ea; border-radius: 6px; gridline-color: #eef0f3; }}
QHeaderView::section {{ background: #eef1f6; border: none; padding: 6px; font-weight: 600; }}
QPlainTextEdit {{ background: white; border: 1px solid #dde2ea; border-radius: 6px; font-family: Consolas, monospace; }}
QProgressBar {{ border: 1px solid #cfd5df; border-radius: 6px; background: white; text-align: center; height: 18px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QFrame#drop {{ border: 2px dashed #b9c3d3; border-radius: 10px; background: #fbfcfe; }}
QLabel#hint {{ color: #5b6578; }}
QLabel#imgcap {{ color: #5b6578; font-weight: 600; }}
"""


def _to_pixmap(im) -> QPixmap:
    im = im.convert("RGB")
    w, h = im.size
    qimg = QImage(im.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ImageBox(QWidget):
    """A caption and a picture that scales with the window."""

    def __init__(self, caption: str):
        super().__init__()
        self._pix: QPixmap | None = None
        self.cap = QLabel(caption)
        self.cap.setObjectName("imgcap")
        self.img = QLabel()
        self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img.setMinimumSize(QSize(200, 150))
        self.img.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.img.setStyleSheet("background: #e9ecf1; border-radius: 6px; color: #7b8496;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.cap)
        lay.addWidget(self.img, 1)

    def show_image(self, im, empty_text: str = "") -> None:
        self._pix = _to_pixmap(im) if im is not None else None
        if self._pix is None:
            self.img.setPixmap(QPixmap())
            self.img.setText(empty_text)
        else:
            self.img.setText("")
            self._rescale()

    def resizeEvent(self, e):  # noqa: N802 (Qt naming)
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is not None:
            self.img.setPixmap(self._pix.scaled(self.img.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))


class RepairWorker(QObject):
    """Runs the repair in a background thread and reports each finished photo."""

    result = Signal(object)
    ai_done = Signal(object)
    status = Signal(str)
    ai_status = Signal(str)
    finished = Signal(float, str)

    def __init__(self, paths: list[Path], out: Path, options: dict):
        super().__init__()
        self.paths, self.out, self.options = paths, out, options
        self.stop = threading.Event()

    def run(self) -> None:
        t = time.perf_counter()
        err = ""
        try:
            runner.run(self.paths, self.out, self.options, on_result=self.result.emit,
                       on_status=self.status.emit, stop=self.stop, on_ai=self.ai_done.emit,
                       on_ai_status=self.ai_status.emit)
        except Exception as e:  # shown to the user, the app keeps running
            err = f"{type(e).__name__}: {e}"
        self.finished.emit(time.perf_counter() - t, err)


class UpdateChecker(QObject):
    """Asks GitHub for a newer build without blocking the window."""

    found = Signal(object)

    def start(self) -> None:
        def work():
            try:
                upd = updater.check()
            except Exception:
                upd = None  # offline or not reachable: no update offered
            if upd is not None:
                self.found.emit(upd)

        threading.Thread(target=work, daemon=True).start()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        build = updater.current_build()
        self.setWindowTitle(f"{APP_NAME} — repair damaged photos" + (f"  (build {build})" if build else ""))
        self.resize(1280, 820)
        self.setAcceptDrops(True)
        self.results: list[runner.Result] = []
        self.thread: QThread | None = None
        self.worker: RepairWorker | None = None
        self.total = 0

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 12)
        outer.setSpacing(10)

        title = QLabel(APP_NAME)
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        title.setFont(f)
        sub = QLabel("Diagnoses and repairs damaged JPEG photos. Originals are never changed.")
        sub.setObjectName("hint")
        head = QVBoxLayout()
        head.setSpacing(2)
        head.addWidget(title)
        head.addWidget(sub)
        outer.addLayout(head)

        # ---- update banner (shown when GitHub has a newer build)
        self.banner = QFrame()
        self.banner.setStyleSheet("QFrame { background: #e7f0ff; border: 1px solid #b7cdf5; border-radius: 8px; }")
        bl = QHBoxLayout(self.banner)
        bl.setContentsMargins(12, 8, 12, 8)
        self.banner_text = QLabel("")
        bl.addWidget(self.banner_text, 1)
        self.update_btn = QPushButton("Update and restart")
        self.update_btn.setObjectName("primary")
        self.update_btn.clicked.connect(self._do_update)
        bl.addWidget(self.update_btn)
        later = QPushButton("Later")
        later.clicked.connect(self.banner.hide)
        bl.addWidget(later)
        self.banner.hide()
        outer.addWidget(self.banner)
        self.pending_update = None
        self.checker = UpdateChecker()
        self.checker.found.connect(self._offer_update)
        if updater.is_packaged():
            updater.cleanup_old()
            self.checker.start()

        # ---- input / output
        drop = QFrame()
        drop.setObjectName("drop")
        dl = QVBoxLayout(drop)
        dl.setContentsMargins(14, 12, 14, 12)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Photos:"))
        self.inp = QLineEdit()
        self.inp.setPlaceholderText("Drop a folder or photos here, or choose with the buttons")
        self.inp.textChanged.connect(self._suggest_output)
        row1.addWidget(self.inp, 1)
        b = QPushButton("Folder…")
        b.clicked.connect(self._pick_folder)
        row1.addWidget(b)
        b = QPushButton("Photos…")
        b.clicked.connect(self._pick_files)
        row1.addWidget(b)
        dl.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Save to:"))
        self.outp = QLineEdit()
        self.outp.setPlaceholderText("Folder for the repaired copies")
        row2.addWidget(self.outp, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self._pick_output)
        row2.addWidget(b)
        dl.addLayout(row2)
        outer.addWidget(drop)

        # ---- actions
        act = QHBoxLayout()
        self.go = QPushButton("Repair")
        self.go.setObjectName("primary")
        self.go.clicked.connect(self._start)
        act.addWidget(self.go)
        self.stopb = QPushButton("Stop")
        self.stopb.setEnabled(False)
        self.stopb.clicked.connect(self._stop)
        act.addWidget(self.stopb)
        self.mark = QCheckBox("Mark unrecoverable areas in magenta")
        act.addWidget(self.mark)
        self.reports = QCheckBox("Save a report file for each photo")
        self.reports.setToolTip("Writes <photo>.report.json next to every repaired copy")
        act.addWidget(self.reports)
        act.addStretch(1)
        self.openb = QPushButton("Open results folder")
        self.openb.setEnabled(False)
        self.openb.clicked.connect(self._open_output)
        act.addWidget(self.openb)
        outer.addLayout(act)

        # ---- enhanced copy (<photo>_enhanced.jpg): optional steps
        ext = QHBoxLayout()
        lab = QLabel("Enhanced copy:")
        lab.setToolTip("Also save <photo>_enhanced.jpg next to the repaired copy.\n"
                       "The repaired copy itself always stays faithful to the file.")
        ext.addWidget(lab)
        self.steps = {}
        tips = {
            "burst": "Fill holes with REAL pixels from another shot of the same scene\n"
                     "(taken within 3 minutes), aligned and colour-matched. Nothing invented.",
            "deblock": "Remove JPEG blocks and noise from the whole picture (FBCNN, ~290 MB once).",
            "ai": "Sharpen areas rebuilt from the thumbnail (Real-ESRGAN) and fill holes nothing\n"
                  "is known about (LaMa). Invented detail. ~275 MB once.",
            "faces": "Restore faces (GFPGAN, ~340 MB once). Can make a face look like someone else.",
            "diffusion": "Maximum quality: holes repainted by Stable Diffusion following the thumbnail.\n"
                         "Needs a graphics card (minutes per photo on the processor). ~2.1 GB once.",
        }
        from ..ai.run import STEP_NAMES

        for key in ("burst", "deblock", "ai", "faces", "diffusion"):
            cb = QCheckBox(STEP_NAMES[key].capitalize())
            cb.setToolTip(tips[key])
            ext.addWidget(cb)
            self.steps[key] = cb
        self.steps["burst"].setChecked(True)
        self.ai = self.steps["ai"]  # older name (smoke test)
        self.quality = QComboBox()
        for key, label in (("fast", "Fast"), ("balanced", "Balanced"), ("full", "Full size")):
            self.quality.addItem(label, key)
        self.quality.setCurrentIndex(1)
        self.quality.setToolTip("Maximum quality repaints a reduced copy of the photo, then enlarges the holes "
                                "back with Real-ESRGAN.\n"
                                "Fast: quarter size, 6 passes per window (~6 windows on a 12 MP photo).\n"
                                "Balanced: half size, 8 passes (~20 windows) — the network sees more of the "
                                "scene in each window.\n"
                                "Full size: 12 passes, ~80 windows; slowest.")
        self.quality.setEnabled(False)
        self.steps["diffusion"].toggled.connect(self.quality.setEnabled)
        ext.addWidget(self.quality)
        ext.addStretch(1)
        outer.addLayout(ext)
        scene = QHBoxLayout()
        lab = QLabel("Scene:")
        lab.setToolTip("A few words on what the photos show, for Maximum quality: Stable Diffusion\n"
                       "paints the holes to match (e.g. \"people on a square in front of a cathedral\").\n"
                       "Applies to every photo of this run; empty means a generic photograph.")
        scene.addWidget(lab)
        self.scene = QLineEdit()
        self.scene.setPlaceholderText("what the photos show, e.g. people on a square in front of a cathedral (optional)")
        self.scene.setEnabled(False)
        self.steps["diffusion"].toggled.connect(self.scene.setEnabled)
        scene.addWidget(self.scene, 1)
        outer.addLayout(scene)

        prog = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        prog.addWidget(self.bar, 1)
        self.stat = QLabel("Ready.")
        self.stat.setObjectName("hint")
        self.stat.setMinimumWidth(320)
        prog.addWidget(self.stat)
        outer.addLayout(prog)
        self.aistat = QLabel("")  # the enhanced copies run beside the repairs: their own line
        self.aistat.setObjectName("hint")
        self.aistat.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)  # long text is cut, never widens the window
        self.aistat.hide()
        outer.addWidget(self.aistat)

        # ---- results: list | before/after + report
        split = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Photo", "Real data", "Result", "Time", "Enhanced"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3, 4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._show_selected)
        split.addWidget(self.table)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        pics = QHBoxLayout()
        self.before = ImageBox("Before — as a normal viewer shows it")
        self.after = ImageBox("After — repaired copy")
        self.aibox = ImageBox("Enhanced copy")
        pics.addWidget(self.before, 1)
        pics.addWidget(self.after, 1)
        pics.addWidget(self.aibox, 1)
        self.aibox.hide()
        rl.addLayout(pics, 3)
        tools = QHBoxLayout()
        self.handb = QPushButton("Fix by hand…")
        self.handb.setToolTip("Open the selected photo in the manual repair window:\n"
                              "shift colours, move the picture or cut junk from a chosen place on.")
        self.handb.setEnabled(False)
        self.handb.clicked.connect(self._fix_by_hand)
        self.stepsb = QPushButton("Step by step…")
        self.stepsb.setToolTip("How this photo was recovered: the picture after every repair step")
        self.stepsb.setEnabled(False)
        self.stepsb.clicked.connect(self._show_steps)
        self.cmpb = QPushButton("Compare with the original…")
        self.cmpb.setToolTip("Have an intact copy of this photo (another backup, an edited copy)?\n"
                             "Measures the repair against it: which decoded parts are right, how far the\n"
                             "filled holes are from the truth, with a map of the error.")
        self.cmpb.setEnabled(False)
        self.cmpb.clicked.connect(self._compare)
        tools.addWidget(self.stepsb)
        tools.addWidget(self.handb)
        tools.addWidget(self.cmpb)
        tools.addStretch(1)
        rl.addLayout(tools)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("Select a photo to see what was found and repaired.")
        rl.addWidget(self.report, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        outer.addWidget(split, 1)

    # ------------------------------------------------------------------ input
    def _paths(self) -> list[Path]:
        return [Path(p.strip()) for p in self.inp.text().split(";") if p.strip()]

    def _set_inputs(self, paths: list[Path]) -> None:
        self.inp.setText("; ".join(str(p) for p in paths))

    def _suggest_output(self) -> None:
        paths = self._paths()
        if not paths or self.outp.isModified():
            return
        p = paths[0]
        if p.is_dir():
            self.outp.setText(str(p.parent / f"{p.name} (repaired)"))
        else:
            self.outp.setText(str(p.parent / "repaired"))

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder with damaged photos")
        if d:
            self._set_inputs([Path(d)])

    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Damaged photos", "",
                                                "Photos (*.jpg *.jpeg *.jpe *.~cr);;All files (*)")
        if files:
            self._set_inputs([Path(f) for f in files])

    def _pick_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Save repaired copies to")
        if d:
            self.outp.setText(d)
            self.outp.setModified(True)

    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self._set_inputs(paths)

    # ------------------------------------------------------------------ run
    def _start(self) -> None:
        paths = self._paths()
        if not paths or not all(p.exists() for p in paths):
            QMessageBox.warning(self, APP_NAME, "Choose a folder or photos that exist.")
            return
        if not self.outp.text().strip():
            self._suggest_output()
        out = Path(self.outp.text().strip())
        if any(p.is_dir() and p.resolve() == out.resolve() for p in paths):
            QMessageBox.warning(self, APP_NAME, "Save the repaired copies to a different folder than the originals.")
            return
        self.total = len(runner.collect(paths, out))
        if not self.total:
            QMessageBox.information(self, APP_NAME, "No JPEG photos found there.")
            return
        self.results.clear()
        self.table.setRowCount(0)
        self.before.show_image(None)
        self.after.show_image(None)
        self.report.clear()
        self.bar.setRange(0, self.total)
        self.bar.setValue(0)
        self.go.setEnabled(False)
        self.stopb.setEnabled(True)
        self.openb.setEnabled(True)
        self.out_dir = out

        steps = [k for k, cb in self.steps.items() if cb.isChecked()]
        options = {"mark_holes": self.mark.isChecked(), "write_reports": self.reports.isChecked(),
                   "enhance": steps, "quality": self.quality.currentData(),
                   "prompt": self.scene.text().strip() or None}
        self.aibox.setVisible(bool(steps))
        self.aibox.show_image(None)
        self.aistat.setVisible(bool(steps))
        self.aistat.setText("Enhanced copies: waiting for the first repaired photo…" if steps else "")
        self.thread = QThread(self)
        self.worker = RepairWorker(paths, out, options)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.result.connect(self._on_result)
        self.worker.ai_done.connect(self._on_ai)
        self.worker.status.connect(self.stat.setText)
        self.worker.ai_status.connect(self.aistat.setText)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()

    def _stop(self) -> None:
        if self.worker is not None:
            self.worker.stop.set()
            self.stat.setText("Stopping after the photos already in progress…")
            self.stopb.setEnabled(False)

    def _on_result(self, r: runner.Result) -> None:
        self.results.append(r)
        row = self.table.rowCount()
        self.table.insertRow(row)
        rep = r.report
        pct = rep.recovered_pct
        res = runner.verdict(rep, r.output is not None)
        color = {"Healthy": "#1e7b34", "Repaired": "#1e7b34", "Partly recovered": "#9a6700"}.get(res, "#b3261e")
        enh = "—" if (res in runner.LOST and self.aibox.isVisible()) else ""
        items = [QTableWidgetItem(str(r.source.name)), QTableWidgetItem("—" if pct is None else f"{pct:.0f} %"),
                 QTableWidgetItem(res), QTableWidgetItem(f"{r.seconds:.1f} s"), QTableWidgetItem(enh)]
        if enh:
            items[4].setToolTip("No enhanced copy: too little of this photo survived")
        items[2].setForeground(QColor(color))
        items[0].setToolTip(str(r.source))
        for c, it in enumerate(items):
            it.setData(Qt.ItemDataRole.UserRole, len(self.results) - 1)
            self.table.setItem(row, c, it)
        self.bar.setValue(len(self.results))
        self.stat.setText(f"{len(self.results)} / {self.total} photos")
        if len(self.results) == 1:
            self.table.selectRow(0)

    def _on_ai(self, r: runner.Result) -> None:
        for row in range(self.table.rowCount()):
            idx = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if self.results[idx] is r:
                it = QTableWidgetItem("✓")
                it.setData(Qt.ItemDataRole.UserRole, idx)
                it.setToolTip(str(r.ai_output))
                self.table.setItem(row, 4, it)
                if self.table.selectionModel().isRowSelected(row):
                    self._show_selected()
                break

    def _on_finished(self, secs: float, err: str) -> None:
        self.go.setEnabled(True)
        self.stopb.setEnabled(False)
        n = len(self.results)
        if err:
            self.stat.setText("Stopped by an error.")
            QMessageBox.critical(self, APP_NAME, err)
        else:
            self.stat.setText(f"Done: {n} photo(s) in {secs:.1f} s")

    def _show_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        idx = self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        r = self.results[idx]
        self.selected = r
        self.handb.setEnabled(r.output is not None)
        self.cmpb.setEnabled(r.output is not None)
        self.stepsb.setEnabled(True)
        self.before.show_image(load_preview(r.source), "Does not open in a normal viewer")
        self.after.show_image(load_preview(r.output) if r.output else None, "Nothing could be saved")
        if self.aibox.isVisible():
            ai = r.ai_output
            self.aibox.show_image(load_preview(ai) if ai and ai.exists() else None,
                                  "Nothing to enhance" if not r.report.holes else "Not made (yet)")
        text = r.report.format_text("en")
        if r.output:
            text += f"\nSaved as: {r.output}"
        if r.ai_output:
            text += f"\nEnhanced copy: {r.ai_output}"
            for d in r.report.enhanced:
                text += f"\n  - {d}"
        self.report.setPlainText(text)

    def _show_steps(self) -> None:
        r = getattr(self, "selected", None)
        if r is None:
            return
        from .steps_dialog import StepsDialog

        dlg = StepsDialog(self, r)
        self.steps_dlg = dlg
        dlg.show()

    def _fix_by_hand(self) -> None:
        r = getattr(self, "selected", None)
        if r is None or r.output is None:
            return
        from .manual_dialog import ManualDialog

        try:
            dlg = ManualDialog(self, r.output, r.source, r.output.parent)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"This photo cannot be edited by hand: {e}")
            return
        self.manual = dlg
        dlg.show()

    def _compare(self) -> None:
        r = getattr(self, "selected", None)
        if r is None or r.output is None:
            return
        f, _ = QFileDialog.getOpenFileName(self, f"Intact copy of {r.source.name}", str(r.source.parent),
                                           "Photos (*.jpg *.jpeg *.png *.tif *.tiff);;All files (*)")
        if not f:
            return
        from .compare_dialog import CompareDialog

        try:
            dlg = CompareDialog(self, r, Path(f))
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"The photos cannot be compared: {e}")
            return
        self.compare_dlg = dlg
        dlg.show()

    def _open_output(self) -> None:
        d = getattr(self, "out_dir", None)
        if d is None or not Path(d).exists():
            return
        if sys.platform == "win32":
            os.startfile(str(d))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(d)])
        else:
            subprocess.Popen(["xdg-open", str(d)])

    # ------------------------------------------------------------------ updates
    def _offer_update(self, upd) -> None:
        self.pending_update = upd
        self.banner_text.setText(f"A new version of {APP_NAME} is available (build {upd.build}, "
                                 f"you have build {updater.current_build()}).")
        self.banner.show()

    def _do_update(self) -> None:
        upd = self.pending_update
        if upd is None:
            return
        if self.worker is not None and self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, APP_NAME, "Wait until the repair finishes, then update.")
            return
        new_exe = Path(sys.executable).with_name("FotoMedik.new.exe")
        dlg = QProgressDialog("Downloading the new version…", "Cancel", 0, 100, self)
        dlg.setWindowTitle(APP_NAME)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        def progress(done, total):
            dlg.setValue(int(100 * done / total) if total else 0)
            QApplication.processEvents()

        try:
            updater.download(upd, new_exe, progress, cancelled=dlg.wasCanceled)
            dlg.close()
            updater.install_and_restart(new_exe)
        except InterruptedError:
            dlg.close()
            return
        except Exception as e:
            dlg.close()
            QMessageBox.warning(self, APP_NAME, f"The update did not work: {e}")
            return
        QApplication.quit()

    def closeEvent(self, e):  # noqa: N802
        if self.worker is not None:
            self.worker.stop.set()
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(5000)
        super().closeEvent(e)


def main(argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(STYLE)
    w = MainWindow()
    args = (sys.argv if argv is None else argv)[1:]
    if args and args[0] in ("--smoke", "--smoke-ai"):
        return _smoke(app, w, Path(args[1]), Path(args[2]), Path(args[3]), ai=args[0] == "--smoke-ai")
    if args:
        w._set_inputs([Path(a) for a in args])
    w.show()
    return app.exec()


def _smoke(app: QApplication, w: MainWindow, src: Path, out: Path, shot: Path, ai: bool = False) -> int:
    """Self-test for CI: repair `src` into `out` through the window, save a screenshot,
    exit 0 when every photo came back with a result (with `ai`: and an AI copy was made).
    Status lines go to <shot>.txt (a windowed exe has no console)."""
    from PySide6.QtCore import QTimer

    log = open(shot.with_suffix(".txt"), "w", encoding="utf-8")  # noqa: SIM115
    for cb in w.steps.values():
        cb.setChecked(False)
    if ai:
        for k in ("burst", "ai", "deblock", "faces"):
            w.steps[k].setChecked(True)
    w._set_inputs([src])
    w.outp.setText(str(out))
    w.outp.setModified(True)
    w.show()
    w._start()
    w.worker.status.connect(lambda s: (log.write(s + "\n"), log.flush()))
    w.worker.ai_status.connect(lambda s: (log.write("[enhance] " + s + "\n"), log.flush()))
    started = time.perf_counter()
    code = {"v": 2}

    def poll():
        if w.go.isEnabled() and w.results:
            ai_rows = [r for r in w.results if r.ai_output]
            with_output = [r for r in w.results if r.output is not None and r.report.holes] or \
                [r for r in w.results if r.output is not None] or w.results
            pick = ai_rows[0] if ai_rows else with_output[0]
            w.table.selectRow(w.results.index(pick))
            app.processEvents()
            w.grab().save(str(shot))
            ok = len(w.results) == w.total and (not ai or bool(ai_rows))
            try:  # the step-by-step window replays the repair
                from .steps_dialog import StepsDialog

                sd = StepsDialog(w, pick)
                sd.show()
                t0 = time.perf_counter()
                while not sd.steps and time.perf_counter() - t0 < 120:
                    app.processEvents()
                    time.sleep(0.05)
                app.processEvents()
                sd.grab().save(str(shot.with_name(shot.stem + "_steps.png")))
                log.write(f"steps window: {len(sd.steps)} step(s): " + " | ".join(x.title for x in sd.steps) + "\n")
                ok = ok and len(sd.steps) >= 2
                sd.close()
            except Exception as e:  # noqa: BLE001
                ok = False
                log.write(f"steps window failed: {type(e).__name__}: {e}\n")
            try:  # the manual repair window opens and edits
                w._fix_by_hand()
                dlg = getattr(w, "manual", None)
                if dlg is not None:
                    dlg.sliders["dy"].setValue(20)
                    dlg._colour()
                    app.processEvents()
                    dlg.grab().save(str(shot.with_name(shot.stem + "_manual.png")))
                    dlg.close()
                    log.write(f"manual window: {len(dlg.editor.history)} edit(s)\n")
            except Exception as e:  # noqa: BLE001
                ok = False
                log.write(f"manual window failed: {type(e).__name__}: {e}\n")
            code["v"] = 0 if ok else 1
            log.write(f"smoke: {len(w.results)}/{w.total} photos, {len(ai_rows)} AI copies\n")
            app.quit()
        elif time.perf_counter() - started > (1500 if ai else 300):
            app.quit()

    t = QTimer()
    t.timeout.connect(poll)
    t.start(200)
    app.exec()
    print(f"smoke: {len(w.results)}/{w.total} photos, exit {code['v']}")
    return code["v"]
