"""FotoMedik.exe: the small launcher. It installs the app on first run, swaps in a
downloaded update while the app is closed, and starts the app (see fotomedik.install).

Run it from anywhere: the app itself lives in %LOCALAPPDATA%\\FotoMedik.
"""

import sys
import threading
from pathlib import Path

from fotomedik import install


def _first_install(args) -> bool:
    """Download and unpack the app, with a small progress window. False on failure."""
    import os

    if os.environ.get("FOTOMEDIK_NO_WINDOW"):  # the CI tests
        install.stage(install.Release())
        install.apply_pending()
        return install.installed() is not None
    import tkinter as tk
    from tkinter import ttk

    state = {"err": None, "done": False}
    root = tk.Tk()
    root.title("FotoMedik")
    root.resizable(False, False)
    tk.Label(root, text="Installing FotoMedik (about 200 MB, only the first time)…", padx=16, pady=10).pack()
    bar = ttk.Progressbar(root, length=360, mode="determinate", maximum=1000)
    bar.pack(padx=16, pady=(0, 6))
    note = tk.Label(root, text="", padx=16, pady=6)
    note.pack()

    def progress(done, total, name):
        state["p"] = (done, total, name)

    def work():
        try:
            rel = install.Release()
            install.stage(rel, progress)
            install.apply_pending()
        except Exception as e:  # shown in the window
            state["err"] = f"{type(e).__name__}: {e}"
        state["done"] = True

    def poll():
        p = state.get("p")
        if p:
            done, total, name = p
            bar["value"] = int(1000 * done / total) if total else 0
            note["text"] = f"{name}: {done >> 20} / {total >> 20} MB"
        if state["done"]:
            if state["err"]:
                note["text"] = "The installation did not work: " + state["err"]
                tk.Button(root, text="Close", command=root.destroy).pack(pady=8)
            else:
                root.destroy()
            return
        root.after(200, poll)

    threading.Thread(target=work, daemon=True).start()
    root.after(200, poll)
    root.mainloop()
    return state["err"] is None and install.installed() is not None


def main() -> int:
    args = sys.argv[1:]
    if getattr(sys, "frozen", False):
        install.keep_launcher(Path(sys.executable))
    if install.installed() is None and not install.pending():
        if not _first_install(args):
            return 1
    try:
        install.apply_pending()
    except OSError:
        pass  # the app is still running (or a file is locked): the update waits for the next start
    if install.installed() is None:
        return 1
    smoke = bool(args) and args[0].startswith("--smoke")
    return install.start_app(args, wait=smoke)


if __name__ == "__main__":
    sys.exit(main())
