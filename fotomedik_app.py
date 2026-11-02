"""Entry point of the packaged app (FotoMedik.exe of the app folder, or the one-file build).

In the installed folder app the fotomedik package is not frozen into the program: it
sits in app\\code next to app\\base (see fotomedik.install), so an update replaces a
few megabytes of code instead of the whole program. Worker processes start here too.
"""

import multiprocessing
import sys
from pathlib import Path


def _code_on_path() -> None:
    if getattr(sys, "frozen", False):
        code = Path(sys.executable).resolve().parent.parent / "code"
        if (code / "fotomedik").is_dir() and str(code) not in sys.path:
            sys.path.insert(0, str(code))


_code_on_path()

def _crash_log():
    """Where an error that ends the program is written (the windowed program has no console):
    FotoMedik-crash.txt in the program's home; a hard crash goes there too (faulthandler)."""
    import faulthandler
    import os

    try:
        base = os.environ.get("FOTOMEDIK_HOME") or os.path.join(
            os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local"), "FotoMedik")
        os.makedirs(base, exist_ok=True)
        f = open(os.path.join(base, "FotoMedik-crash.txt"), "w", encoding="utf-8")  # noqa: SIM115
        faulthandler.enable(f, all_threads=True)
        return f
    except OSError:
        return None


if __name__ == "__main__":
    multiprocessing.freeze_support()  # worker processes of the packaged app start here
    crash = _crash_log()
    from fotomedik.gui.app import main

    try:
        rc = main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback

        if crash is not None:
            crash.write(traceback.format_exc())
            crash.flush()
        raise
    # The window is closed and its threads are done (gui.app._settle); the process ends here
    # without Python's and Qt's teardown, which crashed at random (an access violation, an
    # abort) after everything had gone well - builds 149-155.
    import os

    for f in (sys.stdout, sys.stderr, crash):
        try:
            if f is not None:
                f.flush()
        except Exception:  # noqa: BLE001
            pass
    if sys.platform == "win32":
        # os._exit still lets Windows unload the DLLs (Qt, onnxruntime) and one of them
        # crashed in its detach routine (build 157: access violation inside os._exit);
        # TerminateProcess ends the process without notifying them
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.TerminateProcess(k32.GetCurrentProcess(), int(rc or 0))
    os._exit(int(rc or 0))
