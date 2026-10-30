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

if __name__ == "__main__":
    multiprocessing.freeze_support()  # worker processes of the packaged app start here
    from fotomedik.gui.app import main

    sys.exit(main())
