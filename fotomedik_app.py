"""Entry point of the packaged app (FotoMedik.exe)."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()  # worker processes of the packaged app start here
    from fotomedik.gui.app import main

    sys.exit(main())
