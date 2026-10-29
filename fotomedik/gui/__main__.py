import multiprocessing
import sys

from fotomedik.gui.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # worker processes of the packaged .exe start here
    sys.exit(main())
