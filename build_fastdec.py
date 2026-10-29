"""Compile the C decoding kernel next to fotomedik/jpeg/fastdec.py.

    python build_fastdec.py

Uses gcc/clang (Linux, macOS, MSYS2/mingw on Windows) or MSVC `cl` when found.
Without the library FotoMedik still works, ~15x slower (pure Python decoder).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "fotomedik" / "jpeg" / "_fastdec.c"
OUT_DIR = SRC.parent


def main() -> int:
    if sys.platform == "win32":
        out = OUT_DIR / "_fastdec.dll"
        if shutil.which("gcc"):
            cmd = ["gcc", "-O2", "-shared", "-o", str(out), str(SRC)]
        elif shutil.which("clang"):
            cmd = ["clang", "-O2", "-shared", "-o", str(out), str(SRC)]
        elif shutil.which("cl"):
            cmd = ["cl", "/O2", "/LD", str(SRC), f"/Fe{out}", f"/Fo{OUT_DIR}\\_fastdec.obj"]
        else:
            print("no C compiler found (install MSYS2 mingw-w64 gcc, LLVM clang, or VS Build Tools)")
            print("or download _fastdec.dll from the GitHub Actions artifacts of this repo")
            return 1
    else:
        out = OUT_DIR / ("_fastdec.dylib" if sys.platform == "darwin" else "_fastdec.so")
        cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
        if not cc:
            print("no C compiler found")
            return 1
        cmd = [cc, "-O2", "-shared", "-fPIC", "-o", str(out), str(SRC)]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print(f"built {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
