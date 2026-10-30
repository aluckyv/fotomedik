# PyInstaller spec of the folder app's base (app\base): Python, Qt, numpy, Pillow,
# OpenCV... without the fotomedik package itself, which ships separately as app\code
# (see fotomedik/install.py) - so an update of the code does not need a new base.
# Build: pyinstaller --noconfirm fotomedik.spec  ->  dist/FotoMedik/
import sys

block_cipher = None

a = Analysis(
    ["fotomedik_app.py"],
    pathex=["."],
    hiddenimports=[
        # used by the code but found only through it: keep them in the base
        "concurrent.futures", "multiprocessing", "ctypes", "zipfile", "urllib.request", "hashlib",
        "sqlite3", "xml.etree.ElementTree", "packaging", "packaging.version", "onnx",
        "PIL.Image", "PIL.ImageFilter", "PIL.ImageOps", "PIL.ImageDraw", "numpy", "cv2", "click",
    ],
    excludes=["onnxruntime", "tkinter", "pytest", "torch"],
    noarchive=False,
)
# the fotomedik package stays out of the base: it is loaded from app\code
a.pure = [m for m in a.pure if not (m[0] == "fotomedik" or m[0].startswith("fotomedik."))]
a.binaries = [b for b in a.binaries if "_fastdec" not in b[0]]
# what the base holds: part of its id, so code that needs a module the base lacks brings a new base
with open("base_manifest.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(sorted({m[0] for m in a.pure} | {b[0] for b in a.binaries})))
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="FotoMedik", console=False,
          upx=False, debug=False, strip=False)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="FotoMedik")
