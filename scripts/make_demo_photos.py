"""Write a few synthetic damaged photos (for the app's smoke test in CI)."""

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from synth import make_image, make_jpeg, xor_prefix  # noqa: E402


def with_thumbnail(data: bytes, img: Image.Image) -> bytes:
    t = img.copy()
    t.thumbnail((160, 120))
    tb = io.BytesIO()
    t.save(tb, format="JPEG", quality=85)
    seg = b"\xff\xe1" + (len(tb.getvalue()) + 8).to_bytes(2, "big") + b"Exif\x00\x00" + tb.getvalue()
    return data[:2] + seg + data[2:]


def rot(data: bytes, frac: float, length: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    buf = bytearray(data)
    a = int(len(data) * frac)
    for o in range(a, a + length):
        if rng.random() < 0.3:
            buf[o] ^= 1 << int(rng.integers(0, 8))
    return bytes(buf)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    img = make_image(640, 480, seed=11)
    plain = with_thumbnail(make_jpeg(img), img)
    rst = with_thumbnail(make_jpeg(img, restart_rows=1), img)
    (out / "healthy.jpg").write_bytes(rst)
    (out / "rotted_rst.jpg").write_bytes(rot(rst, 0.5, 800, 1))
    (out / "rotted_no_rst.jpg").write_bytes(rot(rot(plain, 0.3, 600, 2), 0.7, 600, 3))
    (out / "cardrecovery.~CR").write_bytes(xor_prefix(rot(rst, 0.6, 500, 4)))
    (out / "overwritten.jpg").write_bytes((b"\\\\?\\D:\\x.JPG".decode().encode("utf-16-le") + bytes(range(64))) * 3000)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
