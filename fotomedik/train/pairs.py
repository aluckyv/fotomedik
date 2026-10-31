"""Training pairs for the deblocker: what FotoMedik makes of a damaged photo, next to the
photo as it was.

    python -m fotomedik.train.pairs GOOD_FOLDER OUT_FOLDER [--per 3] [--patches 48]

Every healthy photo in GOOD_FOLDER is brought to camera size (1600 px on the long side,
as the 2005 cameras wrote it, with or without an EXIF thumbnail), damaged the way a dying
card damages photos (dense rot, sometimes the other kinds), repaired by FotoMedik exactly
as the program does it, and cut into patches: the repaired picture is the network's
input, the good photo its target. Most patches come from where the repair differs from
the photo (holes, rotted blocks, colour bands), the rest from anywhere, so the network
also learns to leave good parts alone.

The patches go into OUT_FOLDER as numbered .npz files (train_*.npz, and val_*.npz from
about a tenth of the photos, for checking the training on photos it never saw). Nothing
leaves the computer.
"""

from __future__ import annotations

import argparse
import io
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from .damage import damage, dense_rot, exif_with_thumbnail, healthy

ImageFile.LOAD_TRUNCATED_IMAGES = True

PATCH = 128  # pixels, a multiple of 16 (the network halves three times; MCU-aligned)
DAMAGED_SHARE = 0.75  # share of the patches drawn where the repair differs from the photo
DIFF = 12  # gray levels: a block that differs this much on average counts as damaged
CLOSE = 24  # gray levels: a pixel this close to the photo is right
MIN_RIGHT = 0.25  # a patch needs this share of right pixels: a patch of nothing but hole
MIN_STD = 3.0  # (or one flat colour) teaches inventing a whole scene, which is LaMa's job
SHARD = 512  # patches per file
KINDS = (("dense", 0.45), ("dense-nothumb", 0.35), ("mixed", 0.20))  # how the copies are damaged


def _decode(data: bytes) -> np.ndarray | None:
    try:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:  # noqa: BLE001 - an undecodable repair is no pair
        return None


def make_variants(path: Path, per: int, long: int, seed: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """(repaired RGB, good RGB, kind) for `per` damaged copies of one photo; [] when the
    photo is not healthy itself."""
    from ..cli import repair_bytes

    data = path.read_bytes()
    if not healthy(data, path.name):
        return []
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((long, long), Image.LANCZOS)
    rng = np.random.default_rng(seed)
    out = []
    for k in range(per):
        kind = rng.choice([kd for kd, _p in KINDS], p=[p for _kd, p in KINDS])
        b = io.BytesIO()
        if kind == "dense-nothumb":
            im.save(b, "JPEG", quality=90)
        else:
            im.save(b, "JPEG", quality=90, exif=exif_with_thumbnail(im))
        good = b.getvalue()
        if kind == "mixed":
            bad, _what = damage(good, good, seed * 100 + k)
        else:
            bad, _what = dense_rot(good, seed * 100 + k)
        try:
            ctx, _rep = repair_bytes(bad, path.name, {"time_budget": 2.0})
        except Exception:  # noqa: BLE001 - a crash is no pair (and is the bench's business)
            continue
        rep_px, good_px = _decode(ctx.data), _decode(good)
        if rep_px is None or good_px is None or rep_px.shape != good_px.shape:
            continue
        out.append((rep_px, good_px, str(kind)))
    return out


def cut_patches(rep: np.ndarray, good: np.ndarray, n: int, rng, patch: int = PATCH) -> tuple[np.ndarray, np.ndarray]:
    """Up to n patches (n, patch, patch, 3) of both pictures at the same 16-pixel aligned
    places: DAMAGED_SHARE of them centred on blocks where the repair differs from the
    photo; patches that are mostly hole (see MIN_RIGHT) are left out."""
    h, w = rep.shape[:2]
    if h < patch or w < patch:
        return np.zeros((0, patch, patch, 3), np.uint8), np.zeros((0, patch, patch, 3), np.uint8)
    bh, bw = h // 16, w // 16
    d = np.abs(rep[: bh * 16, : bw * 16].astype(np.int16) - good[: bh * 16, : bw * 16]).mean(axis=2)
    bad = d.reshape(bh, 16, bw, 16).mean(axis=(1, 3)) > DIFF
    ys, xs = np.nonzero(bad)
    places = []
    for _try in range(6 * n):
        if len(places) == n:
            break
        if len(ys) and rng.random() < DAMAGED_SHARE:
            j = int(rng.integers(len(ys)))
            cy, cx = ys[j] * 16 + 8, xs[j] * 16 + 8
            y = int(np.clip(cy - patch // 2 + int(rng.integers(-patch // 4, patch // 4 + 1)), 0, h - patch))
            x = int(np.clip(cx - patch // 2 + int(rng.integers(-patch // 4, patch // 4 + 1)), 0, w - patch))
        else:
            y, x = int(rng.integers(0, h - patch + 1)), int(rng.integers(0, w - patch + 1))
        y, x = y - y % 16, x - x % 16
        a = rep[y : y + patch, x : x + patch]
        b = good[y : y + patch, x : x + patch]
        right = (np.abs(a.astype(np.int16) - b).max(axis=2) < CLOSE).mean()
        if right < MIN_RIGHT or a.reshape(-1, 3).std(axis=0).mean() < MIN_STD:
            continue
        places.append((y, x))
    if not places:
        return np.zeros((0, patch, patch, 3), np.uint8), np.zeros((0, patch, patch, 3), np.uint8)
    a = np.stack([rep[y : y + patch, x : x + patch] for y, x in places])
    b = np.stack([good[y : y + patch, x : x + patch] for y, x in places])
    return a, b


def _one(args):
    path, per, long, n, seed = args
    rng = np.random.default_rng(seed + 7)
    got = []
    for rep, good, _kind in make_variants(Path(path), per, long, seed):
        got.append(cut_patches(rep, good, n, rng))
    if not got:
        return Path(path).name, None, None
    return Path(path).name, np.concatenate([g[0] for g in got]), np.concatenate([g[1] for g in got])


class _Shards:
    def __init__(self, out: Path, prefix: str):
        self.out, self.prefix = out, prefix
        self.inp, self.tgt, self.n, self.files = [], [], 0, 0

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        self.inp.append(a)
        self.tgt.append(b)
        self.n += len(a)
        if self.n >= SHARD:
            self.flush()

    def flush(self) -> None:
        if not self.n:
            return
        a, b = np.concatenate(self.inp), np.concatenate(self.tgt)
        np.savez_compressed(self.out / f"{self.prefix}_{self.files:03d}.npz", inp=a, tgt=b)
        self.files += 1
        self.inp, self.tgt, self.n = [], [], 0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("good", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--per", type=int, default=3, help="damaged copies of every photo")
    ap.add_argument("--patches", type=int, default=48, help="patches cut from every damaged copy")
    ap.add_argument("--long", type=int, default=1600, help="long side the photos are brought to (pixels)")
    ap.add_argument("--jobs", type=int, default=0, help="photos worked on at once (default: all cores)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in a.good.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
    if not files:
        raise SystemExit(f"no photos in {a.good}")
    jobs = [(str(p), a.per, a.long, a.patches, i) for i, p in enumerate(files)]
    train, val = _Shards(a.out, "train"), _Shards(a.out, "val")
    n_ok = n_pairs = 0
    with ProcessPoolExecutor(a.jobs or os.cpu_count() or 1) as ex:
        for i, (name, inp, tgt) in enumerate(ex.map(_one, jobs)):
            if inp is None:
                print(f"skip {name} (not a healthy photo, or no repair could be made)", flush=True)
                continue
            n_ok += 1
            n_pairs += len(inp)
            (val if i % 10 == 9 else train).add(inp, tgt)
            print(f"{i + 1}/{len(files)} {name}: {len(inp)} patches", flush=True)
    train.flush()
    val.flush()
    print(f"{n_ok} photos, {n_pairs} patches -> {a.out} ({train.files} train + {val.files} val files)")


if __name__ == "__main__":
    main()
