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

Each patch also carries what the program knows besides the picture, for the guided
network (finetune --guided): the photo's thumbnail as the program would enlarge it (at a
quarter of the size; nothing when the photo has no usable thumbnail) and the holes (one
value per 8x8 block: synthesised by the repair, not decoded).

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


def guide_quarter(data: bytes, shape) -> np.ndarray | None:
    """The photo's thumbnail brought to a quarter of the picture's size (h/4, w/4, RGB), as
    the guided network gets it in the program, or None without a usable thumbnail."""
    from ..ai.enhance import quarter_guide, thumbnail_rgb

    return quarter_guide(thumbnail_rgb(data), shape)


def hole_blocks(data: bytes, rep, shape) -> np.ndarray:
    """(h/8, w/8) bool: the 8x8 blocks the repair synthesised (holes, not decoded data)."""
    from ..ai.enhance import photo_masks

    h, w = shape[:2]
    holes = [(x["mcu_first"], x["mcu_count"]) if isinstance(x, dict) else (x.mcu_first, x.mcu_count)
             for x in (rep.holes or [])]
    if not holes:
        return np.zeros((h // 8, w // 8), bool)
    hm, _um = photo_masks(data, holes, rep.unknown)
    hm = hm[: h // 8 * 8, : w // 8 * 8]
    return hm.reshape(h // 8, 8, w // 8, 8).any(axis=(1, 3))


def make_variants(path: Path, per: int, long: int, seed: int) -> list[dict]:
    """{rep, good, guide, holes, kind} for `per` damaged copies of one photo (rep and good
    RGB, guide see guide_quarter, holes see hole_blocks); [] when the photo is not healthy
    itself."""
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
        try:
            if kind == "mixed":
                bad, _what = damage(good, good, seed * 100 + k)
            else:
                bad, _what = dense_rot(good, seed * 100 + k)
            ctx, rep = repair_bytes(bad, path.name, {"time_budget": 2.0})
            rep_px, good_px = _decode(ctx.data), _decode(good)
            if rep_px is None or good_px is None or rep_px.shape != good_px.shape:
                continue
            guide = guide_quarter(ctx.data, rep_px.shape)
            holes = hole_blocks(ctx.data, rep, rep_px.shape)
        except Exception:  # noqa: BLE001 - a crash is no pair (and is the bench's business)
            continue
        out.append({"rep": rep_px, "good": good_px, "guide": guide, "holes": holes, "kind": str(kind)})
    return out


def cut_patches(rep: np.ndarray, good: np.ndarray, n: int, rng, patch: int = PATCH, places_out: list | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
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
    if places_out is not None:
        places_out.extend(places)
    a = np.stack([rep[y : y + patch, x : x + patch] for y, x in places])
    b = np.stack([good[y : y + patch, x : x + patch] for y, x in places])
    return a, b


def extras_at(v: dict, places: list, patch: int = PATCH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(guide (n, patch/4, patch/4, 3), has guide (n,), holes (n, patch/8, patch/8)) of a
    variant at the patches' places (16-pixel aligned, so the quarter and block grids fit)."""
    q, b = patch // 4, patch // 8
    n = len(places)
    guide = np.zeros((n, q, q, 3), np.uint8)
    has = np.zeros(n, np.uint8)
    holes = np.zeros((n, b, b), bool)
    for i, (y, x) in enumerate(places):
        if v["guide"] is not None:
            g = v["guide"][y // 4 : y // 4 + q, x // 4 : x // 4 + q]
            if g.shape[:2] == (q, q):
                guide[i], has[i] = g, 1
        hb = v["holes"][y // 8 : y // 8 + b, x // 8 : x // 8 + b]
        holes[i, : hb.shape[0], : hb.shape[1]] = hb
    return guide, has, holes


def _one(args):
    path, per, long, n, seed = args
    rng = np.random.default_rng(seed + 7)
    parts = []
    for v in make_variants(Path(path), per, long, seed):
        places: list = []
        a, b = cut_patches(v["rep"], v["good"], n, rng, places_out=places)
        if len(a):
            parts.append((a, b) + extras_at(v, places))
    if not parts:
        return Path(path).name, None
    return Path(path).name, tuple(np.concatenate([p[k] for p in parts]) for k in range(5))


class _Shards:
    KEYS = ("inp", "tgt", "guide", "has_guide", "holes")

    def __init__(self, out: Path, prefix: str):
        self.out, self.prefix = out, prefix
        self.parts: list = []
        self.n, self.files = 0, 0

    def add(self, a: np.ndarray, b: np.ndarray, *extras) -> None:
        self.parts.append((a, b) + tuple(extras))
        self.n += len(a)
        if self.n >= SHARD:
            self.flush()

    def flush(self) -> None:
        if not self.n:
            return
        arrays = {k: np.concatenate([p[i] for p in self.parts]) for i, k in enumerate(self.KEYS)
                  if all(len(p) > i for p in self.parts)}
        np.savez_compressed(self.out / f"{self.prefix}_{self.files:03d}.npz", **arrays)
        self.files += 1
        self.parts, self.n = [], 0


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
        for i, (name, got) in enumerate(ex.map(_one, jobs)):
            if got is None:
                print(f"skip {name} (not a healthy photo, or no repair could be made)", flush=True)
                continue
            n_ok += 1
            n_pairs += len(got[0])
            (val if i % 10 == 9 else train).add(*got)
            print(f"{i + 1}/{len(files)} {name}: {len(got[0])} patches", flush=True)
    train.flush()
    val.flush()
    print(f"{n_ok} photos, {n_pairs} patches -> {a.out} ({train.files} train + {val.files} val files)")


if __name__ == "__main__":
    main()
