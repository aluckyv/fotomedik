"""Measuring ground: good photos are damaged the way real cards and recoveries damage
them, repaired, and compared with the photo they were.

    python scripts/bench.py make  GOOD_FOLDER  BENCH_FOLDER [--per 2] [--long 1600] [--kind mixed|dense]
    python scripts/bench.py run   BENCH_FOLDER RESULT.json [--ai MODELS_FOLDER] [--only N]
    python scripts/bench.py score BENCH_FOLDER OUT_FOLDER RESULT.json   (pictures made by the app)
    python scripts/bench.py compare OLD.json NEW.json

`make` keeps only photos the scanner finds healthy, brings them to `--long` pixels on
the long side (keeping their EXIF and its thumbnail, as a camera of 2005 would have
written them) and writes `--per` damaged copies of each with a manifest of what was
done. `run` repairs every damaged copy (and with --ai heals it, LaMa from the models
folder) and scores the result against the good photo:

  psnr     the whole picture (dB, higher is better)
  good     share of the damaged area that came out within 24 gray levels of the truth
  stripes  share of the damaged area's 8x8 blocks whose colour is off by more than 40
           (the coloured dashes and bands one notices first; lower is better)
  ssim     structural similarity in the damaged area: edges and detail in the right
           places (a blur scores low, invented wrong detail too)
  faces    the same `good` inside faces the detector finds in the good photo

Nothing of the photos goes into the repository: the bench folder lives outside it.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOL = 24  # gray levels: a pixel this close to the truth counts as right
STRIPE = 40  # a block whose average is off by more than this is a visible defect
SECTOR = 512

# ---------------------------------------------------------------- damage


def scan_start(data: bytes) -> int:
    """Offset of the main image's entropy-coded data (after its SOS header)."""
    p = 2
    while p + 4 <= len(data):
        if data[p] != 0xFF:
            raise ValueError("lost in the header")
        m = data[p + 1]
        if m == 0xFF:
            p += 1
            continue
        n = int.from_bytes(data[p + 2 : p + 4], "big")
        if m == 0xDA:
            return p + 2 + n
        p += 2 + n
    raise ValueError("no SOS")


def _aligned(rng, lo: int, hi: int) -> int:
    return int(rng.integers(lo // SECTOR, hi // SECTOR + 1)) * SECTOR


def damage(data: bytes, other: bytes, seed: int, spans: list | None = None) -> tuple[bytes, list[str]]:
    """One or two kinds of damage seen on real photos, in the scan data. `spans` (when
    given) receives the destroyed byte ranges in the good file's offsets."""
    rng = np.random.default_rng(seed)
    s = scan_start(data)
    end = len(data) - 2
    kinds = ["rot", "flips", "foreign", "zeros", "missing", "truncate", "rot", "foreign"]
    chosen = list(rng.choice(kinds, size=int(rng.integers(1, 3)), replace=False))
    buf = bytearray(data)
    done = []
    deleted: list[tuple[int, int]] = []

    def mark(a: int, b: int) -> None:  # a range of buf, in the good file's offsets
        if spans is not None:
            shift = sum(n for p, n in deleted if p <= a)
            spans.append((a + shift, b + shift))

    for k in chosen:
        span = end - s
        at = s + int(rng.uniform(0.1, 0.85) * span)
        if k == "rot":  # a stretch where ~30 % of the bytes have one bit wrong
            n = int(rng.integers(300, 3000))
            for o in range(at, min(at + n, len(buf) - 2)):
                if rng.random() < 0.3:
                    buf[o] ^= 1 << int(rng.integers(0, 8))
            mark(at, at + n)
            done.append(f"rot {n} B")
        elif k == "flips":
            n = int(rng.integers(1, 4))
            for _ in range(n):
                o = s + int(rng.integers(0, span))
                buf[o] ^= 1 << int(rng.integers(0, 8))
                mark(o, o + 1)
            done.append(f"{n} bit flip(s)")
        elif k == "foreign":  # sectors of another photo where this one's were
            n = _aligned(rng, 4096, 32768)
            at -= at % SECTOR
            os_ = scan_start(other)
            src = os_ + _aligned(rng, 0, max(SECTOR, len(other) - os_ - n - SECTOR))
            chunk = other[src : src + n]
            buf[at : at + len(chunk)] = chunk
            mark(at, at + len(chunk))
            done.append(f"foreign {n // 1024} KB")
        elif k == "zeros":
            n = _aligned(rng, 4096, 16384)
            at -= at % SECTOR
            buf[at : at + n] = bytes(min(n, len(buf) - 2 - at))
            mark(at, at + n)
            done.append(f"zeros {n // 1024} KB")
        elif k == "missing":  # a fragment the recovery lost: everything after it moves up
            n = _aligned(rng, 4096, 16384)
            at -= at % SECTOR
            mark(at, at + n)
            del buf[at : at + n]
            deleted.append((at, n))
            done.append(f"missing {n // 1024} KB")
        elif k == "truncate":
            cut = int(rng.uniform(0.55, 0.9) * (len(buf) - s)) + s
            mark(cut, len(buf))
            del buf[cut:]
            done.append(f"truncated at {100 * (cut - s) // (len(data) - s)} %")
    return bytes(buf), done


DENSE_SHARE = (0.15, 0.40)  # share of the scan's sectors rotted, as measured on photos from a dying card
DENSE_STAY = 0.5  # a rotted sector is followed by another one this often (runs of ~2 sectors)
DENSE_BIT = 0.08  # bits flipped in a rotted sector (half of its FF00 pairs break, as on the real photos)


def dense_rot(data: bytes, seed: int, spans: list | None = None, head: bool = False) -> tuple[bytes, list[str]]:
    """Rot spread over the whole scan the way a dying memory card leaves it: about a
    quarter of the 512-byte sectors, in runs of one to a few, every bit of them wrong
    with a probability of 8 % (measured on the 2005 photos: 26 % of the sectors show
    broken FF00 pairs, half of the pairs in them broken, runs of 2 rotted and 2 good
    sectors at the median)."""
    rng = np.random.default_rng(seed)
    s = scan_start(data)
    share = float(rng.uniform(*DENSE_SHARE))
    p_in = share * (1 - DENSE_STAY) / (1 - share)  # good -> rotted, for that share in the long run
    buf = bytearray(data)
    first = 0 if head else (s // SECTOR + 1) * SECTOR
    bad = False
    n_bad = 0
    for a in range(first, len(buf) - 2, SECTOR):
        bad = rng.random() < (DENSE_STAY if bad else p_in)
        if not bad:
            continue
        n_bad += 1
        b = min(a + SECTOR, len(buf) - 2)
        flips = rng.random((b - a, 8)) < DENSE_BIT
        mask = np.packbits(flips, axis=1).ravel()
        buf[a:b] = (np.frombuffer(bytes(buf[a:b]), np.uint8) ^ mask).tobytes()
        if spans is not None:
            spans.append((a, b))
    total = max(1, (len(buf) - 2 - first) // SECTOR)
    return bytes(buf), [f"dense rot: {n_bad} of {total} sectors ({100 * n_bad // total} %)"]


def exif_with_thumbnail(img: Image.Image) -> bytes:
    """A camera-like EXIF block (TIFF, IFD1) carrying a 160x120 JPEG thumbnail, the way
    cameras of the time wrote it - photos that went through other software often lost it."""
    import struct

    t = img.copy()
    t.thumbnail((160, 120), Image.LANCZOS)
    tb = io.BytesIO()
    t.save(tb, "JPEG", quality=85)
    thumb = tb.getvalue()
    ifd0 = struct.pack("<H", 0) + struct.pack("<I", 8 + 6)  # no entries, next IFD right after
    ifd1_at = 8 + len(ifd0)
    data_at = ifd1_at + 2 + 2 * 12 + 4
    ifd1 = struct.pack("<H", 2)
    ifd1 += struct.pack("<HHII", 0x0201, 4, 1, data_at)  # JPEGInterchangeFormat
    ifd1 += struct.pack("<HHII", 0x0202, 4, 1, len(thumb))  # JPEGInterchangeFormatLength
    ifd1 += struct.pack("<I", 0)
    return b"Exif\x00\x00" + b"II*\x00" + struct.pack("<I", 8) + ifd0 + ifd1 + thumb


def healthy(data: bytes, name: str) -> bool:
    from fotomedik.scan.scanner import scan_bytes

    d = scan_bytes(data, name)
    return not d.findings and d.recovered_fraction >= 0.999


def make(args) -> None:
    src, out = Path(args.good), Path(args.bench)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    (out / "damaged").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
    goods = []
    for p in files:
        data = p.read_bytes()
        if not healthy(data, p.name):
            print(f"skip (not healthy): {p.name}")
            continue
        im = Image.open(io.BytesIO(data))
        im = im.convert("RGB")
        im.thumbnail((args.long, args.long), Image.LANCZOS)
        b = io.BytesIO()
        im.save(b, "JPEG", quality=90, exif=exif_with_thumbnail(im))
        name = p.stem + ".jpg"
        (out / "truth" / name).write_bytes(b.getvalue())
        goods.append(name)
    manifest = []
    for i, name in enumerate(goods):
        data = (out / "truth" / name).read_bytes()
        other = (out / "truth" / goods[(i + 1) % len(goods)]).read_bytes()
        for k in range(args.per):
            seed = 1000 * i + k
            if args.kind in ("dense", "dense-head"):
                bad, what = dense_rot(data, seed, head=args.kind == "dense-head")
            else:
                bad, what = damage(data, other, seed)
            case = f"{Path(name).stem}__{k}.jpg"
            (out / "damaged" / case).write_bytes(bad)
            manifest.append({"case": case, "truth": name, "damage": what, "seed": seed, "kind": args.kind})
            print(f"{case}: {', '.join(what)}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(goods)} good photos, {len(manifest)} damaged copies")


# ---------------------------------------------------------------- scoring


def decode(data: bytes | None) -> np.ndarray | None:
    if not data:
        return None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:
        return None


def _block_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    h8, w8 = h // 8 * 8, w // 8 * 8
    d = np.abs(a[:h8, :w8].astype(np.float32) - b[:h8, :w8].astype(np.float32))
    return d.reshape(h8 // 8, 8, w8 // 8, 8, 3).mean(axis=(1, 3)).max(axis=-1)


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Structural similarity of the luma (Wang et al. 2004, Gaussian window 1.5): detail and
    edges where they belong count, a plain blur does not pass for them."""
    import cv2

    def y(x):
        x = x.astype(np.float64)
        return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

    a, b = y(a), y(b)
    g = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)  # noqa: E731
    ma, mb = g(a), g(b)
    va, vb, cov = g(a * a) - ma * ma, g(b * b) - mb * mb, g(a * b) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))


def score(truth: np.ndarray, got: np.ndarray | None, area: np.ndarray, faces: np.ndarray) -> dict:
    if got is None or got.shape != truth.shape:
        return {"psnr": 0.0, "good": 0.0, "stripes": 1.0, "ssim": 0.0, "faces": 0.0 if faces.any() else None}
    err = np.abs(got.astype(np.int16) - truth.astype(np.int16)).max(axis=-1)
    mse = float(((got.astype(np.float32) - truth) ** 2).mean())
    ok = err <= TOL
    blocks = _block_err(got, truth) > STRIPE
    h8, w8 = truth.shape[0] // 8 * 8, truth.shape[1] // 8 * 8
    area_b = area[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).any(axis=(1, 3))
    ss = ssim_map(truth, got)
    return {
        "psnr": round(10 * np.log10(255.0**2 / max(mse, 1e-6)), 2),
        "ssim": round(float(ss[area].mean()), 4) if area.any() else 1.0,
        "good": round(float(ok[area].mean()), 4) if area.any() else 1.0,
        "stripes": round(float(blocks[area_b].mean()), 4) if area_b.any() else 0.0,
        "faces": round(float(ok[area & faces].mean()), 4) if (area & faces).any() else None,
    }


def face_area(truth: np.ndarray, model: Path | None) -> np.ndarray:
    if model is None or not model.exists():
        return np.zeros(truth.shape[:2], bool)
    from fotomedik.ai.guard import face_mask

    return face_mask(truth, model)


def _repair_one(args) -> dict:
    bench, item = args
    from fotomedik.cli import repair_bytes

    data = (Path(bench) / "damaged" / item["case"]).read_bytes()
    t0 = time.perf_counter()
    try:
        ctx, rep = repair_bytes(data, item["case"], {"time_budget": 2.0})
        out = ctx.data
        holes = [(h["mcu_first"], h["mcu_count"]) if isinstance(h, dict) else (h.mcu_first, h.mcu_count)
                 for h in (rep.to_dict().get("holes") or [])]
        unknown = list(rep.unknown or [])
        err = None
    except Exception as e:  # a crash is a result too
        out, holes, unknown, err = None, [], [], f"{type(e).__name__}: {e}"
    return dict(item, repaired=out, holes=holes, unknown=unknown, secs=round(time.perf_counter() - t0, 2), error=err)


def run(args) -> None:
    bench = Path(args.bench)
    manifest = json.loads((bench / "manifest.json").read_text(encoding="utf-8"))
    if args.cases:
        want = {c.strip() for c in args.cases.split(",")}
        manifest = [m for m in manifest if m["case"] in want or Path(m["case"]).stem in want]
    if args.only:
        manifest = manifest[: args.only]
    models = Path(args.ai) if args.ai else None
    face_model = models / "face_detection_yunet_2023mar.onnx" if models else Path(args.faces) if args.faces else None
    engine = None
    if models:
        import onnxruntime as ort

        from fotomedik.ai.enhance import Engine

        engine = Engine(inpaint=ort.InferenceSession(str(models / "lama_fp32.onnx"), providers=["CPUExecutionProvider"]),
                        face_det=str(face_model) if face_model and face_model.exists() else None)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.jobs or None) as ex:
        repaired = list(ex.map(_repair_one, [(str(bench), m) for m in manifest]))
    print(f"repaired {len(repaired)} in {time.perf_counter() - t0:.0f} s")
    results = []
    for r in repaired:
        truth = decode((bench / "truth" / r["truth"]).read_bytes())
        naive = decode((bench / "damaged" / r["case"]).read_bytes())
        if naive is not None and naive.shape == truth.shape:
            area = np.abs(naive.astype(np.int16) - truth.astype(np.int16)).max(axis=-1) > TOL
            area = np.asarray(Image.fromarray(area).resize((area.shape[1] // 8, area.shape[0] // 8), Image.BOX)
                              .resize((area.shape[1], area.shape[0]), Image.NEAREST)) > 0
        else:
            area = np.ones(truth.shape[:2], bool)
        faces = face_area(truth, face_model)
        got = decode(r["repaired"])
        row = {"case": r["case"], "damage": r["damage"], "area": round(float(area.mean()), 4), "secs": r["secs"],
               "error": r["error"], "naive": score(truth, naive, area, faces), "repair": score(truth, got, area, faces)}
        if engine is not None and got is not None and got.shape == truth.shape and r["holes"]:
            from fotomedik.ai.enhance import enhance_steps, photo_masks, thumbnail_picture

            holes, unknown = photo_masks(r["repaired"], r["holes"], r["unknown"])
            t1 = time.perf_counter()
            healed, _done = enhance_steps(got, holes, unknown, engine, ["ai"],
                                          thumb=thumbnail_picture(r["repaired"], got.shape))
            row["ai"] = score(truth, healed, area, faces)
            row["ai_secs"] = round(time.perf_counter() - t1, 1)
            if args.keep:
                (Path(args.keep)).mkdir(parents=True, exist_ok=True)
                Image.fromarray(healed).save(Path(args.keep) / r["case"], quality=90)
        elif "ai" not in row and engine is not None:
            row["ai"] = row["repair"]
        if args.keep and r["repaired"]:
            Path(args.keep).mkdir(parents=True, exist_ok=True)
            (Path(args.keep) / ("rep_" + r["case"])).write_bytes(r["repaired"])
        results.append(row)
        print(f"{r['case']:<34} {', '.join(r['damage']):<34} " + "  ".join(
            f"{k} {row[k]['good']:.2f}/{row[k]['stripes']:.2f}" for k in ("naive", "repair", "ai") if k in row))
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "cases": results, "summary": summary(results)}
    Path(args.result).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print_summary(out["summary"])


def summary(rows: list[dict]) -> dict:
    out = {}
    for stage in ("naive", "repair", "ai"):
        rs = [r[stage] for r in rows if stage in r]
        if not rs:
            continue
        fs = [r["faces"] for r in rs if r["faces"] is not None]
        out[stage] = {"n": len(rs), "psnr": round(float(np.mean([r["psnr"] for r in rs])), 2),
                      "good": round(float(np.mean([r["good"] for r in rs])), 4),
                      "stripes": round(float(np.mean([r["stripes"] for r in rs])), 4),
                      "ssim": round(float(np.mean([r.get("ssim", 0.0) for r in rs])), 4),
                      "faces": round(float(np.mean(fs)), 4) if fs else None}
    out["errors"] = sum(1 for r in rows if r.get("error"))
    return out


def print_summary(s: dict) -> None:
    print(f"\n{'':8}{'psnr':>8}{'good':>8}{'stripes':>9}{'ssim':>8}{'faces':>8}")
    for stage in ("naive", "repair", "ai"):
        if stage in s:
            v = s[stage]
            f = "-" if v["faces"] is None else f"{v['faces']:.3f}"
            print(f"{stage:8}{v['psnr']:8.2f}{v['good']:8.3f}{v['stripes']:9.3f}{v.get('ssim', 0):8.3f}{f:>8}")
    print(f"errors: {s['errors']}")


def compare(args) -> None:
    a = json.loads(Path(args.old).read_text(encoding="utf-8"))
    b = json.loads(Path(args.new).read_text(encoding="utf-8"))
    print("old:"), print_summary(a["summary"])
    print("new:"), print_summary(b["summary"])
    old = {r["case"]: r for r in a["cases"]}
    for stage in ("repair", "ai"):
        worse = [(r["case"], old[r["case"]][stage]["good"], r[stage]["good"]) for r in b["cases"]
                 if r["case"] in old and stage in r and stage in old[r["case"]]
                 and r[stage]["good"] < old[r["case"]][stage]["good"] - 0.02]
        for c, o, n in worse:
            print(f"worse ({stage}): {c}  good {o:.3f} -> {n:.3f}")


def score_folder(args) -> None:
    """Score pictures made elsewhere (the app on a graphics card): OUT holds <case>.jpg
    (repaired) and/or <case>_enhanced.jpg for damaged copies of the bench."""
    bench, out = Path(args.bench), Path(args.out)
    manifest = {m["case"]: m for m in json.loads((bench / "manifest.json").read_text(encoding="utf-8"))}
    face_model = Path(args.faces) if args.faces else None
    rows = []
    for case, m in manifest.items():
        stem = Path(case).stem
        rep, enh = out / f"{stem}.jpg", out / f"{stem}_enhanced.jpg"
        if not rep.exists() and not enh.exists():
            continue
        truth = decode((bench / "truth" / m["truth"]).read_bytes())
        naive = decode((bench / "damaged" / case).read_bytes())
        if naive is not None and naive.shape == truth.shape:
            area = np.abs(naive.astype(np.int16) - truth.astype(np.int16)).max(axis=-1) > TOL
            area = np.asarray(Image.fromarray(area).resize((area.shape[1] // 8, area.shape[0] // 8), Image.BOX)
                              .resize((area.shape[1], area.shape[0]), Image.NEAREST)) > 0
        else:
            area = np.ones(truth.shape[:2], bool)
        faces = face_area(truth, face_model)
        row = {"case": case, "damage": m["damage"], "area": round(float(area.mean()), 4),
               "naive": score(truth, naive, area, faces)}
        if rep.exists():
            row["repair"] = score(truth, decode(rep.read_bytes()), area, faces)
        if enh.exists():
            row["ai"] = score(truth, decode(enh.read_bytes()), area, faces)
        rows.append(row)
        print(f"{case:<34} " + "  ".join(f"{k} {row[k]['good']:.2f}/{row[k]['stripes']:.2f}/{row[k]['ssim']:.2f}"
                                         for k in ("naive", "repair", "ai") if k in row))
    res = {"when": time.strftime("%Y-%m-%d %H:%M"), "cases": rows, "summary": summary(rows)}
    Path(args.result).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print_summary(res["summary"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("good")
    m.add_argument("bench")
    m.add_argument("--per", type=int, default=2)
    m.add_argument("--long", type=int, default=1600)
    m.add_argument("--kind", choices=("mixed", "dense", "dense-head"), default="mixed",
                   help="mixed: one or two kinds of damage per copy; dense: rot over the whole scan (a dying card); "
                        "dense-head: the same from the first byte (header and thumbnail rot too; repair with the "
                        "app's batch and judge with `score`)")
    r = sub.add_parser("run")
    r.add_argument("bench")
    r.add_argument("result")
    r.add_argument("--ai", help="folder with lama_fp32.onnx (and the YuNet face model)")
    r.add_argument("--faces", help="YuNet face model, to score faces without --ai")
    r.add_argument("--only", type=int, default=0)
    r.add_argument("--cases", help="comma separated case names (default: all)")
    r.add_argument("--jobs", type=int, default=0)
    r.add_argument("--keep", help="folder to save the repaired and healed pictures in")
    sc = sub.add_parser("score")
    sc.add_argument("bench")
    sc.add_argument("out")
    sc.add_argument("result")
    sc.add_argument("--faces", help="YuNet face model, to score faces")
    c = sub.add_parser("compare")
    c.add_argument("old")
    c.add_argument("new")
    a = p.parse_args()
    {"make": make, "run": run, "score": score_folder, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
