"""Fine-tune FBCNN (the "JPEG clean-up" network) on FotoMedik's own repairs.

    python -m fotomedik.train.finetune PAIRS_FOLDER OUT_FOLDER [--iters 20000] [--install]

PAIRS_FOLDER holds the patches of fotomedik.train.pairs. The network starts from the
authors' weights (fbcnn_color.pth, downloaded once into OUT_FOLDER) and learns, patch by
patch, to turn what the repair left (flat squares, rotted blocks, colour bands, blur) into
the photo it was. Every --check iterations it is measured on the val_* patches (photos it
never trains on): PSNR of the repair as it is, of the network before training, and now.
The best state is kept (OUT_FOLDER/best.pt) and exported as fbcnn-fotomedik.onnx, the
file the program looks for; --install copies it into the program's model folder, where
the "JPEG clean-up" step then uses it instead of the original.

A graphics card (CUDA) is used when torch sees one; on the processor it works, slowly.
Stop it any time (Ctrl+C): the best state so far is exported. --resume continues.
"""

from __future__ import annotations

import argparse
import glob
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np

WEIGHTS_URL = "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth"
ONNX_NAME = "fbcnn-fotomedik.onnx"
CROP = 96  # training crop (from the 128-pixel patches), a multiple of 8: the JPEG grid stays put


def load_pairs(folder: Path, prefix: str, limit: int = 0) -> tuple[np.ndarray, np.ndarray]:
    inp, tgt = [], []
    for f in sorted(glob.glob(str(folder / f"{prefix}_*.npz"))):
        z = np.load(f)
        inp.append(z["inp"])
        tgt.append(z["tgt"])
    if not inp:
        return np.zeros((0, 8, 8, 3), np.uint8), np.zeros((0, 8, 8, 3), np.uint8)
    a, b = np.concatenate(inp), np.concatenate(tgt)
    if limit and len(a) > limit:
        a, b = a[:limit], b[:limit]
    return a, b


def batch_of(inp: np.ndarray, tgt: np.ndarray, n: int, rng, crop: int = CROP):
    """n random crops (8-pixel aligned offsets) with random left-right flips, as
    float tensors (n, 3, crop, crop) in 0..1."""
    import torch

    idx = rng.integers(0, len(inp), n)
    size = inp.shape[1]
    xs, ys = [], []
    for i in idx:
        oy = int(rng.integers(0, (size - crop) // 8 + 1)) * 8
        ox = int(rng.integers(0, (size - crop) // 8 + 1)) * 8
        a = inp[i, oy : oy + crop, ox : ox + crop]
        b = tgt[i, oy : oy + crop, ox : ox + crop]
        if rng.random() < 0.5:  # a mirrored photo is a photo too (the 8x8 grid mirrors onto itself)
            a, b = a[:, ::-1], b[:, ::-1]
        xs.append(a)
        ys.append(b)
    x = torch.from_numpy(np.ascontiguousarray(np.stack(xs).transpose(0, 3, 1, 2))).float() / 255.0
    y = torch.from_numpy(np.ascontiguousarray(np.stack(ys).transpose(0, 3, 1, 2))).float() / 255.0
    return x, y


def psnr_of(net, inp: np.ndarray, tgt: np.ndarray, device, chunk: int = 16) -> tuple[float, float]:
    """(PSNR of the network's output, PSNR of the input as it is) over all patches, dB."""
    import torch

    se_out = se_in = 0.0
    n = 0
    net.eval()
    with torch.no_grad():
        for k in range(0, len(inp), chunk):
            x = torch.from_numpy(inp[k : k + chunk].transpose(0, 3, 1, 2).copy()).float().to(device) / 255.0
            y = torch.from_numpy(tgt[k : k + chunk].transpose(0, 3, 1, 2).copy()).float().to(device) / 255.0
            out = net(x)[0].clamp(0, 1)
            se_out += float(((out - y) ** 2).sum())
            se_in += float(((x - y) ** 2).sum())
            n += y.numel()
    net.train()
    return 10 * np.log10(n / max(se_out, 1e-12)), 10 * np.log10(n / max(se_in, 1e-12))


def export_onnx(net, path: Path) -> None:
    """The network as the program runs it: "input" (N, 3, H, W) in 0..1, H and W multiples
    of 8; outputs the cleaned picture and the estimated JPEG quality."""
    import torch

    net = net.cpu().eval()
    x = torch.rand(1, 3, 128, 128)
    torch.onnx.export(net, x, str(path), input_names=["input"], output_names=["output", "qf"], opset_version=17,
                      dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h", 3: "w"}},
                      dynamo=False)


def model_dir() -> Path:
    from ..ai.models import model_dir as md

    return md()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--check", type=int, default=500, help="iterations between measurements on the val patches")
    ap.add_argument("--max-patches", type=int, default=0, help="use at most this many training patches (memory)")
    ap.add_argument("--resume", action="store_true", help="continue from OUT/last.pt")
    ap.add_argument("--install", action="store_true", help="copy the result into the program's model folder")
    ap.add_argument("--cpu", action="store_true", help="do not use the graphics card")
    a = ap.parse_args(argv)

    import torch

    from .fbcnn_net import FBCNN

    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    tr_in, tr_tgt = load_pairs(a.pairs, "train", a.max_patches)
    va_in, va_tgt = load_pairs(a.pairs, "val")
    if not len(tr_in):
        raise SystemExit(f"no train_*.npz in {a.pairs} (make them with: python -m fotomedik.train.pairs)")
    if not len(va_in):  # few photos: check on a slice of the training patches
        va_in, va_tgt = tr_in[-64:], tr_tgt[-64:]
    print(f"{len(tr_in)} training patches, {len(va_in)} for checking")

    net = FBCNN()
    weights = a.out / "fbcnn_color.pth"
    if not weights.exists():
        print("downloading the original FBCNN weights (~290 MB, once)...")
        urllib.request.urlretrieve(WEIGHTS_URL, weights)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters, eta_min=a.lr * 0.1)
    start, best = 0, -1.0
    last = a.out / "last.pt"
    if a.resume and last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best = st["iter"], st["best"]
        print(f"resumed at iteration {start} (best {best:.2f} dB)")
    else:
        p_net, p_in = psnr_of(net, va_in, va_tgt, device)
        print(f"before training: repair as it is {p_in:.2f} dB, original FBCNN {p_net:.2f} dB")
        best = p_net
        torch.save(net.state_dict(), a.out / "best.pt")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(start + 1)
    l1 = torch.nn.L1Loss()
    t0 = time.time()
    it = start
    try:
        net.train()
        for it in range(start + 1, a.iters + 1):
            x, y = batch_of(tr_in, tr_tgt, a.batch, rng)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = net(x)[0]
                loss = l1(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if it % 50 == 0:
                rate = (it - start) / max(time.time() - t0, 1e-6)
                left = (a.iters - it) / max(rate, 1e-6)
                print(f"iteration {it}/{a.iters}  loss {float(loss):.4f}  ({rate:.1f}/s, ~{left / 60:.0f} min left)",
                      flush=True)
            if it % a.check == 0 or it == a.iters:
                p_net, p_in = psnr_of(net, va_in, va_tgt, device)
                mark = ""
                if p_net > best:
                    best = p_net
                    torch.save(net.state_dict(), a.out / "best.pt")
                    mark = "  <- best so far"
                print(f"check at {it}: {p_net:.2f} dB (repair as it is {p_in:.2f} dB){mark}", flush=True)
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "iter": it, "best": best}, last)
    except KeyboardInterrupt:
        print(f"stopped at iteration {it}")

    net.load_state_dict(torch.load(a.out / "best.pt", map_location="cpu", weights_only=True))
    onnx_path = a.out / ONNX_NAME
    export_onnx(net, onnx_path)
    print(f"best: {best:.2f} dB -> {onnx_path}")
    if a.install:
        dst = model_dir() / ONNX_NAME
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, dst)
        for old in dst.parent.glob(ONNX_NAME.replace(".onnx", "") + "*fp16*"):
            old.unlink(missing_ok=True)  # a half-precision copy of an earlier version
        print(f"installed: {dst} (the JPEG clean-up step uses it from now on)")


if __name__ == "__main__":
    main()
