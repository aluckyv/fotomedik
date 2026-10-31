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

--guided trains the guided network instead (fbcnn-fotomedik-guided.onnx): FBCNN that
sees, besides the repaired picture, the photo's enlarged thumbnail and where the holes
are, so it can take colours and shapes from the thumbnail and keep the decoded detail.
It starts from the fine-tuned network when --init gives one (e.g. izlaz/best.pt), with the
new inputs weighted zero, i.e. exactly where the plain network stands.

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
ONNX_GUIDED = "fbcnn-fotomedik-guided.onnx"
EXTRA = 5  # guided input channels besides the picture: thumbnail RGB, "has a thumbnail", holes
CROP = 96  # training crop (from the 128-pixel patches), a multiple of 8: the JPEG grid stays put


def load_pairs(folder: Path, prefix: str, limit: int = 0, guided: bool = False) -> dict:
    """{inp, tgt[, guide, has_guide, holes]} of all prefix_*.npz files (see pairs)."""
    keys = ("inp", "tgt") + (("guide", "has_guide", "holes") if guided else ())
    got: dict = {k: [] for k in keys}
    for f in sorted(glob.glob(str(folder / f"{prefix}_*.npz"))):
        z = np.load(f)
        if guided and "guide" not in z.files:
            raise SystemExit(f"{f} has no thumbnails: make the pairs again (delete the folder, run pairs)")
        for k in keys:
            got[k].append(z[k])
    out = {k: (np.concatenate(v) if v else np.zeros((0,), np.uint8)) for k, v in got.items()}
    if limit and len(out["inp"]) > limit:
        out = {k: v[:limit] for k, v in out.items()}
    return out


def guided_net(base):
    """The guided network around an FBCNN with 3 + EXTRA input channels: inputs picture
    (N, 3, H, W), guide (N, 3, H/4, W/4), has_guide (N, 1), holes (N, 1, H/8, W/8); the
    guide and holes are brought to full size inside, so the program and the training do
    it the same way."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Guided(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, img, guide, has_guide, holes):
            g = F.interpolate(guide, size=img.shape[-2:], mode="bilinear", align_corners=False)
            m = F.interpolate(holes, size=img.shape[-2:], mode="nearest")
            v = has_guide[:, :, None, None] * torch.ones_like(img[:, :1])
            return self.net(torch.cat([img, g * v, v, m], dim=1))

    return Guided(base)


def widen_head(state: dict, extra: int = EXTRA) -> dict:
    """A 3-channel FBCNN state for the guided network: the first convolution gets zero
    weights for the new inputs, so it starts out doing exactly what it did."""
    import torch

    w = state["m_head.weight"]
    state = dict(state)
    state["m_head.weight"] = torch.cat([w, torch.zeros(w.shape[0], extra, *w.shape[2:], dtype=w.dtype)], dim=1)
    return state


def _tensors(d: dict, idx, crops, flips, guided: bool, crop: int):
    import torch

    def cut(arr, scale):
        out = []
        for i, (oy, ox), fl in zip(idx, crops, flips):
            a = arr[i, oy // scale : (oy + crop) // scale, ox // scale : (ox + crop) // scale]
            out.append(a[:, ::-1] if fl else a)
        return np.ascontiguousarray(np.stack(out))

    x = torch.from_numpy(cut(d["inp"], 1).transpose(0, 3, 1, 2)).float() / 255.0
    y = torch.from_numpy(cut(d["tgt"], 1).transpose(0, 3, 1, 2)).float() / 255.0
    if not guided:
        return (x,), y
    g = torch.from_numpy(cut(d["guide"], 4).transpose(0, 3, 1, 2)).float() / 255.0
    hv = torch.from_numpy(d["has_guide"][idx].astype(np.float32))[:, None]
    m = torch.from_numpy(cut(d["holes"], 8).astype(np.float32))[:, None]
    return (x, g, hv, m), y


def batch_of(d: dict, n: int, rng, crop: int = CROP, guided: bool = False):
    """n random crops (8-pixel aligned offsets) with random left-right flips: (inputs, target)
    as float tensors in 0..1 (inputs: the picture, and for the guided network its extras)."""
    size = d["inp"].shape[1]
    idx = rng.integers(0, len(d["inp"]), n)
    crops = [(int(rng.integers(0, (size - crop) // 8 + 1)) * 8, int(rng.integers(0, (size - crop) // 8 + 1)) * 8)
             for _ in idx]
    flips = [rng.random() < 0.5 for _ in idx]  # a mirrored photo is a photo too (the 8x8 grid mirrors onto itself)
    return _tensors(d, idx, crops, flips, guided, crop)


def psnr_of(net, d: dict, device, guided: bool = False, chunk: int = 16) -> tuple[float, float]:
    """(PSNR of the network's output, PSNR of the input as it is) over all patches, dB."""
    import torch

    se_out = se_in = 0.0
    n = 0
    size = d["inp"].shape[1]
    net.eval()
    with torch.no_grad():
        for k in range(0, len(d["inp"]), chunk):
            idx = np.arange(k, min(k + chunk, len(d["inp"])))
            xs, y = _tensors(d, idx, [(0, 0)] * len(idx), [False] * len(idx), guided, size)
            xs = tuple(t.to(device) for t in xs)
            y = y.to(device)
            out = net(*xs)[0].clamp(0, 1)
            se_out += float(((out - y) ** 2).sum())
            se_in += float(((xs[0] - y) ** 2).sum())
            n += y.numel()
    net.train()
    return 10 * np.log10(n / max(se_out, 1e-12)), 10 * np.log10(n / max(se_in, 1e-12))


def export_onnx(net, path: Path, guided: bool = False) -> None:
    """The network as the program runs it: "input" (N, 3, H, W) in 0..1, H and W multiples
    of 8 (guided: also "guide" (N, 3, H/4, W/4), "has_guide" (N, 1), "holes" (N, 1, H/8,
    W/8)); outputs the cleaned picture and the estimated JPEG quality."""
    import torch

    net = net.cpu().eval()
    x = torch.rand(1, 3, 128, 128)
    if guided:
        args = (x, torch.rand(1, 3, 32, 32), torch.ones(1, 1), torch.zeros(1, 1, 16, 16))
        names = ["input", "guide", "has_guide", "holes"]
        axes = {"input": {0: "n", 2: "h", 3: "w"}, "guide": {0: "n", 2: "h4", 3: "w4"}, "has_guide": {0: "n"},
                "holes": {0: "n", 2: "h8", 3: "w8"}, "output": {0: "n", 2: "h", 3: "w"}}
    else:
        args = (x,)
        names = ["input"]
        axes = {"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h", 3: "w"}}
    torch.onnx.export(net, args, str(path), input_names=names, output_names=["output", "qf"], opset_version=17,
                      dynamic_axes=axes, dynamo=False)


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
    ap.add_argument("--guided", action="store_true", help="train the guided network (sees the thumbnail and the holes)")
    ap.add_argument("--init", type=Path, help="start from this state (a best.pt of the plain fine-tuning)")
    a = ap.parse_args(argv)

    import torch

    from .fbcnn_net import FBCNN

    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    tr = load_pairs(a.pairs, "train", a.max_patches, a.guided)
    va = load_pairs(a.pairs, "val", 0, a.guided)
    if not len(tr["inp"]):
        raise SystemExit(f"no train_*.npz in {a.pairs} (make them with: python -m fotomedik.train.pairs)")
    if not len(va["inp"]):  # few photos: check on a slice of the training patches
        va = {k: v[-64:] for k, v in tr.items()}
    print(f"{len(tr['inp'])} training patches, {len(va['inp'])} for checking"
          + (f" ({100 * float(tr['has_guide'].mean()):.0f} % with a thumbnail)" if a.guided else ""))

    if a.init is not None:
        state = torch.load(a.init, map_location="cpu", weights_only=True)
        state = {k[4:] if k.startswith("net.") else k: v for k, v in state.items()}  # a guided state
        print(f"starting from {a.init}")
    else:
        weights = a.out / "fbcnn_color.pth"
        if not weights.exists():
            print("downloading the original FBCNN weights (~290 MB, once)...")
            urllib.request.urlretrieve(WEIGHTS_URL, weights)
        state = torch.load(weights, map_location="cpu", weights_only=True)
    if a.guided:
        base = FBCNN(in_nc=3 + EXTRA)
        if state["m_head.weight"].shape[1] == 3:
            state = widen_head(state)
        base.load_state_dict(state, strict=True)
        net = guided_net(base)
    else:
        net = FBCNN()
        net.load_state_dict(state, strict=True)
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
        start = st["iter"]
        # the patches may be new (made again, more photos): the best so far is measured anew on
        # today's check patches, or no new state could ever beat a number from other patches
        p_now, p_in = psnr_of(net, va, device, a.guided)
        best_state = torch.load(a.out / "best.pt", map_location="cpu", weights_only=True)
        cur_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        best, _ = psnr_of(net, va, device, a.guided)
        net.load_state_dict(cur_state)
        print(f"resumed at iteration {start}: now {p_now:.2f} dB, best so far {best:.2f} dB "
              f"(repair as it is {p_in:.2f} dB, on today's check patches)")
        if start >= sched.T_max:  # more iterations asked for than the last run had: a new, gentler schedule
            lr = a.lr * 0.5
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] = lr
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.iters - start), eta_min=lr * 0.1)
            print(f"continuing to iteration {a.iters} with learning rate {lr:g}")
    else:
        p_net, p_in = psnr_of(net, va, device, a.guided)
        print(f"before training: repair as it is {p_in:.2f} dB, the network as it starts {p_net:.2f} dB")
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
            xs, y = batch_of(tr, a.batch, rng, guided=a.guided)
            xs = tuple(t.to(device, non_blocking=True) for t in xs)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = net(*xs)[0]
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
                p_net, p_in = psnr_of(net, va, device, a.guided)
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
    name = ONNX_GUIDED if a.guided else ONNX_NAME
    onnx_path = a.out / name
    export_onnx(net, onnx_path, a.guided)
    print(f"best: {best:.2f} dB -> {onnx_path}")
    if a.install:
        dst = model_dir() / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, dst)
        for old in dst.parent.glob(name.replace(".onnx", "") + "*fp16*"):
            old.unlink(missing_ok=True)  # a half-precision copy of an earlier version
        print(f"installed: {dst} (the JPEG clean-up step uses it from now on)")


if __name__ == "__main__":
    main()
