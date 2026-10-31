"""Export super-resolution networks (PyTorch weights) to ONNX, for trying them as the
enlarger of the thumbnail step. Run where torch is installed (the convert-models
workflow does it):

    python scripts/export_sr.py WEIGHTS_DIR OUT_DIR [SWINIR_REPO]

Every network takes float32 RGB 0..1 (N, 3, H, W) as "input" and gives 4x (N, 3, 4H, 4W).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


# --- RRDBNet (ESRGAN / Real-ESRGAN / Real-ESRNet), as in BasicSR -----------------
class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# --- SRVGGNetCompact (realesr-general-x4v3) ------------------------------------------
class SRVGGNetCompact(nn.Module):
    def __init__(self, nf=64, nc=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
        for _ in range(nc):
            body += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
        body.append(nn.Conv2d(nf, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for m in self.body:
            out = m(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def _state(path: Path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("params_ema", "params"):
        if isinstance(sd, dict) and k in sd:
            return sd[k]
    return sd


SWIN_TILE = 144  # the enlarger's tile side in the program (TILE + 2 * PAD)


class Padded(nn.Module):
    """SwinIR needs sides that are a multiple of its window: pad (reflect) and crop."""

    def __init__(self, net, window=8, scale=4):
        super().__init__()
        self.net, self.window, self.scale = net, window, scale

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        ph = (self.window - h % self.window) % self.window
        pw = (self.window - w % self.window) % self.window
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        y = self.net(x)
        return y[:, :, : h * self.scale, : w * self.scale]


def export(net: nn.Module, out: Path, size=64, fixed=False):
    """fixed: the side is always `size` (SwinIR computes its attention masks for one size;
    the program feeds the enlarger tiles of one size anyway)."""
    net.eval()
    x = torch.rand(1, 3, size, size)
    axes = {"input": {0: "n"}, "output": {0: "n"}} if fixed else \
        {"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h4", 3: "w4"}}
    torch.onnx.export(net, x, str(out), input_names=["input"], output_names=["output"], opset_version=17,
                      dynamic_axes=axes, dynamo=False)
    print("wrote", out, out.stat().st_size // 1024, "KB")


def main():
    wdir, odir = Path(sys.argv[1]), Path(sys.argv[2])
    odir.mkdir(parents=True, exist_ok=True)
    import traceback

    def safe(fn):
        try:
            fn()
        except Exception:  # noqa: BLE001 - one network failing must not stop the others
            traceback.print_exc()

    safe(lambda: _rrdb_and_compact(wdir, odir))
    if len(sys.argv) > 3:
        safe(lambda: _swinir(wdir, odir, sys.argv[3]))


def _rrdb_and_compact(wdir: Path, odir: Path):
    for name in ("RealESRNet_x4plus", "RealESRGAN_x4plus"):
        p = wdir / f"{name}.pth"
        if p.exists():
            net = RRDBNet()
            net.load_state_dict(_state(p), strict=True)
            export(net, odir / f"{name}.onnx")
    for name in ("realesr-general-x4v3", "realesr-general-wdn-x4v3"):
        p = wdir / f"{name}.pth"
        if p.exists():
            net = SRVGGNetCompact()
            net.load_state_dict(_state(p), strict=True)
            export(net, odir / f"{name}.onnx")


def _swinir(wdir: Path, odir: Path, repo: str):  # with the architecture from its repository
    sys.path.insert(0, repo)
    from models.network_swinir import SwinIR  # noqa: PLC0415
    p = wdir / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    if p.exists():
        net = SwinIR(upscale=4, in_chans=3, img_size=SWIN_TILE, window_size=8, img_range=1.0,
                     depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
                     upsampler="pixelshuffle", resi_connection="1conv")
        net.load_state_dict(_state(p), strict=True)
        export(Padded(net), odir / "SwinIR-M_classical_x4.onnx", SWIN_TILE, fixed=True)
    p = wdir / "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth"
    if p.exists():
        net = SwinIR(upscale=4, in_chans=3, img_size=SWIN_TILE, window_size=8, img_range=1.0,
                     depths=[6] * 9, embed_dim=240, num_heads=[8] * 9, mlp_ratio=2,
                     upsampler="nearest+conv", resi_connection="3conv")
        net.load_state_dict(_state(p), strict=True)
        export(Padded(net), odir / "SwinIR-L_real_x4_PSNR.onnx", SWIN_TILE, fixed=True)


if __name__ == "__main__":
    main()
