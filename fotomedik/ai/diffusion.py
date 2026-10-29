"""Maximum-quality hole filling with Stable Diffusion 1.5 inpainting (ONNX).

The holes are repainted window by window (512x512 pixels at the photo's own scale).
The diffusion does not start from noise but from the current picture with noise
added part of the way (SDEdit, `STRENGTH`), so the shapes and colours the thumbnail
gave the holes stay; the network adds texture and detail. Real pixels are passed to
the inpainting network as known and are never changed. No text prompt is used.

Made for a graphics card: on the processor every window takes minutes.
"""

from __future__ import annotations

import numpy as np

WINDOW = 512
STRIDE = 384
STEPS = 20
STRENGTH = 0.6  # how far into the noise the start goes: higher invents more
LATENT_SCALE = 0.18215
BOS, EOS = 49406, 49407  # CLIP tokens: an empty prompt is BOS, then EOS padding


def _dtype(inp):
    from .enhance import ORT_DTYPES

    return ORT_DTYPES.get(getattr(inp, "type", ""), np.float32)


class Diffusion:
    def __init__(self, unet, vae_enc, vae_dec, text):
        self.unet, self.vae_enc, self.vae_dec, self.text = unet, vae_enc, vae_dec, text
        self._emb = None
        betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float64) ** 2
        self.alphas = np.cumprod(1.0 - betas)

    def embedding(self) -> np.ndarray:
        if self._emb is None:
            inp = self.text.get_inputs()[0]
            ids = np.array([[BOS] + [EOS] * 76], dtype=_dtype(inp))
            self._emb = self.text.run(None, {inp.name: ids})[0]
        return self._emb

    def encode(self, img: np.ndarray) -> np.ndarray:
        """uint8 (H, W, 3) -> latents (1, 4, H/8, W/8)."""
        inp = self.vae_enc.get_inputs()[0]
        x = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None].astype(_dtype(inp))
        lat = self.vae_enc.run(None, {inp.name: x})[0].astype(np.float32)
        if lat.shape[1] == 8:  # moments (mean, logvar): take the mean
            lat = lat[:, :4]
        return lat * LATENT_SCALE

    def decode(self, lat: np.ndarray) -> np.ndarray:
        inp = self.vae_dec.get_inputs()[0]
        x = self.vae_dec.run(None, {inp.name: (lat / LATENT_SCALE).astype(_dtype(inp))})[0].astype(np.float32)
        return np.clip(np.rint((x[0].transpose(1, 2, 0) + 1.0) * 127.5), 0, 255).astype(np.uint8)

    def inpaint(self, img: np.ndarray, mask: np.ndarray, steps: int = STEPS, strength: float = STRENGTH,
                seed: int = 0) -> np.ndarray:
        """img uint8 (S, S, 3) and mask bool (S, S), S a multiple of 64; masked part repainted."""
        rng = np.random.default_rng(seed)
        s = img.shape[0]
        m = mask.astype(np.float32)
        x0 = self.encode(img)
        masked = self.encode((img * (1 - m[..., None]) + 127.5 * m[..., None]).astype(np.uint8))
        lm = m.reshape(s // 8, 8, s // 8, 8).max(axis=(1, 3))[None, None]
        ratio = 1000 // steps
        timesteps = (np.arange(steps) * ratio)[::-1] + 1
        k = max(1, int(round(steps * strength)))
        timesteps = timesteps[steps - k :]
        t0 = int(timesteps[0])
        noise = rng.standard_normal(x0.shape).astype(np.float32)
        lat = np.sqrt(self.alphas[t0]) * x0 + np.sqrt(1 - self.alphas[t0]) * noise
        ins = {i.name: i for i in self.unet.get_inputs()}
        names = list(ins)
        sample_n = next((n for n in names if "sample" in n), names[0])
        t_n = next((n for n in names if "time" in n), names[1])
        enc_n = next((n for n in names if "hidden" in n or "encoder" in n), names[2])
        emb = self.embedding().astype(_dtype(ins[enc_n]))
        for t in timesteps:
            t = int(t)
            model_in = np.concatenate([lat, lm, masked], axis=1).astype(_dtype(ins[sample_n]))
            eps = self.unet.run(None, {sample_n: model_in, t_n: np.array([t], dtype=_dtype(ins[t_n])),
                                       enc_n: emb})[0].astype(np.float32)
            a_t = self.alphas[t]
            prev = t - ratio
            a_prev = self.alphas[prev] if prev >= 0 else self.alphas[0]
            x0_pred = (lat - np.sqrt(1 - a_t) * eps) / np.sqrt(a_t)
            lat = np.sqrt(a_prev) * x0_pred + np.sqrt(1 - a_prev) * eps
        out = self.decode(lat)
        return np.where(mask[..., None], out, img)


def windows(holes: np.ndarray, size: int = WINDOW, stride: int = STRIDE) -> list[tuple[int, int]]:
    """Top-left corners of the windows that cover all holes."""
    h, w = holes.shape
    ys = list(range(0, max(1, h - size) + 1, stride))
    xs = list(range(0, max(1, w - size) + 1, stride))
    if ys[-1] + size < h:
        ys.append(h - size)
    if xs[-1] + size < w:
        xs.append(w - size)
    return [(max(0, y), max(0, x)) for y in ys for x in xs if holes[max(0, y) : y + size, max(0, x) : x + size].any()]


def diffusion_fill(arr: np.ndarray, holes: np.ndarray, unknown: np.ndarray, sd: Diffusion, progress=None,
                   size: int = WINDOW, steps: int = STEPS) -> np.ndarray:
    """Holes of `arr` repainted window by window; real pixels stay."""
    from .enhance import _soft

    h, w = holes.shape
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    work = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect") if pad_h or pad_w else arr.copy()
    hm = np.pad(holes, ((0, pad_h), (0, pad_w))) if pad_h or pad_w else holes
    done = np.zeros_like(hm)
    wins = windows(hm, size, min(STRIDE, size * 3 // 4))
    for i, (y, x) in enumerate(wins):
        tile = work[y : y + size, x : x + size]
        m = hm[y : y + size, x : x + size] & ~done[y : y + size, x : x + size]
        if not m.any():
            continue
        res = sd.inpaint(tile, m, steps=steps, seed=i)
        a = _soft(m, 2.0)[..., None]
        work[y : y + size, x : x + size] = np.clip(np.rint(tile * (1 - a) + res * a), 0, 255).astype(np.uint8)
        # the centre of a window counts as done; its border may be redone by the next one
        inner = np.zeros_like(m)
        b = (size - min(STRIDE, size * 3 // 4)) // 2
        inner[b : size - b, b : size - b] = True
        done[y : y + size, x : x + size] |= m & inner
        if progress:
            progress(i + 1, len(wins))
    return work[:h, :w]
