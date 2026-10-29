"""Maximum-quality hole filling with Stable Diffusion 1.5 inpainting (ONNX).

The holes are repainted window by window (512x512 pixels). The diffusion does not
start from noise but from the current picture with noise added part of the way
(SDEdit, `STRENGTH`), so the shapes and colours the thumbnail gave the holes stay;
the network adds texture and detail. Real pixels are passed to the inpainting
network as known and are never changed. No text prompt is used.

The work is done on a reduced copy of the photo (`PRESETS`): a 12 MP photo is
repainted at half size, in about 20 windows instead of 80, and each window then
shows the network a bigger part of the scene. The repainted holes are brought back
to full size by Real-ESRGAN (see enhance.enlarge_holes).

Made for a graphics card: on the processor every window takes minutes.
"""

from __future__ import annotations

import numpy as np

WINDOW = 512
STRIDE = 384
STEPS = 20
STRENGTH = 0.6  # how far into the noise the start goes: higher invents more
GUIDANCE = 5.0  # classifier-free guidance scale when a prompt is given
DEFAULT_PROMPT = "a sharp, detailed photograph"
LATENT_SCALE = 0.18215
BOS, EOS = 49406, 49407  # CLIP tokens: an empty prompt is BOS, then EOS padding

# quality presets: (the photo is reduced this many times, length of the DDIM schedule);
# with STRENGTH 0.6 the schedule lengths mean 6, 8 and 12 passes of the network per window
PRESETS = {"fast": (4, 10), "balanced": (2, 14), "full": (1, 20)}
PRESET_NAMES = {"fast": "fast", "balanced": "balanced", "full": "full size"}
DEFAULT_PRESET = "balanced"
MIN_SIDE = 768  # the reduced copy keeps at least this many pixels on its long side


def working_divisor(long_side: int, divisor: int) -> int:
    """The preset's reduction, lowered for small photos (a 2 MP photo is not quartered)."""
    while divisor > 1 and long_side // divisor < MIN_SIDE:
        divisor //= 2
    return divisor


def _dtype(inp):
    from .enhance import ORT_DTYPES

    return ORT_DTYPES.get(getattr(inp, "type", ""), np.float32)


class Diffusion:
    """The four networks of Stable Diffusion inpainting and the DDIM loop around them.
    With a tokenizer, a prompt steers what is painted (classifier-free guidance: every
    step runs the network with and without the prompt and goes `guidance` times further
    in the prompt's direction); without one the painting is unconditioned."""

    def __init__(self, unet, vae_enc, vae_dec, text, tokenizer=None):
        self.unet, self.vae_enc, self.vae_dec, self.text = unet, vae_enc, vae_dec, text
        self.tokenizer = tokenizer
        self._emb: dict[str, np.ndarray] = {}
        self._pair_ok: bool | None = None  # whether the network takes prompt and no-prompt in one batch
        betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float64) ** 2
        self.alphas = np.cumprod(1.0 - betas)

    def embedding(self, prompt: str = "") -> np.ndarray:
        """Text encoder output (1, 77, 768) for `prompt` ("" is the empty prompt)."""
        key = " ".join(prompt.lower().split()) if (prompt and self.tokenizer is not None) else ""
        if key not in self._emb:
            inp = self.text.get_inputs()[0]
            ids = self.tokenizer.encode(key) if key else [BOS] + [EOS] * 76
            self._emb[key] = self.text.run(None, {inp.name: np.array([ids], dtype=_dtype(inp))})[0]
        return self._emb[key]

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

    def _noise(self, sample_n: str, t_n: str, enc_n: str, model_in, t: int, embs: list) -> list:
        """The network's noise estimate for every embedding in `embs` (one batch when it can)."""
        ins = {i.name: i for i in self.unet.get_inputs()}
        tt = np.array([t], dtype=_dtype(ins[t_n]))
        if len(embs) > 1 and self._pair_ok is not False:
            try:
                batch = np.concatenate([model_in] * len(embs), axis=0)
                emb = np.concatenate(embs, axis=0).astype(_dtype(ins[enc_n]))
                out = self.unet.run(None, {sample_n: batch, t_n: tt, enc_n: emb})[0].astype(np.float32)
                if out.shape[0] == len(embs):
                    self._pair_ok = True
                    return [out[i : i + 1] for i in range(len(embs))]
            except Exception:
                pass
            self._pair_ok = False
        return [self.unet.run(None, {sample_n: model_in, t_n: tt, enc_n: e.astype(_dtype(ins[enc_n]))})[0]
                .astype(np.float32) for e in embs]

    def inpaint(self, img: np.ndarray, mask: np.ndarray, steps: int = STEPS, strength: float = STRENGTH,
                seed: int = 0, prompt: str | None = None, guidance: float = GUIDANCE) -> np.ndarray:
        """img uint8 (S, S, 3) and mask bool (S, S), S a multiple of 64; masked part repainted.
        prompt: what the holes should show (needs the tokenizer), guidance: how strongly."""
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
        guided = bool(prompt) and self.tokenizer is not None and guidance > 1.0
        embs = [self.embedding(prompt), self.embedding("")] if guided else [self.embedding("")]
        for t in timesteps:
            t = int(t)
            model_in = np.concatenate([lat, lm, masked], axis=1).astype(_dtype(ins[sample_n]))
            eps_all = self._noise(sample_n, t_n, enc_n, model_in, t, embs)
            eps = eps_all[1] + guidance * (eps_all[0] - eps_all[1]) if guided else eps_all[0]
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
                   size: int = WINDOW, steps: int = STEPS, stop=None, stats: dict | None = None,
                   prompt: str | None = None) -> np.ndarray:
    """Holes of `arr` repainted window by window; real pixels stay."""
    from .enhance import _check, _soft

    h, w = holes.shape
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    work = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect") if pad_h or pad_w else arr.copy()
    hm = np.pad(holes, ((0, pad_h), (0, pad_w))) if pad_h or pad_w else holes
    done = np.zeros_like(hm)
    wins = windows(hm, size, min(STRIDE, size * 3 // 4))
    if stats is not None:
        stats["windows"] = len(wins)
    for i, (y, x) in enumerate(wins):
        _check(stop)
        tile = work[y : y + size, x : x + size]
        m = hm[y : y + size, x : x + size] & ~done[y : y + size, x : x + size]
        if not m.any():
            continue
        res = sd.inpaint(tile, m, steps=steps, seed=i, prompt=prompt)
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


def repaint_holes(arr: np.ndarray, holes: np.ndarray, unknown: np.ndarray, engine, quality: str | None = None,
                  progress=None, stop=None, prompt: str | None = None) -> tuple[np.ndarray, str]:
    """Holes of `arr` repainted at the preset's working size and brought back to full
    size (Real-ESRGAN where the engine has it, bicubic otherwise). Real pixels stay.
    prompt: what the scene shows (None: the default prompt). Returns (rgb, how it was done)."""
    from PIL import Image

    from .enhance import enlarge_holes, reduce, reduce_mask

    divisor, steps = PRESETS[quality or DEFAULT_PRESET]
    prompt = DEFAULT_PROMPT if prompt is None else prompt
    h, w = holes.shape
    divisor = working_divisor(max(h, w), divisor)
    st: dict = {}
    if divisor == 1:
        out = diffusion_fill(arr, holes, unknown, engine.sd, progress, steps=steps, stop=stop, stats=st, prompt=prompt)
        return out, f"full size, {st.get('windows', 0)} windows"
    lo, lo_holes = reduce(arr, holes, divisor)
    lo_unknown = reduce_mask(unknown & holes, divisor) if unknown.any() else np.zeros_like(lo_holes)
    lo_out = diffusion_fill(lo, lo_holes, lo_unknown, engine.sd, progress, steps=steps, stop=stop, stats=st,
                            prompt=prompt)
    if engine.sr is not None:
        hi = enlarge_holes(lo_out, lo_holes, engine, divisor, None, stop) * 255.0
        how = "Real-ESRGAN"
    else:
        hi = np.asarray(Image.fromarray(lo_out).resize((lo_out.shape[1] * divisor, lo_out.shape[0] * divisor),
                                                       Image.BICUBIC), dtype=np.float32)
        how = "bicubic"
    hi = np.clip(np.rint(hi[:h, :w]), 0, 255).astype(np.uint8)
    out = np.where(holes[..., None], hi, arr)
    return out, f"1/{divisor} size, {st.get('windows', 0)} windows, enlarged by {how}"
