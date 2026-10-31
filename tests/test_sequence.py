"""Short chains between rotted sectors placed as a sequence (r11_segments.place_sequence)."""

import numpy as np

from fotomedik.recipes.r11_segments import place_sequence


class FakePlacer:
    """Misfit of a unit against a known reference (best offset removed), like Placer.costs."""

    def __init__(self, ref, noise=1.0):
        self.ref, self.noise = ref, noise

    def costs(self, rel, lo, hi):
        n = len(rel)
        lo, hi = max(0, lo), min(len(self.ref), hi)
        out = []
        for m in range(lo, hi - n + 1):
            d = self.ref[m : m + n] - rel
            out.append(float(((d - d.mean()) ** 2).mean()) / self.noise**2)
        return np.array(out), lo


def test_sequence_puts_short_chains_back_in_order():
    rng = np.random.default_rng(3)
    total = 3000
    ref = np.cumsum(rng.normal(0, 6, (total, 1)), axis=0)  # a textured picture: DC along the MCUs
    rate = 20.0  # bytes per MCU
    units, truth = [], []
    m, b = 100, 2000.0
    while m < total - 200:
        n = int(rng.integers(14, 40))
        lost = int(rng.integers(5, 40))  # MCUs in the rotted sectors before it
        m += lost
        b += lost * rate
        rel = ref[m : m + n] + 50.0 + rng.normal(0, 0.5, (n, 1))  # unknown DC offset, a little noise
        units.append({"byte": b, "end_byte": b + n * rate, "n": n, "rel": rel})
        truth.append(m)
        m += n
        b += n * rate
    got = place_sequence(FakePlacer(ref), units, 2000.0, 100, b + 10 * rate, m + 10, rate)
    right = sum(1 for j, pos in got if pos == truth[j])
    assert len(got) >= 0.8 * len(units)
    assert right >= 0.95 * len(got)
