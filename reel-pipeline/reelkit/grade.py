"""Colour look: measure it from a sample reel, transfer it onto new footage as a 3D LUT.

A look is summarised in CIELAB, ignoring text overlays:
  - the lightness distribution (101 percentiles): contrast, black/white points, tone curve
  - per tone zone (shadows / mids / highlights, by lightness rank): mean a*, b* (split
    toning, casts) and colour spread (saturation)
The LUT maps footage → look: lightness by percentile matching (a monotone tone curve);
colour, zone by zone, by re-centring (a*, b*) on the target zone's mean and scaling
its spread (Reinhard transfer per zone). ffmpeg's lut3d applies it at render time.
"""
import cv2
import numpy as np

PCTS = list(range(0, 101))                 # lightness percentiles kept (tone curve + ranks)
ZONES = {"shadows": 10.0, "mids": 50.0, "highlights": 90.0}   # centres, in lightness-rank %
ZONE_WIDTH = 22.0


def rank_weights(r):
    """Soft memberships (sum 1) of lightness ranks r (0..100) in shadows / mids / highlights.
    Zones are defined by rank, not absolute lightness, so a tone curve (monotone)
    keeps every pixel in the same zone on both sides of the transfer."""
    w = np.stack([np.exp(-0.5 * ((r - c) / ZONE_WIDTH) ** 2) for c in ZONES.values()], -1)
    return w / w.sum(-1, keepdims=True)


def zone_weights(L):
    """Memberships by absolute lightness (for synthetic looks in tests)."""
    return rank_weights(np.asarray(L, np.float64))


def to_lab(bgr_u8):
    return cv2.cvtColor(bgr_u8.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).reshape(-1, 3)


class Look:
    """Accumulate Lab samples from frames (with optional exclusion masks)."""

    def __init__(self, cap=1_500_000, step=3):
        self.parts, self.n, self.cap, self.step = [], 0, cap, step

    def add(self, bgr, exclude=None):
        sub = bgr[::self.step, ::self.step]
        lab = to_lab(sub)
        if exclude is not None:
            keep = ~exclude[::self.step, ::self.step].reshape(-1)
            lab = lab[keep]
        if self.n < self.cap and len(lab):
            self.parts.append(lab)
            self.n += len(lab)

    def stats(self):
        lab = np.concatenate(self.parts) if self.parts else np.zeros((1, 3), np.float32)
        return stats_of(lab)


def _rank(L, pct):
    return np.interp(L, np.maximum.accumulate(np.asarray(pct) + np.arange(len(pct)) * 1e-6), np.arange(len(pct)))


def stats_of(lab):
    L, a, b = lab[:, 0].astype(np.float64), lab[:, 1].astype(np.float64), lab[:, 2].astype(np.float64)
    pct = np.percentile(L, PCTS)
    w = rank_weights(_rank(L, pct))
    zones = {}
    for i, name in enumerate(ZONES):
        wi = w[:, i]
        tot = wi.sum()
        ma, mb = (a * wi).sum() / tot, (b * wi).sum() / tot
        sd = np.sqrt((((a - ma) ** 2 + (b - mb) ** 2) * wi).sum() / tot)
        zones[name] = {"a": float(ma), "b": float(mb), "sd": float(sd)}
    C = np.hypot(a, b)
    return {"L": [float(x) for x in pct], "zones": zones,
            "chroma": {str(p): float(np.percentile(C, p)) for p in (25, 50, 75, 90, 95)},
            "mean_lab": [float(L.mean()), float(a.mean()), float(b.mean())], "samples": int(len(L))}


def _tone_curve(src_L, tgt_L):
    """Monotone lightness map through the percentile pairs, linear tails, clipped 0..100."""
    xs = np.array([0.0] + list(src_L) + [100.0])
    lo_slope = (tgt_L[1] - tgt_L[0]) / max(1e-3, src_L[1] - src_L[0])
    hi_slope = (tgt_L[-1] - tgt_L[-2]) / max(1e-3, src_L[-1] - src_L[-2])
    ys = np.array([max(0.0, tgt_L[0] - src_L[0] * np.clip(lo_slope, 0.3, 3))] + list(tgt_L) +
                  [min(100.0, tgt_L[-1] + (100 - src_L[-1]) * np.clip(hi_slope, 0.3, 3))])
    xs = np.maximum.accumulate(xs + np.arange(len(xs)) * 1e-6)
    ys = np.maximum.accumulate(ys)
    return xs, ys


def build_lut(src, tgt, size=33, strength=1.0, sd_clamp=(0.5, 2.0)):
    """3D LUT (size³ × RGB in 0..1, indexed [r, g, b]) taking the `src` look to `tgt`.
    Lightness: percentile matching. Colour, per rank zone: (a, b) about the zone mean,
    scaled by the zone's spread ratio, re-centred on the target zone mean."""
    g = np.linspace(0, 1, size, dtype=np.float32)
    R, G, B = np.meshgrid(g, g, g, indexing="ij")
    rgb = np.stack([R, G, B], -1).reshape(-1, 3)
    lab = cv2.cvtColor(rgb[:, None, ::-1].copy(), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    xs, ys = _tone_curve(src["L"], tgt["L"])
    L2 = np.interp(L, xs, ys)
    w = rank_weights(_rank(L, src["L"]))
    a2 = np.zeros_like(a)
    b2 = np.zeros_like(b)
    for i, z in enumerate(ZONES):
        s_, t_ = src["zones"][z], tgt["zones"][z]
        k = np.clip(t_["sd"] / max(1e-3, s_["sd"]), *sd_clamp)
        a2 += w[:, i] * (t_["a"] + (a - s_["a"]) * k)
        b2 += w[:, i] * (t_["b"] + (b - s_["b"]) * k)
    lab2 = np.stack([L2, a2, b2], -1).astype(np.float32)
    out = cv2.cvtColor(lab2[:, None, :], cv2.COLOR_LAB2BGR).reshape(-1, 3)[:, ::-1]
    out = np.clip(out, 0, 1)
    out = rgb + (out - rgb) * strength
    return out.reshape(size, size, size, 3)


def apply_lut(bgr_u8, lut):
    """Trilinear LUT lookup in numpy (tests / previews; renders use ffmpeg lut3d)."""
    n = lut.shape[0]
    rgb = bgr_u8[..., ::-1].astype(np.float32) / 255.0 * (n - 1)
    i0 = np.clip(np.floor(rgb).astype(int), 0, n - 2)
    f = rgb - i0
    out = np.zeros_like(rgb)
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                w = (f[..., 0] if dr else 1 - f[..., 0]) * (f[..., 1] if dg else 1 - f[..., 1]) * \
                    (f[..., 2] if db else 1 - f[..., 2])
                out += w[..., None] * lut[i0[..., 0] + dr, i0[..., 1] + dg, i0[..., 2] + db]
    return np.clip(out[..., ::-1] * 255 + 0.5, 0, 255).astype(np.uint8)


def write_cube(lut, path, title="reelkit look"):
    n = lut.shape[0]
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        for b in range(n):            # .cube order: red fastest
            for g in range(n):
                for r in range(n):
                    f.write("%.6f %.6f %.6f\n" % tuple(lut[r, g, b]))


def delta_e(bgr1, bgr2):
    """Mean CIE76 ΔE between two images."""
    return float(np.linalg.norm(to_lab(bgr1) - to_lab(bgr2), axis=1).mean())
