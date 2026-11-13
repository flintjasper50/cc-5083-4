"""Synthetic ground-truth images and spatially varying degradation.

Every sample in the shipped dataset is produced by
generate_sample(): a sharp image x, a kernel field F, a forward pass
A(F) x with a declared boundary rule, and Gaussian noise with a declared
sigma.  Nothing is hand-drawn: the blur is the operator's own forward model,
so operator consistency and restoration are measured against a real,
reproducible degradation rather than a plausible-looking fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .field import KernelField, make_control_field
from .operator import SVForward


# --------------------------------------------------------------------------- sources
def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render(size: tuple[int, int], draw_fn) -> np.ndarray:
    W, H = size
    canvas = Image.new("L", (W, H), 0)
    draw_fn(ImageDraw.Draw(canvas), W, H)
    return np.asarray(canvas, dtype=np.float64) / 255.0


def texture_checker(W: int, H: int) -> np.ndarray:
    """Fine checkerboard + sinusoids: high-frequency texture for ringing tests."""
    yy, xx = np.mgrid[0:H, 0:W]
    base = 0.5 + 0.20 * np.sin(xx * 0.55) * np.cos(yy * 0.41)
    checker = (((xx // 6) + (yy // 6)) % 2).astype(float) * 0.18
    fine = 0.12 * np.sin((xx * 1.7 + yy * 1.3))
    return np.clip(base + checker + fine, 0, 1)


def texture_bars(W: int, H: int) -> np.ndarray:
    """Vertical bars of increasing width plus a diagonal edge."""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 0.35)
    x = 0
    width = 2
    while x < W:
        img[:, x:x + width] = 0.85
        x += width + 4
        width = 2 + (width * 2) % 14
    diag = (xx + yy * 0.6) > W * 0.65
    img[diag] = np.where(img[diag] > 0.6, 0.15, 0.95)
    return np.clip(img, 0, 1)


def texture_text(W: int, H: int) -> np.ndarray:
    """Dense glyphs at two scales: strong edges in both axes."""
    font_big = _font(max(20, H // 4))
    font_small = _font(max(11, H // 9))

    def draw(d: ImageDraw.ImageDraw, w: int, h: int):
        d.rectangle([0, 0, w - 1, h - 1], fill=0)
        d.text((w * 0.03, h * 0.10), "SVDB", font=font_big, fill=255)
        d.text((w * 0.03, h * 0.48), "kernel field", font=font_small, fill=200)
        for k in range(4):
            d.text((w * 0.03, h * 0.60 + k * h * 0.09),
                   f"{['inverse','adjoint','tent','TV'][k]} 0123456789",
                   font=font_small, fill=160 + 20 * k)
        d.line([(0, h - 1), (w - 1, 0)], fill=255, width=2)

    return _render((W, H), draw)


def texture_rings(W: int, H: int) -> np.ndarray:
    """Concentric rings and dots: curvature / isotropy probe."""
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    img = 0.5 + 0.25 * np.sin(r * 0.55)
    mask = ((yy // 11 + xx // 11) % 3 == 0) & (r > 0.12 * min(H, W))
    img[mask] = 1.0 - img[mask]
    return np.clip(img, 0, 1)


def texture_mixed(W: int, H: int) -> np.ndarray:
    """Composite: text block, bar block and a smooth gradient."""
    img = np.zeros((H, W))
    half = H // 2
    img[:half] = texture_text(W, half).repeat(1, 1)
    img[half:] = texture_bars(W, H - half)
    grad = np.linspace(0.12, 0.0, H)[:, None]
    img = np.clip(img * 0.92 + grad, 0, 1)
    return img


TEXTURES = {
    "checker": texture_checker,
    "bars": texture_bars,
    "text": texture_text,
    "rings": texture_rings,
    "mixed": texture_mixed,
}


# --------------------------------------------------------------------------- PSF shapes
def gaussian_psf(support: tuple[int, int], sy: float, sx: float,
                 cy: float | None = None, cx: float | None = None) -> np.ndarray:
    kh, kw = support
    y0, x0 = (kh - 1) / 2, (kw - 1) / 2
    cy = y0 if cy is None else cy
    cx = x0 if cx is None else cx
    yy, xx = np.mgrid[0:kh, 0:kw]
    g = np.exp(-((yy - cy) ** 2 / (2 * sy ** 2) + (xx - cx) ** 2 / (2 * sx ** 2)))
    return g / g.sum()


def motion_psf(support: tuple[int, int], length: float, angle_deg: float) -> np.ndarray:
    """Anti-aliased line PSF (linear motion), energy normalised."""
    kh, kw = support
    c = np.array([(kh - 1) / 2, (kw - 1) / 2])
    ang = np.deg2rad(angle_deg)
    d = np.array([np.sin(ang), np.cos(ang)])
    yy, xx = np.mgrid[0:kh, 0:kw]
    p = np.stack([yy - c[0], xx - c[1]], axis=-1)
    along = p[..., 0] * d[0] + p[..., 1] * d[1]
    perp = np.abs(p[..., 0] * d[1] - p[..., 1] * d[0])
    k = np.exp(-(perp ** 2) / (2 * 0.55 ** 2)) * (np.abs(along) <= length / 2)
    return k / k.sum()


def disc_psf(support: tuple[int, int], radius: float) -> np.ndarray:
    kh, kw = support
    yy, xx = np.mgrid[0:kh, 0:kw]
    cy = cx = (kh - 1) / 2, (kw - 1) / 2
    cy, cx = cx, cy
    k = (np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) <= radius).astype(float)
    if k.sum() == 0:
        k[kh // 2, kw // 2] = 1.0
    return k / k.sum()


def signed_psf(support: tuple[int, int]) -> np.ndarray:
    """A signed (unphysical) kernel: sharp centre with negative lobes."""
    k = gaussian_psf(support, 0.9, 0.9)
    kh, kw = support
    yy, xx = np.mgrid[0:kh, 0:kw]
    cy = cx = (kh - 1) / 2
    ring = np.exp(-((np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) - 2.2) ** 2) / (2 * 0.6 ** 2))
    ring /= ring.sum()
    return 1.35 * k - 0.35 * ring


# --------------------------------------------------------------------------- fields
KERNEL_KINDS = ("gaussian_gradient", "motion_rotation", "mixed_modes",
                "checker_swap", "signed_demo", "oversized_demo")


def make_field(kind: str, image_shape: tuple[int, int],
               grid_shape: tuple[int, int], support: tuple[int, int]) -> KernelField:
    """Deterministic kernel fields used by the shipped samples."""
    gh, gw = grid_shape
    if kind == "gaussian_gradient":
        def fn(yn, xn, i, j):
            return gaussian_psf(support, 0.9 + 1.9 * yn, 0.9 + 1.9 * xn,
                                cy=2 + 0.9 * yn, cx=2 + 0.9 * xn)
    elif kind == "motion_rotation":
        def fn(yn, xn, i, j):
            ang = -35 + 70 * (0.5 * yn + 0.5 * xn)
            return motion_psf(support, 3.0 + 5.0 * (xn), ang)
    elif kind == "mixed_modes":
        def fn(yn, xn, i, j):
            if (i + j) % 2 == 0:
                return gaussian_psf(support, 1.1 + 1.6 * yn, 1.1 + 1.6 * (1 - xn))
            return motion_psf(support, 4.0 + 4.0 * yn, 20 + 120 * xn)
    elif kind == "checker_swap":
        def fn(yn, xn, i, j):
            return gaussian_psf(support, 2.2, 2.2,
                                cy=2.0 + 1.2 * ((i % 2) * 2 - 1),
                                cx=2.0 + 1.2 * ((j % 2) * 2 - 1))
    elif kind == "signed_demo":
        def fn(yn, xn, i, j):
            base = signed_psf(support)
            mix = gaussian_psf(support, 1.4 + 0.8 * yn, 1.4 + 0.8 * xn)
            return 0.55 * base + 0.45 * mix
    elif kind == "oversized_demo":
        def fn(yn, xn, i, j):
            return gaussian_psf(support, 0.45 * support[0] / 2,
                                0.45 * support[1] / 2)
    else:
        raise ValueError(f"unknown kernel kind {kind!r}")
    return make_control_field(image_shape, grid_shape, support, fn)


# --------------------------------------------------------------------------- samples
@dataclass
class DegradedSample:
    sharp: np.ndarray
    blurred: np.ndarray
    observed: np.ndarray          # blurred + noise, clipped
    field: KernelField
    boundary: str
    sigma: float
    meta: dict[str, Any]


def generate_sample(
    texture: str = "mixed", size: tuple[int, int] = (192, 192),
    grid_shape: tuple[int, int] = (3, 3), support: tuple[int, int] = (11, 11),
    kernel_kind: str = "gaussian_gradient", sigma: float = 0.004,
    boundary: str = "reflect", seed: int = 0,
    missing_cells: list[tuple[int, int]] | None = None,
    normalized: bool = True,
) -> DegradedSample:
    H, W = size
    if texture not in TEXTURES:
        raise ValueError(f"unknown texture {texture!r}; choose from {sorted(TEXTURES)}")
    sharp = TEXTURES[texture](W, H)
    field = make_field(kernel_kind, (H, W), grid_shape, support)
    if not normalized:
        # deliberately mis-scaled kernels to exercise the diagnostics path
        kernels = field.kernels.copy()
        kernels[field.mask] *= 1.07
        field = KernelField(kernels, field.mask, "unit_sum", field.interpolation)
    # the degradation is produced by the complete field, so the sample itself
    # stays physically well-posed; missing cells are removed afterwards to
    # model a kernel field that the restorer is handed with holes.  A support
    # that is oversized for the image is likewise a *restoration* diagnostic,
    # so the generation pass is run without the size gate.
    A = SVForward(field, (H, W), boundary=boundary, allow_signed=True,
                  strict_support=False)
    blurred = A.apply(sharp)
    if missing_cells:
        mask = field.mask.copy()
        for i, j in missing_cells:
            mask[i, j] = False
        field = KernelField(field.kernels, mask, field.normalization, field.interpolation)
    rng = np.random.default_rng(seed)
    noise = sigma * rng.standard_normal((H, W))
    observed = np.clip(blurred + noise, 0.0, 1.0)
    meta = {
        "texture": texture, "size": [H, W], "grid_shape": list(grid_shape),
        "support": list(support), "kernel_kind": kernel_kind,
        "sigma": sigma, "boundary": boundary, "seed": seed,
        "missing_cells": [list(c) for c in (missing_cells or [])],
        "normalized": normalized,
        "generator": "svdeblur.synth.generate_sample",
    }
    return DegradedSample(sharp, blurred, observed, field, boundary, sigma, meta)


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 1.0) -> float:
    mse = float(np.mean((np.clip(a, 0, peak) - np.clip(b, 0, peak)) ** 2))
    if mse <= 1e-20:
        return float("inf")
    return 10.0 * np.log10(peak ** 2 / mse)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(img) * 255.0), 0, 255).astype(np.uint8)


def save_png(path, img: np.ndarray, scale: int = 1):
    """Deterministic PNG write (no timestamps) for reproducible exports."""
    arr = to_uint8(img)
    pic = Image.fromarray(arr, mode="L")
    if scale > 1:
        pic = pic.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    pic.save(path, format="PNG", optimize=False, compress_level=6)
