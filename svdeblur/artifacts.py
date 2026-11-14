"""Atomic artwork writers used by the release runner.

Every file is written to a sibling temporary name and then ``os.replace``-d,
so a crash cannot leave a half-written PNG that a downstream reader could
mistake for a finished restoration.  No timestamps are embedded, keeping
artefact bytes deterministic for a fixed input.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .field import KernelField
from .synth import to_uint8

# Residual display: signed residual mapped around mid-grey so sign and
# structure stay visible.  Scale is declared so the report can invert it.
RESIDUAL_VIS_GAIN = 4.0


def atomic_png(path: Path, img: np.ndarray, scale: int = 1) -> None:
    arr = to_uint8(img)
    pic = Image.fromarray(arr, mode="L")
    if scale > 1:
        pic = pic.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    tmp = path.with_name(f".{path.name}.tmp")
    pic.save(tmp, format="PNG", optimize=False, compress_level=6)
    os.replace(tmp, path)


def atomic_json(path: Path, body: dict) -> None:
    import json
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def residual_visual(residual: np.ndarray) -> np.ndarray:
    return np.clip(0.5 + residual * RESIDUAL_VIS_GAIN, 0.0, 1.0)


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def kernel_sheet(field: KernelField, shape: tuple[int, int], path: Path,
                 title: str) -> None:
    gh, gw = field.grid_shape
    kh, kw = field.support
    cell = max(48, kh * 4)
    pad = 18
    width = gw * cell + 2 * pad + 190
    height = gh * cell + 2 * pad + 44
    pic = Image.new("L", (width, height), 24)
    d = ImageDraw.Draw(pic)
    d.text((pad, 8), title, fill=235)
    pos = field.cell_positions(shape)
    normalised = field.normalized_kernels()
    for i in range(gh):
        for j in range(gw):
            x0, y0 = pad + j * cell, 34 + i * cell
            if not field.mask[i, j]:
                d.rectangle([x0, y0, x0 + cell - 8, y0 + cell - 8], outline=110)
                d.line([x0, y0, x0 + cell - 8, y0 + cell - 8], fill=110)
                d.text((x0 + 4, y0 + cell // 2 - 6), "missing", fill=180)
                continue
            k = normalised[i, j]
            lo, hi = float(k.min()), float(k.max())
            vis = np.clip((k - lo) / max(hi - lo, 1e-9), 0, 1)
            pic_k = Image.fromarray(to_uint8(vis)).resize((cell - 8, cell - 8),
                                                          Image.NEAREST)
            pic.paste(pic_k, (x0, y0))
            d.text((x0 + 2, y0 + cell - 20),
                   f"({int(pos[i, j, 0])},{int(pos[i, j, 1])}) s={k.sum():.3f}",
                   fill=200)
    tx = pad + gw * cell + 10
    d.text((tx, 34), "cell -> sampled", fill=210)
    d.text((tx, 50), "at image (y,x)", fill=170)
    d.text((tx, 76), "s = kernel sum", fill=170)
    d.text((tx, 102), "bilinear tent", fill=170)
    d.text((tx, 118), "weights between", fill=170)
    d.text((tx, 134), "control points", fill=170)
    atomic_png(path, np.asarray(pic, dtype=np.float64) / 255.0)


def curves_png(residuals: list[float], objectives: list[float], tvs: list[float],
               stop_iteration: int, path: Path, title: str) -> None:
    W, H = 720, 300
    pic = Image.new("L", (W, H), 18)
    d = ImageDraw.Draw(pic)
    d.text((12, 8), title, fill=235)

    def panel(series, label, top, height):
        d.rectangle([40, top, W - 14, top + height], outline=60)
        lo, hi = min(series), max(series)
        if hi <= lo:
            hi = lo + 1.0
        n = len(series)
        for i in range(n - 1):
            x0 = 40 + (W - 54) * i / max(n - 1, 1)
            x1 = 40 + (W - 54) * (i + 1) / max(n - 1, 1)
            v0 = height - height * (series[i] - lo) / (hi - lo)
            v1 = height - height * (series[i + 1] - lo) / (hi - lo)
            d.line([x0, top + v0, x1, top + v1], fill=225, width=1)
        d.text((4, top + height // 2 - 6), label, fill=200)
        sx = 40 + (W - 54) * (stop_iteration - 1) / max(n - 1, 1)
        d.line([sx, top, sx, top + height], fill=245)
        d.text((min(sx + 4, W - 70), top + 2), f"stop {stop_iteration}", fill=245)

    panel(residuals, "res", 34, 96)
    panel(objectives, "obj", 150, 96)
    panel(tvs, "tv", 266, 34)
    d.text((40, 268), "vertical bar = declared stop iteration", fill=200)
    atomic_png(path, np.asarray(pic, dtype=np.float64) / 255.0)
