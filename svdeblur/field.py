"""Kernel field: control grid of local PSFs with explicit bilinear interpolation.

The field stores one PSF per control point on a (gh, gw) grid laid over the
image rectangle.  A kernel at continuous image coordinate (y, x) is the
bilinear combination

    k(y, x) = sum_c  a_c(y, x) * k_c

with non-negative coefficients a_c that partition unity.  This is the single
interpolation rule used by the forward operator, its adjoint, the simulator
and the local-kernel viewer, so all four agree exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable

import numpy as np


INTERP_MODES = ("bilinear",)
NORMALIZATIONS = ("unit_sum", "none")
BOUNDARY_MODES = ("zero", "reflect", "symmetric", "replicate")


@dataclass
class Diagnostic:
    code: str
    severity: str  # "error" | "warning" | "info"
    message: str
    cells: list[tuple[int, int]] = dc_field(default_factory=list)


def _tent(coord: np.ndarray, n_pixels: int, n_control: int) -> np.ndarray:
    """Per-axis bilinear (tent) weights, shape (n_control, n_pixels).

    Control point c sits at t = c and pixel p maps to
    t = p * (n_control - 1) / (n_pixels - 1); weights are max(0, 1 - |t - c|),
    so they are non-negative and sum to one for every pixel.  A single control
    row degenerates to the constant weight one.
    """
    if n_control == 1:
        return np.ones((1, n_pixels), dtype=np.float64)
    t = np.asarray(coord, dtype=np.float64) * (n_control - 1) / max(n_pixels - 1, 1)
    c = np.arange(n_control, dtype=np.float64)[:, None]
    return np.maximum(0.0, 1.0 - np.abs(t[None, :] - c))


def bilinear_weights(
    points: Iterable[tuple[float, float]], grid_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, list[list[tuple[int, int, float]]]]:
    """Bilinear coefficients for arbitrary (y, x) points.

    Returns a (n_points, gh, gw) coefficient array plus a per-point list of
    (ci, cj, coefficient) for the at-most-four contributing control cells.
    """
    gh, gw = grid_shape
    H, W = image_shape
    pts = np.asarray(list(points), dtype=np.float64).reshape(-1, 2)
    coeff = np.zeros((pts.shape[0], gh, gw), dtype=np.float64)
    details: list[list[tuple[int, int, float]]] = []
    for n, (py, px) in enumerate(pts):
        wy = _tent(np.array([min(max(py, 0.0), H - 1.0)]), H, gh)[:, 0]
        wx = _tent(np.array([min(max(px, 0.0), W - 1.0)]), W, gw)[:, 0]
        nz_y = [(int(i), float(w)) for i, w in enumerate(wy) if w > 1e-12]
        nz_x = [(int(j), float(w)) for j, w in enumerate(wx) if w > 1e-12]
        terms = [(i, j, wy0 * wx0) for i, wy0 in nz_y for j, wx0 in nz_x]
        for ci, cj, w in terms:
            coeff[n, ci, cj] += w
        details.append(terms)
    return coeff, details


def _resize_kernel(k: np.ndarray, support: tuple[int, int]) -> np.ndarray:
    """Center-crop or zero-pad a kernel to a new odd support."""
    kh, kw = support
    oh, ow = k.shape
    out = np.zeros((kh, kw), dtype=np.float64)
    sy = (oh - kh) // 2
    sx = (ow - kw) // 2
    y0, y1 = max(0, sy), min(oh, sy + kh)
    x0, x1 = max(0, sx), min(ow, sx + kw)
    ty0, tx0 = y0 - sy, x0 - sx
    out[ty0:ty0 + (y1 - y0), tx0:tx0 + (x1 - x0)] = k[y0:y1, x0:x1]
    return out


@dataclass
class KernelField:
    """(gh, gw) control PSFs over an image, with missing cells allowed.

    kernels: float array (gh, gw, kh, kw); cells whose mask is False are
    ignored (their values are kept only to round-trip through the editor).
    """

    kernels: np.ndarray
    mask: np.ndarray
    normalization: str = "unit_sum"
    interpolation: str = "bilinear"
    control_positions: np.ndarray | None = None  # (gh, gw, 2) explicit pixel coords

    def __post_init__(self):
        if self.kernels.ndim != 4:
            raise ValueError("kernels must have shape (gh, gw, kh, kw)")
        gh, gw = self.kernels.shape[:2]
        if self.mask.shape != (gh, gw):
            raise ValueError("mask shape must match the control grid")
        if self.interpolation not in INTERP_MODES:
            raise ValueError(f"unknown interpolation {self.interpolation!r}")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"unknown normalization {self.normalization!r}")

    # ------------------------------------------------------------------ shapes
    @property
    def grid_shape(self) -> tuple[int, int]:
        return int(self.kernels.shape[0]), int(self.kernels.shape[1])

    @property
    def support(self) -> tuple[int, int]:
        return int(self.kernels.shape[2]), int(self.kernels.shape[3])

    @property
    def n_missing(self) -> int:
        return int((~self.mask).sum())

    def cell_positions(self, image_shape: tuple[int, int]) -> np.ndarray:
        """Continuous (y, x) position of every control point."""
        gh, gw = self.grid_shape
        H, W = image_shape
        ys = np.linspace(0, H - 1, gh) if gh > 1 else np.array([(H - 1) / 2.0])
        xs = np.linspace(0, W - 1, gw) if gw > 1 else np.array([(W - 1) / 2.0])
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        return np.stack([yy, xx], axis=-1)

    # -------------------------------------------------------------- normalizing
    def normalized_kernels(self) -> np.ndarray:
        """Copy of the kernels after the declared normalization policy."""
        out = np.array(self.kernels, dtype=np.float64, copy=True)
        if self.normalization == "none":
            return out
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if not self.mask[i, j]:
                    continue
                s = float(out[i, j].sum())
                if abs(s) <= 1e-12:
                    raise ValueError(f"kernel at cell ({i},{j}) sums to zero")
                out[i, j] /= s
        return out

    def kernel_sums(self) -> np.ndarray:
        sums = np.zeros(self.grid_shape, dtype=np.float64)
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if self.mask[i, j]:
                    sums[i, j] = self.kernels[i, j].sum()
                else:
                    sums[i, j] = np.nan
        return sums

    # ------------------------------------------------------------ interpolation
    def interpolate_at(self, y: float, x: float, image_shape: tuple[int, int],
                       normalized: bool = True) -> dict[str, Any]:
        """Kernel at a continuous pixel position, with the four cell weights."""
        coeff, details = bilinear_weights([(y, x)], self.grid_shape, image_shape)
        c = coeff[0]
        contrib = details[0]
        missing = [(i, j) for i, j, w in contrib if not self.mask[i, j]]
        kernels = self.normalized_kernels() if normalized else np.array(self.kernels)
        k = np.tensordot(c, kernels, axes=([0, 1], [0, 1]))
        missing_kernel = bool(missing) or not np.isfinite(k).all()
        return {
            "y": float(y), "x": float(x),
            "weights": [[int(i), int(j), float(w)] for i, j, w in contrib],
            "kernel": k,
            "sum": float(np.nansum(k)) if not missing_kernel else float("nan"),
            "missing": missing_kernel,
            "missing_cells": [[int(i), int(j)] for i, j in missing],
        }

    def weight_maps(self, image_shape: tuple[int, int]) -> list[dict[str, Any]]:
        """Coefficient maps a_c(y,x) for every control cell that participates.

        Returns a list of {cell, map}.  Only valid cells occur; interpolation
        onto a missing cell is treated as an error by diagnostics, never
        silently skipped here.
        """
        H, W = image_shape
        gh, gw = self.grid_shape
        wy = _tent(np.arange(H, dtype=np.float64), H, gh)
        wx = _tent(np.arange(W, dtype=np.float64), W, gw)
        maps: list[dict[str, Any]] = []
        for ci in range(gh):
            for cj in range(gw):
                if not self.mask[ci, cj]:
                    continue
                maps.append({"cell": (ci, cj), "map": np.outer(wy[ci], wx[cj])})
        return maps

    # --------------------------------------------------------------- diagnostics
    def diagnostics(
        self,
        image_shape: tuple[int, int] | None = None,
        allow_signed: bool = False,
        oversized_warn_ratio: float = 1.0 / 8.0,
        oversized_error_ratio: float = 1.0 / 4.0,
    ) -> list[Diagnostic]:
        gh, gw = self.grid_shape
        kh, kw = self.support
        out: list[Diagnostic] = []

        if gh < 2 or gw < 2:
            out.append(Diagnostic(
                "SINGLETON_GRID", "error",
                f"control grid {gh}x{gw}: a spatially varying field needs at least 2x2 cells"))
        if kh % 2 == 0 or kw % 2 == 0:
            out.append(Diagnostic(
                "EVEN_SUPPORT", "error",
                f"support {kh}x{kw} must be odd on both axes (centred PSF)"))
        if kh <= 0 or kw <= 0:
            out.append(Diagnostic("BAD_SUPPORT", "error", f"invalid support {kh}x{kw}"))

        bad_finite, negative_cells, zero_sum_cells, off_sum_cells = [], [], [], []
        for i in range(gh):
            for j in range(gw):
                if not self.mask[i, j]:
                    continue
                k = self.kernels[i, j]
                if not np.isfinite(k).all():
                    bad_finite.append((i, j))
                    continue
                s = float(k.sum())
                if s <= 0 and abs(s) <= 1e-12:
                    zero_sum_cells.append((i, j))
                elif abs(s - 1.0) > 1e-3:
                    off_sum_cells.append((i, j))
                if k.min() < -1e-9:
                    negative_cells.append((i, j))

        missing = [(i, j) for i in range(gh) for j in range(gw) if not self.mask[i, j]]
        if missing:
            out.append(Diagnostic(
                "MISSING_KERNEL", "error",
                f"{len(missing)} control cell(s) have no kernel: "
                + ", ".join(f"({i},{j})" for i, j in missing)
                + ". Fill them or remove them from the field.",
                missing))
        if bad_finite:
            out.append(Diagnostic("NONFINITE_KERNEL", "error",
                                  "non-finite kernel weights", bad_finite))
        if zero_sum_cells:
            out.append(Diagnostic("ZERO_SUM_KERNEL", "error",
                                  "kernel sums to zero; cannot normalize", zero_sum_cells))
        if negative_cells:
            sev = "warning" if allow_signed else "error"
            out.append(Diagnostic(
                "NEGATIVE_WEIGHT", sev,
                f"{len(negative_cells)} cell(s) contain negative weights (min "
                f"{min(float(self.kernels[i, j].min()) for i, j in negative_cells):.3g}). "
                + ("Signed kernels accepted explicitly; expect ringing."
                   if allow_signed else
                   "A physical PSF must be non-negative; clamp/renormalize or "
                   "enable signed kernels deliberately."),
                negative_cells))
        if off_sum_cells:
            worst = max(abs(float(self.kernels[i, j].sum()) - 1.0)
                        for i, j in off_sum_cells)
            out.append(Diagnostic(
                "KERNEL_NOT_NORMALIZED", "warning",
                f"{len(off_sum_cells)} cell(s) deviate from unit sum (worst {worst:.4f}); "
                f"the operator applies the declared '{self.normalization}' policy.",
                off_sum_cells))

        if image_shape is not None:
            H, W = image_shape
            dim = min(H, W)
            big = max(kh, kw)
            if kh > H or kw > W:
                out.append(Diagnostic(
                    "SUPPORT_LARGER_THAN_IMAGE", "error",
                    f"support {kh}x{kw} does not fit image {H}x{W}"))
            elif big >= oversized_error_ratio * dim:
                out.append(Diagnostic(
                    "SUPPORT_TOO_LARGE", "error",
                    f"support {kh}x{kw} reaches {big / dim:.1%} of the short image side "
                    f"({dim}px; limit {oversized_error_ratio:.0%}). Reduce the support."))
            elif big >= oversized_warn_ratio * dim:
                out.append(Diagnostic(
                    "SUPPORT_LARGE", "warning",
                    f"support {kh}x{kw} reaches {big / dim:.1%} of the short image side "
                    f"({dim}px); boundary extension affects a large fraction of pixels."))
        return out

    @staticmethod
    def blocking_errors(diags: Iterable[Diagnostic]) -> list[Diagnostic]:
        return [d for d in diags if d.severity == "error"]

    @staticmethod
    def summarize(diags: Iterable[Diagnostic]) -> dict[str, Any]:
        items = list(diags)
        return {
            "errors": [d.__dict__ for d in items if d.severity == "error"],
            "warnings": [d.__dict__ for d in items if d.severity == "warning"],
            "info": [d.__dict__ for d in items if d.severity == "info"],
            "ok": not any(d.severity == "error" for d in items),
        }

    # ------------------------------------------------------------- (de)serialize
    def to_dict(self) -> dict[str, Any]:
        cells = []
        for i in range(self.grid_shape[0]):
            row = []
            for j in range(self.grid_shape[1]):
                row.append(self.kernels[i, j].tolist() if self.mask[i, j] else None)
            cells.append(row)
        return {
            "grid_shape": list(self.grid_shape),
            "support": list(self.support),
            "interpolation": self.interpolation,
            "normalization": self.normalization,
            "cells": cells,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KernelField":
        gh, gw = data["grid_shape"]
        kh, kw = data["support"]
        kernels = np.zeros((gh, gw, kh, kw), dtype=np.float64)
        mask = np.zeros((gh, gw), dtype=bool)
        for i in range(gh):
            for j in range(gw):
                cell = data["cells"][i][j]
                if cell is not None:
                    arr = np.asarray(cell, dtype=np.float64)
                    if arr.shape != (kh, kw):
                        arr = _resize_kernel(arr, (kh, kw))
                    kernels[i, j] = arr
                    mask[i, j] = True
        return cls(kernels=kernels, mask=mask,
                   normalization=data.get("normalization", "unit_sum"),
                   interpolation=data.get("interpolation", "bilinear"))

    def resized(self, support: tuple[int, int]) -> "KernelField":
        kh, kw = int(support[0]), int(support[1])
        if kh == self.support[0] and kw == self.support[1]:
            return self
        kernels = np.zeros((*self.grid_shape, kh, kw), dtype=np.float64)
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                kernels[i, j] = _resize_kernel(self.kernels[i, j], (kh, kw))
        return KernelField(kernels, self.mask.copy(), self.normalization,
                           self.interpolation)

    def cleared(self, i: int, j: int) -> "KernelField":
        mask = self.mask.copy()
        mask[i, j] = False
        return KernelField(self.kernels.copy(), mask, self.normalization,
                           self.interpolation)

    def filled_missing(self) -> "KernelField":
        """Fill missing control cells from the nearest valid cell (explicit action)."""
        mask = self.mask.copy()
        kernels = self.kernels.copy()
        if mask.any():
            valid = np.argwhere(mask)
            for i, j in np.argwhere(~mask):
                d = ((valid[:, 0] - i) ** 2 + (valid[:, 1] - j) ** 2)
                src = tuple(valid[int(np.argmin(d))])
                kernels[i, j] = kernels[src]
                mask[i, j] = True
        return KernelField(kernels, mask, self.normalization, self.interpolation)


def make_control_field(
    image_shape: tuple[int, int], grid_shape: tuple[int, int],
    support: tuple[int, int], kernel_fn: Any,
    normalization: str = "unit_sum",
) -> KernelField:
    """Build a field by sampling kernel_fn(y_norm, x_norm, ci, cj) per cell."""
    gh, gw = grid_shape
    kh, kw = support
    kernels = np.zeros((gh, gw, kh, kw), dtype=np.float64)
    mask = np.ones((gh, gw), dtype=bool)
    for i in range(gh):
        for j in range(gw):
            yn = i / max(gh - 1, 1)
            xn = j / max(gw - 1, 1)
            k = np.asarray(kernel_fn(yn, xn, i, j), dtype=np.float64)
            if k.shape != (kh, kw):
                k = _resize_kernel(k, (kh, kw))
            kernels[i, j] = k
    return KernelField(kernels, mask, normalization)
