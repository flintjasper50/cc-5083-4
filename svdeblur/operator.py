"""Forward operator and its exact adjoint for a spatially varying blur field.

Model (all on a single image grid, no block splitting):

    A x = b,   A x = sum_c  m_c ⊙ (k_c ⊛ E x)

where E is a boundary extension (zero / reflect / symmetric / replicate),
k_c is the normalised PSF of control cell c, m_c the tent weight map of that
cell, and ⊙ is elementwise multiplication.  Because the tent weights partition
unity, the operator equals the pointwise interpolated kernel convolution
k(y, x) ⊛ x, but is assembled from a fixed number of grid-wide convolutions,
which keeps the adjoint exact and cheap.

The adjoint is the exact transpose

    A^T y = E^T ( sum_c  k_c^T ⊛ (m_c ⊙ y) )

with k^T the mirrored kernel; E^T is the index-fold of E, implemented by
np.add.at so that reflection/replication folds add back correctly.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .field import KernelField, Diagnostic

BoundaryMode = str

_ADJOINT_OF = {"zero": "zero", "reflect": "reflect",
               "symmetric": "symmetric", "replicate": "replicate"}


def _axis_index_map(n: int, pad: int, mode: str) -> np.ndarray:
    idx = np.arange(-pad, n + pad)
    if mode == "zero":
        return np.full_like(idx, -1)
    if mode == "replicate":
        return np.clip(idx, 0, n - 1)
    if mode == "reflect":
        period = 2 * (n - 1) if n > 1 else 1
        folded = np.mod(idx, period) if n > 1 else np.zeros_like(idx)
        return np.where(folded < n, folded, period - folded)
    if mode == "symmetric":
        period = 2 * n
        folded = np.mod(idx, period)
        return np.where(folded < n, folded, period - 1 - folded)
    raise ValueError(f"unknown boundary mode {mode!r}")


def extend(img: np.ndarray, pad: tuple[int, int], mode: str) -> np.ndarray:
    """Pad (pad_y, pad_x) with the chosen boundary rule; pad may be asymmetric."""
    py, px = pad
    if mode == "zero":
        return np.pad(img, ((py, py), (px, px)), mode="constant")
    if mode == "replicate":
        return np.pad(img, ((py, py), (px, px)), mode="edge")
    if mode == "reflect":
        return np.pad(img, ((py, py), (px, px)), mode="reflect")
    if mode == "symmetric":
        return np.pad(img, ((py, py), (px, px)), mode="symmetric")
    raise ValueError(f"unknown boundary mode {mode!r}")


def fold_extend(ext: np.ndarray, shape: tuple[int, int], mode: str) -> np.ndarray:
    """Exact adjoint of :func:`extend` (sums the overlapping boundary copies)."""
    H, W = shape
    ey, ex = ext.shape
    py, px = (ey - H) // 2, (ex - W) // 2
    if mode == "zero":
        return ext[py:py + H, px:px + W].copy()
    iy = _axis_index_map(H, py, mode)
    ix = _axis_index_map(W, px, mode)
    # fold rows/cols separately with duplicate-index accumulation
    tmp = np.zeros((H, ex), dtype=np.float64)
    np.add.at(tmp, iy, ext)
    out = np.zeros((H, W), dtype=np.float64)
    np.add.at(out, (slice(None), ix), tmp)
    return out


def mirror(k: np.ndarray) -> np.ndarray:
    return k[::-1, ::-1].copy()


def _shift_for(k: np.ndarray, which: str) -> tuple[int, int]:
    """Pixel offsets that pad between full convolution and 'same' result."""
    kh, kw = k.shape
    if which == "forward":
        return (kh - 1) // 2, (kw - 1) // 2
    return kh // 2, kw // 2


class SVForward:
    """Assembled forward operator A and its adjoint A^T for a kernel field."""

    def __init__(self, field: KernelField, image_shape: tuple[int, int],
                 boundary: str = "reflect", allow_signed: bool = False,
                 strict_support: bool = True):
        self.field = field
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        if boundary not in _ADJOINT_OF:
            raise ValueError(f"unknown boundary {boundary!r}")
        self.boundary = boundary
        self.allow_signed = allow_signed
        self.diagnostics: list[Diagnostic] = field.diagnostics(
            self.image_shape, allow_signed=allow_signed)
        self._blocking = KernelField.blocking_errors(self.diagnostics)
        # simulator use may bypass the size policy only; structural errors
        # (missing / non-finite / zero-sum) always block
        if strict_support:
            blocking = self._blocking
        else:
            blocking = [d for d in self._blocking if not d.code.startswith("SUPPORT")]
        if blocking:
            raise ValueError("; ".join(d.message for d in blocking))
        self.kernels = field.normalized_kernels()
        self.weights = field.weight_maps(self.image_shape)
        self.pad = (_shift_for(self.kernels[0, 0], "forward")[0],
                    _shift_for(self.kernels[0, 0], "forward")[1])

    # ------------------------------------------------------------------ forward
    def _conv_same(self, img_ext: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Centred convolution  out[p] = sum_u k[u] img_ext[p - u]  on the crop.

        Full convolution of the extended image and the PSF has a surplus of
        (kh-1)//2 pixels on each side; cropping that surplus yields a PSF
        whose centre stays at the same pixel, i.e. no shift is introduced.
        """
        H, W = self.image_shape
        ry, rx = self.pad
        full = fftconvolve(img_ext, k, mode="full")
        return full[2 * ry:2 * ry + H, 2 * rx:2 * rx + W]

    def apply(self, x: np.ndarray) -> np.ndarray:
        H, W = self.image_shape
        xe = extend(x, self.pad, self.boundary)
        acc = np.zeros((H, W), dtype=np.float64)
        for entry in self.weights:
            ci, cj = entry["cell"]
            acc += entry["map"] * self._conv_same(xe, self.kernels[ci, cj])
        return acc

    def apply_adjoint(self, y: np.ndarray) -> np.ndarray:
        """Exact transpose of apply().

        apply() = sum_c m_c ⊙ crop_{2r}( full( extend(x), k_c ) ).  The
        transpose embeds (m_c ⊙ y) at the crop offset, convolves with the
        mirrored kernel (adjoint of full convolution), crops to the extended
        window, sums cells, and folds the boundary extension back.
        """
        H, W = self.image_shape
        ry, rx = self.pad
        eh, ew = H + 2 * ry, W + 2 * rx
        acc_ext = np.zeros((eh, ew), dtype=np.float64)
        for entry in self.weights:
            ci, cj = entry["cell"]
            z = np.zeros((H + 4 * ry, W + 4 * rx), dtype=np.float64)
            z[2 * ry:2 * ry + H, 2 * rx:2 * rx + W] = entry["map"] * y
            full = fftconvolve(z, mirror(self.kernels[ci, cj]), mode="full")
            acc_ext += full[2 * ry:2 * ry + eh, 2 * rx:2 * rx + ew]
        return fold_extend(acc_ext, self.image_shape, self.boundary)

    __call__ = apply

    # ---------------------------------------------------------------- utilities
    def pointwise_kernel(self, y: float, x: float) -> np.ndarray:
        return self.field.interpolate_at(y, x, self.image_shape)["kernel"]

    def describe(self) -> dict:
        return {
            "image_shape": list(self.image_shape),
            "grid_shape": list(self.field.grid_shape),
            "support": list(self.field.support),
            "interpolation": self.field.interpolation,
            "normalization": self.field.normalization,
            "boundary": self.boundary,
            "n_active_cells": len(self.weights),
            "allow_signed": self.allow_signed,
        }


def apply_kernel_at(kernel: np.ndarray, x: np.ndarray, boundary: str = "reflect") -> np.ndarray:
    """Convenience: blur one image by one kernel with the same conventions."""
    pad = ((kernel.shape[0] - 1) // 2, (kernel.shape[1] - 1) // 2)
    return fftconvolve(extend(x, pad, boundary), kernel, mode="same")
