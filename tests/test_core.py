"""Numerical verification of the SVDB operator and solver.

Run:  python -m pytest tests -q      (or  python tests/test_core.py)
"""

from __future__ import annotations

import numpy as np
import pytest

from svdeblur.field import KernelField, bilinear_weights, make_control_field
from svdeblur.operator import SVForward, extend, fold_extend, mirror
from svdeblur.solve import SolveConfig, TVSolver, tv_gradient, _grad
from svdeblur.synth import generate_sample, psnr

BOUNDARIES = ["zero", "reflect", "symmetric", "replicate"]


def gauss(kh, kw, sy, sx, cy=None, cx=None):
    cy = (kh - 1) / 2 if cy is None else cy
    cx = (kw - 1) / 2 if cx is None else cx
    yy, xx = np.mgrid[0:kh, 0:kw]
    g = np.exp(-((yy - cy) ** 2 / (2 * sy * sy) + (xx - cx) ** 2 / (2 * sx * sx)))
    return g / g.sum()


def varying_field(H, W, grid=(3, 3), support=(7, 7)):
    return make_control_field(
        (H, W), grid, support,
        lambda yn, xn, i, j: gauss(support[0], support[1],
                                   0.8 + 1.4 * yn, 0.8 + 1.4 * xn,
                                   cy=2.6 + 0.8 * (yn - 0.5), cx=2.6 + 0.8 * (xn - 0.5)))


# ---------------------------------------------------------------- interpolation
def test_tent_weights_partition_unity():
    H, W = 37, 43
    for gh, gw in [(2, 2), (3, 5), (6, 2)]:
        f = varying_field(H, W, (gh, gw))
        maps = f.weight_maps((H, W))
        total = np.zeros((H, W))
        for m in maps:
            assert (m["map"] >= -1e-15).all()
            total += m["map"]
        assert np.allclose(total, 1.0, atol=1e-12), (gh, gw)


def test_pointwise_interpolation_matches_weight_maps():
    H, W, grid = 29, 31, (3, 4)
    f = varying_field(H, W, grid)
    coeff, _ = bilinear_weights([(7.3, 11.8), (0, 0), (H - 1, W - 1)], grid, (H, W))
    maps = {(m["cell"]): m["map"] for m in f.weight_maps((H, W))}
    k = f.normalized_kernels()
    # pointwise coefficients must equal the weight maps at integer pixels
    yix, xix = 7, 11
    expected = sum(maps[(i, j)][yix, xix] * k[i, j]
                   for i in range(grid[0]) for j in range(grid[1]))
    got = f.interpolate_at(yix, xix, (H, W))["kernel"]
    assert np.allclose(got, expected, atol=1e-12)
    assert np.allclose(coeff.sum(axis=(1, 2)), 1.0, atol=1e-12)


# ---------------------------------------------------------------------- adjoint
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_adjoint_is_exact_for_every_boundary(boundary):
    H, W = 33, 39
    A = SVForward(varying_field(H, W, (3, 4)), (H, W), boundary=boundary)
    rng = np.random.default_rng(5)
    x, y = rng.standard_normal((H, W)), rng.standard_normal((H, W))
    lhs = float(np.sum(A.apply(x) * y))
    rhs = float(np.sum(x * A.apply_adjoint(y)))
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), 1.0)


@pytest.mark.parametrize("mode", BOUNDARIES)
def test_boundary_pad_adjoint_folds_correctly(mode):
    H, W, pad = 17, 21, 4
    rng = np.random.default_rng(9)
    x = rng.standard_normal((H, W))
    ext = rng.standard_normal((H + 2 * pad, W + 2 * pad))
    lhs = float(np.sum(extend(x, (pad, pad), mode) * ext))
    rhs = float(np.sum(x * fold_extend(ext, (H, W), mode)))
    assert abs(lhs - rhs) < 1e-11 * max(abs(lhs), 1.0)


# ----------------------------------------------------- operator = local kernels
def test_operator_equals_pointwise_local_kernel_convolution():
    """A impulse response must equal the interpolated local PSF, pixel by pixel."""
    H, W = 41, 45
    f = varying_field(H, W, (3, 3), (7, 7))
    A = SVForward(f, (H, W), boundary="zero")
    m = (H // 2, W // 2)
    imp = np.zeros((H, W))
    imp[m] = 1.0
    out = A.apply(imp)
    kh, kw = f.support
    r = (kh // 2, kw // 2)
    worst = 0.0
    for p in range(H):
        for q in range(W):
            dy, dx = p - m[0], q - m[1]
            if abs(dy) <= r[0] and abs(dx) <= r[1]:
                k = f.interpolate_at(p, q, (H, W))["kernel"]
                worst = max(worst, abs(out[p, q] - k[dy + r[0], dx + r[1]]))
    assert worst < 1e-12, worst


def test_uniform_field_matches_single_convolution():
    """A field with identical cells reduces exactly to one ordinary convolution."""
    H, W = 31, 27
    k = gauss(5, 5, 0.9, 1.1)
    f = KernelField(np.tile(k, (2, 2, 1, 1)), np.ones((2, 2), bool))
    A = SVForward(f, (H, W), boundary="reflect")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((H, W))
    out = A.apply(x)
    from scipy.signal import fftconvolve
    ref = fftconvolve(extend(x, (2, 2), "reflect"), k, mode="full")[4:4 + H, 4:4 + W]
    assert np.allclose(out, ref, atol=1e-12)


# --------------------------------------------------------------- normalization
def test_unit_sum_normalization_and_diagnostics():
    H, W = 32, 32
    f = varying_field(H, W)
    assert np.allclose(f.normalized_kernels().sum(axis=(2, 3)), 1.0)

    # mis-scaled kernels: warning, not silently renormalised by the report
    bad = KernelField(f.kernels * 1.07, f.mask)
    codes = {d.code for d in bad.diagnostics((H, W))}
    assert "KERNEL_NOT_NORMALIZED" in codes
    assert "NEGATIVE_WEIGHT" not in codes

    # negative weights are blocking unless explicitly allowed
    signed = f.kernels.copy()
    signed[0, 0, 0, 0] = -0.05
    sf = KernelField(signed, f.mask)
    normal = {d.code: d.severity for d in sf.diagnostics((H, W), allow_signed=False)}
    relaxed = {d.code: d.severity for d in sf.diagnostics((H, W), allow_signed=True)}
    assert normal.get("NEGATIVE_WEIGHT") == "error"
    assert relaxed.get("NEGATIVE_WEIGHT") == "warning"


def test_zero_sum_kernel_is_blocking():
    H, W = 16, 16
    f = varying_field(H, W, (2, 2), (5, 5))
    k = f.kernels.copy()
    k[0, 0] = 0.0
    cf = KernelField(k, f.mask)
    codes = {d.code for d in cf.diagnostics((H, W))}
    assert "ZERO_SUM_KERNEL" in codes


# ------------------------------------------------------------------ diagnostics
def test_missing_kernel_diagnostic_and_blocking_operator():
    H, W = 24, 24
    f = varying_field(H, W, (3, 3))
    f = f.cleared(1, 1)
    diags = f.diagnostics((H, W))
    assert any(d.code == "MISSING_KERNEL" and d.severity == "error" for d in diags)
    with pytest.raises(ValueError):
        SVForward(f, (H, W))


def test_oversized_support_diagnostic():
    f = varying_field(64, 64, (2, 2), (41, 41))
    codes = {d.code for d in f.diagnostics((64, 64))}
    assert "SUPPORT_TOO_LARGE" in codes

    small = varying_field(256, 256, (2, 2), (41, 41))
    codes_small = {d.code for d in small.diagnostics((256, 256))}
    assert "SUPPORT_TOO_LARGE" not in codes_small


def test_even_support_and_singleton_grid_diagnostics():
    f = varying_field(20, 20, (1, 3), (4, 4))
    codes = {d.code for d in f.diagnostics((20, 20))}
    assert "EVEN_SUPPORT" in codes
    assert "SINGLETON_GRID" in codes


# ------------------------------------------------------------ edge handling
def test_edge_residual_is_comparable_to_interior():
    """The declared reflect boundary must not create an edge artefact ring."""
    s = generate_sample(texture="bars", size=(128, 128), grid_shape=(2, 2),
                        support=(9, 9), kernel_kind="gaussian_gradient", sigma=0.0, seed=0)
    H, W = s.sharp.shape
    A = SVForward(s.field, (H, W), boundary="reflect")
    r = A.apply(s.sharp) - s.blurred
    dist = np.minimum(np.minimum(np.arange(H)[:, None], np.arange(H)[::-1][:, None]),
                      np.minimum(np.arange(W)[None, :], np.arange(W)[::-1][None, :]))
    edge = float(np.sqrt(np.mean(r[dist < 3] ** 2)))
    inner = float(np.sqrt(np.mean(r[dist >= 12] ** 2)))
    assert edge < 1e-10 and inner < 1e-10


# ------------------------------------------------------------------- solver
def _run(cfg, sample, allow_signed=False):
    A = SVForward(sample.field, sample.sharp.shape, boundary="reflect",
                  allow_signed=allow_signed)
    return TVSolver(A, cfg, reference=sample.observed).run(sample.observed)


def test_solver_improves_psnr_and_never_increases_tv_to_blowup():
    s = generate_sample(texture="text", size=(128, 128), grid_shape=(2, 2),
                        support=(7, 7), kernel_kind="gaussian_gradient", sigma=0.003, seed=1)
    cfg = SolveConfig(iters=250, sigma=0.003, tol=1e-9)
    r = _run(cfg, s)
    assert psnr(r.image, s.sharp) > psnr(s.observed, s.sharp) + 2.0
    assert r.residuals[-1] < r.residuals[0]
    assert r.stop_reason in {"noise_floor", "rel_residual", "semi_convergence", "max_iter"}


def test_more_iterations_do_not_blow_up_noise():
    """Iteration budget must not change the guarded output: it stops at the floor."""
    s = generate_sample(texture="rings", size=(96, 96), grid_shape=(2, 2),
                        support=(7, 7), kernel_kind="mixed_modes", sigma=0.005, seed=2)
    outs = {}
    for iters in (150, 400, 900):
        cfg = SolveConfig(iters=iters, sigma=0.005, tol=1e-9)
        r = _run(cfg, s)
        outs[iters] = r
        assert r.stop_reason in {"noise_floor", "semi_convergence", "rel_residual"}
    # guarded stops at the same iteration regardless of the budget
    stops = {r.stop_iteration for r in outs.values()}
    assert len(stops) == 1, stops
    imgs = [r.image for r in outs.values()]
    assert np.allclose(imgs[0], imgs[1], atol=1e-9)
    assert np.allclose(imgs[1], imgs[2], atol=1e-9)


def test_noise_floor_stopping_is_recorded():
    s = generate_sample(texture="bars", size=(128, 128), grid_shape=(2, 2),
                        support=(9, 9), kernel_kind="gaussian_gradient", sigma=0.006, seed=3)
    r = _run(SolveConfig(iters=400, sigma=0.006, tol=1e-9), s)
    assert r.stop_reason == "noise_floor"
    assert r.residual_abs[r.stop_iteration - 1] <= r.noise_floor * (1 + 1e-9)
    assert "further iterations would fit noise" in r.stop_detail


def test_semi_convergence_guard_stops_before_noise_growth():
    """Tight noise floor: the residual turns upward and the guard stops there."""
    s = generate_sample(texture="bars", size=(96, 96), grid_shape=(2, 2),
                        support=(9, 9), kernel_kind="gaussian_gradient", sigma=0.006, seed=3)
    r = _run(SolveConfig(iters=400, sigma=0.006, tol=1e-9, noise_floor_factor=0.8), s)
    assert r.stop_reason == "semi_convergence"
    assert r.stop_detail
    assert r.residual_abs[-1] >= min(r.residual_abs)


def test_tv_gradient_is_adjoint_of_grad():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((13, 17))
    py, px = rng.standard_normal((2, 13, 17))
    gy, gx = _grad(x)
    lhs = float(np.sum(gy * py + gx * px))
    from svdeblur.solve import _grad_adjoint
    rhs = float(np.sum(x * _grad_adjoint(py, px)))
    assert abs(lhs - rhs) < 1e-12 * max(abs(lhs), 1.0)


# ------------------------------------------------------------------ reproducibility
def test_sample_generation_is_deterministic():
    a = generate_sample(texture="text", size=(64, 64), grid_shape=(2, 2),
                        support=(7, 7), kernel_kind="motion_rotation", sigma=0.004, seed=11)
    b = generate_sample(texture="text", size=(64, 64), grid_shape=(2, 2),
                        support=(7, 7), kernel_kind="motion_rotation", sigma=0.004, seed=11)
    assert np.array_equal(a.observed, b.observed)
    assert np.array_equal(a.field.kernels, b.field.kernels)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
